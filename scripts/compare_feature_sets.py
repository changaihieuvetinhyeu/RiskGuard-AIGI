#!/usr/bin/env python3
"""Focused M2/M3/M4 OOF ablation with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from selective_detection.calibration_metrics import calibrator_metrics
from selective_detection.calibrator_artifact_io import DETECTORS, SPLITS, combo_slug, load_config, phase4_feature_path


MODELS: dict[str, tuple[str, ...]] = {
    "M2": ("margin_distance", "orbit_logit_variance"),
    "M3": ("margin_distance", "orbit_logit_variance", "embedding_drift_mean"),
    "M4": ("margin_distance", "orbit_logit_variance", "embedding_drift_mean", "orbit_support_distance_max"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase5" / "riskguard_calibrator.yaml"))
    parser.add_argument("--phase5-root", default=str(PROJECT_ROOT / "artifacts" / "phase5"))
    parser.add_argument("--phase6-root", default=str(PROJECT_ROOT / "artifacts" / "phase6"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "reports" / "feature_audit" / "m2_m3_m4"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def transform_frame(df: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    values = df.loc[:, list(features)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < -1.0e-6).any():
        raise ValueError(f"non-finite or negative feature in {features}")
    out = np.log1p(values)
    for idx, name in enumerate(features):
        if name == "margin_distance":
            out[:, idx] *= -1.0
    return out


def scaler_from_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    scales = x.std(axis=0, ddof=0)
    if (scales < 1.0e-12).any():
        raise RuntimeError(f"feature standard deviation below 1e-12: {scales.tolist()}")
    return means, scales


def fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float, cfg: dict[str, Any]) -> tuple[LogisticRegression, bool]:
    model_cfg = cfg["model"]
    clf = LogisticRegression(
        C=float(c_value),
        penalty=model_cfg["penalty"],
        solver=model_cfg["solver"],
        fit_intercept=bool(model_cfg["fit_intercept"]),
        class_weight=model_cfg["class_weight"],
        max_iter=int(model_cfg["max_iter"]),
        tol=float(model_cfg["tolerance"]),
        random_state=int(cfg["seed"]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    return clf, not any(issubclass(item.category, ConvergenceWarning) for item in caught)


def select_candidate(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    candidates = sorted(rows, key=lambda r: float(r["candidate_C"]))
    best = candidates[0]
    for row in candidates[1:]:
        for metric in ("binary_nll", "brier_score", "AURC"):
            delta = float(row[metric]) - float(best[metric])
            if delta < -tolerance:
                best = row
                break
            if abs(delta) > tolerance:
                break
        else:
            if float(row["candidate_C"]) < float(best["candidate_C"]):
                best = row
    return best


def evaluate_model(df: pd.DataFrame, features: tuple[str, ...], cfg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    y = df["base_error"].to_numpy(dtype=np.int64)
    folds = df["cv_fold"].to_numpy(dtype=int)
    sample_ids = df["sample_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    predictions: dict[float, np.ndarray] = {}
    for c_value in [float(c) for c in cfg["regularization"]["candidate_C"]]:
        probs = np.full(len(df), np.nan, dtype=np.float64)
        converged = 0
        for fold in sorted(np.unique(folds)):
            train = folds != fold
            val = ~train
            train_t = transform_frame(df.loc[train], features)
            means, scales = scaler_from_train(train_t)
            train_z = (train_t - means) / scales
            val_z = (transform_frame(df.loc[val], features) - means) / scales
            clf, ok = fit_logistic(train_z, y[train], c_value, cfg)
            converged += int(ok)
            probs[val] = clf.predict_proba(val_z)[:, 1]
        metrics = calibrator_metrics(y, probs, sample_ids=sample_ids, n_bins=int(cfg["calibration"]["ece_bins"]))
        rows.append({"candidate_C": c_value, "converged_fold_count": converged, **metrics})
        predictions[c_value] = probs
    best = select_candidate(rows, float(cfg["selection"]["tie_tolerance"]))
    return best, predictions[float(best["candidate_C"])]


def aurc(errors: np.ndarray, risks: np.ndarray, ids: np.ndarray) -> float:
    order = np.lexsort((ids.astype(str), risks))
    sorted_errors = errors[order].astype(np.float64)
    return float((np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1)).mean())


def metric_value(metric: str, y: np.ndarray, probs: np.ndarray, ids: np.ndarray) -> float:
    if metric == "AURC":
        return aurc(y, probs, ids)
    if metric == "AUROC":
        return float(roc_auc_score(y, probs))
    if metric == "NLL":
        return float(log_loss(y, np.clip(probs, 1.0e-15, 1.0 - 1.0e-15), labels=[0, 1]))
    raise ValueError(metric)


def paired_bootstrap(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    ids: np.ndarray,
    *,
    metric: str,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    point = metric_value(metric, y, left, ids) - metric_value(metric, y, right, ids)
    samples = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        idx = rng.integers(0, n, size=n)
        samples[rep] = metric_value(metric, y[idx], left[idx], ids[idx]) - metric_value(metric, y[idx], right[idx], ids[idx])
    return {
        "point_delta_left_minus_right": float(point),
        "ci_lower_2p5": float(np.quantile(samples, 0.025)),
        "ci_upper_97p5": float(np.quantile(samples, 0.975)),
        "bootstrap_replicates": int(reps),
        "bootstrap_seed": int(seed),
    }


def certification_lookup(phase6_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = phase6_root / "certified_threshold_registry.csv"
    cert = pd.read_csv(path)
    cert = cert[(cert["method"] == "riskguard") & (cert["alpha"] == 0.05) & (cert["policy"] == "source_group_cp")]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in cert.to_dict("records"):
        out[(str(row["detector"]), str(row["split"]))] = {
            "certified_status": row["policy_status"],
            "certified_coverage": row["overall_certification_coverage"],
            "max_group_cp_bound": row["max_group_cp_upper"],
        }
    return out


def main() -> None:
    args = parse_args()
    started = time.time()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(args.config))
    certs = certification_lookup(Path(args.phase6_root))
    metric_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str, str], np.ndarray] = {}

    for detector in DETECTORS:
        for split in SPLITS:
            slug = combo_slug(detector, split)
            print(f"[{detector}/{split}] loading risk_fit", flush=True)
            df = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"))
            folds = pd.read_parquet(Path(args.phase5_root) / "cv_fold_assignments" / f"{slug}.parquet")
            df = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
            df["cv_fold"] = df["cv_fold"].astype(int)
            y = df["base_error"].to_numpy(dtype=np.int64)
            ids = df["sample_id"].astype(str).to_numpy()
            for model_name, features in MODELS.items():
                print(f"[{detector}/{split}] fitting {model_name}", flush=True)
                best, probs = evaluate_model(df, features, cfg)
                predictions[(detector, split, model_name)] = probs
                cert = certs.get((detector, split), {})
                if model_name != "M4":
                    cert = {"certified_status": "NOT_REBUILT", "certified_coverage": np.nan, "max_group_cp_bound": np.nan}
                metric_rows.append(
                    {
                        "detector": detector,
                        "split": split,
                        "model": model_name,
                        "features": json.dumps(list(features), sort_keys=False),
                        "selected_C": best["candidate_C"],
                        "AURC": best["AURC"],
                        "AUROC": best["error_detection_AUROC"],
                        "NLL": best["binary_nll"],
                        **cert,
                        "row_count": best["row_count"],
                        "error_count": best["error_count"],
                        "converged_fold_count": best["converged_fold_count"],
                    }
                )
                pd.DataFrame(metric_rows).to_csv(output_root / "m2_m3_m4_metrics.csv", index=False)
            for left, right in [("M3", "M2"), ("M4", "M3"), ("M4", "M2"), ("M2", "M4"), ("M3", "M4")]:
                for metric in ["AURC", "AUROC", "NLL"]:
                    print(f"[{detector}/{split}] bootstrap {left}-{right} {metric}", flush=True)
                    stats = paired_bootstrap(
                        y,
                        predictions[(detector, split, left)],
                        predictions[(detector, split, right)],
                        ids,
                        metric=metric,
                        reps=args.bootstrap_replicates,
                        seed=args.seed,
                    )
                    boot_rows.append({"detector": detector, "split": split, "left_model": left, "right_model": right, "metric": metric, **stats})
                    pd.DataFrame(boot_rows).to_csv(output_root / "m2_m3_m4_paired_bootstrap.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    boots = pd.DataFrame(boot_rows)
    metrics.to_csv(output_root / "m2_m3_m4_metrics.csv", index=False)
    boots.to_csv(output_root / "m2_m3_m4_paired_bootstrap.csv", index=False)
    summary = [
        "# M2/M3/M4 Focused Ablation",
        "",
        f"Runtime seconds: {time.time() - started:.3f}",
        f"Bootstrap replicates: {args.bootstrap_replicates}",
        "OOF folds: existing Phase 5 SHA-disjoint risk_fit folds.",
        "Certification fields: frozen Phase 6 riskguard alpha=0.05 source_group_cp for unchanged M4 only; M2/M3 are NOT_REBUILT.",
        "",
        "Files:",
        "- `m2_m3_m4_metrics.csv`",
        "- `m2_m3_m4_paired_bootstrap.csv`",
    ]
    (output_root / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
