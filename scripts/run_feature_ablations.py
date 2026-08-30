#!/usr/bin/env python3
"""Run required Phase 5 ablation OOF experiments using the fixed CV folds."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from selective_detection.error_probability_calibrator import PRIMARY_FEATURES, transform_features
from selective_detection.calibration_metrics import calibrator_metrics
from selective_detection.calibrator_artifact_io import (
    DETECTORS,
    SPLITS,
    combo_slug,
    load_config,
    phase4_feature_path,
    sha256_file,
    verify_frozen_inputs,
    write_default_config,
)


ABLATIONS: dict[str, tuple[str, ...]] = {
    "full_four": PRIMARY_FEATURES,
    "no_margin": ("orbit_logit_variance", "embedding_drift_mean", "orbit_support_distance_max"),
    "no_variance": ("margin_distance", "embedding_drift_mean", "orbit_support_distance_max"),
    "no_drift": ("margin_distance", "orbit_logit_variance", "orbit_support_distance_max"),
    "no_support": ("margin_distance", "orbit_logit_variance", "embedding_drift_mean"),
    "margin_only": ("margin_distance",),
    "variance_only": ("orbit_logit_variance",),
    "drift_only": ("embedding_drift_mean",),
    "support_only": ("orbit_support_distance_max",),
    "orbit_only": ("orbit_logit_variance", "embedding_drift_mean"),
    "geometry_support": ("margin_distance", "orbit_support_distance_max"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=[*DETECTORS, "all"], default="all")
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase5" / "riskguard_calibrator.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "artifacts" / "phase5"))
    return parser.parse_args()


def selected(values: tuple[str, ...], requested: str) -> tuple[str, ...]:
    return values if requested == "all" else (requested,)


def scaler_from_train(transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = transformed.mean(axis=0)
    scales = transformed.std(axis=0, ddof=0)
    if (scales < 1.0e-12).any():
        raise RuntimeError(f"feature standard deviation below 1e-12: {scales.tolist()}")
    return means, scales


def fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float, config: dict[str, Any]) -> tuple[LogisticRegression, bool]:
    model_cfg = config["model"]
    clf = LogisticRegression(
        C=float(c_value),
        penalty=model_cfg["penalty"],
        solver=model_cfg["solver"],
        fit_intercept=bool(model_cfg["fit_intercept"]),
        class_weight=model_cfg["class_weight"],
        max_iter=int(model_cfg["max_iter"]),
        tol=float(model_cfg["tolerance"]),
        random_state=int(config["seed"]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return clf, converged


def select_candidate(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    candidates = sorted(rows, key=lambda r: float(r["candidate_C"]))
    best = candidates[0]
    for row in candidates[1:]:
        better = False
        for metric in ("NLL", "Brier", "AURC"):
            delta = float(row[metric]) - float(best[metric])
            if delta < -tolerance:
                better = True
                break
            if abs(delta) > tolerance:
                break
        else:
            if float(row["candidate_C"]) < float(best["candidate_C"]):
                better = True
        if better:
            best = row
    return best


def run_ablation(
    df: pd.DataFrame,
    feature_order: tuple[str, ...],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_count = int(config["cross_validation"]["folds"])
    y = df["base_error"].to_numpy(dtype=np.int64)
    sample_ids = df["sample_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for c_value in [float(c) for c in config["regularization"]["candidate_C"]]:
        probabilities = np.full(len(df), np.nan, dtype=np.float64)
        converged_count = 0
        for fold in range(fold_count):
            train_mask = df["cv_fold"].to_numpy() != fold
            val_mask = ~train_mask
            train_t = transform_features(df.loc[train_mask, list(feature_order)], feature_order, as_frame=False)
            means, scales = scaler_from_train(train_t)
            train_z = (train_t - means) / scales
            val_t = transform_features(df.loc[val_mask, list(feature_order)], feature_order, as_frame=False)
            val_z = (val_t - means) / scales
            clf, converged = fit_logistic(train_z, y[train_mask], c_value, config)
            converged_count += int(converged)
            probabilities[val_mask] = clf.predict_proba(val_z)[:, 1]
        metrics = calibrator_metrics(y, probabilities, sample_ids=sample_ids, n_bins=int(config["calibration"]["ece_bins"]))
        rows.append(
            {
                "candidate_C": c_value,
                "fold_count": fold_count,
                "converged_fold_count": converged_count,
                "row_count": int(len(df)),
                "error_count": int(y.sum()),
                "NLL": metrics["binary_nll"],
                "Brier": metrics["brier_score"],
                "ECE": metrics["ece"],
                "error_detection_AUROC": metrics["error_detection_AUROC"],
                "error_detection_AUPR": metrics["error_detection_AUPR"],
                "AURC": metrics["AURC"],
                "E_AURC": metrics["E_AURC"],
                "mean_predicted_risk": metrics["mean_predicted_risk"],
                "observed_error_prevalence": metrics["observed_error_prevalence"],
            }
        )
    best = select_candidate(rows, float(config["selection"]["tie_tolerance"]))
    return best, rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    config_path = Path(args.config)
    if args.force or not config_path.exists():
        write_default_config(PROJECT_ROOT)
    verify_frozen_inputs(PROJECT_ROOT, output_root, raise_on_fail=True)
    config = load_config(config_path)
    metric_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    for detector in selected(DETECTORS, args.detector):
        for split in selected(SPLITS, args.split):
            slug = combo_slug(detector, split)
            df = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"))
            folds = pd.read_parquet(output_root / "cv_fold_assignments" / f"{slug}.parquet")
            fold_hash = sha256_file(output_root / "cv_fold_assignments" / f"{slug}.parquet")
            df = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
            full_metrics: dict[str, Any] | None = None
            combo_results: list[dict[str, Any]] = []
            for ablation, features in ABLATIONS.items():
                selected_row, _ = run_ablation(df, features, config)
                row = {
                    "detector": detector,
                    "split": split,
                    "ablation": ablation,
                    "feature_order_json": jsonable_feature_order(features),
                    "selected_C": float(selected_row["candidate_C"]),
                    **selected_row,
                }
                combo_results.append(row)
                registry_rows.append(
                    {
                        "detector": detector,
                        "split": split,
                        "ablation": ablation,
                        "feature_order_json": jsonable_feature_order(features),
                        "feature_count": int(len(features)),
                        "selected_C": float(selected_row["candidate_C"]),
                        "fold_assignment_sha256": fold_hash,
                        "uses_risk_fit_only": True,
                        "converged_fold_count": int(selected_row["converged_fold_count"]),
                        "status": "pass" if int(selected_row["converged_fold_count"]) == int(config["cross_validation"]["folds"]) else "fail",
                    }
                )
                if ablation == "full_four":
                    full_metrics = row
            if full_metrics is None:
                raise RuntimeError("missing full_four ablation")
            for row in combo_results:
                for metric in ("NLL", "Brier", "ECE", "error_detection_AUROC", "error_detection_AUPR", "AURC", "E_AURC"):
                    row[f"delta_{metric}_from_full_four"] = float(row[metric]) - float(full_metrics[metric])
                metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(output_root / "ablation_oof_metrics.csv", index=False)
    pd.DataFrame(registry_rows).to_csv(output_root / "ablation_model_registry.csv", index=False)
    failures = [row for row in registry_rows if row["status"] != "pass"]
    if failures:
        raise SystemExit(f"ablation convergence failure: {failures[:5]}")


def jsonable_feature_order(features: tuple[str, ...]) -> str:
    import json

    return json.dumps(list(features), sort_keys=False)


if __name__ == "__main__":
    main()
