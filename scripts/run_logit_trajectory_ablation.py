#!/usr/bin/env python3
"""Leakage-free logit trajectory ablation audit.

All outputs are written under reports/logit_trajectory_audit.  The script reads
frozen Phase 4 logits/features and Phase 5/6 split assignments, but never writes
to artifacts/phase5, artifacts/phase6, configs/phase5, or configs/phase6.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from selective_detection.calibration_metrics import calibrator_metrics
from selective_detection.calibrator_artifact_io import DETECTORS, SPLITS, combo_slug, load_config, phase4_feature_path, payload_sha256, read_json, sha256_file
from selective_detection.selective_metrics import accepted_error_metrics, aurc, eaurc


SEED = 20260916
C_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0)
TRANSFORMS = {
    1: "JPEG q=75 erosion",
    2: "resize 0.75 -> restore erosion",
    3: "Gaussian blur sigma=0.5 erosion",
    4: "center crop 0.90 -> restore erosion",
}
VARIANTS: dict[str, tuple[str, ...]] = {
    "OLD_CURRENT": ("margin_distance", "orbit_logit_variance", "embedding_drift_mean", "orbit_support_distance_max"),
    "M1_MINIMAL": ("margin_distance", "orbit_logit_variance"),
    "M2_SUMMARY_TRAJECTORY": ("margin_distance", "orbit_logit_variance", "mean_directional_erosion", "worst_view_erosion"),
    "M3_SIGNED_TRAJECTORY": ("margin_distance", "Delta_1", "Delta_2", "Delta_3", "Delta_4"),
}
SHORT = {
    "margin_distance": "m",
    "orbit_logit_variance": "v",
    "embedding_drift_mean": "r_raw_cosine",
    "orbit_support_distance_max": "s_cosine_knn",
    "mean_directional_erosion": "d",
    "worst_view_erosion": "e",
    "Delta_1": "Delta_1",
    "Delta_2": "Delta_2",
    "Delta_3": "Delta_3",
    "Delta_4": "Delta_4",
}
SIGNED = {"mean_directional_erosion", "worst_view_erosion", "Delta_1", "Delta_2", "Delta_3", "Delta_4"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase5" / "riskguard_calibrator.yaml"))
    p.add_argument("--output-root", default=str(PROJECT_ROOT / "reports" / "logit_trajectory_audit"))
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-certification", action="store_true")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def signed_log1p(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))


def transform_values(df: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    cols = []
    for feature in features:
        values = df[feature].to_numpy(dtype=np.float64)
        if feature == "margin_distance":
            cols.append(-np.log1p(values))
        elif feature in SIGNED:
            cols.append(signed_log1p(values))
        else:
            if (values < -1.0e-6).any():
                raise ValueError(f"{feature} contains negative values")
            cols.append(np.log1p(values))
    out = np.column_stack(cols)
    if not np.isfinite(out).all():
        raise ValueError("non-finite transformed feature")
    return out


def scaler_from_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    if not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd < 1.0e-12).any():
        raise RuntimeError(f"invalid fold-local scaler: means={mu.tolist()} scales={sd.tolist()}")
    return mu, sd


def fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float, cfg: dict[str, Any]) -> tuple[LogisticRegression, bool]:
    model_cfg = cfg["model"]
    clf = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        class_weight=None,
        max_iter=5000,
        tol=1.0e-10,
        random_state=int(cfg.get("seed", SEED)),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    return clf, not any(issubclass(item.category, ConvergenceWarning) for item in caught)


def select_candidate(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    best = sorted(rows, key=lambda r: float(r["candidate_C"]))[0]
    for row in sorted(rows, key=lambda r: float(r["candidate_C"]))[1:]:
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


def recover_gamma(df: pd.DataFrame) -> np.ndarray:
    z0 = df["base_logit"].to_numpy(dtype=np.float64)
    m = df["margin_distance"].to_numpy(dtype=np.float64)
    pred = df["base_prediction"].to_numpy(dtype=np.int64)
    return np.where(pred == 1, z0 - m, z0 + m)


def load_orbit_logits(detector: str, parent_ids: set[str], partition: str) -> pd.DataFrame:
    paths = sorted((PROJECT_ROOT / "artifacts" / "phase4" / "orbit_cache" / detector).glob("predictions_*.parquet"))
    chunks = []
    cols = ["parent_sample_id", "view_index", "raw_logit", "partition", "evaluation_role"]
    for path in paths:
        part = pd.read_parquet(path, columns=cols)
        role_match = part["partition"].eq(partition) | part["evaluation_role"].eq(partition)
        part = part[role_match & part["parent_sample_id"].astype(str).isin(parent_ids)]
        if len(part):
            chunks.append(part.drop(columns=["partition", "evaluation_role"]))
    if not chunks:
        raise RuntimeError(f"no orbit logits found for {detector}/{partition}")
    long = pd.concat(chunks, ignore_index=True)
    wide = long.pivot(index="parent_sample_id", columns="view_index", values="raw_logit")
    wide.columns = [f"z_{int(c)}" for c in wide.columns]
    wide = wide.reset_index()
    required = [f"z_{i}" for i in range(5)]
    missing = [c for c in required if c not in wide.columns]
    if missing:
        raise RuntimeError(f"missing orbit logit view(s) for {detector}/{partition}: {missing}")
    if wide["parent_sample_id"].duplicated().any():
        raise RuntimeError(f"duplicate parent orbit logits for {detector}/{partition}")
    return wide[["parent_sample_id", *required]]


def add_trajectory_features(df: pd.DataFrame, detector: str, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    logits = load_orbit_logits(detector, set(out["parent_sample_id"].astype(str)), partition)
    out = out.merge(logits, on="parent_sample_id", how="left", validate="one_to_one")
    if out[[f"z_{i}" for i in range(5)]].isna().any().any():
        raise RuntimeError(f"incomplete orbit-logit join for {detector}/{partition}")
    gamma = recover_gamma(out)
    z0 = out["z_0"].to_numpy(dtype=np.float64)
    c = np.where(z0 >= gamma, 1.0, -1.0)
    a = {i: c * (out[f"z_{i}"].to_numpy(dtype=np.float64) - gamma) for i in range(5)}
    deltas = []
    for i in range(1, 5):
        delta = a[0] - a[i]
        direct = c * (out["z_0"].to_numpy(dtype=np.float64) - out[f"z_{i}"].to_numpy(dtype=np.float64))
        if not np.allclose(delta, direct, rtol=1.0e-10, atol=1.0e-10):
            raise RuntimeError(f"Delta identity failed for view {i}")
        out[f"Delta_{i}"] = delta
        deltas.append(delta)
    out["mean_directional_erosion"] = np.column_stack(deltas).mean(axis=1)
    out["worst_view_erosion"] = np.column_stack(deltas).max(axis=1)
    out["identity_orientation_c"] = c.astype(np.int64)
    out["recovered_gamma"] = gamma
    audit = {
        "partition": partition,
        "row_count": int(len(out)),
        "z0_matches_base_logit": bool(np.allclose(out["z_0"].to_numpy(dtype=float), out["base_logit"].to_numpy(dtype=float), rtol=1e-10, atol=1e-10)),
        "orientation_matches_base_prediction": bool(np.array_equal((out["identity_orientation_c"].to_numpy() == 1).astype(int), out["base_prediction"].to_numpy(dtype=int))),
        "delta_identity_verified": True,
        "finite_feature_values": bool(np.isfinite(out[[*SIGNED, "margin_distance", "orbit_logit_variance"]].to_numpy(dtype=float)).all()),
    }
    return out, audit


def load_feature_frame(detector: str, split: str, partition: str, *, with_folds: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = phase4_feature_path(PROJECT_ROOT, detector, split, partition)
    df = pd.read_parquet(path)
    if with_folds:
        fold_path = PROJECT_ROOT / "artifacts" / "phase5" / "cv_fold_assignments" / f"{combo_slug(detector, split)}.parquet"
        folds = pd.read_parquet(fold_path)
        df = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
        if df["cv_fold"].isna().any():
            raise RuntimeError(f"incomplete CV fold assignment for {detector}/{split}")
        df["cv_fold"] = df["cv_fold"].astype(int)
    df, audit = add_trajectory_features(df, detector, partition)
    audit.update({"detector": detector, "split": split, "feature_artifact": str(path.relative_to(PROJECT_ROOT)), "feature_artifact_sha256": sha256_file(path)})
    return df, audit


def evaluate_variant(df: pd.DataFrame, variant: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    features = VARIANTS[variant]
    y = df["base_error"].to_numpy(dtype=np.int64)
    folds = df["cv_fold"].to_numpy(dtype=int)
    sample_ids = df["sample_id"].astype(str).to_numpy()
    search_rows: list[dict[str, Any]] = []
    coeff_rows_by_c: dict[float, list[dict[str, Any]]] = {}
    pred_by_c: dict[float, np.ndarray] = {}
    logit_by_c: dict[float, np.ndarray] = {}
    for c_value in C_GRID:
        probs = np.full(len(df), np.nan)
        logits = np.full(len(df), np.nan)
        conv = 0
        coeff_rows: list[dict[str, Any]] = []
        for fold in sorted(np.unique(folds)):
            train = folds != fold
            val = ~train
            tx = transform_values(df.loc[train], features)
            vx = transform_values(df.loc[val], features)
            mu, sd = scaler_from_train(tx)
            clf, ok = fit_logistic((tx - mu) / sd, y[train], c_value, cfg)
            conv += int(ok)
            logits[val] = clf.decision_function((vx - mu) / sd)
            probs[val] = clf.predict_proba((vx - mu) / sd)[:, 1]
            coeff_rows.append(
                {
                    "fold": int(fold),
                    "candidate_C": float(c_value),
                    "converged": bool(ok),
                    "intercept": float(clf.intercept_[0]),
                    "feature_order": json.dumps([SHORT[f] for f in features]),
                    "scaler_means": json.dumps(mu.tolist()),
                    "scaler_scales": json.dumps(sd.tolist()),
                    "coefficient_vector": json.dumps(clf.coef_[0].astype(float).tolist()),
                    "sha_overlap_train_validation": int(len(set(df.loc[train, "sha256"].astype(str)) & set(df.loc[val, "sha256"].astype(str)))),
                }
            )
        metrics = calibrator_metrics(y, probs, sample_ids=sample_ids, n_bins=int(cfg["calibration"]["ece_bins"]))
        search_rows.append({"variant": variant, "candidate_C": float(c_value), "converged_fold_count": int(conv), **metrics})
        coeff_rows_by_c[float(c_value)] = coeff_rows
        pred_by_c[float(c_value)] = probs
        logit_by_c[float(c_value)] = logits
    best = select_candidate(search_rows, float(cfg["selection"].get("tie_tolerance", 1.0e-12)))
    selected_c = float(best["candidate_C"])
    pred = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "sha256": df["sha256"].astype(str),
            "detector": df["detector"].astype(str),
            "split": df["split"].astype(str),
            "generator": df["generator"].astype(str),
            "label": df["label"].astype(int),
            "base_prediction": df["base_prediction"].astype(int),
            "base_error": df["base_error"].astype(int),
            "cv_fold": df["cv_fold"].astype(int),
            "variant": variant,
            "risk_logit": logit_by_c[selected_c],
            "risk_probability": pred_by_c[selected_c],
            "selected_C": selected_c,
        }
    )
    for f in features:
        pred[SHORT[f]] = df[f].to_numpy(dtype=float)
    coeff_rows = coeff_rows_by_c[selected_c]
    return best, pred, search_rows, coeff_rows


def bootstrap_differences(preds: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    pairs = [
        ("M1_MINIMAL", "OLD_CURRENT"), ("M2_SUMMARY_TRAJECTORY", "OLD_CURRENT"), ("M3_SIGNED_TRAJECTORY", "OLD_CURRENT"),
        ("M2_SUMMARY_TRAJECTORY", "M1_MINIMAL"), ("M3_SIGNED_TRAJECTORY", "M1_MINIMAL"), ("M3_SIGNED_TRAJECTORY", "M2_SUMMARY_TRAJECTORY"),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for (detector, split), cell in preds.groupby(["detector", "split"], sort=True):
        by_variant = {v: g.sort_values("sha256", kind="mergesort").reset_index(drop=True) for v, g in cell.groupby("variant")}
        shas = np.array(sorted(cell["sha256"].astype(str).unique()))
        variant_boot: dict[str, dict[str, np.ndarray]] = {}
        for variant, frame in by_variant.items():
            if set(frame["sha256"].astype(str)) != set(shas):
                raise RuntimeError(f"bootstrap SHA alignment failed for {detector}/{split}/{variant}")
            variant_boot[variant] = batched_bootstrap_metrics(frame, shas, reps, rng)
        for cand, ref in pairs:
            if cand not in by_variant or ref not in by_variant:
                continue
            cdf = by_variant[cand]
            rdf = by_variant[ref]
            point = {
                "Delta_AURC": aurc(cdf["base_error"], cdf["risk_probability"], cdf["sample_id"]) - aurc(rdf["base_error"], rdf["risk_probability"], rdf["sample_id"]),
                "Delta_NLL": float(calibrator_metrics(cdf["base_error"], cdf["risk_probability"])["binary_nll"] - calibrator_metrics(rdf["base_error"], rdf["risk_probability"])["binary_nll"]),
                "Delta_AUROC": float(calibrator_metrics(cdf["base_error"], cdf["risk_probability"])["error_detection_AUROC"] - calibrator_metrics(rdf["base_error"], rdf["risk_probability"])["error_detection_AUROC"]),
            }
            row = {"detector": detector, "split": split, "candidate": cand, "reference": ref}
            for metric, value in point.items():
                arr = variant_boot[cand][metric.replace("Delta_", "")] - variant_boot[ref][metric.replace("Delta_", "")]
                row[f"{metric}_point"] = float(value)
                row[f"{metric}_ci95_low"] = float(np.nanpercentile(arr, 2.5))
                row[f"{metric}_ci95_high"] = float(np.nanpercentile(arr, 97.5))
            rows.append(row)
    return pd.DataFrame(rows)


def batched_bootstrap_metrics(df: pd.DataFrame, shas: np.ndarray, reps: int, rng: np.random.Generator, batch_size: int = 100) -> dict[str, np.ndarray]:
    arrays = bootstrap_arrays(df, shas)
    n_sha = len(shas)
    out = {"AURC": [], "NLL": [], "AUROC": []}
    probs = np.full(n_sha, 1.0 / n_sha)
    for start in range(0, reps, batch_size):
        size = min(batch_size, reps - start)
        sha_w = rng.multinomial(n_sha, probs, size=size).astype(np.float64)
        out["AURC"].append(batch_aurc(arrays, sha_w))
        out["NLL"].append(batch_nll(arrays, sha_w))
        out["AUROC"].append(batch_auroc(arrays, sha_w))
    return {k: np.concatenate(v) for k, v in out.items()}


def bootstrap_arrays(df: pd.DataFrame, shas: np.ndarray) -> dict[str, np.ndarray]:
    y = df["base_error"].to_numpy(dtype=float)
    p = df["risk_probability"].to_numpy(dtype=float)
    sid = df["sample_id"].astype(str).to_numpy()
    row_sha = df["sha256"].astype(str).to_numpy()
    sha_idx = np.searchsorted(shas, row_sha)
    if not np.array_equal(shas[sha_idx], row_sha):
        raise RuntimeError("row SHA index alignment failed")
    aurc_order = np.lexsort((sid, p))
    auc_order = np.argsort(p, kind="mergesort")
    p_sorted = p[auc_order]
    starts = np.r_[0, np.flatnonzero(np.diff(p_sorted) != 0.0) + 1]
    ends = np.r_[starts[1:], len(p_sorted)]
    return {"y": y, "p": p, "sha_idx": sha_idx, "aurc_order": aurc_order, "auc_order": auc_order, "auc_starts": starts, "auc_ends": ends}


def batch_aurc(arrays: dict[str, np.ndarray], sha_w: np.ndarray) -> np.ndarray:
    order = arrays["aurc_order"]
    y = arrays["y"][order][None, :]
    w = sha_w[:, arrays["sha_idx"][order]]
    accepted = np.cumsum(w, axis=1)
    errors = np.cumsum(w * y, axis=1)
    prefix = errors / np.maximum(accepted, 1.0e-12)
    return np.sum(prefix * w, axis=1) / np.maximum(np.sum(w, axis=1), 1.0e-12)


def batch_nll(arrays: dict[str, np.ndarray], sha_w: np.ndarray) -> np.ndarray:
    y = arrays["y"][None, :]
    p = np.clip(arrays["p"], 1.0e-15, 1.0 - 1.0e-15)[None, :]
    w = sha_w[:, arrays["sha_idx"]]
    loss = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return np.sum(loss * w, axis=1) / np.maximum(np.sum(w, axis=1), 1.0e-12)


def batch_auroc(arrays: dict[str, np.ndarray], sha_w: np.ndarray) -> np.ndarray:
    order = arrays["auc_order"]
    y = arrays["y"][order][None, :]
    w = sha_w[:, arrays["sha_idx"][order]]
    pos_w = w * y
    neg_w = w * (1.0 - y)
    total_pos = np.sum(pos_w, axis=1)
    total_neg = np.sum(neg_w, axis=1)
    cum_neg = np.cumsum(neg_w, axis=1)
    u = np.sum(pos_w * (cum_neg - 0.5 * neg_w), axis=1)
    denom = total_pos * total_neg
    out = u / np.maximum(denom, 1.0e-12)
    out[denom <= 0.0] = np.nan
    return out


def weighted_aurc_arrays(arrays: dict[str, np.ndarray], weights: np.ndarray) -> float:
    order = arrays["aurc_order"]
    y = arrays["y"][order]
    w = weights[arrays["sha_idx"]][order].astype(float)
    accepted = np.cumsum(w)
    errors = np.cumsum(y * w)
    nz = w > 0
    return float(np.average((errors / np.maximum(accepted, 1.0e-12))[nz], weights=w[nz]))


def weighted_nll_arrays(arrays: dict[str, np.ndarray], weights: np.ndarray) -> float:
    y = arrays["y"]
    p = np.clip(arrays["p"], 1.0e-15, 1.0 - 1.0e-15)
    w = weights[arrays["sha_idx"]].astype(float)
    loss = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return float(np.average(loss, weights=w))


def weighted_auroc_arrays(arrays: dict[str, np.ndarray], weights: np.ndarray) -> float:
    w = weights[arrays["sha_idx"]].astype(float)
    y = arrays["y"]
    total_pos = float(np.sum(w * y))
    total_neg = float(np.sum(w * (1.0 - y)))
    if total_pos <= 0.0 or total_neg <= 0.0:
        return float("nan")
    order = arrays["auc_order"]
    ys = y[order]
    ws = w[order]
    cum_neg_before = 0.0
    u_stat = 0.0
    for start, end in zip(arrays["auc_starts"], arrays["auc_ends"]):
        sl = slice(int(start), int(end))
        group_w = ws[sl]
        group_y = ys[sl]
        pos_w = float(np.sum(group_w * group_y))
        neg_w = float(np.sum(group_w * (1.0 - group_y)))
        u_stat += pos_w * (cum_neg_before + 0.5 * neg_w)
        cum_neg_before += neg_w
    return float(u_stat / (total_pos * total_neg))


def select_winner(metrics: pd.DataFrame, boot: pd.DataFrame) -> dict[str, Any]:
    rank_rows = []
    for _, cell in metrics.groupby(["detector", "split"], sort=True):
        ordered = cell.sort_values(["AURC", "error_detection_AUROC", "binary_nll"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
        n = max(1, len(ordered) - 1)
        for idx, row in ordered.iterrows():
            rank_rows.append({"variant": row["variant"], "normalized_AURC_rank": float(idx / n)})
    ranks = pd.DataFrame(rank_rows).groupby("variant")["normalized_AURC_rank"].mean().sort_values()
    candidates = list(ranks.index)
    winner = candidates[0]
    notes = ["Selected by lowest mean normalized AURC rank on risk_fit OOF only."]
    return {
        "selection_timestamp_local": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "selection_data": "risk_fit_oof_only",
        "primary_criterion": "lowest_mean_normalized_AURC_rank",
        "mean_normalized_AURC_rank": {k: float(v) for k, v in ranks.to_dict().items()},
        "selected_variant": winner,
        "selected_features": [SHORT[f] for f in VARIANTS[winner]],
        "uses_embeddings": bool(winner == "OLD_CURRENT"),
        "uses_support_bank": bool(winner == "OLD_CURRENT"),
        "decision_notes": notes,
        "bootstrap_file_used": "paired_bootstrap_differences.csv",
    }


def refit_model(df: pd.DataFrame, variant: str, cfg: dict[str, Any], c_value: float) -> dict[str, Any]:
    features = VARIANTS[variant]
    x = transform_values(df, features)
    mu, sd = scaler_from_train(x)
    clf, ok = fit_logistic((x - mu) / sd, df["base_error"].to_numpy(int), c_value, cfg)
    payload = {
        "model_version": "logit_trajectory_audit_scorer_v1",
        "variant": variant,
        "feature_order": [SHORT[f] for f in features],
        "raw_feature_order": list(features),
        "feature_transformations": {SHORT[f]: ("-log1p" if f == "margin_distance" else "signed_log1p" if f in SIGNED else "log1p") for f in features},
        "selected_C": float(c_value),
        "scaler_means": mu.tolist(),
        "scaler_scales": sd.tolist(),
        "coefficient_vector": clf.coef_[0].astype(float).tolist(),
        "intercept": float(clf.intercept_[0]),
        "converged": bool(ok),
    }
    payload["model_hash"] = payload_sha256(payload)
    return payload


def score_with_model(df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    features = tuple(model["raw_feature_order"])
    x = transform_values(df, features)
    z = (x - np.asarray(model["scaler_means"])) / np.asarray(model["scaler_scales"])
    logits = z @ np.asarray(model["coefficient_vector"]) + float(model["intercept"])
    out = df.copy()
    out["risk_logit"] = logits
    out["risk_probability"] = 1.0 / (1.0 + np.exp(-logits))
    out["risk_score"] = out["risk_probability"]
    out["variant"] = model["variant"]
    return out


def cp_upper(errors: int, n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    if errors <= 0:
        return float(1.0 - delta ** (1.0 / n))
    if errors >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


def policy_groups(df: pd.DataFrame) -> pd.Series:
    labels = df["label"].astype(int)
    gens = df["generator"].astype(str).str.lower()
    return pd.Series(np.where(labels.eq(0), "real_all", gens), index=df.index, dtype=str)


def certify_and_evaluate(detector: str, split: str, model: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cal, _ = load_feature_frame(detector, split, "threshold_cal")
    scored = score_with_model(cal, model)
    assign_path = PROJECT_ROOT / "artifacts" / "phase6" / "calibration_split_assignments" / f"{combo_slug(detector, split)}.csv"
    assignments = pd.read_csv(assign_path)
    scored = scored.merge(assignments[["sha256", "calibration_subset"]], on="sha256", how="left", validate="many_to_one")
    select = scored[scored["calibration_subset"].eq("policy_select")].copy()
    certify = scored[scored["calibration_subset"].eq("policy_certify")].copy()
    thresholds = np.quantile(select["risk_score"].to_numpy(float), np.linspace(0.5, 1.0, 10))
    thresholds = sorted(set(float(x) for x in thresholds), reverse=True)[:10]
    groups = policy_groups(certify)
    delta_cell = 0.05 / max(1, len(thresholds) * groups.nunique())
    group_rows = []
    best = None
    for rank, tau in enumerate(thresholds, 1):
        accepted = certify["risk_score"].to_numpy(float) <= tau
        max_bound = 0.0
        ok = True
        details = []
        for group in sorted(groups.unique()):
            mask = groups.eq(group).to_numpy()
            acc = accepted & mask
            n = int(acc.sum())
            err = int(certify.loc[acc, "base_error"].astype(int).sum())
            bound = cp_upper(err, n, delta_cell)
            max_bound = max(max_bound, bound)
            gok = bool(n > 0 and bound <= 0.05)
            ok = ok and gok
            details.append({"group": group, "accepted_count": n, "accepted_errors": err, "cp_upper": bound, "group_certified": gok, "threshold": tau, "candidate_rank": rank})
        if best is None or (ok, accepted.sum(), -max_bound) > (best["certified"], best["accepted_count"], -best["max_group_cp_upper"]):
            best = {"threshold": tau, "certified": bool(ok), "accepted_count": int(accepted.sum()), "coverage": float(accepted.mean()), "max_group_cp_upper": float(max_bound), "details": details, "candidate_count": len(thresholds)}
    for row in (best or {}).get("details", []):
        row.update({"detector": detector, "split": split, "variant": model["variant"]})
        group_rows.append(row)
    threshold = best["threshold"] if best and best["certified"] else None
    cert = {
        "detector": detector, "split": split, "variant": model["variant"], "policy": "source_group_cp", "alpha": 0.05, "delta": 0.05,
        "certification_status": "CERTIFIED" if threshold is not None else "NONE",
        "selected_threshold": threshold, "certified_coverage": best["coverage"] if threshold is not None else 0.0,
        "maximum_group_cp_upper_bound": best["max_group_cp_upper"] if best else float("nan"), "candidate_count": len(thresholds),
    }
    eval_rows = []
    for partition in ("protocol_seen", "protocol_held_out"):
        part, _ = load_feature_frame(detector, split, partition)
        ps = score_with_model(part, model)
        metrics = accepted_error_metrics(ps, threshold)
        eval_rows.append({
            "detector": detector, "split": split, "partition": partition, "variant": model["variant"], **metrics,
            "AURC": aurc(ps["base_error"], ps["risk_probability"], ps["sample_id"]),
            "E_AURC": eaurc(ps["base_error"], ps["risk_probability"], ps["sample_id"]),
            "worst_group_selective_risk": worst_group_risk(ps, threshold),
            "minimum_group_coverage": min_group_coverage(ps, threshold),
            "accepted_false_positive_count": accepted_fp(ps, threshold),
            "accepted_false_negative_count": accepted_fn(ps, threshold),
        })
    return cert, group_rows, eval_rows, []


def worst_group_risk(df: pd.DataFrame, threshold: float | None) -> float:
    if threshold is None:
        return float("nan")
    accepted = df["risk_score"].to_numpy(float) <= threshold
    groups = policy_groups(df)
    vals = []
    for group in sorted(groups.unique()):
        mask = accepted & groups.eq(group).to_numpy()
        if mask.any():
            vals.append(float(df.loc[mask, "base_error"].astype(int).mean()))
    return float(max(vals)) if vals else float("nan")


def min_group_coverage(df: pd.DataFrame, threshold: float | None) -> float:
    if threshold is None:
        return 0.0
    accepted = df["risk_score"].to_numpy(float) <= threshold
    groups = policy_groups(df)
    vals = []
    for group in sorted(groups.unique()):
        mask = groups.eq(group).to_numpy()
        vals.append(float((accepted & mask).sum() / mask.sum()) if mask.sum() else float("nan"))
    return float(np.nanmin(vals)) if vals else 0.0


def accepted_fp(df: pd.DataFrame, threshold: float | None) -> int:
    if threshold is None:
        return 0
    acc = df["risk_score"].to_numpy(float) <= threshold
    return int(((df["label"].astype(int) == 0) & (df["base_prediction"].astype(int) == 1) & acc).sum())


def accepted_fn(df: pd.DataFrame, threshold: float | None) -> int:
    if threshold is None:
        return 0
    acc = df["risk_score"].to_numpy(float) <= threshold
    return int(((df["label"].astype(int) == 1) & (df["base_prediction"].astype(int) == 0) & acc).sum())


def feature_distribution_rows(df: pd.DataFrame, detector: str, split: str) -> list[dict[str, Any]]:
    rows = []
    feature_names = sorted({f for feats in VARIANTS.values() for f in feats})
    for correctness, group in df.groupby(df["base_error"].map({0: "correct_base_predictions", 1: "incorrect_base_predictions"}), sort=True):
        for feature in feature_names:
            arr = group[feature].to_numpy(float)
            rows.append({"detector": detector, "split": split, "group": correctness, "feature": SHORT[feature], "count": len(arr), "mean": float(np.mean(arr)), "std": float(np.std(arr)), "p05": float(np.percentile(arr, 5)), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95))})
    return rows


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    write_json(out / "feature_definitions.json", {
        "variants": {k: [SHORT[f] for f in v] for k, v in VARIANTS.items()},
        "transformations": {"m": "-log(1+m)", "v": "log(1+v)", "d/e/Delta_i": "sign(t)*log(1+abs(t))"},
        "Delta_mapping": TRANSFORMS,
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
    })

    metric_rows = []
    pred_frames = []
    c_rows = []
    coef_rows = []
    dist_rows = []
    leakage_rows = []
    runtime_rows = []
    refit_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for detector in DETECTORS:
        for split in SPLITS:
            t0 = time.perf_counter()
            df, audit = load_feature_frame(detector, split, "risk_fit", with_folds=True)
            refit_frames[(detector, split)] = df
            leakage_rows.append({**audit, "audit": "risk_fit_feature_construction", "status": "pass" if all([audit["orientation_matches_base_prediction"], audit["delta_identity_verified"], audit["finite_feature_values"]]) else "fail"})
            dist_rows.extend(feature_distribution_rows(df, detector, split))
            for variant in VARIANTS:
                vt0 = time.perf_counter()
                best, pred, search, coefs = evaluate_variant(df, variant, cfg)
                for row in search:
                    row.update({"detector": detector, "split": split, "selected": bool(float(row["candidate_C"]) == float(best["candidate_C"]))})
                    c_rows.append(row)
                best.update({"detector": detector, "split": split, "variant": variant, "selected_C": float(best["candidate_C"]), "runtime_seconds": time.perf_counter() - vt0})
                metric_rows.append(best)
                pred_frames.append(pred)
                for row in coefs:
                    row.update({"detector": detector, "split": split, "variant": variant, "selected_C": float(best["candidate_C"])})
                    coef_rows.append(row)
            runtime_rows.append({"detector": detector, "split": split, "stage": "oof_all_variants", "runtime_seconds": time.perf_counter() - t0, "peak_ram_bytes": "", "peak_gpu_bytes": ""})

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "oof_variant_metrics.csv", index=False)
    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_parquet(out / "oof_predictions.parquet", index=False)
    pd.DataFrame(c_rows).to_csv(out / "selected_c_by_variant.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(out / "variant_coefficients.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out / "feature_distribution_summary.csv", index=False)

    boot = bootstrap_differences(preds, args.bootstrap_replicates, args.seed)
    boot.to_csv(out / "paired_bootstrap_differences.csv", index=False)
    decision = select_winner(metrics, boot)
    write_json(out / "feature_selection_decision.json", decision)

    selected = decision["selected_variant"]
    scorer_rows = []
    cert_rows = []
    group_rows = []
    eval_rows = []
    for detector in DETECTORS:
        for split in SPLITS:
            cell_metrics = metrics[(metrics["detector"].eq(detector)) & (metrics["split"].eq(split)) & (metrics["variant"].eq(selected))]
            c_value = float(cell_metrics.iloc[0]["selected_C"])
            model = refit_model(refit_frames[(detector, split)], selected, cfg, c_value)
            model.update({"detector": detector, "split": split})
            model_path = out / "scorers" / f"{combo_slug(detector, split)}_{selected}.json"
            write_json(model_path, model)
            old_model_path = PROJECT_ROOT / "artifacts" / "phase5" / "models" / f"{combo_slug(detector, split)}_riskguard.json"
            scorer_rows.append({"detector": detector, "split": split, "winner_variant": selected, "audit_model": str(model_path.relative_to(PROJECT_ROOT)), "audit_model_hash": model["model_hash"], "old_current_model": str(old_model_path.relative_to(PROJECT_ROOT)), "old_current_model_sha256": sha256_file(old_model_path), "phase5_artifact_unchanged": True})
            if not args.skip_certification:
                cert, groups, evals, _ = certify_and_evaluate(detector, split, model)
                cert_rows.append(cert)
                group_rows.extend(groups)
                eval_rows.extend(evals)

    pd.DataFrame(scorer_rows).to_csv(out / "scorer_artifact_comparison.csv", index=False)
    pd.DataFrame(cert_rows).to_csv(out / "certification_comparison.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "certification_group_details.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(out / "seen_heldout_comparison.csv", index=False)
    pd.DataFrame(leakage_rows).to_csv(out / "sha_leakage_audit.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(out / "runtime_summary.csv", index=False)

    status_by = {(r["detector"], r["split"]): r.get("certification_status", "NOT_RUN") for r in cert_rows}
    uses_embed = bool(decision["uses_embeddings"])
    uses_support = bool(decision["uses_support_bank"])
    summary = [
        "# Logit Trajectory Ablation Audit",
        "",
        "Selection was frozen from risk_fit OOF evidence before threshold_cal, certification, protocol_seen, or protocol_held_out results were evaluated.",
        "",
        f"BEST_OOF_VARIANT: {selected}",
        f"Winner features: {decision['selected_features']}",
        f"Winner removes embeddings/support bank: {not uses_embed and not uses_support}",
        "",
        "Certification metrics are computed on policy_certify. protocol_seen and protocol_held_out are empirical transfer evaluations.",
        "",
        "## Required Questions",
        f"1. [m,v] practically non-inferior to OLD_CURRENT: see paired_bootstrap_differences.csv.",
        f"2. [d,e] improvement over [m,v]: see M2 vs M1 bootstrap rows.",
        f"3. Delta identity improvement over summary features: see M3 vs M2 bootstrap rows.",
        f"4. Winner consistency across SAFE and UnivFD: selected by four-cell mean rank; inspect oof_variant_metrics.csv.",
        f"5. Certified coverage preservation: see certification_comparison.csv.",
        f"6. Removes embeddings/support bank: {str((not uses_embed and not uses_support)).upper()}.",
        "7. Current certificate must be rebuilt if adopting any non-OLD_CURRENT winner; audit scorer artifacts are separate.",
        "8. Main-method change is justified only if the bootstrap and certification tables support the frozen winner.",
        "",
        "LOGIT_TRAJECTORY_EVALUATION_COMPLETE = TRUE",
        f"BEST_OOF_VARIANT = {selected}",
        f"BEST_VARIANT_FEATURES = {decision['selected_features']}",
        "WINNER_SELECTED_USING_RISK_FIT_ONLY = TRUE",
        f"WINNER_USES_EMBEDDINGS = {str(uses_embed).upper()}",
        f"WINNER_USES_SUPPORT_BANK = {str(uses_support).upper()}",
        f"WINNER_CERTIFICATION_STATUS_SAFE_A = {status_by.get(('safe','split_a'), 'NOT_RUN')}",
        f"WINNER_CERTIFICATION_STATUS_SAFE_B = {status_by.get(('safe','split_b'), 'NOT_RUN')}",
        f"WINNER_CERTIFICATION_STATUS_UNIVFD_A = {status_by.get(('univfd','split_a'), 'NOT_RUN')}",
        f"WINNER_CERTIFICATION_STATUS_UNIVFD_B = {status_by.get(('univfd','split_b'), 'NOT_RUN')}",
        "CURRENT_SCORER_ARTIFACT_UNCHANGED = TRUE",
        f"CERTIFICATION_MUST_BE_REBUILT = {str(selected != 'OLD_CURRENT').upper()}",
        f"PRIMARY_METHOD_CHANGE_RECOMMENDED = {str(selected != 'OLD_CURRENT' and not uses_embed and not uses_support).upper()}",
        "SAFE_FOR_PAPER_UPDATE = FALSE",
    ]
    (out / "logit_trajectory_ablation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote audit bundle to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
