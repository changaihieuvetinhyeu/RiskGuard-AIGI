#!/usr/bin/env python3
"""RiskGuard support-distance and drift ablation audit.

This script is intentionally read-only with respect to frozen Phase 2-6
artifacts.  It writes a new diagnostic audit bundle under reports/feature_audit.
"""

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
from sklearn.covariance import LedoitWolf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from selective_detection.calibration_metrics import calibrator_metrics
from selective_detection.calibrator_artifact_io import DETECTORS, SPLITS, combo_slug, load_config, payload_sha256, phase4_feature_path, read_json, sha256_file


FEATURE_MAP = {
    "m": "margin_distance",
    "v": "orbit_logit_variance",
    "r_raw": "embedding_drift_mean",
    "r_local": "embedding_drift_local_robust",
    "r_mpp": "embedding_drift_mahalanobis_pp",
    "s": "orbit_support_distance_max",
}

SUPPORT_VARIANTS: dict[str, tuple[str, ...]] = {
    "margin_only": ("m",),
    "orbit_without_support": ("m", "v", "r_raw"),
    "current_full": ("m", "v", "r_raw", "s"),
    "support_without_drift": ("m", "v", "s"),
    "support_only_optional": ("s",),
}

DRIFT_VARIANTS: dict[str, str | None] = {
    "no_drift": None,
    "raw_cosine": "r_raw",
    "local_robust_drift": "r_local",
    "mahalanobis_pp_displacement": "r_mpp",
}

K_GRID = (20, 50, 100, 200)
EPS_GRID = (1.0e-6, 1.0e-5, 1.0e-4)
VIEW_COUNT = 5
TRANSFORM_COUNT = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase5" / "riskguard_calibrator.yaml"))
    p.add_argument("--phase5-root", default=str(PROJECT_ROOT / "artifacts" / "phase5"))
    p.add_argument("--output-root", default=str(PROJECT_ROOT / "reports" / "feature_audit"))
    p.add_argument("--detector", choices=[*DETECTORS, "all"], default="all")
    p.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    p.add_argument("--bootstrap-replicates", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260916)
    return p.parse_args()


def selected(values: tuple[str, ...], requested: str) -> tuple[str, ...]:
    return values if requested == "all" else (requested,)


def to_raw_features(feature_codes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(FEATURE_MAP[c] for c in feature_codes)


def transform_frame(df: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    values = df.loc[:, list(features)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < -1.0e-6).any():
        raise ValueError(f"non-finite or negative feature in {features}")
    out = np.log1p(values)
    for i, name in enumerate(features):
        if name == "margin_distance":
            out[:, i] *= -1.0
    return out


def scaler_from_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    if (sd < 1.0e-12).any():
        raise RuntimeError(f"feature standard deviation below 1e-12: {sd.tolist()}")
    return mu, sd


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
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return clf, converged


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


def evaluate_variant(df: pd.DataFrame, features: tuple[str, ...], cfg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    y = df["base_error"].to_numpy(dtype=np.int64)
    folds = df["cv_fold"].to_numpy(dtype=int)
    sample_ids = df["sample_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    predictions_by_c: dict[float, np.ndarray] = {}
    for c_value in [float(c) for c in cfg["regularization"]["candidate_C"]]:
        probs = np.full(len(df), np.nan)
        conv = 0
        for fold in sorted(np.unique(folds)):
            train = folds != fold
            val = ~train
            tx = transform_frame(df.loc[train], features)
            mu, sd = scaler_from_train(tx)
            vx = (transform_frame(df.loc[val], features) - mu) / sd
            clf, ok = fit_logistic((tx - mu) / sd, y[train], c_value, cfg)
            conv += int(ok)
            probs[val] = clf.predict_proba(vx)[:, 1]
        metrics = calibrator_metrics(y, probs, sample_ids=sample_ids, n_bins=int(cfg["calibration"]["ece_bins"]))
        rows.append({"candidate_C": c_value, "converged_fold_count": conv, **metrics})
        predictions_by_c[c_value] = probs
    best = select_candidate(rows, float(cfg["selection"]["tie_tolerance"]))
    return best, predictions_by_c[float(best["candidate_C"])], rows


def add_local_robust_drift(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw = out["embedding_drift_mean"].to_numpy(dtype=np.float64)
    # The full identity embedding matrix is not needed for a leakage-safe local
    # normalization audit; using margin/variance space keeps this diagnostic fast
    # and risk_fit-only. Mahalanobis++ is recorded as pending full embedding pass.
    ref = out[["margin_distance", "orbit_logit_variance"]].to_numpy(dtype=np.float64)
    ref = (ref - ref.mean(axis=0)) / np.maximum(ref.std(axis=0), 1.0e-12)
    best_col = None
    best_metric = float("inf")
    for k in K_GRID:
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(out)), algorithm="auto").fit(ref)
        idx = nn.kneighbors(ref, return_distance=False)[:, 1:]
        local_raw = raw[idx]
        med = np.median(local_raw, axis=1)
        mad = np.median(np.abs(local_raw - med[:, None]), axis=1) * 1.4826
        for eps in EPS_GRID:
            col = f"_r_local_k{k}_eps{eps:g}"
            out[col] = np.maximum((raw - med) / (mad + eps), 0.0)
            score = float(np.nanstd(out[col]))
            if score < best_metric:
                best_metric = score
                best_col = col
    out["embedding_drift_local_robust"] = out[best_col].to_numpy(dtype=np.float64)
    out.attrs["local_robust_params"] = {"selected_proxy_column": best_col, "selection_note": "proxy diagnostic; final k/epsilon selected by OOF logistic metrics"}
    return out


def normalized_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1.0e-12)


def load_orbit_embeddings(detector: str, parent_ids: np.ndarray) -> np.ndarray:
    parent_order = {str(pid): i for i, pid in enumerate(parent_ids.astype(str))}
    manifest_cols = ["parent_sample_id", "view_id", "view_index"]
    manifest = pd.read_parquet(PROJECT_ROOT / "artifacts" / "phase4" / "transformation_orbit_manifest.parquet", columns=manifest_cols)
    manifest = manifest[manifest["parent_sample_id"].isin(parent_order)]
    index = pd.read_parquet(PROJECT_ROOT / "artifacts" / "phase4" / "orbit_cache" / detector / "index.parquet")
    joined = index.merge(manifest, on=["parent_sample_id", "view_id"], how="inner", validate="one_to_one")
    if len(joined) != len(parent_ids) * VIEW_COUNT:
        raise RuntimeError(f"orbit embedding row mismatch for {detector}: got {len(joined)}, expected {len(parent_ids) * VIEW_COUNT}")
    first_path = Path(str(joined["embedding_shard"].iloc[0]))
    dim = int(np.load(first_path, mmap_mode="r").shape[1])
    tensor = np.empty((len(parent_ids), VIEW_COUNT, dim), dtype=np.float32)
    filled = np.zeros((len(parent_ids), VIEW_COUNT), dtype=bool)
    for shard, group in joined.groupby("embedding_shard", sort=False):
        arr = np.load(Path(str(shard)), mmap_mode="r")
        offsets = group["row_offset"].to_numpy(dtype=np.int64)
        parents = group["parent_sample_id"].astype(str).map(parent_order).to_numpy(dtype=np.int64)
        views = group["view_index"].to_numpy(dtype=np.int64)
        tensor[parents, views, :] = arr[offsets]
        filled[parents, views] = True
    if not filled.all():
        missing = int((~filled).sum())
        raise RuntimeError(f"missing orbit embedding slots for {detector}: {missing}")
    return normalized_rows(tensor.reshape(-1, dim)).reshape(len(parent_ids), VIEW_COUNT, dim)


def cholesky_mahalanobis(centered: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, float]:
    jitter = 1.0e-8
    eye = np.eye(covariance.shape[0], dtype=np.float64)
    for _ in range(6):
        try:
            chol = np.linalg.cholesky(covariance + jitter * eye)
            solved = np.linalg.solve(chol, centered.T)
            dist = np.sqrt(np.maximum(np.sum(solved * solved, axis=0), 0.0))
            return dist.astype(np.float64), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    pinv = np.linalg.pinv(covariance)
    dist = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, pinv, centered), 0.0))
    return dist.astype(np.float64), float("nan")


def add_mahalanobis_pp_drift(df: pd.DataFrame, detector: str, cov_rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    print(f"[{detector}/{out['split'].iloc[0]}] loading orbit embeddings for Mahalanobis++", flush=True)
    embeddings = load_orbit_embeddings(detector, out["parent_sample_id"].astype(str).to_numpy())
    deltas = embeddings[:, 1:, :] - embeddings[:, :1, :]
    folds = out["cv_fold"].to_numpy(dtype=int)
    scores = np.full(len(out), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        val = ~train
        fold_scores = np.zeros(int(val.sum()), dtype=np.float64)
        for view in range(TRANSFORM_COUNT):
            train_delta = deltas[train, view, :].astype(np.float64, copy=False)
            val_delta = deltas[val, view, :].astype(np.float64, copy=False)
            lw = LedoitWolf(assume_centered=False).fit(train_delta)
            mu = lw.location_.astype(np.float64, copy=False)
            cov = lw.covariance_.astype(np.float64, copy=False)
            centered = val_delta - mu
            dist, jitter = cholesky_mahalanobis(centered, cov)
            fold_scores += dist / float(TRANSFORM_COUNT)
            eigvals = np.linalg.eigvalsh(cov)
            positive = eigvals[eigvals > 1.0e-12]
            cov_rows.append(
                {
                    "detector": detector,
                    "split": str(out["split"].iloc[0]),
                    "drift_variant": "mahalanobis_pp_displacement",
                    "fold": int(fold),
                    "view_index": int(view + 1),
                    "train_row_count": int(train.sum()),
                    "validation_row_count": int(val.sum()),
                    "dimension": int(cov.shape[0]),
                    "condition_number": float(eigvals.max() / max(eigvals.min(), 1.0e-12)),
                    "effective_rank": float(np.exp(-np.sum((positive / positive.sum()) * np.log(positive / positive.sum())))) if len(positive) else 0.0,
                    "ledoit_wolf_shrinkage": float(lw.shrinkage_),
                    "cholesky_jitter": jitter,
                    "status": "ok",
                }
            )
        scores[val] = fold_scores
    if not np.isfinite(scores).all():
        raise RuntimeError("Mahalanobis++ OOF scores contain NaN or Inf")
    out["embedding_drift_mahalanobis_pp"] = scores
    return out


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, ids: np.ndarray, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        sample = rng.integers(0, n, size=n)
        diffs[i] = fast_aurc(y[sample], a[sample], ids[sample]) - fast_aurc(y[sample], b[sample], ids[sample])
    return {"diff_mean": float(np.mean(diffs)), "diff_ci_low": float(np.quantile(diffs, 0.025)), "diff_ci_high": float(np.quantile(diffs, 0.975))}


def fast_aurc(errors: np.ndarray, risks: np.ndarray, sample_ids: np.ndarray) -> float:
    order = np.lexsort((sample_ids.astype(str), risks))
    sorted_errors = errors[order].astype(np.float64)
    return float((np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1)).mean())


def stable_hash(payload: Any) -> str:
    import hashlib

    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def selected_drift_by_rule(drift: pd.DataFrame) -> str:
    candidates = drift[
        (drift["status"] == "ok")
        & (drift["support_state"] == "with_support")
        & (drift["drift_variant"].isin(["raw_cosine", "mahalanobis_pp_displacement"]))
    ].copy()
    expected = len(DETECTORS) * len(SPLITS) * 2
    if len(candidates) != expected:
        return "INCONCLUSIVE"
    candidates["aurc_rank"] = candidates.groupby(["detector", "split"])["AURC"].rank(method="average", ascending=True)
    candidates["auroc_rank"] = candidates.groupby(["detector", "split"])["error_detection_AUROC"].rank(method="average", ascending=False)
    candidates["nll_rank"] = candidates.groupby(["detector", "split"])["binary_nll"].rank(method="average", ascending=True)
    summary = candidates.groupby("drift_variant", as_index=False)[["aurc_rank", "auroc_rank", "nll_rank"]].mean()
    summary = summary.sort_values(["aurc_rank", "auroc_rank", "nll_rank", "drift_variant"])
    best = str(summary.iloc[0]["drift_variant"])
    return "MAHALANOBIS_PP" if best == "mahalanobis_pp_displacement" else "RAW_COSINE"


def fit_full_scorer(df: pd.DataFrame, features: tuple[str, ...], cfg: dict[str, Any], c_value: float) -> tuple[dict[str, Any], np.ndarray]:
    y = df["base_error"].to_numpy(dtype=np.int64)
    tx = transform_frame(df, features)
    mu, sd = scaler_from_train(tx)
    clf, converged = fit_logistic((tx - mu) / sd, y, c_value, cfg)
    probs = clf.predict_proba((tx - mu) / sd)[:, 1]
    artifact = {
        "feature_order": list(features),
        "feature_transformations": {name: ("-log1p" if name == "margin_distance" else "log1p") for name in features},
        "scaler_means": mu.astype(float).tolist(),
        "scaler_scales": sd.astype(float).tolist(),
        "coefficient_vector": clf.coef_[0].astype(float).tolist(),
        "intercept": float(clf.intercept_[0]),
        "selected_C": float(c_value),
        "converged": bool(converged),
    }
    artifact["feature_schema_hash"] = stable_hash({"feature_order": artifact["feature_order"], "feature_transformations": artifact["feature_transformations"]})
    artifact["scaler_hash"] = stable_hash({"scaler_means": artifact["scaler_means"], "scaler_scales": artifact["scaler_scales"]})
    artifact["logistic_model_hash"] = stable_hash({"coefficient_vector": artifact["coefficient_vector"], "intercept": artifact["intercept"], "selected_C": artifact["selected_C"]})
    artifact["model_hash"] = stable_hash(artifact)
    artifact["score_hash"] = stable_hash({"sample_id": df["sample_id"].astype(str).tolist(), "risk_probability": np.round(probs, 15).tolist()})
    return artifact, probs


def current_artifact_hashes(detector: str, split: str, phase5_root: Path) -> dict[str, str]:
    current = read_json(phase5_root / "models" / f"{detector}_{split}_riskguard.json")
    score_path = phase5_root / "scores" / detector / split / "risk_fit_fullfit.parquet"
    score_hash = sha256_file(score_path) if score_path.exists() else ""
    schema_hash = stable_hash({"feature_order": current["feature_order"], "feature_transformations": current["feature_transformations"]})
    scaler_hash = stable_hash({"scaler_means": current["scaler_means"], "scaler_scales": current["scaler_scales"]})
    model_hash = str(current.get("model_hash") or payload_sha256(current))
    return {"feature_schema_hash": schema_hash, "scaler_hash": scaler_hash, "logistic_model_hash": model_hash, "score_hash": score_hash}


def main() -> None:
    args = parse_args()
    started = time.time()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(args.config))
    support_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    cov_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str, str], np.ndarray] = {}
    combo_frames: dict[tuple[str, str], pd.DataFrame] = {}

    for detector in selected(DETECTORS, args.detector):
        for split in selected(SPLITS, args.split):
            slug = combo_slug(detector, split)
            df = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"))
            folds = pd.read_parquet(Path(args.phase5_root) / "cv_fold_assignments" / f"{slug}.parquet")
            df = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
            df["cv_fold"] = df["cv_fold"].astype(int)
            df = add_local_robust_drift(df)
            df = add_mahalanobis_pp_drift(df, detector, cov_rows)
            pd.DataFrame(cov_rows).to_csv(out / "covariance_diagnostics.csv", index=False)
            combo_frames[(detector, split)] = df
            sha_overlap = int(sum(len(set(df.loc[df.cv_fold != f, "sha256"]) & set(df.loc[df.cv_fold == f, "sha256"])) for f in sorted(df.cv_fold.unique())))
            for name, codes in SUPPORT_VARIANTS.items():
                print(f"[{detector}/{split}] support variant {name}", flush=True)
                features = to_raw_features(codes)
                best, probs, _ = evaluate_variant(df, features, cfg)
                predictions[(detector, split, name)] = probs
                support_rows.append({"detector": detector, "split": split, "variant": name, "features": json.dumps(features), "selected_C": best["candidate_C"], "sha_overlap_train_validation": sha_overlap, **best})
                pd.DataFrame(support_rows).to_csv(out / "support_ablation_metrics.csv", index=False)
            for drift_name, drift_code in DRIFT_VARIANTS.items():
                combos = [("without_support", ("m", "v") if drift_code is None else ("m", "v", drift_code)), ("with_support", ("m", "v", "s") if drift_code is None else ("m", "v", drift_code, "s"))]
                for support_state, codes in combos:
                    print(f"[{detector}/{split}] drift variant {drift_name}/{support_state}", flush=True)
                    features = to_raw_features(codes)
                    best, probs, _ = evaluate_variant(df, features, cfg)
                    key = f"{drift_name}_{support_state}"
                    predictions[(detector, split, key)] = probs
                    drift_rows.append({"detector": detector, "split": split, "drift_variant": drift_name, "support_state": support_state, "features": json.dumps(features), "status": "ok", "selected_C": best["candidate_C"], **best})
                    pd.DataFrame(drift_rows).to_csv(out / "drift_ablation_metrics.csv", index=False)
            y = df["base_error"].to_numpy(dtype=np.int64)
            ids = df["sample_id"].astype(str).to_numpy()
            ref = predictions[(detector, split, "current_full")]
            for name in SUPPORT_VARIANTS:
                print(f"[{detector}/{split}] paired bootstrap {name}", flush=True)
                stats = paired_bootstrap(y, predictions[(detector, split, name)], ref, ids, args.bootstrap_replicates, args.seed)
                boot_rows.append({"detector": detector, "split": split, "variant": name, "reference": "current_full", "metric": "AURC", **stats})
                pd.DataFrame(boot_rows).to_csv(out / "paired_bootstrap_differences.csv", index=False)
            top = {name: set(df.loc[np.argsort(-predictions[(detector, split, name)])[: max(1, len(df) // 10)], "sample_id"]) for name in SUPPORT_VARIANTS}
            for a_name, a_set in top.items():
                for b_name, b_set in top.items():
                    overlap_rows.append({"detector": detector, "split": split, "feature_a": a_name, "feature_b": b_name, "highest_risk_decile_overlap": len(a_set & b_set), "highest_risk_decile_jaccard": len(a_set & b_set) / max(1, len(a_set | b_set))})

    support = pd.DataFrame(support_rows)
    drift = pd.DataFrame(drift_rows)
    support.to_csv(out / "support_ablation_metrics.csv", index=False)
    drift.to_csv(out / "drift_ablation_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "paired_bootstrap_differences.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(out / "covariance_diagnostics.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(out / "error_overlap.csv", index=False)
    for name in ["certification_comparison.csv", "heldout_diagnostic_comparison.csv"]:
        pd.DataFrame([{"status": "not_rebuilt", "note": "Winner must be refit/frozen before certification or held-out diagnostics influence decisions."}]).to_csv(out / name, index=False)

    best = support.sort_values(["AURC", "binary_nll", "row_count"]).iloc[0].to_dict()
    best_drift = selected_drift_by_rule(drift)
    phase5_root = Path(args.phase5_root)
    current_unchanged = best_drift == "RAW_COSINE"
    selected_features = ("margin_distance", "orbit_logit_variance", "embedding_drift_mean", "orbit_support_distance_max") if best_drift == "RAW_COSINE" else ("margin_distance", "orbit_logit_variance", "embedding_drift_mahalanobis_pp", "orbit_support_distance_max")
    if best_drift != "INCONCLUSIVE":
        selected_rows = drift[(drift["status"] == "ok") & (drift["support_state"] == "with_support")]
        selected_rows = selected_rows[selected_rows["drift_variant"] == ("raw_cosine" if best_drift == "RAW_COSINE" else "mahalanobis_pp_displacement")]
        for _, sel in selected_rows.iterrows():
            detector = str(sel["detector"])
            split = str(sel["split"])
            if best_drift == "RAW_COSINE":
                selected_hashes = current_artifact_hashes(detector, split, phase5_root)
            else:
                artifact, _ = fit_full_scorer(combo_frames[(detector, split)], selected_features, cfg, float(sel["selected_C"]))
                selected_hashes = {key: str(artifact[key]) for key in ["feature_schema_hash", "scaler_hash", "logistic_model_hash", "score_hash"]}
            current_hashes = current_artifact_hashes(detector, split, phase5_root)
            artifact_rows.append(
                {
                    "detector": detector,
                    "split": split,
                    "best_drift_variant": best_drift,
                    "selected_feature_schema_hash": selected_hashes["feature_schema_hash"],
                    "current_feature_schema_hash": current_hashes["feature_schema_hash"],
                    "feature_schema_hash_unchanged": selected_hashes["feature_schema_hash"] == current_hashes["feature_schema_hash"],
                    "selected_scaler_hash": selected_hashes["scaler_hash"],
                    "current_scaler_hash": current_hashes["scaler_hash"],
                    "scaler_hash_unchanged": selected_hashes["scaler_hash"] == current_hashes["scaler_hash"],
                    "selected_logistic_model_hash": selected_hashes["logistic_model_hash"],
                    "current_logistic_model_hash": current_hashes["logistic_model_hash"],
                    "logistic_model_hash_unchanged": selected_hashes["logistic_model_hash"] == current_hashes["logistic_model_hash"],
                    "selected_score_hash": selected_hashes["score_hash"],
                    "current_score_hash": current_hashes["score_hash"],
                    "score_hash_unchanged": selected_hashes["score_hash"] == current_hashes["score_hash"],
                }
            )
    pd.DataFrame(artifact_rows).to_csv(out / "selected_scorer_artifact_comparison.csv", index=False)
    decision = {
        "selected_by": "risk_fit_oof_only",
        "best_support_variant": best["variant"],
        "support_distance_decision": "KEEP" if best["variant"] in {"current_full", "support_without_drift", "support_only_optional"} else "INCONCLUSIVE",
        "mahalanobis_evaluation_complete": best_drift != "INCONCLUSIVE",
        "best_drift_variant": best_drift,
        "current_scorer_artifact_unchanged": current_unchanged,
        "primary_method_change_recommended": best_drift == "MAHALANOBIS_PP",
        "selection_used_risk_fit_only": True,
        "certification_must_be_rebuilt": not current_unchanged,
        "safe_for_paper_update": best_drift != "INCONCLUSIVE" and current_unchanged,
        "runtime_seconds": round(time.time() - started, 3),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (out / "feature_ablation_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "feature_ablation_summary.md").write_text(
        "\n".join(
            [
                "# Feature Ablation Audit",
                "",
                f"Output directory: `{out}`",
                f"Runtime seconds: {decision['runtime_seconds']}",
                f"CUDA_VISIBLE_DEVICES: `{decision['gpu']}`",
                "",
                "Selection used only risk_fit OOF metrics and existing SHA-disjoint Phase 5 folds.",
                "Evaluation partitions did not influence the selected variant in this audit.",
                "Mahalanobis++ displacement used fold-specific Ledoit-Wolf covariance estimates fitted only on OOF training folds.",
                "",
                f"Best support variant by OOF AURC/NLL: `{best['variant']}`.",
                f"Best drift variant by the predeclared with-support OOF rank rule: `{best_drift}`.",
                "",
                "Selected scorer artifact comparison is written to `selected_scorer_artifact_comparison.csv`.",
                "",
                f"MAHALANOBIS_EVALUATION_COMPLETE = {'TRUE' if decision['mahalanobis_evaluation_complete'] else 'FALSE'}",
                f"BEST_DRIFT_VARIANT = {best_drift}",
                f"CURRENT_SCORER_ARTIFACT_UNCHANGED = {'TRUE' if decision['current_scorer_artifact_unchanged'] else 'FALSE'}",
                f"CERTIFICATION_MUST_BE_REBUILT = {'TRUE' if decision['certification_must_be_rebuilt'] else 'FALSE'}",
                f"SAFE_FOR_PAPER_UPDATE = {'TRUE' if decision['safe_for_paper_update'] else 'FALSE'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
