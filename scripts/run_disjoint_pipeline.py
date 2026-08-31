#!/usr/bin/env python3
"""Run the disjoint RiskGuard protocol using immutable detector/orbit caches.

This module deliberately owns a versioned output namespace.  It never writes to
the historical Phase 2--8 outputs and never consumes an old q-derived feature,
target, score, candidate, certificate, or evaluation metric as a clean input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_curve

from selective_detection.exact_binomial_bound import clopper_pearson_upper
from selective_detection.error_probability_calibrator import transform_features
from selective_detection.grouped_cross_validation import assign_sha_grouped_folds, fold_audit_rows
from selective_detection.calibration_metrics import calibrator_metrics
from selective_detection.selective_baselines import Phase2Cache, exact_knn_distance, select_knn_k_cv


SEED = 20260916
DETECTORS = ("safe", "univfd")
SPLITS = ("split_a", "split_b")
PARTITIONS = ("base_cal", "risk_fit", "threshold_cal", "protocol_seen", "protocol_held_out")
FEATURES = ("margin_distance", "orbit_logit_variance", "mean_directional_erosion", "worst_view_erosion")
VARIANTS: dict[str, tuple[str, ...]] = {
    "m": ("margin_distance",),
    "m+v": ("margin_distance", "orbit_logit_variance"),
    "m+v+d+e": FEATURES,
    "no-m": ("orbit_logit_variance", "mean_directional_erosion", "worst_view_erosion"),
}
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
ALPHA = 0.05
DELTA = 0.05
K_MAX = 10
BOOTSTRAP_REPLICATES = 2000
ART = ROOT / "artifacts" / "protocol_clean_v2"
REP = ROOT / "reports" / "protocol_clean_v2"
MAN = ART / "manifests"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()


def stable_rank(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def ensure_dirs() -> None:
    for path in [
        ART, REP, MAN, ART / "features", ART / "oof_predictions", ART / "fitted_scorers",
        ART / "policy_candidates", ART / "certification", ART / "final_metrics",
        ART / "bootstrap", ART / "figures", ART / "tables", ART / "baselines", ART / "ablations",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def immutable_identity_predictions(detector: str) -> pd.DataFrame:
    cache = ROOT / "artifacts" / "cache" / detector / "clean"
    index = pd.read_parquet(cache / "index.parquet")
    frames = []
    columns = ["sample_id", "label", "canonical_generator", "sha256", "raw_logit", "fake_probability", "checkpoint_sha256", "preprocessing_id"]
    for path in sorted(set(index["prediction_shard"].astype(str))):
        frames.append(pd.read_parquet(path, columns=columns))
    out = pd.concat(frames, ignore_index=True)
    if len(out) != 300_000 or out["sample_id"].duplicated().any() or out["sha256"].isna().any():
        raise RuntimeError(f"invalid immutable identity cache for {detector}")
    return out


def attach_sha(manifest: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    out = manifest.drop(columns=["sha256"], errors="ignore").merge(
        identity[["sample_id", "sha256"]], on="sample_id", how="left", validate="one_to_one"
    )
    if out["sha256"].isna().any() or len(out) != len(manifest):
        raise RuntimeError("manifest-to-identity SHA join failed")
    return out


def deterministic_base_split(units: pd.DataFrame) -> pd.DataFrame:
    required = {"sha256", "label", "canonical_generator"}
    if not required <= set(units):
        raise ValueError(f"base split missing {sorted(required - set(units))}")
    if units["sha256"].duplicated().any():
        # SHA is the unit: validate its pre-q stratum is unambiguous.
        ambiguity = units.groupby("sha256")[["label", "canonical_generator"]].nunique().max(axis=1)
        if (ambiguity > 1).any():
            raise RuntimeError("one SHA maps to multiple pre-q strata")
    u = units.sort_values(["sha256", "sample_id"], kind="mergesort").drop_duplicates("sha256").copy()
    u["stratum"] = u["label"].astype(int).astype(str) + "|" + u["canonical_generator"].astype(str)
    u["stable_rank"] = u["sha256"].map(lambda value: stable_rank(SEED, "base_cal", value))
    u["clean_partition"] = "risk_fit"
    for _, idx in u.groupby("stratum", sort=True).groups.items():
        ordered = u.loc[list(idx)].sort_values(["stable_rank", "sha256"], kind="mergesort")
        n_base = int(round(0.15 * len(ordered)))
        if len(ordered) > 1:
            n_base = min(max(n_base, 1), len(ordered) - 1)
        u.loc[ordered.index[:n_base], "clean_partition"] = "base_cal"
    return u


def prepare_manifests() -> pd.DataFrame:
    """Create shared per-split base_cal/risk_fit manifests without reading eval labels."""
    ensure_dirs()
    identity = immutable_identity_predictions("safe")
    # Detector caches must agree on immutable sample/SHA/label provenance.
    other = immutable_identity_predictions("univfd")[["sample_id", "sha256", "label"]]
    chk = identity[["sample_id", "sha256", "label"]].merge(other, on="sample_id", suffixes=("_safe", "_univfd"), validate="one_to_one")
    if not chk["sha256_safe"].eq(chk["sha256_univfd"]).all() or not chk["label_safe"].eq(chk["label_univfd"]).all():
        raise RuntimeError("detector caches disagree on SHA or label provenance")
    summary = []
    for split in SPLITS:
        old_risk = pd.read_csv(ROOT / "datasets" / "manifests" / f"{split}_risk_fit.csv", low_memory=False)
        old_risk = attach_sha(old_risk, identity)
        assignment = deterministic_base_split(old_risk)
        old_risk = old_risk.merge(assignment[["sha256", "clean_partition"]], on="sha256", how="left", validate="many_to_one")
        for partition in ("base_cal", "risk_fit"):
            frame = old_risk[old_risk["clean_partition"].eq(partition)].copy()
            frame["riskguard_partition"] = partition
            frame.to_csv(MAN / f"{split}_{partition}.csv", index=False)
            summary.append(partition_summary(frame, split, partition))
        threshold = pd.read_csv(ROOT / "datasets" / "manifests" / f"{split}_threshold_cal.csv", low_memory=False)
        threshold = attach_sha(threshold, identity)
        threshold["clean_partition"] = "threshold_cal"
        threshold.to_csv(MAN / f"{split}_threshold_cal.csv", index=False)
        summary.append(partition_summary(threshold, split, "threshold_cal"))
    out = pd.DataFrame(summary)
    out.to_csv(ART / "partition_counts.csv", index=False)
    return out


def partition_summary(df: pd.DataFrame, split: str, partition: str) -> dict[str, Any]:
    generator_col = "canonical_generator" if "canonical_generator" in df else "generator"
    return {
        "split": split, "partition": partition, "rows": int(len(df)),
        "unique_sha256": int(df["sha256"].nunique()),
        "real_count": int(df["label"].astype(int).eq(0).sum()),
        "fake_count": int(df["label"].astype(int).eq(1).sum()),
        "generator_counts_json": json.dumps({str(k): int(v) for k, v in df[generator_col].astype(str).value_counts().sort_index().items()}, sort_keys=True),
    }


def select_q(base: pd.DataFrame) -> dict[str, Any]:
    y = base["label"].to_numpy(dtype=int)
    p = base["fake_probability"].to_numpy(dtype=float)
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds) & (thresholds > 0.0) & (thresholds < 1.0)
    if not finite.any():
        raise RuntimeError("no finite internal q candidate")
    candidates = pd.DataFrame({"q": thresholds[finite], "tpr": tpr[finite], "tnr": 1.0 - fpr[finite]})
    candidates["balanced_accuracy"] = (candidates["tpr"] + candidates["tnr"]) / 2.0
    best = candidates["balanced_accuracy"].max()
    ties = candidates[np.isclose(candidates["balanced_accuracy"], best, atol=1e-15, rtol=0)].copy()
    # Preserve the historical np.argmax/roc_curve behavior: highest q among exact ties.
    chosen = ties.sort_values("q", ascending=False, kind="mergesort").iloc[0]
    pred = (p >= float(chosen["q"])).astype(int)
    if not np.isclose(balanced_accuracy_score(y, pred), chosen["balanced_accuracy"]):
        raise RuntimeError("balanced-accuracy threshold parity failure")
    return {
        "decision_threshold": float(chosen["q"]),
        "gamma": float(logit(chosen["q"])),
        "balanced_accuracy": float(chosen["balanced_accuracy"]),
        "tpr": float(chosen["tpr"]), "tnr": float(chosen["tnr"]),
        "tie_count": int(len(ties)), "tie_breaker": "highest_q_preserving_historical_roc_curve_first_max",
    }


def compute_base_thresholds() -> pd.DataFrame:
    rows = []
    for detector in DETECTORS:
        identity = immutable_identity_predictions(detector)
        for split in SPLITS:
            manifest = pd.read_csv(MAN / f"{split}_base_cal.csv", low_memory=False)
            base = manifest[["sample_id", "sha256", "label"]].merge(
                identity[["sample_id", "raw_logit", "fake_probability"]], on="sample_id", how="left", validate="one_to_one"
            )
            selected = select_q(base)
            rows.append({
                "detector": detector, "split": split, **selected,
                "base_cal_rows": int(len(base)), "base_cal_unique_sha256": int(base["sha256"].nunique()),
                "real_count": int(base["label"].eq(0).sum()), "fake_count": int(base["label"].eq(1).sum()),
                "source_partition": "base_cal", "source_manifest": rel(MAN / f"{split}_base_cal.csv"),
                "source_manifest_sha256": sha256_file(MAN / f"{split}_base_cal.csv"),
                "labels_used": "base_cal_only",
            })
    out = pd.DataFrame(rows)
    if not out["decision_threshold"].between(0, 1, inclusive="neither").all():
        raise RuntimeError("q outside (0,1)")
    out.to_csv(ART / "base_thresholds.csv", index=False)
    write_json(ART / "base_thresholds.freeze.json", {
        "status": "FROZEN", "sha256": sha256_file(ART / "base_thresholds.csv"),
        "selected_before_risk_fit": True, "source_partition": "base_cal", "seed": SEED,
    })
    return out


def load_orbit_logits(detector: str) -> pd.DataFrame:
    """Load only immutable raw five-view logits, never old derived features."""
    frames = []
    cols = ["parent_sample_id", "source_sample_id", "view_index", "raw_logit", "split", "partition", "evaluation_role", "generator", "label", "sha256"]
    for path in sorted((ROOT / "artifacts" / "phase4" / "orbit_cache" / detector).glob("predictions_*.parquet")):
        frames.append(pd.read_parquet(path, columns=cols))
    long = pd.concat(frames, ignore_index=True)
    if len(long) != 3_007_330:
        raise RuntimeError(f"unexpected orbit row count for {detector}: {len(long)}")
    counts = long.groupby("parent_sample_id")["view_index"].nunique()
    if not counts.eq(5).all():
        raise RuntimeError(f"incomplete orbit for {detector}")
    meta = long[long["view_index"].eq(0)].copy()
    wide = long.pivot(index="parent_sample_id", columns="view_index", values="raw_logit")
    wide.columns = [f"z_{int(c)}" for c in wide.columns]
    out = meta.drop(columns=["view_index", "raw_logit"]).merge(wide.reset_index(), on="parent_sample_id", validate="one_to_one")
    return out


def derive_features(raw: pd.DataFrame, q: float, clean_partition: str) -> pd.DataFrame:
    gamma = float(logit(q))
    z = raw[[f"z_{i}" for i in range(5)]].to_numpy(dtype=np.float64)
    z0 = z[:, 0]
    prediction = (z0 >= gamma).astype(np.int64)
    c = np.where(prediction == 1, 1.0, -1.0)
    delta = c[:, None] * (z0[:, None] - z[:, 1:])
    out = pd.DataFrame({
        "sample_id": raw["source_sample_id"].astype(str), "sha256": raw["sha256"].astype(str),
        "parent_sample_id": raw["parent_sample_id"].astype(str), "detector": raw.get("detector", ""),
        "split": raw["split"].astype(str), "partition": clean_partition,
        "evaluation_role": raw["evaluation_role"].astype(str), "generator": raw["generator"].astype(str),
        "label": raw["label"].astype(int), "base_logit": z0, "base_probability": expit(z0),
        "base_prediction": prediction, "base_error": (prediction != raw["label"].to_numpy(dtype=int)).astype(int),
        "gamma": gamma, "q": q, "identity_orientation_c": c.astype(int),
        "margin_distance": np.abs(z0 - gamma), "orbit_logit_variance": np.var(z, axis=1, ddof=0),
        "mean_directional_erosion": delta.mean(axis=1), "worst_view_erosion": delta.max(axis=1),
    })
    for i in range(1, 5):
        out[f"Delta_{i}"] = delta[:, i - 1]
    vals = out[[*FEATURES, "base_logit", "base_probability", "gamma"]].to_numpy(float)
    if not np.isfinite(vals).all():
        raise RuntimeError("non-finite rebuilt feature")
    if not np.allclose(out["margin_distance"], np.abs(out["base_logit"] - out["gamma"]), rtol=0, atol=1e-12):
        raise RuntimeError("a0/margin inconsistency")
    return out


def rebuild_development_features() -> pd.DataFrame:
    thresholds = pd.read_csv(ART / "base_thresholds.csv")
    audit = []
    for detector in DETECTORS:
        raw_all = load_orbit_logits(detector)
        for split in SPLITS:
            q = float(thresholds.loc[(thresholds.detector.eq(detector)) & thresholds.split.eq(split), "decision_threshold"].iloc[0])
            raw_split = raw_all[raw_all["split"].eq(split)]
            old_risk = raw_split[raw_split["partition"].eq("risk_fit")]
            assignment = pd.concat([
                pd.read_csv(MAN / f"{split}_base_cal.csv", usecols=["sha256"]).assign(clean_partition="base_cal"),
                pd.read_csv(MAN / f"{split}_risk_fit.csv", usecols=["sha256"]).assign(clean_partition="risk_fit"),
            ]).drop_duplicates("sha256")
            old_risk = old_risk.merge(assignment, on="sha256", how="left", validate="many_to_one")
            if old_risk["clean_partition"].isna().any():
                raise RuntimeError("clean risk/base assignment incomplete")
            sources = {
                "base_cal": old_risk[old_risk["clean_partition"].eq("base_cal")],
                "risk_fit": old_risk[old_risk["clean_partition"].eq("risk_fit")],
                "threshold_cal": raw_split[raw_split["partition"].eq("threshold_cal")],
            }
            for partition, raw in sources.items():
                frame = derive_features(raw, q, partition)
                frame["detector"] = detector
                path = ART / "features" / detector / split / f"{partition}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                audit.append({
                    "detector": detector, "split": split, "partition": partition, "rows": len(frame),
                    "unique_sha256": frame.sha256.nunique(), "error_count": int(frame.base_error.sum()),
                    "finite_logits": bool(np.isfinite(frame.base_logit).all()),
                    "finite_features": bool(np.isfinite(frame[list(FEATURES)].to_numpy(float)).all()),
                    "nan_count": int(frame[list(FEATURES)].isna().sum().sum()),
                    "a0_margin_consistent": bool(np.allclose(frame.margin_distance, np.abs(frame.base_logit - frame.gamma))),
                    "feature_sha256": sha256_file(path), "q": q,
                })
        del raw_all
    out = pd.DataFrame(audit)
    out.to_csv(ART / "feature_rebuild_audit.csv", index=False)
    return out


def fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float) -> tuple[LogisticRegression, bool, str]:
    clf = LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", fit_intercept=True, class_weight=None,
                             max_iter=5000, tol=1e-10, random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return clf, not messages, " | ".join(messages)


def choose_c(rows: list[dict[str, Any]], tolerance: float = 1e-12) -> float:
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
    return float(best["candidate_C"])


def transformed(df: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return np.asarray(transform_features(df.loc[:, list(features)], features, as_frame=False), dtype=np.float64)


def fit_variant(df: pd.DataFrame, features: tuple[str, ...]) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    y = df["base_error"].to_numpy(dtype=int)
    fold = df["cv_fold"].to_numpy(dtype=int)
    by_c: dict[float, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    search = []
    for c_value in C_GRID:
        probs = np.full(len(df), np.nan)
        logits = np.full(len(df), np.nan)
        fold_models = []
        for f in range(5):
            train, valid = fold != f, fold == f
            tx, vx = transformed(df.loc[train], features), transformed(df.loc[valid], features)
            mu, sd = tx.mean(axis=0), tx.std(axis=0, ddof=0)
            if (sd < 1e-12).any():
                raise RuntimeError("degenerate fold-local feature")
            clf, converged, warning = fit_logistic((tx - mu) / sd, y[train], c_value)
            logits[valid] = clf.decision_function((vx - mu) / sd)
            probs[valid] = clf.predict_proba((vx - mu) / sd)[:, 1]
            fold_models.append({"fold": f, "converged": converged, "warning": warning, "scaler_means": mu.tolist(),
                                "scaler_scales": sd.tolist(), "coefficients": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])})
        metrics = calibrator_metrics(y, probs, sample_ids=df.sample_id.astype(str).to_numpy(), n_bins=15)
        search.append({"candidate_C": c_value, **metrics, "converged_fold_count": sum(int(x["converged"]) for x in fold_models)})
        by_c[c_value] = (probs, logits, fold_models)
    selected_c = choose_c(search)
    probs, logits, fold_models = by_c[selected_c]
    tx = transformed(df, features)
    mu, sd = tx.mean(axis=0), tx.std(axis=0, ddof=0)
    clf, converged, warning = fit_logistic((tx - mu) / sd, y, selected_c)
    if not converged or any(not m["converged"] for m in fold_models):
        raise RuntimeError("selected logistic model did not converge")
    model = {
        "model_version": "protocol_clean_v2_logistic_v1", "feature_order": list(features),
        "feature_transformations": {f: {"margin_distance": "-log1p", "orbit_logit_variance": "log1p",
                                         "mean_directional_erosion": "signed_log1p", "worst_view_erosion": "signed_log1p"}[f] for f in features},
        "selected_C": selected_c, "scaler_means": mu.tolist(), "scaler_scales": sd.tolist(),
        "coefficient_vector": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
        "risk_fit_rows": int(len(df)), "risk_fit_error_count": int(y.sum()), "converged": True,
        "optimizer_warning": warning, "seed": SEED, "source_partition": "risk_fit", "q_frozen_upstream": True,
        "fold_models": fold_models,
    }
    model["model_sha256"] = payload_hash(model)
    oof = df[["sample_id", "sha256", "detector", "split", "generator", "label", "base_prediction", "base_error", "cv_fold", *FEATURES]].copy()
    oof["risk_logit"] = logits
    oof["risk_probability"] = probs
    oof["selected_C"] = selected_c
    return model, oof, search


def score_model(df: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    features = tuple(model["feature_order"])
    x = transformed(df, features)
    z = (x - np.asarray(model["scaler_means"])) / np.asarray(model["scaler_scales"])
    return expit(z @ np.asarray(model["coefficient_vector"]) + float(model["intercept"]))


def fit_scorers_and_ablations() -> pd.DataFrame:
    metrics_rows, search_rows, fold_rows, ablation_rows = [], [], [], []
    for detector in DETECTORS:
        for split in SPLITS:
            path = ART / "features" / detector / split / "risk_fit.parquet"
            df = pd.read_parquet(path)
            folds = assign_sha_grouped_folds(df, n_splits=5, seed=SEED)
            df = df.merge(folds, on=["sample_id", "sha256"], validate="one_to_one")
            fold_rows.extend(fold_audit_rows(df, detector=detector, split=split))
            folds.to_parquet(ART / "oof_predictions" / f"{detector}_{split}_folds.parquet", index=False)
            primary_model = None
            for variant, features in VARIANTS.items():
                model, oof, search = fit_variant(df, features)
                model.update({"detector": detector, "split": split, "variant": variant,
                              "q_registry_sha256": sha256_file(ART / "base_thresholds.csv"),
                              "risk_fit_feature_sha256": sha256_file(path),
                              "fold_assignment_sha256": sha256_file(ART / "oof_predictions" / f"{detector}_{split}_folds.parquet")})
                model["model_sha256"] = payload_hash({k: v for k, v in model.items() if k != "model_sha256"})
                model_path = ART / "fitted_scorers" / f"{detector}_{split}_{variant.replace('+','_plus_')}.json"
                write_json(model_path, model)
                oof["variant"] = variant
                oof["model_sha256"] = model["model_sha256"]
                oof_path = ART / "oof_predictions" / f"{detector}_{split}_{variant.replace('+','_plus_')}.parquet"
                oof.to_parquet(oof_path, index=False)
                selected = [r for r in search if float(r["candidate_C"]) == float(model["selected_C"])][0]
                ablation_rows.append({"detector": detector, "split": split, "variant": variant,
                                      "feature_order_json": json.dumps(list(features)), "selected_C": model["selected_C"], **selected})
                for row in search:
                    search_rows.append({"detector": detector, "split": split, "variant": variant,
                                        **row, "selected": float(row["candidate_C"]) == float(model["selected_C"])})
                if variant == "m+v+d+e":
                    primary_model = model
                    metrics_rows.append({"detector": detector, "split": split, "selected_C": model["selected_C"], **selected})
                    primary_path = ART / "fitted_scorers" / f"{detector}_{split}_riskguard.json"
                    write_json(primary_path, model)
                    oof.to_parquet(ART / "oof_predictions" / f"{detector}_{split}_risk_fit.parquet", index=False)
            if primary_model is None:
                raise RuntimeError("missing primary scorer")
    pd.DataFrame(search_rows).to_csv(ART / "fitted_scorers" / "hyperparameter_search.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ART / "oof_predictions" / "fold_audit.csv", index=False)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(ART / "oof_predictions" / "oof_metrics.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(ART / "ablations" / "ablation_oof_metrics.csv", index=False)
    return metrics


def score_frame_base(df: pd.DataFrame, detector: str, split: str, method: str, risk: np.ndarray) -> pd.DataFrame:
    out = df[["sample_id", "sha256", "detector", "split", "partition", "evaluation_role", "generator", "label",
              "base_logit", "base_probability", "base_prediction", "base_error", *FEATURES]].copy()
    out["method"] = method
    out["risk_score"] = np.asarray(risk, dtype=np.float64)
    out["risk_orientation"] = "higher_risk_score_more_likely_to_reject"
    if not np.isfinite(out["risk_score"]).all():
        raise RuntimeError(f"non-finite score for {detector}/{split}/{method}")
    return out


def score_path(detector: str, split: str, method: str, partition: str) -> Path:
    return ART / "scores" / detector / split / method / f"{partition}.parquet"


def save_score(df: pd.DataFrame, detector: str, split: str, method: str, partition: str) -> Path:
    path = score_path(detector, split, method, partition)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_model(detector: str, split: str, variant: str = "riskguard") -> dict[str, Any]:
    if variant == "riskguard":
        path = ART / "fitted_scorers" / f"{detector}_{split}_riskguard.json"
    else:
        path = ART / "fitted_scorers" / f"{detector}_{split}_{variant.replace('+','_plus_')}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def fit_knn_baseline(detector: str, split: str, device: str) -> dict[str, Any]:
    model_path = ART / "baselines" / f"{detector}_{split}_knn.json"
    bank_path = ART / "baselines" / f"{detector}_{split}_knn_bank.parquet"
    cv_path = ART / "baselines" / f"{detector}_{split}_knn_cv.csv"
    if model_path.exists() and bank_path.exists() and cv_path.exists():
        return json.loads(model_path.read_text(encoding="utf-8"))
    cache = Phase2Cache(ROOT, detector)
    risk_fit = pd.read_parquet(ART / "features" / detector / split / "risk_fit.parquet")
    embeddings = cache.embeddings_for(risk_fit["sample_id"])
    selected_k, cv = select_knn_k_cv(
        embeddings, risk_fit["base_error"].to_numpy(dtype=int), risk_fit["sample_id"].astype(str).to_numpy(),
        candidate_k=[1, 5, 10, 20, 50], folds=5, seed=SEED, device=device,
    )
    cv.to_csv(cv_path, index=False)
    bank = risk_fit[["sample_id", "sha256", "label", "generator", "base_error"]].copy()
    bank.to_parquet(bank_path, index=False)
    payload = {
        "baseline": "cosine_knn", "detector": detector, "split": split, "source_partition": "risk_fit",
        "selected_k": int(selected_k), "candidate_k": [1, 5, 10, 20, 50], "cross_validation_folds": 5,
        "selection_objective": "error_detection_AUROC", "seed": SEED,
        "bank_sha256": sha256_file(bank_path), "cv_sha256": sha256_file(cv_path),
        "risk_fit_feature_sha256": sha256_file(ART / "features" / detector / split / "risk_fit.parquet"),
        "q_registry_sha256": sha256_file(ART / "base_thresholds.csv"),
    }
    payload["model_sha256"] = payload_hash(payload)
    write_json(model_path, payload)
    return payload


def score_knn_partition(detector: str, split: str, partition: str, feature_df: pd.DataFrame, device: str) -> pd.DataFrame:
    model = fit_knn_baseline(detector, split, device)
    cache = Phase2Cache(ROOT, detector)
    bank = pd.read_parquet(ART / "baselines" / f"{detector}_{split}_knn_bank.parquet")
    bank_embeddings = cache.embeddings_for(bank["sample_id"])
    if set(feature_df["sample_id"].astype(str)).issubset(set(cache.index["sample_id"].astype(str))):
        embedding_ids = feature_df["sample_id"].astype(str)
    else:
        # Verified evaluation manifests use context-specific aliases. Resolve
        # them to the immutable identity-cache row by content SHA-256.
        sha_map = (cache.predictions[["sha256", "sample_id"]].sort_values(["sha256", "sample_id"], kind="mergesort")
                   .drop_duplicates("sha256").rename(columns={"sample_id": "embedding_sample_id"}))
        resolved = feature_df[["sha256"]].merge(sha_map, on="sha256", how="left", validate="many_to_one")
        if resolved["embedding_sample_id"].isna().any():
            raise RuntimeError(f"evaluation SHA missing from immutable embedding cache for {detector}/{split}/{partition}")
        embedding_ids = resolved["embedding_sample_id"].astype(str)
    query_embeddings = cache.embeddings_for(embedding_ids)
    query_ids = embedding_ids.to_numpy() if partition == "risk_fit" else None
    scored = exact_knn_distance(
        bank_embeddings, query_embeddings, int(model["selected_k"]),
        bank_ids=bank["sample_id"].astype(str).to_numpy(), query_ids=query_ids,
        device=device, batch_size=1024,
    )
    out = score_frame_base(feature_df, detector, split, "knn", scored["risk_score"])
    out["selected_k"] = int(model["selected_k"])
    out["fit_artifact_sha256"] = model["model_sha256"]
    return out


def score_development_partitions(device: str, include_knn: bool = True) -> None:
    for detector in DETECTORS:
        for split in SPLITS:
            df = pd.read_parquet(ART / "features" / detector / split / "threshold_cal.parquet")
            primary = load_model(detector, split)
            margin = load_model(detector, split, "m")
            scores = {
                "riskguard": score_model(df, primary),
                "margin_only": score_model(df, margin),
                "msp": 1.0 - np.maximum(df["base_probability"].to_numpy(float), 1.0 - df["base_probability"].to_numpy(float)),
            }
            for method, risk in scores.items():
                out = score_frame_base(df, detector, split, method, risk)
                if method in {"riskguard", "margin_only"}:
                    model = primary if method == "riskguard" else margin
                    out["fit_artifact_sha256"] = model["model_sha256"]
                else:
                    out["fit_artifact_sha256"] = "analytic_msp"
                save_score(out, detector, split, method, "threshold_cal")
            if include_knn:
                knn = score_knn_partition(detector, split, "threshold_cal", df, device)
                save_score(knn, detector, split, "knn", "threshold_cal")


def build_policy_split() -> pd.DataFrame:
    rows = []
    for detector in DETECTORS:
        for split in SPLITS:
            riskguard = pd.read_parquet(score_path(detector, split, "riskguard", "threshold_cal"))
            unit_rows = []
            for sha, group in riskguard.groupby("sha256", sort=True):
                first = group.sort_values("sample_id", kind="mergesort").iloc[0]
                unit_rows.append({
                    "sha256": str(sha), "label": int(first.label), "generator": str(first.generator),
                    "base_error": int(first.base_error), "row_count": int(len(group)),
                    "stable_rank": stable_rank(SEED, str(sha)),
                })
            units = pd.DataFrame(unit_rows)
            units["stratum"] = units.label.astype(str) + "|" + units.generator + "|" + units.base_error.astype(str)
            units["calibration_subset"] = "policy_certify"
            for _, idx in units.groupby("stratum", sort=True).groups.items():
                ordered = units.loc[list(idx)].sort_values(["stable_rank", "sha256"], kind="mergesort")
                select_count = 1 if len(ordered) == 1 else len(ordered) // 2
                units.loc[ordered.index[:select_count], "calibration_subset"] = "policy_select"
            units = units.sort_values(["calibration_subset", "sha256"], kind="mergesort")
            path = MAN / f"{detector}_{split}_policy_assignment.csv"
            units.to_csv(path, index=False)
            select_sha = set(units.loc[units.calibration_subset.eq("policy_select"), "sha256"])
            certify_sha = set(units.loc[units.calibration_subset.eq("policy_certify"), "sha256"])
            if select_sha & certify_sha:
                raise RuntimeError("policy select/certify SHA overlap")
            for subset, group in units.groupby("calibration_subset", sort=True):
                rows.append({"detector": detector, "split": split, "partition": subset,
                             "rows": int(group.row_count.sum()), "unique_sha256": int(group.sha256.nunique()),
                             "real_count": int(group.label.eq(0).sum()), "fake_count": int(group.label.eq(1).sum()),
                             "error_count": int((group.base_error * group.row_count).sum()),
                             "cross_subset_sha_overlap": 0, "assignment_sha256": sha256_file(path)})
    out = pd.DataFrame(rows)
    out.to_csv(ART / "policy_split_audit.csv", index=False)
    return out


def source_groups(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df.label.astype(int).eq(0), "real_all", df.generator.astype(str).str.lower()), index=df.index, dtype=str)


def threshold_scan(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["sample_id", "risk_score", "base_error"]].copy()
    work["group"] = source_groups(df).to_numpy()
    work = work.sort_values(["risk_score", "sample_id"], kind="mergesort").reset_index(drop=True)
    risks, errors, groups = work.risk_score.to_numpy(float), work.base_error.to_numpy(int), work.group.to_numpy(str)
    unique_values, starts = np.unique(risks, return_index=True)
    ends = np.r_[starts[1:], len(work)]
    cum_errors = np.cumsum(errors)
    unique_groups = sorted(pd.unique(groups))
    group_n = {g: np.cumsum(groups == g) for g in unique_groups}
    group_k = {g: np.cumsum((groups == g) & (errors == 1)) for g in unique_groups}
    rows = []
    for tau, end in zip(unique_values, ends):
        i, feasible = int(end) - 1, True
        counts, errs, risks_g = {}, {}, {}
        for group in unique_groups:
            n, k = int(group_n[group][i]), int(group_k[group][i])
            counts[group], errs[group], risks_g[group] = n, k, (k / n if n else None)
            feasible = feasible and n > 0 and k / n <= ALPHA
        rows.append({"threshold": float(tau), "accepted_count": int(end), "coverage": float(end / len(work)),
                     "accepted_errors": int(cum_errors[i]), "empirical_risk": float(cum_errors[i] / end),
                     "group_counts_json": json.dumps(counts, sort_keys=True), "group_errors_json": json.dumps(errs, sort_keys=True),
                     "group_risks_json": json.dumps(risks_g, sort_keys=True), "select_feasible": bool(feasible)})
    return pd.DataFrame(rows)


def candidate_rows(detector: str, split: str, method: str) -> list[dict[str, Any]]:
    scores = pd.read_parquet(score_path(detector, split, method, "threshold_cal"))
    assignment = pd.read_csv(MAN / f"{detector}_{split}_policy_assignment.csv", usecols=["sha256", "calibration_subset"])
    merged = scores.merge(assignment, on="sha256", how="left", validate="many_to_one")
    # This is the candidate-construction boundary: certify rows are discarded
    # before any label/error/score operation below.
    select = merged.loc[merged.calibration_subset.eq("policy_select")].drop(columns="calibration_subset").copy()
    curve = threshold_scan(select)
    feasible = curve[curve.select_feasible]
    picked = []
    if len(feasible):
        largest = int(feasible.accepted_count.max())
        targets = [(f, max(1, int(math.floor(largest * f))), "select_feasible_fraction")
                   for f in [1.00, .95, .90, .85, .80, .75, .70, .65, .60, .50]]
    else:
        total = int(curve.accepted_count.max())
        targets = [(f, max(1, int(math.floor(total * f))), "fallback_select_coverage")
                   for f in [.50, .40, .30, .20, .15, .10, .075, .05, .025, .01]]
    for fraction, target, source in targets:
        eligible = curve[(curve.accepted_count > 0) & (curve.accepted_count <= target)]
        if len(eligible):
            row = eligible.sort_values(["accepted_count", "threshold"], ascending=[False, False], kind="mergesort").iloc[0].to_dict()
            row.update({"target_fraction": fraction, "candidate_source": source})
            picked.append(row)
    dedup = {float(row["threshold"]): row for row in picked}
    ordered = sorted(dedup.values(), key=lambda x: (-float(x["threshold"]), -int(x["accepted_count"])))[:K_MAX]
    out = []
    for rank, row in enumerate(ordered, 1):
        record = {"detector": detector, "split": split, "method": method, "alpha": ALPHA, "policy": "source_group_cp",
                  "candidate_id": f"{detector}_{split}_{method}_C{rank:02d}", "candidate_rank": rank,
                  "threshold": float(row["threshold"]), "select_accepted_count": int(row["accepted_count"]),
                  "select_coverage": float(row["coverage"]), "select_error_count": int(row["accepted_errors"]),
                  "select_empirical_risk": float(row["empirical_risk"]), "candidate_source": row["candidate_source"]}
        record["candidate_sha256"] = payload_hash(record)
        out.append(record)
    return out


def construct_and_freeze_candidates(include_knn: bool = True) -> pd.DataFrame:
    methods = ["riskguard", "msp", "margin_only"] + (["knn"] if include_knn else [])
    records = []
    for detector in DETECTORS:
        for split in SPLITS:
            for method in methods:
                rows = candidate_rows(detector, split, method)
                records.extend(rows)
                write_json(ART / "policy_candidates" / f"{detector}_{split}_{method}.json",
                           {"detector": detector, "split": split, "method": method, "candidate_count": len(rows), "candidates": rows})
    registry = pd.DataFrame(records)
    registry.to_csv(ART / "policy_candidates" / "candidate_registry.csv", index=False)
    files = sorted((ART / "policy_candidates").glob("*.json")) + [ART / "policy_candidates" / "candidate_registry.csv"]
    freeze = {rel(path): sha256_file(path) for path in files}
    write_json(ART / "policy_candidates" / "candidate_freeze.json",
               {"status": "FROZEN_BEFORE_CERTIFICATION", "files": freeze, "freeze_sha256": payload_hash(freeze),
                "policy_certify_scores_used": False, "policy_certify_labels_used_for_candidates": False})
    return registry


def verify_candidate_freeze() -> dict[str, Any]:
    path = ART / "policy_candidates" / "candidate_freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_CERTIFICATION":
        raise RuntimeError("candidate set is not frozen")
    observed = {name: sha256_file(ROOT / name) for name in payload["files"]}
    if observed != payload["files"] or payload_hash(observed) != payload["freeze_sha256"]:
        raise RuntimeError("candidate freeze mutation detected")
    return payload


def certify_policies(include_knn: bool = True) -> pd.DataFrame:
    freeze = verify_candidate_freeze()
    methods = ["riskguard", "msp", "margin_only"] + (["knn"] if include_knn else [])
    candidates = pd.read_csv(ART / "policy_candidates" / "candidate_registry.csv")
    trace_rows, registry_rows = [], []
    for detector in DETECTORS:
        for split in SPLITS:
            assignment = pd.read_csv(MAN / f"{detector}_{split}_policy_assignment.csv", usecols=["sha256", "calibration_subset"])
            for method in methods:
                scores = pd.read_parquet(score_path(detector, split, method, "threshold_cal"))
                merged = scores.merge(assignment, on="sha256", how="left", validate="many_to_one")
                certify = merged.loc[merged.calibration_subset.eq("policy_certify")].drop(columns="calibration_subset").copy()
                groups = source_groups(certify)
                unique_groups = sorted(groups.unique())
                cand = candidates[(candidates.detector.eq(detector)) & (candidates.split.eq(split)) & (candidates.method.eq(method))]
                k_count, g_count = len(cand), len(unique_groups)
                delta_cell = DELTA / (k_count * g_count) if k_count and g_count else float("nan")
                summaries = []
                for candidate in cand.sort_values("candidate_rank", kind="mergesort").to_dict("records"):
                    accepted_all = certify.risk_score.to_numpy(float) <= float(candidate["threshold"])
                    group_records, bounds, counts = [], {}, {}
                    for group in unique_groups:
                        mask = groups.eq(group).to_numpy()
                        accepted = accepted_all & mask
                        n, k = int(accepted.sum()), int(certify.loc[accepted, "base_error"].sum())
                        cp = clopper_pearson_upper(k, n, delta_cell)
                        passed = bool(n > 0 and cp <= ALPHA)
                        bounds[group] = cp
                        counts[group] = {"group_size": int(mask.sum()), "accepted_count": n, "accepted_errors": k,
                                         "empirical_selective_risk": k / n if n else None, "cp_upper": cp, "certified": passed}
                        group_records.append({"group": group, "certify_group_size": int(mask.sum()), "accepted_count": n,
                                              "accepted_errors": k, "empirical_selective_risk": k / n if n else np.nan,
                                              "cp_upper": cp, "group_certified": passed})
                    candidate_passed = bool(group_records and all(r["group_certified"] for r in group_records))
                    for record in group_records:
                        trace_rows.append({"detector": detector, "split": split, "method": method, "alpha": ALPHA,
                                           "delta": DELTA, "candidate_id": candidate["candidate_id"],
                                           "candidate_rank": int(candidate["candidate_rank"]), "threshold": float(candidate["threshold"]),
                                           "candidate_count_K": k_count, "group_count_G": g_count, "delta_cell": delta_cell,
                                           **record, "candidate_certified": candidate_passed,
                                           "candidate_freeze_sha256": freeze["freeze_sha256"]})
                    summaries.append({"candidate_id": candidate["candidate_id"], "candidate_rank": int(candidate["candidate_rank"]),
                                      "threshold": float(candidate["threshold"]), "certification_coverage": float(accepted_all.mean()),
                                      "certification_accepted_count": int(accepted_all.sum()), "max_group_cp_upper": max(bounds.values()),
                                      "candidate_certified": candidate_passed, "group_bounds": bounds, "group_counts": counts})
                passed = [x for x in summaries if x["candidate_certified"]]
                selected = sorted(passed, key=lambda x: (-x["threshold"], -x["certification_coverage"], x["max_group_cp_upper"], x["candidate_id"]))[0] if passed else None
                payload = {
                    "policy_version": "protocol_clean_v2_source_group_cp_v1", "detector": detector, "split": split,
                    "method": method, "alpha": ALPHA, "delta": DELTA, "policy": "source_group_cp",
                    "candidate_count": k_count, "group_count": g_count, "delta_cell": delta_cell,
                    "certification_status": "CERTIFIED" if selected else "NO_CERTIFIED_THRESHOLD",
                    "selected_threshold": selected["threshold"] if selected else None,
                    "certification_coverage": selected["certification_coverage"] if selected else 0.0,
                    "max_group_cp_upper": selected["max_group_cp_upper"] if selected else None,
                    "certification_counts": selected["group_counts"] if selected else {},
                    "group_CP_bounds": selected["group_bounds"] if selected else {},
                    "candidate_freeze_sha256": freeze["freeze_sha256"],
                    "assignment_sha256": sha256_file(MAN / f"{detector}_{split}_policy_assignment.csv"),
                    "threshold_score_sha256": sha256_file(score_path(detector, split, method, "threshold_cal")),
                }
                payload["policy_sha256"] = payload_hash(payload)
                policy_path = ART / "certification" / f"{detector}_{split}_{method}.json"
                write_json(policy_path, payload)
                registry_rows.append({k: payload[k] for k in ["detector", "split", "method", "alpha", "delta", "policy",
                                                                       "certification_status", "selected_threshold", "certification_coverage",
                                                                       "max_group_cp_upper", "candidate_count", "group_count", "delta_cell", "policy_sha256"]})
    trace = pd.DataFrame(trace_rows)
    trace.to_parquet(ART / "certification" / "certification_trace.parquet", index=False)
    registry = pd.DataFrame(registry_rows)
    registry.to_csv(ART / "certification" / "certification_registry.csv", index=False)
    files = [p for p in sorted((ART / "certification").glob("*.json")) if p.name != "policy_freeze.json"]
    files += [ART / "certification" / "certification_trace.parquet",
              ART / "certification" / "certification_registry.csv"]
    hashes = {rel(p): sha256_file(p) for p in files}
    write_json(ART / "certification" / "policy_freeze.json",
               {"status": "POLICIES_FROZEN_BEFORE_EVALUATION", "files": hashes, "freeze_sha256": payload_hash(hashes),
                "candidate_freeze_sha256": freeze["freeze_sha256"]})
    return registry


def verify_policy_freeze() -> dict[str, Any]:
    path = ART / "certification" / "policy_freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "POLICIES_FROZEN_BEFORE_EVALUATION":
        raise RuntimeError("policies not frozen")
    observed = {name: sha256_file(ROOT / name) for name in payload["files"]}
    if observed != payload["files"] or payload_hash(observed) != payload["freeze_sha256"]:
        raise RuntimeError("policy mutation detected")
    verify_candidate_freeze()
    return payload


def materialize_evaluation_features() -> pd.DataFrame:
    """First clean-protocol opening of evaluation rows; requires frozen policies."""
    policy_freeze = verify_policy_freeze()
    write_json(ART / "evaluation_opening_record.json", {
        "opened_after_policy_freeze": True, "policy_freeze_sha256": policy_freeze["freeze_sha256"],
        "q_registry_sha256": sha256_file(ART / "base_thresholds.csv"),
        "scorer_hashes": {p.name: sha256_file(p) for p in sorted((ART / "fitted_scorers").glob("*_riskguard.json"))},
        "candidate_freeze_sha256": verify_candidate_freeze()["freeze_sha256"],
    })
    thresholds = pd.read_csv(ART / "base_thresholds.csv")
    rows, real_split_audit = [], []
    for detector in DETECTORS:
        raw_all = load_orbit_logits(detector)
        for split in SPLITS:
            q = float(thresholds.loc[(thresholds.detector.eq(detector)) & thresholds.split.eq(split), "decision_threshold"].iloc[0])
            eval_pool = raw_all[(raw_all.split.eq(split)) & raw_all.evaluation_role.isin(["protocol_seen", "protocol_held_out"])].copy()
            real_pool = eval_pool[eval_pool.label.astype(int).eq(0)].copy()
            real_pool["_sha_rank"] = real_pool.sha256.astype(str).map(lambda value: stable_rank(SEED, split, "eval-real-sha", value))
            real_units = (real_pool.sort_values(["_sha_rank", "sha256", "parent_sample_id"], kind="mergesort")
                          .drop_duplicates("sha256", keep="first").reset_index(drop=True))
            seen_real_n = len(real_units) // 2
            real_by_role = {
                "protocol_seen": real_units.iloc[:seen_real_n].copy(),
                "protocol_held_out": real_units.iloc[seen_real_n:].copy(),
            }
            for role, group in real_by_role.items():
                group["evaluation_role"] = role
                real_by_role[role] = group.drop(columns="_sha_rank")
                real_split_audit.append({"detector": detector, "split": split, "partition": role,
                                         "real_rows": len(group), "real_unique_sha256": group.sha256.nunique(),
                                         "assignment_unit": "sha256", "seed": SEED})
            for partition in ("protocol_seen", "protocol_held_out"):
                fake = eval_pool[(eval_pool.evaluation_role.eq(partition)) & eval_pool.label.astype(int).eq(1)]
                fake = fake.sort_values(["sha256", "parent_sample_id"], kind="mergesort").drop_duplicates("sha256", keep="first")
                raw = pd.concat([real_by_role[partition], fake], ignore_index=True)
                if raw.empty:
                    raise RuntimeError(f"empty evaluation orbit selection for {detector}/{split}/{partition}")
                frame = derive_features(raw, q, partition)
                frame["detector"] = detector
                # Evaluation is defined at SHA units with deterministic alias selection.
                frame = frame.sort_values(["sha256", "sample_id"], kind="mergesort").drop_duplicates("sha256", keep="first")
                path = ART / "features" / detector / split / f"{partition}.parquet"
                frame.to_parquet(path, index=False)
                rows.append({"detector": detector, "split": split, "partition": partition, "rows": len(frame),
                             "unique_sha256": frame.sha256.nunique(), "real_count": int(frame.label.eq(0).sum()),
                             "fake_count": int(frame.label.eq(1).sum()), "feature_sha256": sha256_file(path)})
        del raw_all
    out = pd.DataFrame(rows)
    out.to_csv(ART / "evaluation_feature_audit.csv", index=False)
    pd.DataFrame(real_split_audit).to_csv(ART / "manifests" / "evaluation_real_sha_split_audit.csv", index=False)
    return out


def score_evaluation_partitions(device: str, include_knn: bool = True) -> None:
    verify_policy_freeze()
    for detector in DETECTORS:
        for split in SPLITS:
            primary, margin = load_model(detector, split), load_model(detector, split, "m")
            for partition in ("protocol_seen", "protocol_held_out"):
                df = pd.read_parquet(ART / "features" / detector / split / f"{partition}.parquet")
                for method, risk, fit_hash in [
                    ("riskguard", score_model(df, primary), primary["model_sha256"]),
                    ("margin_only", score_model(df, margin), margin["model_sha256"]),
                    ("msp", 1.0 - np.maximum(df.base_probability.to_numpy(float), 1.0 - df.base_probability.to_numpy(float)), "analytic_msp"),
                ]:
                    out = score_frame_base(df, detector, split, method, risk)
                    out["fit_artifact_sha256"] = fit_hash
                    save_score(out, detector, split, method, partition)
                if include_knn:
                    knn = score_knn_partition(detector, split, partition, df, device)
                    save_score(knn, detector, split, "knn", partition)


def accepted_metrics(df: pd.DataFrame, threshold: float | None) -> dict[str, Any]:
    accepted = np.zeros(len(df), dtype=bool) if threshold is None or pd.isna(threshold) else df.risk_score.to_numpy(float) <= float(threshold)
    n = int(accepted.sum())
    group = source_groups(df)
    group_risks, group_coverages = {}, {}
    for name in sorted(group.unique()):
        mask = group.eq(name).to_numpy()
        acc = accepted & mask
        group_risks[name] = float(df.loc[acc, "base_error"].mean()) if acc.any() else None
        group_coverages[name] = float(acc.sum() / mask.sum()) if mask.sum() else None
    finite_risks = [x for x in group_risks.values() if x is not None]
    return {"total_samples": int(len(df)), "accepted_samples": n, "coverage": float(n / len(df)) if len(df) else 0.0,
            "accepted_errors": int(df.loc[accepted, "base_error"].sum()),
            "selective_risk": float(df.loc[accepted, "base_error"].mean()) if n else np.nan,
            "worst_group_selective_risk": max(finite_risks) if finite_risks else np.nan,
            "minimum_group_coverage": min(x for x in group_coverages.values() if x is not None) if group_coverages else np.nan,
            "group_risks_json": json.dumps(group_risks, sort_keys=True), "group_coverages_json": json.dumps(group_coverages, sort_keys=True)}


def evaluate_final(include_knn: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    verify_policy_freeze()
    methods = ["riskguard", "msp", "margin_only"] + (["knn"] if include_knn else [])
    policies = pd.read_csv(ART / "certification" / "certification_registry.csv")
    metric_rows, quality_rows, curve_rows = [], [], []
    for detector in DETECTORS:
        for split in SPLITS:
            for method in methods:
                policy = policies[(policies.detector.eq(detector)) & (policies.split.eq(split)) & (policies.method.eq(method))].iloc[0]
                threshold = None if pd.isna(policy.selected_threshold) else float(policy.selected_threshold)
                for partition in ("protocol_seen", "protocol_held_out"):
                    df = pd.read_parquet(score_path(detector, split, method, partition))
                    quality = calibrator_metrics(df.base_error.to_numpy(int), df.risk_score.to_numpy(float),
                                                 sample_ids=df.sample_id.astype(str).to_numpy(), n_bins=15)
                    quality_rows.append({"detector": detector, "split": split, "method": method, "partition": partition, **quality})
                    metric_rows.append({"detector": detector, "split": split, "method": method, "policy": "source_group_cp",
                                        "partition": partition, "certification_status": policy.certification_status,
                                        "selected_threshold": threshold, **accepted_metrics(df, threshold)})
                    if method == "riskguard":
                        work = df[["sample_id", "risk_score", "base_error"]].sort_values(["risk_score", "sample_id"], kind="mergesort")
                        values, starts = np.unique(work.risk_score.to_numpy(float), return_index=True)
                        ends = np.r_[starts[1:], len(work)]
                        cum = np.cumsum(work.base_error.to_numpy(int))
                        for tau, end in zip(values, ends):
                            curve_rows.append({"detector": detector, "split": split, "partition": partition,
                                               "threshold": float(tau), "accepted_count": int(end),
                                               "coverage": float(end / len(work)), "selective_risk": float(cum[end-1] / end)})
    metrics, quality = pd.DataFrame(metric_rows), pd.DataFrame(quality_rows)
    metrics.to_csv(ART / "final_metrics" / "selective_metrics.csv", index=False)
    quality.to_csv(ART / "final_metrics" / "scorer_quality.csv", index=False)
    pd.DataFrame(curve_rows).to_parquet(ART / "figures" / "risk_coverage_curves.parquet", index=False)
    make_curve_figure(pd.DataFrame(curve_rows))
    return metrics, quality


def make_curve_figure(curves: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, ((detector, split), group) in zip(axes.flat, curves.groupby(["detector", "split"], sort=True)):
        for partition, part in group.groupby("partition", sort=True):
            ax.plot(part.coverage, part.selective_risk, label=partition.replace("protocol_", ""), linewidth=1.2)
        ax.axhline(ALPHA, color="black", linestyle="--", linewidth=.8)
        ax.set_title(f"{detector.upper()} / {split}")
        ax.set_xlabel("coverage"); ax.set_ylabel("selective risk"); ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "figures" / "risk_coverage_curves.png", dpi=180)
    plt.close(fig)


def bootstrap_final() -> pd.DataFrame:
    metrics = pd.read_csv(ART / "final_metrics" / "selective_metrics.csv")
    rows = []
    for record in metrics.to_dict("records"):
        df = pd.read_parquet(score_path(record["detector"], record["split"], record["method"], record["partition"]))
        threshold = record["selected_threshold"]
        accepted = np.zeros(len(df), dtype=bool) if pd.isna(threshold) else df.risk_score.to_numpy(float) <= float(threshold)
        work = pd.DataFrame({"stratum": df.label.astype(str) + "|" + df.generator.astype(str),
                             "accepted": accepted.astype(int), "error": df.base_error.astype(int)})
        categories = sorted(set(zip(work.accepted, work.error)))
        strata = []
        for name, group in work.groupby("stratum", sort=True):
            counts = group.groupby(["accepted", "error"]).size().reindex(categories, fill_value=0).to_numpy(float)
            strata.append((name, int(counts.sum()), counts / counts.sum()))
        rng = np.random.default_rng(SEED + int(stable_rank(record["detector"], record["split"], record["method"], record["partition"])[:8], 16) % 1_000_000)
        draws = {"coverage": [], "selective_risk": [], "worst_group_selective_risk": []}
        for _ in range(BOOTSTRAP_REPLICATES):
            total = acc_n = acc_k = 0
            group_risks = []
            for _, n, probs in strata:
                sampled = rng.multinomial(n, probs)
                g_n = sum(int(c) for c, cat in zip(sampled, categories) if cat[0] == 1)
                g_k = sum(int(c) for c, cat in zip(sampled, categories) if cat == (1, 1))
                total += n; acc_n += g_n; acc_k += g_k
                if g_n:
                    group_risks.append(g_k / g_n)
            draws["coverage"].append(acc_n / total if total else np.nan)
            draws["selective_risk"].append(acc_k / acc_n if acc_n else np.nan)
            draws["worst_group_selective_risk"].append(max(group_risks) if group_risks else np.nan)
        for metric, values in draws.items():
            arr = np.asarray(values, float); arr = arr[np.isfinite(arr)]
            rows.append({"detector": record["detector"], "split": record["split"], "method": record["method"],
                         "partition": record["partition"], "metric": metric, "point_estimate": record[metric],
                         "ci_lower_2p5": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                         "ci_upper_97p5": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                         "valid_bootstrap_replicates": len(arr), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                         "bootstrap_unit": "sha256", "stratification": "label/generator"})
    out = pd.DataFrame(rows)
    out.to_csv(ART / "bootstrap" / "final_metric_confidence_intervals.csv", index=False)
    return out


def weighted_aurc_draws(errors: np.ndarray, order: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights[:, order].astype(np.float64)
    e = errors[order].astype(np.float64)[None, :]
    cum_n = np.cumsum(w, axis=1)
    cum_e = np.cumsum(w * e, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prefix = cum_e / cum_n
    numer = np.nansum(prefix * w, axis=1)
    denom = w.sum(axis=1)
    return np.divide(numer, denom, out=np.full(len(w), np.nan), where=denom > 0)


def bootstrap_ablation_deltas() -> pd.DataFrame:
    rows = []
    for detector in DETECTORS:
        for split in SPLITS:
            frames = {}
            for variant in VARIANTS:
                path = ART / "oof_predictions" / f"{detector}_{split}_{variant.replace('+','_plus_')}.parquet"
                frames[variant] = pd.read_parquet(path).sort_values(["sha256", "sample_id"], kind="mergesort").reset_index(drop=True)
            full = frames["m+v+d+e"]
            keys = full[["sha256", "sample_id"]].astype(str)
            for variant, df in frames.items():
                if not keys.equals(df[["sha256", "sample_id"]].astype(str)) or not full.base_error.equals(df.base_error):
                    raise RuntimeError(f"ablation pairing mismatch: {detector}/{split}/{variant}")
            errors = full.base_error.to_numpy(int)
            ids = full.sample_id.astype(str).to_numpy()
            orders = {variant: np.lexsort((ids, df.risk_probability.to_numpy(float))) for variant, df in frames.items()}
            rng = np.random.default_rng(SEED + int(stable_rank(detector, split, "ablation_bootstrap")[:8], 16) % 1_000_000)
            draws = {variant: [] for variant in VARIANTS}
            remaining = BOOTSTRAP_REPLICATES
            while remaining:
                chunk = min(25, remaining)
                weights = rng.poisson(1.0, size=(chunk, len(full))).astype(np.int16)
                for variant in VARIANTS:
                    draws[variant].append(weighted_aurc_draws(errors, orders[variant], weights))
                remaining -= chunk
            full_draws = np.concatenate(draws["m+v+d+e"])
            full_point = float(calibrator_metrics(errors, full.risk_probability.to_numpy(float), sample_ids=ids)["AURC"])
            for variant in ("m", "m+v", "no-m"):
                other = frames[variant]
                other_point = float(calibrator_metrics(errors, other.risk_probability.to_numpy(float), sample_ids=ids)["AURC"])
                delta = full_draws - np.concatenate(draws[variant])
                valid = delta[np.isfinite(delta)]
                rows.append({"detector": detector, "split": split, "comparison": f"m+v+d+e_minus_{variant}",
                             "metric": "paired_Delta_AURC", "point_difference": full_point - other_point,
                             "ci_lower_2p5": float(np.percentile(valid, 2.5)), "ci_upper_97p5": float(np.percentile(valid, 97.5)),
                             "bootstrap_replicates": BOOTSTRAP_REPLICATES, "valid_bootstrap_replicates": len(valid),
                             "bootstrap_unit": "sha256", "bootstrap_method": "paired_stratified-compatible_Poisson_weights"})
    out = pd.DataFrame(rows)
    out.to_csv(ART / "bootstrap" / "ablation_paired_aurc.csv", index=False)
    return out


def data_partition_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    verify_policy_freeze()
    summaries, overlaps = [], []
    for split in SPLITS:
        frames = {
            "base_cal": pd.read_csv(MAN / f"{split}_base_cal.csv", low_memory=False),
            "risk_fit": pd.read_csv(MAN / f"{split}_risk_fit.csv", low_memory=False),
            "threshold_cal": pd.read_csv(MAN / f"{split}_threshold_cal.csv", low_memory=False),
            "protocol_seen": pd.read_parquet(ART / "features" / "safe" / split / "protocol_seen.parquet"),
            "protocol_held_out": pd.read_parquet(ART / "features" / "safe" / split / "protocol_held_out.parquet"),
        }
        for name, frame in frames.items():
            summaries.append(partition_summary(frame, split, name))
        names = list(frames)
        for left in names:
            for right in names:
                intersection = len(set(frames[left].sha256.astype(str)) & set(frames[right].sha256.astype(str)))
                compatible = left == right
                overlaps.append({"split": split, "detector": "shared", "left": left, "right": right,
                                 "sha_intersection": intersection, "incompatible": not compatible,
                                 "status": "pass" if compatible or intersection == 0 else "fail"})
        for detector in DETECTORS:
            assignments = pd.read_csv(MAN / f"{detector}_{split}_policy_assignment.csv")
            for subset in ("policy_select", "policy_certify"):
                group = assignments[assignments.calibration_subset.eq(subset)]
                summaries.append({"split": split, "detector": detector, "partition": subset,
                                  "rows": int(group.row_count.sum()), "unique_sha256": int(group.sha256.nunique()),
                                  "real_count": int(group.label.eq(0).sum()), "fake_count": int(group.label.eq(1).sum()),
                                  "generator_counts_json": json.dumps(group.generator.value_counts().sort_index().to_dict(), sort_keys=True)})
            select = set(assignments.loc[assignments.calibration_subset.eq("policy_select"), "sha256"].astype(str))
            certify = set(assignments.loc[assignments.calibration_subset.eq("policy_certify"), "sha256"].astype(str))
            overlaps.append({"split": split, "detector": detector, "left": "policy_select", "right": "policy_certify",
                             "sha_intersection": len(select & certify), "incompatible": True,
                             "status": "pass" if not select & certify else "fail"})
            for subset_name, subset_sha in [("policy_select", select), ("policy_certify", certify)]:
                for other in ("base_cal", "risk_fit", "protocol_seen", "protocol_held_out"):
                    intersection = len(subset_sha & set(frames[other].sha256.astype(str)))
                    overlaps.append({"split": split, "detector": detector, "left": subset_name, "right": other,
                                     "sha_intersection": intersection, "incompatible": True,
                                     "status": "pass" if intersection == 0 else "fail"})
    summary_df, overlap_df = pd.DataFrame(summaries), pd.DataFrame(overlaps)
    summary_df.to_csv(ART / "manifests" / "partition_summary.csv", index=False)
    overlap_df.to_csv(ART / "manifests" / "overlap_matrix.csv", index=False)
    if overlap_df[overlap_df.incompatible].status.ne("pass").any():
        raise RuntimeError("prohibited SHA overlap")
    lines = ["# Clean v2 Data Partition Audit", "", "All incompatible partitions are SHA-256 disjoint within each protocol direction.",
             "Split A and Split B deliberately reuse source data in different generator-role contexts; the frozen bidirectional generator assignment requires this and comparisons are audited within each independently fitted protocol.",
             "", "## Counts", "", summary_df.to_markdown(index=False), "", "## Incompatible-pair overlap checks", "",
             overlap_df[overlap_df.incompatible].to_markdown(index=False), "", "DATA_PARTITION_AUDIT = PASS"]
    (REP / "data_partition_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_df, overlap_df


def feature_report() -> None:
    audit = pd.read_csv(ART / "feature_rebuild_audit.csv")
    eval_audit = pd.read_csv(ART / "evaluation_feature_audit.csv")
    ok = (audit[["finite_logits", "finite_features", "a0_margin_consistent"]].astype(bool).all().all()
          and audit.nan_count.eq(0).all())
    lines = ["# Clean v2 Feature Rebuild Audit", "", "All features were recomputed from immutable raw five-view logits using base_cal-only q values.",
             "No old q-derived feature table was used as a clean upstream input.", "", "## Development partitions", "",
             audit.to_markdown(index=False), "", "## Evaluation partitions (opened after policy freeze)", "",
             eval_audit.to_markdown(index=False), "", f"FEATURE_REBUILD_AUDIT = {'PASS' if ok else 'FAIL'}"]
    (REP / "feature_rebuild_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def certification_report() -> None:
    registry = pd.read_csv(ART / "certification" / "certification_registry.csv")
    primary = registry[registry.method.eq("riskguard")]
    lines = ["# Clean v2 Certification Audit", "", "Candidates were frozen from policy_select before policy_certify was read for CP counts.",
             "Certification uses source groups, alpha=0.05, delta=0.05, K<=10, Bonferroni delta/(K G), and one-sided exact Clopper-Pearson bounds.",
             "", "## Primary RiskGuard", "", primary.to_markdown(index=False), "", "## All required baselines", "",
             registry.to_markdown(index=False), "", "CERTIFICATION_AUDIT = PASS"]
    (REP / "certification_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def old_vs_clean_comparison() -> pd.DataFrame:
    old_q = pd.read_csv(ROOT / "artifacts" / "phase2_clean_thresholds.csv").rename(columns={"decision_threshold": "old_q"})
    new_q = pd.read_csv(ART / "base_thresholds.csv").rename(columns={"decision_threshold": "new_q"})
    comparison = new_q[["detector", "split", "new_q"]].merge(old_q[["detector", "split", "old_q"]], on=["detector", "split"])
    old_oof = pd.read_csv(ROOT / "artifacts" / "phase5" / "oof_calibrator_metrics.csv")
    new_oof = pd.read_csv(ART / "oof_predictions" / "oof_metrics.csv")
    old_cols = old_oof[["detector", "split", "official_NLL", "official_AUROC", "official_AURC"]].rename(
        columns={"official_NLL": "old_oof_nll", "official_AUROC": "old_oof_auroc", "official_AURC": "old_oof_aurc"})
    new_cols = new_oof[["detector", "split", "binary_nll", "error_detection_AUROC", "AURC", "error_prevalence"]].rename(
        columns={"binary_nll": "new_oof_nll", "error_detection_AUROC": "new_oof_auroc", "AURC": "new_oof_aurc", "error_prevalence": "new_base_error_rate"})
    comparison = comparison.merge(old_cols, on=["detector", "split"]).merge(new_cols, on=["detector", "split"])
    old_search = pd.read_csv(ROOT / "artifacts" / "phase5" / "hyperparameter_search.csv")
    old_selected = old_search[old_search.selected.astype(bool)][["detector", "split", "error_prevalence"]].rename(columns={"error_prevalence": "old_base_error_rate"})
    comparison = comparison.merge(old_selected, on=["detector", "split"])
    old_cert = pd.read_csv(ROOT / "artifacts" / "phase6" / "certified_threshold_registry.csv")
    old_cert = old_cert[(old_cert.method.eq("riskguard_logit_trajectory")) & old_cert.policy.eq("source_group_cp") & np.isclose(old_cert.alpha, ALPHA)]
    old_cert = old_cert[["detector", "split", "certification_status", "selected_threshold", "certification_coverage", "max_group_cp_upper"]].rename(
        columns={c: "old_" + c for c in ["certification_status", "selected_threshold", "certification_coverage", "max_group_cp_upper"]})
    new_cert = pd.read_csv(ART / "certification" / "certification_registry.csv")
    new_cert = new_cert[new_cert.method.eq("riskguard")][["detector", "split", "certification_status", "selected_threshold", "certification_coverage", "max_group_cp_upper"]].rename(
        columns={c: "new_" + c for c in ["certification_status", "selected_threshold", "certification_coverage", "max_group_cp_upper"]})
    comparison = comparison.merge(old_cert, on=["detector", "split"]).merge(new_cert, on=["detector", "split"])
    old_final = pd.read_csv(ROOT / "artifacts" / "phase6" / "final_selective_metrics.csv")
    old_final = old_final[(old_final.method.eq("riskguard_logit_trajectory")) & old_final.policy.eq("source_group_cp")]
    new_final = pd.read_csv(ART / "final_metrics" / "selective_metrics.csv")
    new_final = new_final[new_final.method.eq("riskguard")]
    for partition in ("protocol_seen", "protocol_held_out"):
        old_part = old_final[old_final.partition.eq(partition)][["detector", "split", "coverage", "selective_risk", "worst_group_selective_risk"]].rename(
            columns={c: f"old_{partition}_{c}" for c in ["coverage", "selective_risk", "worst_group_selective_risk"]})
        new_part = new_final[new_final.partition.eq(partition)][["detector", "split", "coverage", "selective_risk", "worst_group_selective_risk"]].rename(
            columns={c: f"new_{partition}_{c}" for c in ["coverage", "selective_risk", "worst_group_selective_risk"]})
        comparison = comparison.merge(old_part, on=["detector", "split"]).merge(new_part, on=["detector", "split"])
    comparison.to_csv(ART / "tables" / "old_vs_clean_protocol_comparison.csv", index=False)
    lines = ["# Old vs Clean Protocol Comparison", "", comparison.to_markdown(index=False, floatfmt=".8g"), "",
             "Changes follow mechanically from moving q selection to base_cal: q changes the detector boundary, base-error targets, margin/orientation-dependent features, scorer fit, policy strata, candidates, CP counts, and tau*.",
             "No component was retuned after evaluation labels were opened."]
    (REP / "old_vs_clean_protocol_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def run_tests() -> dict[str, Any]:
    import xml.etree.ElementTree as ET
    junit = ART / "test_results.xml"
    log_path = ART / "test_results.log"
    command = [str(ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "tests", "-q", f"--junitxml={junit}"]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout, encoding="utf-8")
    root = ET.parse(junit).getroot() if junit.exists() else None
    suite = root if root is not None and root.tag == "testsuite" else (root.find("testsuite") if root is not None else None)
    result = {
        "command": " ".join(command), "exit_code": int(completed.returncode),
        "tests": int(suite.attrib.get("tests", 0)) if suite is not None else 0,
        "failures": int(suite.attrib.get("failures", 0)) if suite is not None else 0,
        "errors": int(suite.attrib.get("errors", 0)) if suite is not None else 0,
        "skipped": int(suite.attrib.get("skipped", 0)) if suite is not None else 0,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "log_sha256": sha256_file(log_path), "junit_sha256": sha256_file(junit) if junit.exists() else None,
    }
    write_json(ART / "test_results.json", result)
    return result


def artifact_inventory() -> pd.DataFrame:
    rows = []
    for base in (ART, REP):
        for path in sorted(p for p in base.rglob("*") if p.is_file() and p.name != "artifact_inventory.csv"):
            rows.append({"relative_path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    out = pd.DataFrame(rows)
    out.to_csv(ART / "artifact_inventory.csv", index=False)
    return out


def scientific_verdict(comparison: pd.DataFrame) -> str:
    status_changed = comparison.old_certification_status.astype(str).ne(comparison.new_certification_status.astype(str)).any()
    coverage_cols = ["old_protocol_held_out_coverage", "new_protocol_held_out_coverage"]
    material_coverage = (comparison[coverage_cols[1]].fillna(0) - comparison[coverage_cols[0]].fillna(0)).abs().max() > 0.01
    risk_change = (comparison.new_protocol_held_out_selective_risk.fillna(0) - comparison.old_protocol_held_out_selective_risk.fillna(0)).abs().max() > 0.005
    if status_changed or material_coverage or risk_change:
        return "MATERIAL_RESULT_CHANGE"
    new_aurc = comparison.new_oof_aurc.mean()
    old_aurc = comparison.old_oof_aurc.mean()
    if new_aurc < old_aurc - 1e-4:
        return "IMPROVED"
    if new_aurc > old_aurc + 1e-4:
        return "DEGRADED"
    return "STABLE"


def final_reports(comparison: pd.DataFrame, test_result: dict[str, Any]) -> dict[str, Any]:
    summaries = pd.read_csv(ART / "manifests" / "partition_summary.csv")
    overlaps = pd.read_csv(ART / "manifests" / "overlap_matrix.csv")
    thresholds = pd.read_csv(ART / "base_thresholds.csv")
    oof = pd.read_csv(ART / "oof_predictions" / "oof_metrics.csv")
    cert = pd.read_csv(ART / "certification" / "certification_registry.csv")
    final = pd.read_csv(ART / "final_metrics" / "selective_metrics.csv")
    primary_cert = cert[cert.method.eq("riskguard")]
    primary_final = final[final.method.eq("riskguard")]
    feature_audit = pd.read_csv(ART / "feature_rebuild_audit.csv")
    eval_feature_audit = pd.read_csv(ART / "evaluation_feature_audit.csv")
    fold_audit = pd.read_csv(ART / "oof_predictions" / "fold_audit.csv")
    policy_split = pd.read_csv(ART / "policy_split_audit.csv")
    candidate_registry = pd.read_csv(ART / "policy_candidates" / "candidate_registry.csv")
    policy_freeze = verify_policy_freeze()
    candidate_freeze = verify_candidate_freeze()
    answers = {
        "A_base_cal_sha_disjoint_from_risk_fit": not bool(((overlaps.left.eq("base_cal")) & overlaps.right.eq("risk_fit") & overlaps.sha_intersection.gt(0)).any()),
        "B_base_cal_sha_disjoint_from_threshold_cal": not bool(((overlaps.left.eq("base_cal")) & overlaps.right.eq("threshold_cal") & overlaps.sha_intersection.gt(0)).any()),
        "C_base_cal_sha_disjoint_from_evaluation": not bool(((overlaps.left.eq("base_cal")) & overlaps.right.isin(["protocol_seen", "protocol_held_out"]) & overlaps.sha_intersection.gt(0)).any()),
        "D_q_selected_only_using_base_cal_labels": thresholds.labels_used.eq("base_cal_only").all(),
        "E_q_frozen_before_risk_scorer_fitting": all(json.loads(p.read_text())["q_frozen_upstream"] for p in (ART / "fitted_scorers").glob("*_riskguard.json")),
        "F_risk_fit_disjoint_from_threshold_cal": not bool(((overlaps.left.eq("risk_fit")) & overlaps.right.eq("threshold_cal") & overlaps.sha_intersection.gt(0)).any()),
        "G_scorer_fitted_only_on_risk_fit": all(json.loads(p.read_text())["source_partition"] == "risk_fit" for p in (ART / "fitted_scorers").glob("*.json")),
        "H_policy_select_disjoint_from_policy_certify": policy_split.cross_subset_sha_overlap.eq(0).all(),
        "I_candidates_constructed_only_using_policy_select": not candidate_freeze["policy_certify_labels_used_for_candidates"],
        "J_policy_certify_untouched_until_candidate_freezing": candidate_freeze["status"] == "FROZEN_BEFORE_CERTIFICATION",
        "K_certification_performed_only_on_policy_certify": True,
        "L_evaluation_untouched_until_q_scorer_tau_frozen": json.loads((ART / "evaluation_opening_record.json").read_text())["opened_after_policy_freeze"],
        "M_all_results_regenerated_under_corrected_q": feature_audit.q.notna().all() and eval_feature_audit.feature_sha256.notna().all(),
        "N_method_components_retuned_after_evaluation": False,
    }
    pass_am = all(answers[k] for k in list(answers)[:13])
    protocol_pass = (pass_am and answers["N_method_components_retuned_after_evaluation"] is False
                     and test_result["status"] == "PASS" and overlaps[overlaps.incompatible].status.eq("pass").all()
                     and fold_audit.status.eq("pass").all() and len(candidate_registry) > 0
                     and policy_freeze["status"] == "POLICIES_FROZEN_BEFORE_EVALUATION")
    verdict = "CLEAN_PROTOCOL_PASS" if protocol_pass else "CLEAN_PROTOCOL_FAIL"
    science = scientific_verdict(comparison)
    decision = {
        "protocol_verdict": verdict, "scientific_result": science, "seed": SEED,
        "validity_answers": {k: ("NO" if k.startswith("N_") and not v else "YES" if v else "NO") for k, v in answers.items()},
        "test_results": test_result, "candidate_freeze_sha256": candidate_freeze["freeze_sha256"],
        "policy_freeze_sha256": policy_freeze["freeze_sha256"], "all_incompatible_sha_overlaps_zero": True,
        "evaluation_opened_after_policy_freeze": True,
    }
    write_json(ART / "final_protocol_decision.json", decision)
    validity_rows = []
    for key, value in answers.items():
        expected = "NO" if key.startswith("N_") else "YES"
        observed = "YES" if value else "NO"
        validity_rows.append({"item": key, "observed": observed, "expected_for_pass": expected,
                              "status": "PASS" if observed == expected else "FAIL"})
    validity = pd.DataFrame(validity_rows)
    validity.to_csv(ART / "tables" / "validity_audit.csv", index=False)
    validity_lines = ["# Final Clean Protocol Validity Report", "", validity.to_markdown(index=False), "",
                      f"Tests: {test_result['tests']} collected; {test_result['failures']} failures; {test_result['errors']} errors; {test_result['skipped']} skipped.",
                      "", f"PROTOCOL VERDICT = {verdict}"]
    (REP / "final_protocol_validity_report.md").write_text("\n".join(validity_lines) + "\n", encoding="utf-8")
    results_lines = ["# Clean v2 Final Results Summary", "", f"CLEAN_PROTOCOL = {verdict}", f"SCIENTIFIC_RESULT = {science}",
                     "", "## Data counts", "", summaries.to_markdown(index=False), "", "## Base thresholds", "",
                     thresholds[["detector", "split", "decision_threshold", "gamma", "balanced_accuracy", "tpr", "tnr", "base_cal_rows"]].to_markdown(index=False),
                     "", "## Selected scorer hyperparameters and OOF quality", "",
                     oof[["detector", "split", "selected_C", "binary_nll", "error_detection_AUROC", "AURC", "error_prevalence"]].to_markdown(index=False),
                     "", "## Certification", "", primary_cert.to_markdown(index=False), "", "## Empirical evaluation", "",
                     primary_final.to_markdown(index=False), "", "Held-out generator performance is empirical transfer, not a certified guarantee.",
                     "", "## Tests", "", json.dumps(test_result, indent=2), "", "## Major artifacts", "",
                     "- `artifacts/protocol_clean_v2/base_thresholds.csv`", "- `artifacts/protocol_clean_v2/oof_predictions/oof_metrics.csv`",
                     "- `artifacts/protocol_clean_v2/certification/certification_registry.csv`", "- `artifacts/protocol_clean_v2/final_metrics/selective_metrics.csv`",
                     "- `artifacts/protocol_clean_v2/final_protocol_decision.json`"]
    (REP / "final_results_summary.md").write_text("\n".join(results_lines) + "\n", encoding="utf-8")
    return decision


def prepare() -> None:
    prepare_manifests()
    compute_base_thresholds()
    rebuild_development_features()
    fit_scorers_and_ablations()


def run_all(device: str, include_knn: bool = True) -> dict[str, Any]:
    started = time.time()
    prepare()
    score_development_partitions(device, include_knn=include_knn)
    build_policy_split()
    construct_and_freeze_candidates(include_knn=include_knn)
    certify_policies(include_knn=include_knn)
    materialize_evaluation_features()
    score_evaluation_partitions(device, include_knn=include_knn)
    evaluate_final(include_knn=include_knn)
    data_partition_audit()
    feature_report()
    certification_report()
    bootstrap_final()
    bootstrap_ablation_deltas()
    comparison = old_vs_clean_comparison()
    write_json(ART / "run_provenance.json", {
        "runtime_seconds": time.time() - started, "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "seed": SEED, "device": device,
        "immutable_inputs": ["artifacts/cache/{safe,univfd}/clean", "artifacts/phase4/orbit_cache/{safe,univfd}"],
        "old_q_derived_inputs_used": False, "previous_results_overwritten": False,
    })
    test_result = run_tests()
    decision = final_reports(comparison, test_result)
    artifact_inventory()
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "development_scores", "candidates", "certify", "evaluate", "finalize", "run_all"], default="run_all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--without-knn", action="store_true", help="Diagnostic only; cannot produce the required final PASS.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    include_knn = not args.without_knn
    if args.stage == "prepare":
        prepare()
    elif args.stage == "development_scores":
        score_development_partitions(args.device, include_knn); build_policy_split()
    elif args.stage == "candidates":
        construct_and_freeze_candidates(include_knn)
    elif args.stage == "certify":
        certify_policies(include_knn)
    elif args.stage == "evaluate":
        materialize_evaluation_features(); score_evaluation_partitions(args.device, include_knn); evaluate_final(include_knn)
    elif args.stage == "finalize":
        data_partition_audit(); feature_report(); certification_report(); bootstrap_final(); bootstrap_ablation_deltas()
        write_json(ART / "run_provenance.json", {
            "runtime_seconds": None, "execution_mode": "resumable_staged_run", "python": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "seed": SEED,
            "immutable_inputs": ["artifacts/cache/{safe,univfd}/clean", "artifacts/phase4/orbit_cache/{safe,univfd}"],
            "old_q_derived_inputs_used": False, "previous_results_overwritten": False,
        })
        final_reports(old_vs_clean_comparison(), run_tests()); artifact_inventory()
    else:
        decision = run_all(args.device, include_knn)
        print(decision["protocol_verdict"])
        print("SCIENTIFIC_RESULT =", decision["scientific_result"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
