#!/usr/bin/env python3
"""Fit Phase 5 RiskGuard primary logistic calibrators."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from selective_detection.error_probability_calibrator import (
    FEATURE_TRANSFORMATIONS,
    PRIMARY_FEATURES,
    TRANSFORMED_FEATURE_NAMES,
    risk_logit as manual_risk_logit,
    risk_probability as manual_risk_probability,
    transform_features,
)
from selective_detection.calibration_metrics import calibrator_metrics
from selective_detection.calibrator_artifact_io import (
    DETECTORS,
    SPLITS,
    combo_slug,
    environment_provenance,
    load_config,
    payload_sha256,
    phase4_feature_path,
    read_json,
    relative_to_root,
    sha256_file,
    verify_frozen_inputs,
    write_default_config,
    write_json,
)


COMMANDS: list[str] = []


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


def fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float, config: dict[str, Any]) -> tuple[LogisticRegression, bool, str]:
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
    optimizer_warning = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    warning_messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    optimizer_warning = " | ".join(warning_messages)
    converged = len(warning_messages) == 0 and int(clf.n_iter_[0]) < int(model_cfg["max_iter"])
    return clf, converged, optimizer_warning


def scaler_from_train(transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = transformed.mean(axis=0)
    scales = transformed.std(axis=0, ddof=0)
    if (scales < 1.0e-12).any():
        raise RuntimeError(f"feature standard deviation below 1e-12: {scales.tolist()}")
    return means, scales


def standardized(raw: pd.DataFrame, feature_order: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = transform_features(raw, feature_order, as_frame=False)
    means, scales = scaler_from_train(transformed)
    return (transformed - means) / scales, means, scales


def select_candidate(rows: list[dict[str, Any]], tolerance: float) -> float:
    candidates = sorted(rows, key=lambda r: float(r["candidate_C"]))
    best = candidates[0]
    for row in candidates[1:]:
        better = False
        for metric in ("binary_nll", "brier_score", "AURC"):
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
    return float(best["candidate_C"])


def transformed_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = transform_features(df, PRIMARY_FEATURES, as_frame=True)
    assert isinstance(out, pd.DataFrame)
    return out


def fit_combo(detector: str, split: str, config: dict[str, Any], config_path: Path, output_root: Path) -> dict[str, Any]:
    slug = combo_slug(detector, split)
    risk_path = phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit")
    df = pd.read_parquet(risk_path)
    feature_hash = sha256_file(risk_path)
    config_hash = sha256_file(config_path)
    fold_path = output_root / "cv_fold_assignments" / f"{slug}.parquet"
    if not fold_path.exists():
        raise RuntimeError(f"missing CV fold assignment: {fold_path}")
    folds = pd.read_parquet(fold_path)
    fold_hash = sha256_file(fold_path)
    df = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
    if df["cv_fold"].isna().any() or len(df) != len(folds):
        raise RuntimeError(f"fold assignment incomplete for {slug}")
    df["cv_fold"] = df["cv_fold"].astype(int)

    raw_features = df.loc[:, list(PRIMARY_FEATURES)]
    selected_c_predictions: dict[float, np.ndarray] = {}
    selected_c_logits: dict[float, np.ndarray] = {}
    search_rows: list[dict[str, Any]] = []
    convergence_rows: dict[float, list[dict[str, Any]]] = {}
    c_values = [float(c) for c in config["regularization"]["candidate_C"]]
    n_bins = int(config["calibration"]["ece_bins"])
    for c_value in c_values:
        probabilities = np.full(len(df), np.nan, dtype=np.float64)
        logits = np.full(len(df), np.nan, dtype=np.float64)
        conv_rows: list[dict[str, Any]] = []
        for fold in range(int(config["cross_validation"]["folds"])):
            train_mask = df["cv_fold"].to_numpy() != fold
            val_mask = ~train_mask
            train_raw = raw_features.loc[train_mask]
            val_raw = raw_features.loc[val_mask]
            train_t = transform_features(train_raw, PRIMARY_FEATURES, as_frame=False)
            means, scales = scaler_from_train(train_t)
            train_z = (train_t - means) / scales
            val_t = transform_features(val_raw, PRIMARY_FEATURES, as_frame=False)
            val_z = (val_t - means) / scales
            clf, converged, warning_text = fit_logistic(train_z, df.loc[train_mask, "base_error"].to_numpy(dtype=np.int64), c_value, config)
            logits[val_mask] = clf.decision_function(val_z)
            probabilities[val_mask] = clf.predict_proba(val_z)[:, 1]
            train_sha = set(df.loc[train_mask, "sha256"].astype(str))
            val_sha = set(df.loc[val_mask, "sha256"].astype(str))
            conv_rows.append(
                {
                    "fold": fold,
                    "converged": bool(converged),
                    "iteration_count": int(clf.n_iter_[0]),
                    "coefficient_norm": float(np.linalg.norm(clf.coef_[0])),
                    "intercept": float(clf.intercept_[0]),
                    "optimizer_warnings": warning_text,
                    "train_row_count": int(train_mask.sum()),
                    "validation_row_count": int(val_mask.sum()),
                    "sha_overlap_train_validation": int(len(train_sha & val_sha)),
                    "scaler_means": means.tolist(),
                    "scaler_scales": scales.tolist(),
                    "coefficient_vector": clf.coef_[0].astype(float).tolist(),
                }
            )
        if np.isnan(probabilities).any():
            raise RuntimeError(f"missing OOF prediction for {slug} C={c_value}")
        metrics = calibrator_metrics(
            df["base_error"].to_numpy(dtype=np.int64),
            probabilities,
            sample_ids=df["sample_id"].astype(str).to_numpy(),
            n_bins=n_bins,
        )
        row = {
            "detector": detector,
            "split": split,
            "candidate_C": c_value,
            "fold_count": int(config["cross_validation"]["folds"]),
            "oof_row_count": int(len(df)),
            "oof_error_count": int(df["base_error"].sum()),
            "binary_nll": metrics["binary_nll"],
            "brier_score": metrics["brier_score"],
            "ece": metrics["ece"],
            "error_detection_AUROC": metrics["error_detection_AUROC"],
            "error_detection_AUPR": metrics["error_detection_AUPR"],
            "AURC": metrics["AURC"],
            "E_AURC": metrics["E_AURC"],
            "converged_fold_count": int(sum(1 for item in conv_rows if item["converged"])),
            "selected": False,
        }
        search_rows.append(row)
        selected_c_predictions[c_value] = probabilities
        selected_c_logits[c_value] = logits
        convergence_rows[c_value] = conv_rows

    selected_c = select_candidate(search_rows, float(config["selection"]["tie_tolerance"]))
    for row in search_rows:
        row["selected"] = bool(float(row["candidate_C"]) == selected_c)

    selected_probs = selected_c_predictions[selected_c]
    selected_logits = selected_c_logits[selected_c]
    transformed = transformed_frame(df)
    oof = df[
        [
            "sample_id",
            "sha256",
            "detector",
            "split",
            "generator",
            "label",
            "base_prediction",
            "base_error",
            "cv_fold",
            *PRIMARY_FEATURES,
        ]
    ].copy()
    for col in transformed.columns:
        oof[col] = transformed[col].to_numpy(dtype=np.float64)
    oof["risk_logit"] = selected_logits
    oof["risk_probability"] = selected_probs
    oof["selected_C"] = selected_c
    oof["config_sha256"] = config_hash
    oof["phase4_feature_artifact_sha256"] = feature_hash
    oof_dir = output_root / "oof_scores"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_dir / f"{slug}_risk_fit.parquet", index=False)

    # Store selected fold model parameters so risk_fit_oof has a concrete model hash.
    fold_model_set = {
        "model_set_version": "riskguard_phase5_oof_fold_models_v1",
        "detector": detector,
        "split": split,
        "feature_order": list(PRIMARY_FEATURES),
        "feature_transformations": {name: FEATURE_TRANSFORMATIONS[name] for name in PRIMARY_FEATURES},
        "selected_C": selected_c,
        "config_sha256": config_hash,
        "phase4_feature_artifact_sha256": feature_hash,
        "fold_assignment_sha256": fold_hash,
        "fold_models": convergence_rows[selected_c],
    }
    fold_model_set["model_set_hash"] = payload_sha256(fold_model_set)
    write_json(output_root / "models" / f"{slug}_riskguard_oof_folds.json", fold_model_set)

    final_t = transform_features(raw_features, PRIMARY_FEATURES, as_frame=False)
    means, scales = scaler_from_train(final_t)
    final_z = (final_t - means) / scales
    final_clf, final_converged, final_warning = fit_logistic(
        final_z,
        df["base_error"].to_numpy(dtype=np.int64),
        selected_c,
        config,
    )
    final_model = {
        "model_version": "riskguard_phase5_logistic_v1",
        "detector": detector,
        "split": split,
        "feature_order": list(PRIMARY_FEATURES),
        "feature_transformations": {name: FEATURE_TRANSFORMATIONS[name] for name in PRIMARY_FEATURES},
        "scaler_means": means.astype(float).tolist(),
        "scaler_scales": scales.astype(float).tolist(),
        "coefficient_vector": final_clf.coef_[0].astype(float).tolist(),
        "intercept": float(final_clf.intercept_[0]),
        "selected_C": selected_c,
        "solver_configuration": config["model"],
        "convergence_status": "converged" if final_converged else "failed",
        "converged": bool(final_converged),
        "iteration_count": int(final_clf.n_iter_[0]),
        "coefficient_norm": float(np.linalg.norm(final_clf.coef_[0])),
        "optimizer_warnings": final_warning,
        "risk_fit_row_count": int(len(df)),
        "risk_fit_error_count": int(df["base_error"].sum()),
        "risk_fit_error_prevalence": float(df["base_error"].mean()),
        "input_manifest_hash": feature_hash,
        "phase4_feature_artifact_sha256": feature_hash,
        "config_hash": config_hash,
        "config_sha256": config_hash,
        "fold_assignment_hash": fold_hash,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    final_model["model_hash"] = payload_sha256(final_model)
    model_dir = output_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(model_dir / f"{slug}_riskguard.json", final_model)

    # Manual scorer parity against scikit-learn decision_function on a deterministic sample.
    sample_n = min(10000, len(df))
    sample = df.sort_values(["sha256", "sample_id"], kind="mergesort").head(sample_n).copy()
    sample_t = transform_features(sample.loc[:, list(PRIMARY_FEATURES)], PRIMARY_FEATURES, as_frame=False)
    sample_z = (sample_t - means) / scales
    library_logits = final_clf.decision_function(sample_z)
    library_probs = expit(library_logits)
    manual_logits = manual_risk_logit(sample.loc[:, list(PRIMARY_FEATURES)], final_model)
    manual_probs = manual_risk_probability(sample.loc[:, list(PRIMARY_FEATURES)], final_model)
    parity = {
        "detector": detector,
        "split": split,
        "sample_count": int(sample_n),
        "max_abs_logit_difference": float(np.max(np.abs(library_logits - manual_logits))),
        "max_abs_probability_difference": float(np.max(np.abs(library_probs - manual_probs))),
        "logit_tolerance": 1.0e-10,
        "probability_tolerance": 1.0e-10,
        "status": "pass"
        if np.max(np.abs(library_logits - manual_logits)) <= 1.0e-10
        and np.max(np.abs(library_probs - manual_probs)) <= 1.0e-10
        else "fail",
    }

    if not final_converged:
        raise RuntimeError(f"final model did not converge for {slug}")
    if any(row["converged_fold_count"] < int(config["cross_validation"]["folds"]) for row in search_rows if row["selected"]):
        raise RuntimeError(f"selected fold model did not converge for {slug}")
    return {
        "search_rows": search_rows,
        "parity_row": parity,
        "model_path": relative_to_root(PROJECT_ROOT, model_dir / f"{slug}_riskguard.json"),
        "selected_C": selected_c,
    }


def upsert_by_combo(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old_marker = old[keys].astype(str).agg("\x1f".join, axis=1)
        new_marker = set(new[keys].astype(str).agg("\x1f".join, axis=1))
        old = old[~old_marker.isin(new_marker)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def main() -> None:
    started = time.time()
    args = parse_args()
    COMMANDS.append(" ".join(sys.argv))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    if args.force or not config_path.exists():
        write_default_config(PROJECT_ROOT)
    verify_frozen_inputs(PROJECT_ROOT, output_root, raise_on_fail=True)
    config = load_config(config_path)
    all_search_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for detector in selected(DETECTORS, args.detector):
        for split in selected(SPLITS, args.split):
            result = fit_combo(detector, split, config, config_path, output_root)
            all_search_rows.extend(result["search_rows"])
            parity_rows.append(result["parity_row"])
    upsert_by_combo(output_root / "hyperparameter_search.csv", all_search_rows, ["detector", "split", "candidate_C"])
    upsert_by_combo(output_root / "manual_scoring_parity_audit.csv", parity_rows, ["detector", "split"])
    provenance = environment_provenance(PROJECT_ROOT, COMMANDS, started)
    write_json(output_root / "environment_provenance.json", provenance)


if __name__ == "__main__":
    main()
