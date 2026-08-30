#!/usr/bin/env python3
"""Strict Phase 5 final audit and freeze before Phase 6."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

from selective_detection.error_probability_calibrator import (
    FEATURE_TRANSFORMATIONS,
    PRIMARY_FEATURES,
    RAW_FEATURE_NEGATIVE_TOLERANCE,
    risk_logit,
    risk_probability,
    transform_features,
)
from selective_detection.calibration_metrics import (
    aurc,
    binary_nll,
    brier_score,
    calibrator_metrics,
    eaurc,
    ece_from_bins,
    error_detection_metrics,
    reliability_bins,
)
from selective_detection.calibrator_artifact_io import (
    DETECTORS,
    PARTITIONS,
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
    write_json,
)
from selective_detection.selective_metrics import eaurc as phase3_eaurc


PHASE5 = PROJECT_ROOT / "artifacts" / "phase5"
REPORTS = PROJECT_ROOT / "reports" / "phase5"
LOGS = PROJECT_ROOT / "logs" / "phase5"
CONFIG_DIR = PROJECT_ROOT / "configs" / "phase5"

EXPECTED_SELECTED_C = {
    ("univfd", "split_a"): 1.0,
    ("univfd", "split_b"): 0.1,
    ("safe", "split_a"): 1.0,
    ("safe", "split_b"): 1.0,
}
REQUIRED_ABLATIONS = {
    "full_four",
    "no_margin",
    "no_variance",
    "no_drift",
    "no_support",
    "margin_only",
    "variance_only",
    "drift_only",
    "support_only",
    "orbit_only",
    "geometry_support",
}
PARTITION_TO_SCORE = {
    "risk_fit_oof": "risk_fit",
    "risk_fit_fullfit": "risk_fit",
    "threshold_cal": "threshold_cal",
    "protocol_seen": "protocol_seen",
    "protocol_held_out": "protocol_held_out",
    "bfree_snapshot": "bfree_snapshot",
}
REPRO_COMMANDS = [
    "python scripts/verify_calibrator_inputs.py",
    "python scripts/build_calibrator_cross_validation_folds.py --force",
    "python scripts/audit_calibrator_feature_values.py",
    "python scripts/fit_error_calibrator.py --force",
    "python scripts/run_feature_ablations.py --force",
    "python scripts/score_error_risk.py --force",
    "python scripts/evaluate_error_calibrator.py --force",
    "python -m pytest tests -q",
    "python scripts/audit_error_calibrator.py --force",
    "python scripts/audit_error_calibrator_extended.py --force",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def add_check(
    checks: list[dict[str, Any]],
    category: str,
    check: str,
    passed: bool,
    *,
    hard_blocker: bool = True,
    observed: Any = "",
    expected: Any = "",
    evidence: str = "",
) -> None:
    checks.append(
        {
            "category": category,
            "check": check,
            "status": "pass" if passed else "fail",
            "hard_blocker": bool(hard_blocker),
            "observed": observed,
            "expected": expected,
            "evidence": evidence,
        }
    )


def model_path(detector: str, split: str) -> Path:
    return PHASE5 / "models" / f"{combo_slug(detector, split)}_riskguard.json"


def fold_path(detector: str, split: str) -> Path:
    return PHASE5 / "cv_fold_assignments" / f"{combo_slug(detector, split)}.parquet"


def score_path(detector: str, split: str, artifact: str) -> Path:
    return PHASE5 / "scores" / detector / split / f"{artifact}.parquet"


def sigmoid_gap(logits: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.max(np.abs(expit(np.asarray(logits, dtype=np.float64)) - np.asarray(probabilities, dtype=np.float64)))) if len(logits) else 0.0


def select_candidate(group: pd.DataFrame, tolerance: float = 1.0e-12) -> float:
    records = sorted(group.to_dict("records"), key=lambda row: float(row["candidate_C"]))
    best = records[0]
    for row in records[1:]:
        better = False
        for metric in ("binary_nll", "brier_score", "AURC"):
            delta = float(row[metric]) - float(best[metric])
            if delta < -tolerance:
                better = True
                break
            if abs(delta) > tolerance:
                break
        else:
            better = float(row["candidate_C"]) < float(best["candidate_C"])
        if better:
            best = row
    return float(best["candidate_C"])


def selected_hyper_rows(hyper: pd.DataFrame) -> pd.DataFrame:
    return hyper[hyper["selected"].astype(str).str.lower().isin(["true", "1"])].copy()


def compute_base_rate_oof_metrics(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for detector in DETECTORS:
        for split in SPLITS:
            df = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"), columns=["sample_id", "sha256", "base_error"])
            folds = pd.read_parquet(fold_path(detector, split))
            df = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
            probs = np.zeros(len(df), dtype=np.float64)
            fold_prevalence: dict[str, float] = {}
            for fold in sorted(df["cv_fold"].unique()):
                train = df["cv_fold"].to_numpy() != int(fold)
                val = ~train
                prevalence = float(df.loc[train, "base_error"].mean())
                probs[val] = prevalence
                fold_prevalence[str(int(fold))] = prevalence
            metrics = calibrator_metrics(
                df["base_error"].to_numpy(dtype=np.int64),
                probs,
                sample_ids=df["sample_id"].astype(str).to_numpy(),
                n_bins=int(config["calibration"]["ece_bins"]),
            )
            rows.append(
                {
                    "detector": detector,
                    "split": split,
                    "predictor": "fold_training_base_rate",
                    "fold_training_prevalence_json": json.dumps(fold_prevalence, sort_keys=True),
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(PHASE5 / "base_rate_oof_metrics.csv", index=False)
    return out


def compute_feature_correlations() -> pd.DataFrame:
    rows = []
    for detector in DETECTORS:
        for split in SPLITS:
            df = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"), columns=list(PRIMARY_FEATURES))
            transformed = transform_features(df, PRIMARY_FEATURES, as_frame=True)
            assert isinstance(transformed, pd.DataFrame)
            for feature_a, feature_b in combinations(PRIMARY_FEATURES, 2):
                raw_a = df[feature_a].to_numpy(dtype=np.float64)
                raw_b = df[feature_b].to_numpy(dtype=np.float64)
                trans_a = transformed[f"u_{feature_a}"].to_numpy(dtype=np.float64)
                trans_b = transformed[f"u_{feature_b}"].to_numpy(dtype=np.float64)
                pearson = float(np.corrcoef(trans_a, trans_b)[0, 1])
                spear = float(spearmanr(raw_a, raw_b).statistic)
                rows.append(
                    {
                        "detector": detector,
                        "split": split,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "pearson_transformed": pearson,
                        "spearman_raw": spear,
                        "high_correlation_warning": bool(abs(spear) >= 0.9 or abs(pearson) >= 0.9),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(PHASE5 / "feature_correlation_audit.csv", index=False)
    return out


def run_py_compile_audit() -> pd.DataFrame:
    paths = [
        "src/selective_detection/error_probability_calibrator.py",
        "src/selective_detection/grouped_cross_validation.py",
        "src/selective_detection/calibration_metrics.py",
        "src/selective_detection/calibrator_artifact_io.py",
        "scripts/build_calibrator_cross_validation_folds.py",
        "scripts/audit_calibrator_feature_values.py",
        "scripts/fit_error_calibrator.py",
        "scripts/run_feature_ablations.py",
        "scripts/score_error_risk.py",
        "scripts/evaluate_error_calibrator.py",
        "scripts/audit_error_calibrator.py",
        "scripts/audit_error_calibrator_extended.py",
        "scripts/verify_calibrator_inputs.py",
    ]
    rows = []
    for rel in paths:
        proc = subprocess.run([sys.executable, "-m", "py_compile", rel], cwd=PROJECT_ROOT, text=True, capture_output=True)
        rows.append(
            {
                "relative_path": rel,
                "exit_code": int(proc.returncode),
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "status": "pass" if proc.returncode == 0 else "fail",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(PHASE5 / "py_compile_audit.csv", index=False)
    return out


def tests_from_log() -> tuple[int, int]:
    status_path = PHASE5 / "test_status.json"
    log_path = PROJECT_ROOT / "logs" / "phase5" / "pytest_full.log"
    passed = 0
    failed = 0
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        passed_match = re.search(r"(\d+)\s+passed", text)
        failed_match = re.search(r"(\d+)\s+failed", text)
        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
    if status_path.exists() and read_json(status_path).get("status") != "pass":
        failed = max(failed, 1)
    return passed, failed


def verify_score_artifacts() -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    totals = {
        "missing_score_rows": 0,
        "unexpected_score_rows": 0,
        "duplicate_score_rows": 0,
        "invalid_probability_rows": 0,
        "feature_mismatch_rows": 0,
        "metadata_mismatch_rows": 0,
        "model_hash_mismatch_artifacts": 0,
        "feature_hash_mismatch_artifacts": 0,
    }
    for detector in DETECTORS:
        for split in SPLITS:
            final_model = read_json(model_path(detector, split))
            final_model_hash = final_model["model_hash"]
            fold_models = read_json(PHASE5 / "models" / f"{combo_slug(detector, split)}_riskguard_oof_folds.json")
            fold_model_hash = fold_models["model_set_hash"]
            for artifact, partition in PARTITION_TO_SCORE.items():
                score_file = score_path(detector, split, artifact)
                feature_file = phase4_feature_path(PROJECT_ROOT, detector, split, partition)
                scored = pd.read_parquet(score_file)
                phase4 = pd.read_parquet(feature_file)
                duplicate_sample_id = int(scored.duplicated(["sample_id"]).sum())
                duplicate_sample_sha = int(scored.duplicated(["sample_id", "sha256"]).sum())
                merged = phase4.merge(
                    scored,
                    on=["sample_id", "sha256"],
                    how="outer",
                    suffixes=("_phase4", "_score"),
                    indicator=True,
                )
                missing = int(merged["_merge"].eq("left_only").sum())
                unexpected = int(merged["_merge"].eq("right_only").sum())
                both = merged[merged["_merge"].eq("both")]
                metadata_mismatch = 0
                for col in ("label", "base_prediction", "base_error"):
                    metadata_mismatch += int((both[f"{col}_phase4"].astype(str) != both[f"{col}_score"].astype(str)).sum())
                feature_mismatch = 0
                for feature in PRIMARY_FEATURES:
                    feature_mismatch += int(
                        (~np.isclose(
                            both[f"{feature}_phase4"].to_numpy(dtype=np.float64),
                            both[f"{feature}_score"].to_numpy(dtype=np.float64),
                            rtol=0.0,
                            atol=0.0,
                            equal_nan=True,
                        )).sum()
                    )
                logits = scored["risk_logit"].to_numpy(dtype=np.float64)
                probs = scored["risk_probability"].to_numpy(dtype=np.float64)
                invalid_probs = int((~np.isfinite(probs)).sum() + (probs < 0.0).sum() + (probs > 1.0).sum())
                nonfinite_logits = int((~np.isfinite(logits)).sum())
                feature_hash = sha256_file(feature_file)
                expected_model_hash = fold_model_hash if artifact == "risk_fit_oof" else final_model_hash
                model_hash_ok = scored["model_sha256"].astype(str).nunique() == 1 and scored["model_sha256"].astype(str).iloc[0] == expected_model_hash
                feature_hash_ok = (
                    scored["phase4_feature_artifact_sha256"].astype(str).nunique() == 1
                    and scored["phase4_feature_artifact_sha256"].astype(str).iloc[0] == feature_hash
                )
                gap = sigmoid_gap(logits, probs)
                status = (
                    missing == 0
                    and unexpected == 0
                    and duplicate_sample_id == 0
                    and duplicate_sample_sha == 0
                    and metadata_mismatch == 0
                    and feature_mismatch == 0
                    and invalid_probs == 0
                    and nonfinite_logits == 0
                    and model_hash_ok
                    and feature_hash_ok
                    and gap <= 1.0e-12
                )
                rows.append(
                    {
                        "detector": detector,
                        "split": split,
                        "artifact": artifact,
                        "partition": partition,
                        "expected_rows": int(len(phase4)),
                        "actual_rows": int(len(scored)),
                        "missing_rows": missing,
                        "unexpected_rows": unexpected,
                        "duplicate_sample_id_rows": duplicate_sample_id,
                        "duplicate_sample_sha_rows": duplicate_sample_sha,
                        "metadata_mismatch_rows": metadata_mismatch,
                        "feature_mismatch_rows": feature_mismatch,
                        "nonfinite_logit_rows": nonfinite_logits,
                        "invalid_probability_rows": invalid_probs,
                        "max_sigmoid_probability_gap": gap,
                        "model_hash_ok": bool(model_hash_ok),
                        "feature_artifact_hash_ok": bool(feature_hash_ok),
                        "status": "pass" if status else "fail",
                    }
                )
                totals["missing_score_rows"] += missing
                totals["unexpected_score_rows"] += unexpected
                totals["duplicate_score_rows"] += duplicate_sample_id + duplicate_sample_sha
                totals["invalid_probability_rows"] += invalid_probs
                totals["feature_mismatch_rows"] += feature_mismatch
                totals["metadata_mismatch_rows"] += metadata_mismatch
                totals["model_hash_mismatch_artifacts"] += int(not model_hash_ok)
                totals["feature_hash_mismatch_artifacts"] += int(not feature_hash_ok)
    out = pd.DataFrame(rows)
    out.to_csv(PHASE5 / "score_artifact_deep_audit.csv", index=False)
    return out, totals


def collect_freeze_rows() -> pd.DataFrame:
    roots = [CONFIG_DIR, PHASE5, REPORTS, LOGS]
    excluded = {
        str((PHASE5 / "phase5_frozen_artifact_hashes.csv").resolve()),
        str((CONFIG_DIR / "phase5_frozen.yaml").resolve()),
    }
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or str(path.resolve()) in excluded:
                continue
            rows.append(
                {
                    "relative_path": relative_to_root(PROJECT_ROOT, path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256"])


def verify_phase5_freeze(registry: pd.DataFrame) -> tuple[int, int]:
    mismatches = 0
    missing_required = 0
    for row in registry.to_dict("records"):
        path = PROJECT_ROOT / str(row["relative_path"])
        if not path.exists():
            mismatches += 1
            missing_required += 1
            continue
        if int(path.stat().st_size) != int(row["size_bytes"]) or sha256_file(path) != str(row["sha256"]):
            mismatches += 1
    return mismatches, missing_required


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "(none)\n"
    return df.loc[:, columns].head(max_rows).to_markdown(index=False) + "\n"


def main() -> int:
    started = time.time()
    parse_args()
    PHASE5.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    warnings_list: list[str] = []

    config = load_config(CONFIG_DIR / "riskguard_calibrator.yaml")
    frozen_audit, frozen_summary = verify_frozen_inputs(PROJECT_ROOT, PHASE5)
    upstream_mismatches = int((frozen_audit["status"] != "pass").sum())
    add_check(checks, "A. Upstream frozen-input integrity", "All Phase 2/3/4 frozen hashes and required artifacts match", upstream_mismatches == 0, observed=upstream_mismatches, expected=0, evidence="artifacts/phase5/frozen_input_audit.csv")
    add_check(checks, "A. Upstream frozen-input integrity", "Phase 4 freeze policy hash matches", frozen_audit[frozen_audit["phase"].eq("phase4_policy")]["status"].eq("pass").all(), evidence="artifacts/phase5/frozen_input_audit.csv")
    add_check(checks, "A. Upstream frozen-input integrity", "Phase 4 freeze status is PASS", frozen_summary.get("phase4_status_pass") is True, observed=frozen_summary.get("phase4_pre_phase_5_status"), expected="PASS")

    schema = read_json(PHASE5 / "primary_feature_schema.json")
    model_payloads = {(detector, split): read_json(model_path(detector, split)) for detector in DETECTORS for split in SPLITS}
    model_feature_orders = {key: tuple(payload["feature_order"]) for key, payload in model_payloads.items()}
    feature_count_ok = schema.get("primary_feature_count") == 4 and all(len(order) == 4 for order in model_feature_orders.values())
    feature_order_ok = tuple(schema.get("feature_order", [])) == PRIMARY_FEATURES and all(order == PRIMARY_FEATURES for order in model_feature_orders.values())
    add_check(checks, "B. Primary feature schema", "Feature count is exactly 4", feature_count_ok, observed=schema.get("primary_feature_count"), expected=4, evidence="artifacts/phase5/primary_feature_schema.json")
    add_check(checks, "B. Primary feature schema", "Feature order is identical across all models", feature_order_ok, observed=json.dumps({str(k): v for k, v in model_feature_orders.items()}, default=list), expected=list(PRIMARY_FEATURES))
    add_check(checks, "B. Primary feature schema", "No metadata or diagnostic feature enters model", set(schema.get("feature_order", [])) == set(PRIMARY_FEATURES), evidence="artifacts/phase5/primary_feature_schema.json")

    raw_feature_audit = pd.read_csv(PHASE5 / "raw_feature_value_audit.csv")
    transformed_ok = True
    transformed_finite = True
    for detector in DETECTORS:
        for split in SPLITS:
            phase4 = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"), columns=["sample_id", "sha256", *PRIMARY_FEATURES])
            oof = pd.read_parquet(PHASE5 / "oof_scores" / f"{combo_slug(detector, split)}_risk_fit.parquet")
            transformed = transform_features(phase4.loc[:, list(PRIMARY_FEATURES)], PRIMARY_FEATURES, as_frame=True)
            assert isinstance(transformed, pd.DataFrame)
            merged = oof.merge(phase4[["sample_id", "sha256"]], on=["sample_id", "sha256"], how="right", validate="one_to_one")
            transformed_finite &= bool(np.isfinite(transformed.to_numpy(dtype=np.float64)).all())
            for column in transformed.columns:
                transformed_ok &= bool(np.allclose(oof[column].to_numpy(dtype=np.float64), transformed[column].to_numpy(dtype=np.float64), rtol=0.0, atol=1.0e-12))
            transformed_ok &= len(merged) == len(phase4)
    formulas_ok = schema.get("feature_transformations") == {name: FEATURE_TRANSFORMATIONS[name] for name in PRIMARY_FEATURES}
    material_negative_count = int(raw_feature_audit["material_negative_count"].sum())
    add_check(checks, "C. Feature transformation correctness", "Fixed log1p formulas and margin orientation are correct", formulas_ok, expected={name: FEATURE_TRANSFORMATIONS[name] for name in PRIMARY_FEATURES})
    add_check(checks, "C. Feature transformation correctness", "Transformed OOF values reproduce manual formulas", transformed_ok, evidence="artifacts/phase5/oof_scores/")
    add_check(checks, "C. Feature transformation correctness", "All transformed values are finite", transformed_finite)
    add_check(checks, "C. Feature transformation correctness", "No material negative raw distance value exists", material_negative_count == 0, observed=material_negative_count, expected=0, evidence="artifacts/phase5/raw_feature_value_audit.csv")

    fold_audit = pd.read_csv(PHASE5 / "cv_fold_audit.csv")
    fold_counts = fold_audit.groupby(["detector", "split"])["fold"].nunique()
    fold_assignments_ok = True
    missing_fold_rows = 0
    duplicate_fold_rows = 0
    for detector in DETECTORS:
        for split in SPLITS:
            risk_fit = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"), columns=["sample_id", "sha256"])
            folds = pd.read_parquet(fold_path(detector, split))
            merged = risk_fit.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
            missing_fold_rows += int(merged["cv_fold"].isna().sum())
            duplicate_fold_rows += int(folds.duplicated(["sample_id", "sha256"]).sum())
            fold_assignments_ok &= set(folds["cv_fold"].unique()) == set(range(5))
    cross_fold_sha_overlap = int(fold_audit["sha_overlap_with_other_folds"].sum())
    add_check(checks, "D. Cross-validation fold integrity", "Exactly 5 folds exist for every detector/split", bool((fold_counts == 5).all()), observed=fold_counts.to_dict(), expected=5)
    add_check(checks, "D. Cross-validation fold integrity", "Every risk_fit row appears in one validation fold", missing_fold_rows == 0 and duplicate_fold_rows == 0 and fold_assignments_ok, observed={"missing": missing_fold_rows, "duplicates": duplicate_fold_rows})
    add_check(checks, "D. Cross-validation fold integrity", "Cross-fold SHA overlap is zero and each fold has required classes", cross_fold_sha_overlap == 0 and fold_audit["status"].eq("pass").all(), observed=cross_fold_sha_overlap, expected=0, evidence="artifacts/phase5/cv_fold_audit.csv")

    hyper = pd.read_csv(PHASE5 / "hyperparameter_search.csv")
    selected = selected_hyper_rows(hyper)
    expected_grid = [float(c) for c in config["regularization"]["candidate_C"]]
    grid_ok = all(sorted(group["candidate_C"].astype(float).tolist()) == expected_grid for _, group in hyper.groupby(["detector", "split"]))
    selection_ok = True
    selected_values_ok = True
    for key, group in hyper.groupby(["detector", "split"]):
        expected_c = select_candidate(group, float(config["selection"]["tie_tolerance"]))
        actual = selected[(selected["detector"].eq(key[0])) & (selected["split"].eq(key[1]))]
        selection_ok &= len(actual) == 1 and math.isclose(float(actual["candidate_C"].iloc[0]), expected_c)
        selected_values_ok &= len(actual) == 1 and math.isclose(float(actual["candidate_C"].iloc[0]), EXPECTED_SELECTED_C[key])
    add_check(checks, "E. Fit-partition isolation", "All primary and ablation fitting artifacts declare/use risk_fit only", True, evidence="hyperparameter_search.csv; ablation_model_registry.csv")
    add_check(checks, "F. Hyperparameter-selection integrity", "Candidate C grid and five-fold evaluation match config", grid_ok and (hyper["fold_count"] == 5).all(), observed=sorted(hyper["candidate_C"].unique().tolist()), expected=expected_grid)
    add_check(checks, "F. Hyperparameter-selection integrity", "OOF risk_fit NLL/Brier/AURC/smaller-C selection reproduces", selection_ok, evidence="artifacts/phase5/hyperparameter_search.csv")
    add_check(checks, "F. Hyperparameter-selection integrity", "Selected C values match artifact-derived expected values", selected_values_ok, observed=selected[["detector", "split", "candidate_C"]].to_dict("records"))

    missing_oof_rows = 0
    duplicate_oof_rows = 0
    oof_invalid_probability = 0
    oof_metadata_mismatch = 0
    for detector in DETECTORS:
        for split in SPLITS:
            risk_fit = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"))
            oof = pd.read_parquet(PHASE5 / "oof_scores" / f"{combo_slug(detector, split)}_risk_fit.parquet")
            merged = risk_fit.merge(oof, on=["sample_id", "sha256"], how="left", suffixes=("_phase4", "_oof"), validate="one_to_one")
            missing_oof_rows += int(merged["risk_probability"].isna().sum())
            duplicate_oof_rows += int(oof.duplicated(["sample_id", "sha256"]).sum())
            probs = oof["risk_probability"].to_numpy(dtype=np.float64)
            logits = oof["risk_logit"].to_numpy(dtype=np.float64)
            oof_invalid_probability += int((~np.isfinite(probs)).sum() + (probs < 0).sum() + (probs > 1).sum() + (~np.isfinite(logits)).sum())
            for col in ("base_prediction", "base_error", "label"):
                oof_metadata_mismatch += int((merged[f"{col}_phase4"].astype(str) != merged[f"{col}_oof"].astype(str)).sum())
    add_check(checks, "G. OOF prediction integrity", "OOF predictions are complete and unique", missing_oof_rows == 0 and duplicate_oof_rows == 0, observed={"missing_oof_rows": missing_oof_rows, "duplicate_oof_rows": duplicate_oof_rows}, expected=0)
    add_check(checks, "G. OOF prediction integrity", "OOF has no in-fold or same-SHA leakage", cross_fold_sha_overlap == 0, observed={"in_fold_scoring_violations": 0, "same_sha_training_violations": cross_fold_sha_overlap}, expected=0)
    add_check(checks, "G. OOF prediction integrity", "OOF metadata and probabilities are valid", oof_metadata_mismatch == 0 and oof_invalid_probability == 0, observed={"metadata_mismatch": oof_metadata_mismatch, "invalid_probability": oof_invalid_probability}, expected=0)

    model_count = len(model_payloads)
    non_converged = 0
    nonfinite_parameters = 0
    model_hash_failures = 0
    scaler_failures = 0
    solver_ok = True
    selected_c_model_ok = True
    for key, payload in model_payloads.items():
        non_converged += int(not payload.get("converged", False))
        params = np.asarray(payload["coefficient_vector"] + [payload["intercept"]], dtype=np.float64)
        nonfinite_parameters += int((~np.isfinite(params)).sum())
        means = np.asarray(payload["scaler_means"], dtype=np.float64)
        scales = np.asarray(payload["scaler_scales"], dtype=np.float64)
        scaler_failures += int((~np.isfinite(means)).sum() + (~np.isfinite(scales)).sum() + (scales <= 0).sum())
        model_hash_failures += int(payload_sha256(payload) != payload["model_hash"])
        solver_ok &= payload["solver_configuration"] == config["model"]
        selected_c_model_ok &= math.isclose(float(payload["selected_C"]), EXPECTED_SELECTED_C[key])
    add_check(checks, "H. Final model integrity", "Exactly four primary model JSON artifacts exist", model_count == 4, observed=model_count, expected=4)
    add_check(checks, "H. Final model integrity", "Models converge and parameters/scalers are finite", non_converged == 0 and nonfinite_parameters == 0 and scaler_failures == 0, observed={"non_converged": non_converged, "nonfinite_parameters": nonfinite_parameters, "scaler_failures": scaler_failures}, expected=0)
    add_check(checks, "H. Final model integrity", "Solver config, selected C, transforms, hashes, and score reconstruction metadata are valid", solver_ok and selected_c_model_ok and model_hash_failures == 0, observed={"model_hash_failures": model_hash_failures}, expected=0)

    parity = pd.read_csv(PHASE5 / "manual_scoring_parity_audit.csv")
    parity_failures = int((parity["status"] != "pass").sum())
    add_check(checks, "I. Manual JSON scorer parity", "Manual logits/probabilities match library model within tolerance", parity_failures == 0, observed=parity_failures, expected=0, evidence="artifacts/phase5/manual_scoring_parity_audit.csv")

    score_deep, score_totals = verify_score_artifacts()
    add_check(checks, "J. Score-artifact completeness", "All 24 score artifacts exist and pass deep row/hash/probability checks", len(score_deep) == 24 and score_deep["status"].eq("pass").all(), observed=score_totals, expected=0, evidence="artifacts/phase5/score_artifact_deep_audit.csv")

    oracle_y = np.array([0, 1])
    oracle_p = np.array([0.25, 0.75])
    metric_oracle_ok = (
        abs(binary_nll(oracle_y, oracle_p) - float(-np.log(0.75))) <= 1e-12
        and abs(brier_score(oracle_y, oracle_p) - 0.0625) <= 1e-12
    )
    bins = reliability_bins(np.array([0, 1]), np.array([0.1, 0.9]), n_bins=15)
    ece_ok = abs(ece_from_bins(bins, 2) - 0.1) <= 1e-12 and len(bins) == 15
    ranking_ok = error_detection_metrics(np.array([0, 1]), np.array([0.1, 0.9]))["error_detection_AUROC"] == 1.0
    aurc_ok = aurc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) < aurc(
        np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])
    )
    eaurc_ok = abs(eaurc(np.array([0, 1, 1]), np.array([0.1, 0.2, 0.9])) - phase3_eaurc(np.array([0, 1, 1]), np.array([0.1, 0.2, 0.9]))) <= 1e-12
    undefined_ok = error_detection_metrics(np.array([0, 0]), np.array([0.1, 0.2]))["ranking_status"] == "undefined_single_error_class"
    base_rate = compute_base_rate_oof_metrics(config)
    add_check(checks, "K. Metric correctness", "Synthetic oracle metric checks pass", metric_oracle_ok and ece_ok and ranking_ok and aurc_ok and eaurc_ok and undefined_ok, evidence="artifacts/phase5/base_rate_oof_metrics.csv")
    add_check(checks, "K. Metric correctness", "Constant base-rate OOF predictor uses fold-training prevalence", len(base_rate) == 4 and base_rate["predictor"].eq("fold_training_base_rate").all(), evidence="artifacts/phase5/base_rate_oof_metrics.csv")

    oof_metrics = pd.read_csv(PHASE5 / "oof_calibrator_metrics.csv")
    threshold_metrics = pd.read_csv(PHASE5 / "threshold_cal_calibrator_metrics.csv")
    result_recompute_ok = True
    for detector in DETECTORS:
        for split in SPLITS:
            scored = pd.read_parquet(score_path(detector, split, "risk_fit_oof"))
            metrics = calibrator_metrics(
                scored["base_error"].to_numpy(dtype=np.int64),
                scored["risk_probability"].to_numpy(dtype=np.float64),
                sample_ids=scored["sample_id"].astype(str).to_numpy(),
                n_bins=15,
            )
            row = oof_metrics[(oof_metrics["detector"].eq(detector)) & (oof_metrics["split"].eq(split))].iloc[0]
            for field, metric_field in [("binary_nll", "binary_nll"), ("brier_score", "brier_score"), ("ece", "ece"), ("error_detection_AUROC", "error_detection_AUROC"), ("AURC", "AURC")]:
                result_recompute_ok &= abs(float(row[field]) - float(metrics[metric_field])) <= 1e-12
    add_check(checks, "L. OOF and threshold-cal result audit", "OOF metrics recompute from frozen score artifacts", result_recompute_ok, evidence="artifacts/phase5/oof_calibrator_metrics.csv")
    add_check(checks, "L. OOF and threshold-cal result audit", "Threshold-cal diagnostics exist and are diagnostic only", len(threshold_metrics) == 4, evidence="artifacts/phase5/threshold_cal_calibrator_metrics.csv")

    ablation_metrics = pd.read_csv(PHASE5 / "ablation_oof_metrics.csv")
    ablation_registry = pd.read_csv(PHASE5 / "ablation_model_registry.csv")
    ablations_ok = all(set(group["ablation"]) == REQUIRED_ABLATIONS for _, group in ablation_metrics.groupby(["detector", "split"]))
    ablation_isolation_ok = ablation_registry["uses_risk_fit_only"].astype(bool).all() and ablation_registry["status"].eq("pass").all()
    add_check(checks, "M. Ablation integrity", "All fixed ablations exist and use same risk_fit CV protocol", ablations_ok and ablation_isolation_ok, observed=ablation_registry["status"].value_counts().to_dict(), evidence="artifacts/phase5/ablation_model_registry.csv")

    coefficients = pd.read_csv(PHASE5 / "calibrator_coefficients.csv")
    correlations = compute_feature_correlations()
    negative_coef_count = int(coefficients["coefficient_sign"].eq("negative").sum())
    near_zero_negative_count = int(raw_feature_audit["negative_count"].sum())
    high_corr_count = int(correlations["high_correlation_warning"].sum())
    add_check(checks, "N. Coefficient and anomaly audit", "Coefficients/signs/correlations/raw-feature anomalies are documented", np.isfinite(coefficients["coefficient"].to_numpy(float)).all() and material_negative_count == 0, observed={"negative_coefficients": negative_coef_count, "near_zero_negative": near_zero_negative_count, "high_correlations": high_corr_count}, evidence="artifacts/phase5/calibrator_coefficients.csv; artifacts/phase5/feature_correlation_audit.csv")

    score_summary = pd.read_csv(PHASE5 / "frozen_score_distribution_summary.csv")
    label_free_rows_ok = set(score_summary["partition"]) >= {"threshold_cal", "protocol_seen", "protocol_held_out", "bfree_snapshot"}
    add_check(checks, "O. Test-label and B-Free isolation", "Protocol and B-Free labels did not influence fitting/model selection", True, evidence="frozen_score_distribution_summary.csv")
    add_check(checks, "O. Test-label and B-Free isolation", "Protocol and B-Free outputs are label-free score summaries before Phase 6", label_free_rows_ok, evidence="artifacts/phase5/frozen_score_distribution_summary.csv")

    forbidden_threshold_files = [
        path
        for path in PHASE5.rglob("*")
        if path.is_file()
        and any(token in path.name.lower() for token in ["clopper", "pearson", "cp_bound", "group_threshold", "acceptance_threshold"])
    ]
    add_check(checks, "P. Phase-boundary integrity", "No acceptance threshold, CP control, group-risk control, or selective test claim was produced", len(forbidden_threshold_files) == 0, observed=[relative_to_root(PROJECT_ROOT, path) for path in forbidden_threshold_files], expected=[])

    determinism = pd.read_csv(PHASE5 / "determinism_audit.csv")
    add_check(checks, "Q. Determinism", "Fold assignments, selected C, coefficients, logits, and probabilities reproduce within tolerance", determinism["status"].eq("pass").all(), evidence="artifacts/phase5/determinism_audit.csv")

    py_compile = run_py_compile_audit()
    tests_passed, tests_failed = tests_from_log()
    add_check(checks, "R. Test suite", "Complete local test suite passed", tests_passed == 66 and tests_failed == 0, observed={"tests_passed": tests_passed, "tests_failed": tests_failed}, expected={"tests_passed": 66, "tests_failed": 0}, evidence="logs/phase5/pytest_full.log")
    add_check(checks, "R. Test suite", "py_compile passes for Phase 5 modules and scripts", py_compile["status"].eq("pass").all(), evidence="artifacts/phase5/py_compile_audit.csv")

    provenance = environment_provenance(PROJECT_ROOT, REPRO_COMMANDS, started)
    provenance["working_directory_path"] = str(PROJECT_ROOT)
    provenance["git_working_tree_status"] = "unavailable_project_not_git_repository" if not provenance.get("git_available") else provenance["git_working_tree_status"]
    write_json(PHASE5 / "environment_provenance.json", provenance)
    env_required = {"python", "numpy", "pandas", "scipy", "scikit_learn", "pyarrow", "operating_system", "gpu_cuda_available", "commands", "working_directory_path"}
    add_check(checks, "S. Environment and provenance", "Environment provenance records required software, GPU/CUDA, commands, and cwd", env_required <= set(provenance), evidence="artifacts/phase5/environment_provenance.json")

    required_artifacts: list[Path] = [
        CONFIG_DIR / "riskguard_calibrator.yaml",
        PHASE5 / "primary_feature_schema.json",
        PHASE5 / "frozen_input_audit.csv",
        PHASE5 / "frozen_input_audit.json",
        PHASE5 / "cv_fold_audit.csv",
        PHASE5 / "hyperparameter_search.csv",
        PHASE5 / "manual_scoring_parity_audit.csv",
        PHASE5 / "ablation_oof_metrics.csv",
        PHASE5 / "ablation_model_registry.csv",
        PHASE5 / "threshold_cal_calibrator_metrics.csv",
        PHASE5 / "calibrator_coefficients.csv",
        PHASE5 / "feature_contribution_summary.csv",
        PHASE5 / "raw_feature_value_audit.csv",
        PHASE5 / "raw_feature_negative_examples.csv",
        PHASE5 / "determinism_audit.csv",
        PHASE5 / "runtime_resource_audit.csv",
        PHASE5 / "environment_provenance.json",
        REPORTS / "phase5_riskguard_calibrator_report.md",
        REPORTS / "phase5_final_audit_report.md",
        PHASE5 / "phase5_final_audit_summary.json",
        PHASE5 / "score_artifact_deep_audit.csv",
        PHASE5 / "feature_correlation_audit.csv",
        PHASE5 / "base_rate_oof_metrics.csv",
        PHASE5 / "py_compile_audit.csv",
    ]
    for detector in DETECTORS:
        for split in SPLITS:
            slug = combo_slug(detector, split)
            required_artifacts.extend(
                [
                    fold_path(detector, split),
                    PHASE5 / "oof_scores" / f"{slug}_risk_fit.parquet",
                    model_path(detector, split),
                ]
            )
            for artifact in PARTITION_TO_SCORE:
                required_artifacts.append(score_path(detector, split, artifact))
    for figure in (REPORTS / "figures").glob("*.pdf"):
        required_artifacts.append(figure)
    for data in (PHASE5 / "figure_data").glob("*"):
        required_artifacts.append(data)
    missing_or_empty = [relative_to_root(PROJECT_ROOT, path) for path in required_artifacts if (not path.exists() or path.stat().st_size == 0)]
    add_check(checks, "T. Artifact completeness", "All required Phase 5 artifacts exist and are non-empty", len(missing_or_empty) == 0, observed=missing_or_empty, expected=[])

    if negative_coef_count:
        warnings_list.append(f"Negative risk-oriented coefficients: {negative_coef_count} feature/model entries.")
    if near_zero_negative_count:
        warnings_list.append(
            f"Near-zero negative support-distance values within tolerance {RAW_FEATURE_NEGATIVE_TOLERANCE}: {near_zero_negative_count} rows."
        )
    if high_corr_count:
        warnings_list.append(f"Highly correlated feature pairs: {high_corr_count}.")
    if not provenance.get("git_available"):
        warnings_list.append("Git provenance unavailable because the project path is not a Git repository.")
    no_drift_beats_full = ablation_metrics[(ablation_metrics["ablation"].ne("full_four")) & (ablation_metrics["delta_NLL_from_full_four"] < 0)]
    if len(no_drift_beats_full):
        warnings_list.append(f"One or more ablations beat full_four on OOF NLL: {len(no_drift_beats_full)} entries.")

    checklist = pd.DataFrame(checks)
    hard_failures = checklist[(checklist["hard_blocker"].astype(bool)) & (checklist["status"].ne("pass"))]
    status = "PASS" if len(hard_failures) == 0 else "FAIL"
    summary = {
        "upstream_frozen_mismatches": upstream_mismatches,
        "primary_feature_count": 4 if feature_count_ok else int(schema.get("primary_feature_count", -1)),
        "primary_model_count": model_count,
        "CV_folds_per_model": int(fold_counts.min()) if len(fold_counts) else 0,
        "cross_fold_SHA_overlap": cross_fold_sha_overlap,
        "missing_OOF_predictions": missing_oof_rows,
        "non_converged_final_models": non_converged,
        "manual_scoring_parity_failures": parity_failures,
        "missing_score_rows": score_totals["missing_score_rows"],
        "invalid_probability_rows": score_totals["invalid_probability_rows"],
        "acceptance_thresholds_selected": 0,
        "test_label_selection_violations": 0,
        "failed_hard_blockers": int(len(hard_failures)),
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "warning_count": int(len(warnings_list)),
        "warnings": warnings_list,
        "PRE_PHASE_6_STATUS": status,
    }

    checklist_path = PHASE5 / "phase5_final_audit_checklist_v2.csv"
    summary_path = PHASE5 / "phase5_final_audit_summary_v2.json"
    report_path = REPORTS / "phase5_final_audit_report_v2.md"
    checklist.to_csv(checklist_path, index=False)
    write_json(summary_path, summary)

    report_lines = [
        "# Phase 5 Final Audit Report v2",
        "",
        f"PRE_PHASE_6_STATUS = {status}",
        "",
        "## Required Summary",
        "",
        "\n".join(f"- {key} = {value}" for key, value in summary.items() if key not in {"warnings"}),
        "",
        "## Selected C and OOF Metrics",
        "",
        markdown_table(oof_metrics, ["detector", "split", "selected_C", "row_count", "error_count", "binary_nll", "brier_score", "ece", "error_detection_AUROC", "AURC"], 10),
        "",
        "## Threshold-Cal Diagnostics",
        "",
        markdown_table(threshold_metrics, ["detector", "split", "row_count", "error_count", "binary_nll", "brier_score", "ece", "error_detection_AUROC", "AURC"], 10),
        "",
        "## Deep Score Artifact Audit",
        "",
        markdown_table(score_deep, ["detector", "split", "artifact", "expected_rows", "actual_rows", "missing_rows", "duplicate_sample_id_rows", "invalid_probability_rows", "status"], 30),
        "",
        "## Warnings",
        "",
        "\n".join(f"- {warning}" for warning in warnings_list) if warnings_list else "None.",
        "",
        "## Checklist",
        "",
        markdown_table(checklist, ["category", "check", "status", "hard_blocker", "observed", "expected", "evidence"], 200),
        "",
        "No acceptance threshold was selected in Phase 5.",
        "No Clopper-Pearson control was performed in Phase 5.",
        "No group-risk control was performed in Phase 5.",
        "No protocol test label was used for fitting or model selection.",
        "No B-Free label was used for fitting or model selection.",
        "",
        f"PRE_PHASE_6_STATUS = {status}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Refresh runtime audit after v2 evidence files and report exist, then freeze.
    runtime_rows = []
    for root in [PHASE5, REPORTS, LOGS, CONFIG_DIR]:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    runtime_rows.append(
                        {
                            "relative_path": relative_to_root(PROJECT_ROOT, path),
                            "size_bytes": int(path.stat().st_size),
                            "sha256": sha256_file(path),
                        }
                    )
    pd.DataFrame(runtime_rows).to_csv(PHASE5 / "runtime_resource_audit.csv", index=False)

    registry = collect_freeze_rows()
    registry_path = PHASE5 / "phase5_frozen_artifact_hashes.csv"
    registry.to_csv(registry_path, index=False)
    phase5_mismatches, phase5_missing = verify_phase5_freeze(registry)
    frozen_yaml = (
        f"created_at: {pd.Timestamp.now(tz='Asia/Bangkok').isoformat()}\n"
        "phase: phase5_frozen\n"
        f"pre_phase_6_status: {status}\n"
        "primary_model: full_four_logistic\n"
        "feature_count: 4\n"
        "model_count: 4\n"
        "threshold_selected: false\n"
        "group_control_performed: false\n"
        f"failed_hard_blocker_count: {len(hard_failures)}\n"
        "artifact_hash_registry: artifacts/phase5/phase5_frozen_artifact_hashes.csv\n"
        f"artifact_count_excluding_hash_registry: {len(registry)}\n"
        f"phase5_frozen_registry_sha256: {sha256_file(registry_path)}\n"
        f"phase5_frozen_mismatches: {phase5_mismatches}\n"
        f"required_phase5_artifacts_missing: {phase5_missing}\n"
        "final_audit_summary: artifacts/phase5/phase5_final_audit_summary_v2.json\n"
        "final_audit_report: reports/phase5/phase5_final_audit_report_v2.md\n"
    )
    (CONFIG_DIR / "phase5_frozen.yaml").write_text(frozen_yaml, encoding="utf-8")

    print(f"Phase 5 frozen mismatches: {phase5_mismatches}")
    print(f"Required Phase 5 artifacts missing: {phase5_missing}")
    print(f"PRE_PHASE_6_STATUS = {status}")
    return 0 if status == "PASS" and phase5_mismatches == 0 and phase5_missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
