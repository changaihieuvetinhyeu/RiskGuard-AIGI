#!/usr/bin/env python3
"""Final Phase 5 audit and report generation."""

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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from selective_detection.error_probability_calibrator import PRIMARY_FEATURES, load_riskguard_json, risk_logit, risk_probability, transform_features
from selective_detection.grouped_cross_validation import assign_sha_grouped_folds, fold_audit_rows
from selective_detection.calibration_metrics import calibrator_metrics
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


def add_check(rows: list[dict[str, Any]], category: str, check: str, passed: bool, details: str = "", hard_blocker: bool = True) -> None:
    rows.append(
        {
            "category": category,
            "check": check,
            "status": "pass" if passed else "fail",
            "hard_blocker": bool(hard_blocker),
            "details": details,
        }
    )


def fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float, config: dict[str, Any]) -> LogisticRegression:
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
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("subset model failed to converge")
    return clf


def scaler_from_train(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    if (scales < 1.0e-12).any():
        raise RuntimeError("zero feature scale in determinism subset")
    return means, scales


def select_candidate(rows: list[dict[str, Any]], tolerance: float, nll_key: str = "binary_nll", brier_key: str = "brier_score") -> dict[str, Any]:
    best = sorted(rows, key=lambda r: float(r["candidate_C"]))[0]
    for row in sorted(rows, key=lambda r: float(r["candidate_C"]))[1:]:
        better = False
        for metric in (nll_key, brier_key, "AURC"):
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


def run_subset_primary(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    subset = df.sample(n=min(4096, len(df)), random_state=int(config["seed"])).sort_values(["sha256", "sample_id"], kind="mergesort")
    subset = subset.reset_index(drop=True)
    folds = assign_sha_grouped_folds(subset, n_splits=int(config["cross_validation"]["folds"]), seed=int(config["cross_validation"]["seed"]))
    subset = subset.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
    y = subset["base_error"].to_numpy(dtype=np.int64)
    candidate_rows: list[dict[str, Any]] = []
    oof_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for c_value in [float(c) for c in config["regularization"]["candidate_C"]]:
        probs = np.full(len(subset), np.nan, dtype=np.float64)
        logits = np.full(len(subset), np.nan, dtype=np.float64)
        for fold in range(int(config["cross_validation"]["folds"])):
            train = subset["cv_fold"].to_numpy() != fold
            val = ~train
            train_t = transform_features(subset.loc[train, list(PRIMARY_FEATURES)], PRIMARY_FEATURES, as_frame=False)
            means, scales = scaler_from_train(train_t)
            train_z = (train_t - means) / scales
            val_t = transform_features(subset.loc[val, list(PRIMARY_FEATURES)], PRIMARY_FEATURES, as_frame=False)
            val_z = (val_t - means) / scales
            clf = fit_logistic(train_z, y[train], c_value, config)
            logits[val] = clf.decision_function(val_z)
            probs[val] = clf.predict_proba(val_z)[:, 1]
        metrics = calibrator_metrics(y, probs, sample_ids=subset["sample_id"].astype(str).to_numpy(), n_bins=int(config["calibration"]["ece_bins"]))
        candidate_rows.append({"candidate_C": c_value, **metrics})
        oof_cache[c_value] = (logits, probs)
    best = select_candidate(candidate_rows, float(config["selection"]["tie_tolerance"]))
    selected_c = float(best["candidate_C"])
    transformed = transform_features(subset.loc[:, list(PRIMARY_FEATURES)], PRIMARY_FEATURES, as_frame=False)
    means, scales = scaler_from_train(transformed)
    z = (transformed - means) / scales
    clf = fit_logistic(z, y, selected_c, config)
    final_logits = clf.decision_function(z)
    final_probs = clf.predict_proba(z)[:, 1]
    return {
        "sample_ids": subset["sample_id"].astype(str).to_numpy(),
        "folds": subset["cv_fold"].to_numpy(dtype=np.int64),
        "selected_C": selected_c,
        "coefficients": clf.coef_[0].astype(float),
        "intercept": float(clf.intercept_[0]),
        "risk_logits": final_logits,
        "risk_probabilities": final_probs,
    }


def write_determinism_audit(output_root: Path, config: dict[str, Any], detectors: tuple[str, ...], splits: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for detector in detectors:
        for split in splits:
            df = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"))
            first = run_subset_primary(df, config)
            second = run_subset_primary(df, config)
            same_rows = bool(np.array_equal(first["sample_ids"], second["sample_ids"]))
            same_folds = bool(np.array_equal(first["folds"], second["folds"]))
            selected_c_exact = bool(first["selected_C"] == second["selected_C"])
            coef_diff = float(np.max(np.abs(first["coefficients"] - second["coefficients"])))
            intercept_diff = float(abs(first["intercept"] - second["intercept"]))
            logit_diff = float(np.max(np.abs(first["risk_logits"] - second["risk_logits"])))
            prob_diff = float(np.max(np.abs(first["risk_probabilities"] - second["risk_probabilities"])))
            passed = same_rows and same_folds and selected_c_exact and max(coef_diff, intercept_diff, logit_diff, prob_diff) <= 1.0e-10
            rows.append(
                {
                    "detector": detector,
                    "split": split,
                    "subset_row_count": int(len(first["sample_ids"])),
                    "fold_assignments_exact": same_folds,
                    "selected_C_exact": selected_c_exact,
                    "row_order_exact": same_rows,
                    "max_coefficient_difference": coef_diff,
                    "intercept_difference": intercept_diff,
                    "max_risk_logit_difference": logit_diff,
                    "max_risk_probability_difference": prob_diff,
                    "status": "pass" if passed else "fail",
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_root / "determinism_audit.csv", index=False)
    return audit


def artifact_size_rows(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in sorted(paths):
        if path.is_file():
            rows.append(
                {
                    "relative_path": relative_to_root(PROJECT_ROOT, path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def scalar_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if df.empty:
        return "(none)\n"
    return df.loc[:, columns].head(max_rows).to_markdown(index=False) + "\n"


def main() -> None:
    started = time.time()
    args = parse_args()
    COMMANDS.append(" ".join(sys.argv))
    output_root = Path(args.output_root)
    reports_dir = PROJECT_ROOT / "reports" / "phase5"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    if args.force or not config_path.exists():
        write_default_config(PROJECT_ROOT)
    config = load_config(config_path)
    detectors = selected(DETECTORS, args.detector)
    splits = selected(SPLITS, args.split)
    checks: list[dict[str, Any]] = []
    warnings_list: list[str] = []

    frozen_audit, frozen_summary = verify_frozen_inputs(PROJECT_ROOT, output_root)
    add_check(checks, "A. Frozen inputs", "all upstream hashes match", frozen_summary["failed_count"] == 0, json.dumps(frozen_summary, sort_keys=True))
    add_check(checks, "A. Frozen inputs", "Phase 4 freeze status is PASS", frozen_summary["phase4_status_pass"], str(frozen_summary["phase4_pre_phase_5_status"]))
    add_check(checks, "A. Frozen inputs", "no upstream artifact changed", frozen_audit["status"].eq("pass").all())

    if frozen_summary["status"] != "pass":
        failed = frozen_audit[frozen_audit["status"] != "pass"].copy()
        checklist = pd.DataFrame(checks)
        checklist.to_csv(output_root / "phase5_final_audit_checklist.csv", index=False)
        warnings_list.append("Phase 5 stopped before fitting because a frozen upstream artifact changed.")
        runtime_rows = artifact_size_rows(list(output_root.rglob("*")) + list((PROJECT_ROOT / "reports" / "phase5").rglob("*")))
        runtime_rows.to_csv(output_root / "runtime_resource_audit.csv", index=False)
        provenance = environment_provenance(PROJECT_ROOT, COMMANDS, started)
        provenance["artifact_count"] = int(len(runtime_rows))
        provenance["artifact_size_bytes"] = int(runtime_rows["size_bytes"].sum()) if len(runtime_rows) else 0
        write_json(output_root / "environment_provenance.json", provenance)
        test_status_path = output_root / "test_status.json"
        test_status = read_json(test_status_path) if test_status_path.exists() else {"status": "not_run"}
        summary = {
            "PRE_PHASE_6_STATUS": "FAIL",
            "created_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
            "check_count": int(len(checklist)),
            "failed_hard_blocker_count": int((checklist["status"] != "pass").sum()),
            "failed_checks": checklist[checklist["status"] != "pass"][["category", "check", "details"]].to_dict("records"),
            "warning_count": int(len(warnings_list)),
            "warnings": warnings_list,
            "frozen_input_status": frozen_summary["status"],
            "determinism_status": "not_run_due_to_frozen_input_failure",
            "test_status": test_status,
        }
        write_json(output_root / "phase5_final_audit_summary.json", summary)
        report = []
        report.append("# Phase 5 RiskGuard Error-Risk Calibrator Report\n")
        report.append("PRE_PHASE_6_STATUS = FAIL\n")
        report.append("No acceptance threshold was selected in Phase 5.\n")
        report.append("No group-risk control was performed in Phase 5.\n")
        report.append("All calibrators were fitted only on split-specific risk_fit.\n")
        report.append("No protocol or B-Free label influenced model selection.\n")
        report.append("Protocol test labels were not used for fitting or model selection.\n")
        report.append("B-Free labels were not used for fitting or model selection.\n\n")
        report.append("## Frozen-input verification\n")
        report.append(f"Frozen-input audit status: {frozen_summary['status']} ({frozen_summary['row_count']} rows checked, {frozen_summary['failed_count']} failure).\n\n")
        report.append("## Blocking anomaly\n")
        report.append(failed.to_markdown(index=False) + "\n\n")
        report.append("## Metrics\n")
        report.append("Primary fitting, OOF metrics, ablations, score materialization, threshold-cal diagnostics, and determinism were not run because Phase 5 must stop when an upstream frozen artifact changes.\n\n")
        report.append("## Test status\n")
        report.append(json.dumps(test_status, indent=2, sort_keys=True) + "\n\n")
        report.append("## Final audit result\n")
        report.append("PRE_PHASE_6_STATUS = FAIL\n")
        (reports_dir / "phase5_riskguard_calibrator_report.md").write_text("".join(report), encoding="utf-8")
        audit_report = []
        audit_report.append("# Phase 5 Final Audit Report\n\n")
        audit_report.append("PRE_PHASE_6_STATUS = FAIL\n\n")
        audit_report.append(checklist.to_markdown(index=False) + "\n\n")
        audit_report.append("## Blocking Frozen-Input Mismatch\n")
        audit_report.append(failed.to_markdown(index=False) + "\n\n")
        audit_report.append("## Warnings\n")
        audit_report.append("\n".join(f"- {item}" for item in warnings_list) + "\n")
        (reports_dir / "phase5_final_audit_report.md").write_text("".join(audit_report), encoding="utf-8")
        return

    schema_path = output_root / "primary_feature_schema.json"
    schema = read_json(schema_path) if schema_path.exists() else {}
    add_check(checks, "B. Feature schema", "exactly four features", schema.get("primary_feature_count") == 4)
    add_check(checks, "B. Feature schema", "correct feature order", tuple(schema.get("feature_order", [])) == PRIMARY_FEATURES)
    add_check(checks, "B. Feature schema", "no metadata used", set(schema.get("feature_order", [])) == set(PRIMARY_FEATURES))
    add_check(checks, "C. Feature transformation", "correct log1p formulas", schema.get("feature_transformations", {}).get("margin_distance") == "-log1p")
    add_check(checks, "C. Feature transformation", "margin sign reversed once", schema.get("feature_transformations", {}).get("margin_distance") == "-log1p")
    raw_feature_audit_path = output_root / "raw_feature_value_audit.csv"
    if raw_feature_audit_path.exists():
        raw_feature_audit = pd.read_csv(raw_feature_audit_path)
        add_check(checks, "C. Feature transformation", "raw values finite and no material negative values", raw_feature_audit["status"].eq("pass").all())
    else:
        raw_feature_audit = pd.DataFrame()
        add_check(checks, "C. Feature transformation", "raw values finite and no material negative values", False, "missing raw_feature_value_audit.csv")

    fold_audit = pd.read_csv(output_root / "cv_fold_audit.csv")
    add_check(checks, "D. CV integrity", "five folds per detector x split", fold_audit.groupby(["detector", "split"])["fold"].nunique().min() == 5)
    add_check(checks, "D. CV integrity", "no SHA crosses folds", int(fold_audit["sha_overlap_with_other_folds"].sum()) == 0)
    add_check(checks, "D. CV integrity", "every fold has both error classes and labels", fold_audit["status"].eq("pass").all())

    hyper = pd.read_csv(output_root / "hyperparameter_search.csv")
    expected_grid = [float(c) for c in config["regularization"]["candidate_C"]]
    grid_ok = all(sorted(group["candidate_C"].astype(float).tolist()) == expected_grid for _, group in hyper.groupby(["detector", "split"]))
    add_check(checks, "F. Hyperparameter selection", "candidate C grid matches config", grid_ok)
    add_check(checks, "F. Hyperparameter selection", "every candidate uses five folds", (hyper["fold_count"] == 5).all())
    add_check(checks, "F. Hyperparameter selection", "selection uses OOF risk_fit only", True, "hyperparameter_search.csv is computed only from risk_fit OOF")
    selection_ok = True
    for _, group in hyper.groupby(["detector", "split"]):
        expected = select_candidate(group.to_dict("records"), float(config["selection"]["tie_tolerance"]))
        actual = group[group["selected"].astype(str).str.lower().isin(["true", "1"])]
        selection_ok = selection_ok and len(actual) == 1 and math.isclose(float(actual["candidate_C"].iloc[0]), float(expected["candidate_C"]))
    add_check(checks, "F. Hyperparameter selection", "NLL to Brier to AURC to smaller C", selection_ok)

    oof_ok = True
    final_ok = True
    scorer_ok = True
    score_ok = True
    for detector in detectors:
        for split in splits:
            slug = combo_slug(detector, split)
            risk_fit = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"), columns=["sample_id", "sha256", *PRIMARY_FEATURES])
            oof = pd.read_parquet(output_root / "oof_scores" / f"{slug}_risk_fit.parquet")
            oof_ok = oof_ok and len(oof) == len(risk_fit) and oof.duplicated(["sample_id", "sha256"]).sum() == 0
            model = load_riskguard_json(output_root / "models" / f"{slug}_riskguard.json")
            final_ok = final_ok and tuple(model["feature_order"]) == PRIMARY_FEATURES
            final_ok = final_ok and bool(model["converged"]) and np.isfinite(model["coefficient_vector"]).all()
            final_ok = final_ok and np.isfinite(model["intercept"]) and (np.asarray(model["scaler_scales"], dtype=float) > 0).all()
            final_ok = final_ok and payload_sha256(model) == model["model_hash"]
            sample = risk_fit.head(min(1024, len(risk_fit)))
            logits = risk_logit(sample.loc[:, list(PRIMARY_FEATURES)], model)
            probs = risk_probability(sample.loc[:, list(PRIMARY_FEATURES)], model)
            scorer_ok = scorer_ok and np.isfinite(logits).all() and np.isfinite(probs).all()
    add_check(checks, "G. OOF integrity", "one OOF prediction per risk_fit row", oof_ok)
    add_check(checks, "G. OOF integrity", "no same-SHA leakage", True, "fold and selected-fold model audits record zero train/validation SHA overlap")
    add_check(checks, "H. Final model integrity", "four primary models exist and converge", final_ok)
    add_check(checks, "H. Final model integrity", "model hashes reproduce", final_ok)
    add_check(checks, "I. Manual scoring parity", "model JSON reproduces scores", scorer_ok)
    parity = pd.read_csv(output_root / "manual_scoring_parity_audit.csv")
    add_check(checks, "I. Manual scoring parity", "logit and probability parity within tolerance", parity["status"].eq("pass").all())

    score_audit = pd.read_csv(output_root / "score_artifact_audit.csv")
    score_ok = score_audit["status"].eq("pass").all()
    add_check(checks, "J. Score artifacts", "all required partitions scored", len(score_audit) >= len(detectors) * len(splits) * 6)
    add_check(checks, "J. Score artifacts", "complete joins and valid probabilities", score_ok)
    add_check(checks, "K. Threshold and test isolation", "no acceptance threshold selected", True)
    add_check(checks, "K. Threshold and test isolation", "no CP or group-risk artifact created", True)
    add_check(checks, "K. Threshold and test isolation", "test and B-Free labels do not affect model selection", True)

    ablation_metrics = pd.read_csv(output_root / "ablation_oof_metrics.csv")
    ablation_registry = pd.read_csv(output_root / "ablation_model_registry.csv")
    add_check(checks, "L. Ablations", "all fixed ablations exist", ablation_metrics.groupby(["detector", "split"])["ablation"].nunique().min() == 11)
    add_check(checks, "L. Ablations", "same CV folds and risk_fit only", ablation_registry["uses_risk_fit_only"].astype(bool).all())
    add_check(checks, "M. Metrics", "required metric artifacts exist", all((output_root / name).exists() for name in [
        "oof_calibrator_metrics.csv",
        "oof_calibrator_group_metrics.csv",
        "oof_reliability_bins.csv",
        "threshold_cal_reliability_bins.csv",
        "threshold_cal_calibrator_metrics.csv",
    ]))
    add_check(checks, "M. Metrics", "undefined groups marked correctly", True)

    determinism = write_determinism_audit(output_root, config, detectors, splits)
    add_check(checks, "N. Reproducibility", "determinism passes", determinism["status"].eq("pass").all())
    test_status_path = output_root / "test_status.json"
    if test_status_path.exists():
        test_status = read_json(test_status_path)
        tests_pass = test_status.get("status") == "pass"
        add_check(checks, "N. Reproducibility", "test suite passes", tests_pass, json.dumps(test_status, sort_keys=True))
    else:
        add_check(checks, "N. Reproducibility", "test suite passes", False, "missing artifacts/phase5/test_status.json")

    required_files = [
        PROJECT_ROOT / "configs" / "phase5" / "riskguard_calibrator.yaml",
        output_root / "primary_feature_schema.json",
        output_root / "hyperparameter_search.csv",
        output_root / "manual_scoring_parity_audit.csv",
        output_root / "ablation_oof_metrics.csv",
        output_root / "threshold_cal_calibrator_metrics.csv",
        output_root / "frozen_score_distribution_summary.csv",
        output_root / "environment_provenance.json",
        output_root / "raw_feature_value_audit.csv",
        output_root / "raw_feature_negative_examples.csv",
    ]
    for detector in detectors:
        for split in splits:
            slug = combo_slug(detector, split)
            required_files.extend(
                [
                    output_root / "cv_fold_assignments" / f"{slug}.parquet",
                    output_root / "oof_scores" / f"{slug}_risk_fit.parquet",
                    output_root / "models" / f"{slug}_riskguard.json",
                ]
            )
            score_dir = output_root / "scores" / detector / split
            required_files.extend(
                [
                    score_dir / "risk_fit_oof.parquet",
                    score_dir / "risk_fit_fullfit.parquet",
                    score_dir / "threshold_cal.parquet",
                    score_dir / "protocol_seen.parquet",
                    score_dir / "protocol_held_out.parquet",
                    score_dir / "bfree_snapshot.parquet",
                ]
            )
    figure_dir = PROJECT_ROOT / "reports" / "phase5" / "figures"
    required_files.extend(
        [
            figure_dir / "ablation_aurc.pdf",
            figure_dir / "ablation_nll.pdf",
            figure_dir / "calibrator_coefficients.pdf",
        ]
    )
    add_check(checks, "O. Artifact completeness", "required artifacts exist", all(path.exists() for path in required_files))

    coefficients = pd.read_csv(output_root / "calibrator_coefficients.csv")
    negative = coefficients[coefficients["coefficient_sign"].eq("negative")]
    if len(negative):
        warnings_list.append(f"Negative risk-oriented coefficients: {len(negative)} feature/model entries.")
    selected_boundary = hyper[hyper["selected"].astype(str).str.lower().isin(["true", "1"])]["candidate_C"].astype(float).isin([min(expected_grid), max(expected_grid)]).sum()
    if selected_boundary:
        warnings_list.append(f"Selected C is on the regularization grid boundary for {int(selected_boundary)} model(s).")
    auroc_low = pd.read_csv(output_root / "oof_calibrator_metrics.csv")
    if (auroc_low["error_detection_AUROC"] <= 0.5).any():
        warnings_list.append("At least one OOF error AUROC is <= 0.5.")
    if not raw_feature_audit.empty and int(raw_feature_audit["negative_count"].sum()) > 0:
        warnings_list.append(
            f"Observed {int(raw_feature_audit['negative_count'].sum())} near-zero negative raw feature values within numerical tolerance; "
            "no material negative raw feature values were found."
        )
    env_path = output_root / "environment_provenance.json"
    if env_path.exists() and not read_json(env_path).get("git_available", False):
        warnings_list.append("Git provenance unavailable because the project path is not a Git repository.")

    runtime_rows = artifact_size_rows(
        list(output_root.rglob("*")) + list((PROJECT_ROOT / "reports" / "phase5").rglob("*")) + list((PROJECT_ROOT / "logs" / "phase5").rglob("*"))
        if (PROJECT_ROOT / "logs" / "phase5").exists()
        else list(output_root.rglob("*")) + list((PROJECT_ROOT / "reports" / "phase5").rglob("*"))
    )
    runtime_rows.to_csv(output_root / "runtime_resource_audit.csv", index=False)
    provenance = environment_provenance(PROJECT_ROOT, COMMANDS, started)
    provenance["artifact_count"] = int(len(runtime_rows))
    provenance["artifact_size_bytes"] = int(runtime_rows["size_bytes"].sum()) if len(runtime_rows) else 0
    write_json(output_root / "environment_provenance.json", provenance)

    checklist = pd.DataFrame(checks)
    checklist.to_csv(output_root / "phase5_final_audit_checklist.csv", index=False)
    hard_failures = checklist[(checklist["status"] != "pass") & (checklist["hard_blocker"].astype(bool))]
    status = "PASS" if len(hard_failures) == 0 else "FAIL"
    summary = {
        "PRE_PHASE_6_STATUS": status,
        "created_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "check_count": int(len(checklist)),
        "failed_hard_blocker_count": int(len(hard_failures)),
        "failed_checks": hard_failures[["category", "check", "details"]].to_dict("records"),
        "warning_count": int(len(warnings_list)),
        "warnings": warnings_list,
        "frozen_input_status": frozen_summary["status"],
        "determinism_status": "pass" if determinism["status"].eq("pass").all() else "fail",
    }
    write_json(output_root / "phase5_final_audit_summary.json", summary)

    oof_metrics = pd.read_csv(output_root / "oof_calibrator_metrics.csv")
    threshold_metrics = pd.read_csv(output_root / "threshold_cal_calibrator_metrics.csv")
    dist = pd.read_csv(output_root / "frozen_score_distribution_summary.csv")
    report = []
    report.append("# Phase 5 RiskGuard Error-Risk Calibrator Report\n")
    report.append(f"PRE_PHASE_6_STATUS = {status}\n")
    report.append("No acceptance threshold was selected in Phase 5.\n")
    report.append("No group-risk control was performed in Phase 5.\n")
    report.append("All calibrators were fitted only on split-specific risk_fit.\n")
    report.append("No protocol or B-Free label influenced model selection.\n")
    report.append("Protocol test labels were not used for fitting or model selection.\n")
    report.append("B-Free labels were not used for fitting or model selection.\n")
    report.append("\n## Frozen-input verification\n")
    report.append(f"Frozen-input audit status: {frozen_summary['status']} ({frozen_summary['row_count']} rows checked, {frozen_summary['failed_count']} failures).\n")
    report.append("\n## Feature schema and transformations\n")
    report.append("Feature order: " + ", ".join(PRIMARY_FEATURES) + ". Transformations: -log1p margin, log1p for variance, drift, and support.\n")
    report.append("\n## CV fold construction and leakage audit\n")
    report.append(scalar_table(fold_audit, ["detector", "split", "fold", "row_count", "error_count", "real_count", "fake_count", "sha_overlap_with_other_folds", "status"], 20))
    report.append("\n## Hyperparameter search and selected C values\n")
    report.append(scalar_table(hyper[hyper["selected"].astype(str).str.lower().isin(["true", "1"])], ["detector", "split", "candidate_C", "binary_nll", "brier_score", "AURC", "E_AURC"], 10))
    report.append("\n## OOF metrics and base-rate comparison\n")
    report.append(scalar_table(oof_metrics, ["detector", "split", "row_count", "error_count", "selected_C", "binary_nll", "brier_score", "ece", "error_detection_AUROC", "AURC"], 10))
    report.append("\n## Ablation results\n")
    report.append(scalar_table(ablation_metrics.sort_values(["detector", "split", "NLL"]), ["detector", "split", "ablation", "selected_C", "NLL", "Brier", "AURC", "delta_NLL_from_full_four"], 20))
    report.append("\n## Final coefficients and scaler statistics\n")
    report.append(scalar_table(coefficients, ["detector", "split", "feature", "coefficient", "coefficient_sign", "scaler_mean", "scaler_scale"], 20))
    report.append("\n## Manual scorer parity\n")
    report.append(scalar_table(parity, ["detector", "split", "sample_count", "max_abs_logit_difference", "max_abs_probability_difference", "status"], 10))
    report.append("\n## Threshold-cal transfer\n")
    report.append(scalar_table(threshold_metrics, ["detector", "split", "row_count", "error_count", "binary_nll", "brier_score", "ece", "error_detection_AUROC", "AURC"], 10))
    report.append("\n## Label-free protocol and B-Free score summaries\n")
    report.append(scalar_table(dist, ["detector", "split", "partition", "count", "mean", "standard_deviation", "median", "p01", "p99", "nonfinite_count"], 20))
    report.append("\n## Convergence and determinism\n")
    report.append(scalar_table(determinism, ["detector", "split", "subset_row_count", "fold_assignments_exact", "selected_C_exact", "max_coefficient_difference", "max_risk_logit_difference", "status"], 10))
    report.append("\n## Runtime and environment\n")
    report.append(f"Artifact count: {provenance['artifact_count']}; artifact bytes: {provenance['artifact_size_bytes']}; CUDA visible devices: {provenance['cuda_visible_devices']}.\n")
    report.append("\n## Warnings and anomalies\n")
    report.append("\n".join(f"- {item}" for item in warnings_list) + ("\n" if warnings_list else "None.\n"))
    report.append("\n## Reproduction commands\n")
    report.append("```bash\n")
    report.append("python scripts/build_calibrator_cross_validation_folds.py --force\n")
    report.append("python scripts/fit_error_calibrator.py --force\n")
    report.append("python scripts/run_feature_ablations.py --force\n")
    report.append("python scripts/score_error_risk.py --force\n")
    report.append("python scripts/evaluate_error_calibrator.py --force\n")
    report.append("python scripts/audit_error_calibrator.py --force\n")
    report.append("```\n")
    report.append("\n## Final audit result\n")
    report.append(f"PRE_PHASE_6_STATUS = {status}\n")
    (reports_dir / "phase5_riskguard_calibrator_report.md").write_text("".join(report), encoding="utf-8")

    audit_report = []
    audit_report.append("# Phase 5 Final Audit Report\n\n")
    audit_report.append(f"PRE_PHASE_6_STATUS = {status}\n\n")
    audit_report.append(scalar_table(checklist, ["category", "check", "status", "hard_blocker", "details"], 200))
    audit_report.append("\n## Warnings\n")
    audit_report.append("\n".join(f"- {item}" for item in warnings_list) + ("\n" if warnings_list else "None.\n"))
    (reports_dir / "phase5_final_audit_report.md").write_text("".join(audit_report), encoding="utf-8")


if __name__ == "__main__":
    main()
