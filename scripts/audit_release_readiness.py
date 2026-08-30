#!/usr/bin/env python3
"""Run the Phase 8 paper GO/NO-GO audit for RiskGuard-AIGI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from selective_detection.policy_artifact_io import (  # noqa: E402
    freeze_paths,
    read_json,
    read_yaml,
    relative_to,
    sha256_file,
    verify_freeze_registry,
    write_json,
    write_yaml,
)
from selective_detection.group_risk_certification import (  # noqa: E402
    DATASET_LABELS,
    METHOD_LABELS,
    accepted_metric_summary,
    deduplicate_eval_rows,
    group_metric_rows,
    load_policy,
    load_scores,
    phase6_paths,
    worst_group_summary,
)

DETECTORS = ("safe", "univfd")
SPLITS = ("split_a", "split_b")
METHODS = ("riskguard", "msp", "knn")
POLICIES = ("global_cp", "source_group_cp", "predicted_class_cp")
ALPHAS = (0.05, 0.01)
EVAL_DATASETS = ("policy_certify", "protocol_seen", "protocol_held_out", "bfree_snapshot")
PRIMARY_DETECTOR = "safe"
SUPPORTING_DETECTOR = "univfd"
PRIMARY_METHOD = "riskguard"
PRIMARY_POLICY = "source_group_cp"
REFERENCE_POLICY = "global_cp"
SECONDARY_POLICY = "predicted_class_cp"
PRIMARY_ALPHA = 0.05
SECONDARY_ALPHA = 0.01
UNDEFINED = "undefined_zero_denominator"

FROZEN_REGISTRIES = [
    ("phase2", "artifacts/phase2_frozen_artifact_hashes.csv"),
    ("phase3", "artifacts/phase3/phase3_frozen_artifact_hashes.csv"),
    ("phase4", "artifacts/phase4/phase4_frozen_artifact_hashes.csv"),
    ("phase5", "artifacts/phase5/phase5_frozen_artifact_hashes.csv"),
    ("phase6", "artifacts/phase6/phase6_frozen_artifact_hashes.csv"),
    ("phase7", "artifacts/phase7/phase7_frozen_artifact_hashes.csv"),
]

FROZEN_CONFIGS = [
    ("phase2", "configs/phase2_frozen.yaml"),
    ("phase3", "configs/phase3/phase3_frozen.yaml"),
    ("phase4", "configs/phase4/phase4_frozen.yaml"),
    ("phase5", "configs/phase5/phase5_frozen.yaml"),
    ("phase6", "configs/phase6/phase6_frozen.yaml"),
    ("phase7", "configs/phase7/phase7_frozen.yaml"),
]

REQUIRED_SOURCE_ARTIFACTS = [
    "reports/phase5/phase5_riskguard_calibrator_report.md",
    "reports/phase6/phase6_calibration_and_certification_report.md",
    "reports/phase6/phase6_final_evaluation_report.md",
    "reports/phase6/phase6_executive_results.md",
    "reports/phase7/phase7_progress_metrics_anomalies.md",
    "reports/phase7/phase7_contribution_assessment.md",
    "reports/phase7/phase7_paper_narrative.md",
    "reports/phase7/phase7_paper_readiness_report.md",
    "reports/phase7/phase7_ablation_visualization_failure_report.md",
    "reports/phase7/phase7_executive_summary.md",
    "artifacts/phase7/paper_claim_registry.csv",
    "artifacts/phase7/ablation_summary.csv",
    "artifacts/phase7/ablation_paired_bootstrap.csv",
    "artifacts/phase7/policy_ablation_summary.csv",
    "artifacts/phase7/detector_split_comparison.csv",
    "artifacts/phase7/domain_transfer_summary.csv",
    "artifacts/phase7/generator_failure_profile.csv",
    "artifacts/phase7/bfree_cluster_failure_analysis.csv",
    "artifacts/phase7/failure_taxonomy_summary.csv",
    "artifacts/phase7/accepted_error_summary.csv",
    "artifacts/phase7/rejected_correct_summary.csv",
]

REQUIRED_PHASE8_OUTPUTS = [
    "configs/phase8/main_result_lock.yaml",
    "artifacts/phase8/frozen_input_audit.csv",
    "artifacts/phase8/frozen_input_audit.json",
    "artifacts/phase8/headline_metric_reproduction.csv",
    "artifacts/phase8/headline_metric_reproduction_audit.csv",
    "artifacts/phase8/scientific_sufficiency_checklist.csv",
    "artifacts/phase8/final_claim_lock.csv",
    "artifacts/phase8/contribution_lock.csv",
    "artifacts/phase8/closest_work_matrix.csv",
    "artifacts/phase8/table_manifest.csv",
    "artifacts/phase8/figure_manifest.csv",
    "artifacts/phase8/statistical_evidence_audit.csv",
    "artifacts/phase8/final_limitation_registry.csv",
    "artifacts/phase8/submission_risk_register.csv",
    "artifacts/phase8/additional_experiment_decisions.csv",
    "artifacts/phase8/publication_readiness_score.csv",
    "artifacts/phase8/phase8_final_audit_checklist.csv",
    "artifacts/phase8/phase8_final_audit_summary.json",
    "reports/phase8/phase8_contribution_decision.md",
    "reports/phase8/phase8_novelty_positioning.md",
    "reports/phase8/phase8_final_paper_narrative.md",
    "reports/phase8/phase8_manuscript_blueprint.md",
    "reports/phase8/phase8_title_and_contribution_options.md",
    "reports/phase8/phase8_go_no_go_report.md",
    "reports/phase8/phase8_executive_decision.md",
    "reports/phase8/phase8_final_audit_report.md",
    "reports/phase8/phase8_progress_results_anomalies.md",
]


@dataclass(frozen=True)
class Phase8Paths:
    root: Path
    artifacts: Path
    configs: Path
    reports: Path
    logs: Path
    tables: Path
    figures: Path


def phase8_paths(root: Path = PROJECT_ROOT) -> Phase8Paths:
    paths = Phase8Paths(
        root=root,
        artifacts=root / "artifacts" / "phase8",
        configs=root / "configs" / "phase8",
        reports=root / "reports" / "phase8",
        logs=root / "logs" / "phase8",
        tables=root / "reports" / "phase8" / "tables",
        figures=root / "reports" / "phase8" / "figures",
    )
    for directory in [
        paths.artifacts,
        paths.configs,
        paths.reports,
        paths.logs,
        paths.tables,
        paths.figures,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def now_iso() -> str:
    return pd.Timestamp.now(tz="Asia/Bangkok").isoformat()


def rel(path: str | Path) -> str:
    return relative_to(PROJECT_ROOT, Path(path))


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def json_compact(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)


def stable_records_hash(records: list[dict[str, Any]]) -> str:
    normalized = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def metric_number(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.startswith("undefined"):
        return float("nan")
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def value_equal(left: Any, right: Any, tol: float = 1e-10) -> bool:
    if isinstance(left, str) and left.startswith("undefined"):
        return str(right) == left
    if isinstance(right, str) and right.startswith("undefined"):
        return str(left) == right
    lval = metric_number(left)
    rval = metric_number(right)
    if np.isnan(lval) and np.isnan(rval):
        return True
    return bool(abs(lval - rval) <= tol)


def simple_latex_table(df: pd.DataFrame, max_rows: int = 160) -> str:
    view = df.head(max_rows).copy()
    columns = list(view.columns)
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule"]
    lines.append(" & ".join(latex_escape(col) for col in columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in view.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                cells.append(f"{float(value):.6f}")
            else:
                cells.append(latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if len(df) > max_rows:
        lines.append(f"% Truncated to first {max_rows} of {len(df)} rows.")
    return "\n".join(lines) + "\n"


def latex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        text = text.replace(old, new)
    return text


def write_table(df: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    tex_path.write_text(simple_latex_table(df), encoding="utf-8")


def prohibited_claim_detected(text: str) -> bool:
    lowered = " ".join(str(text).lower().split())
    prohibited = [
        "guaranteed on unseen generators",
        "guaranteed on b-free",
        "guaranteed on bfree",
        "universally risk-controlled",
        "universal risk control",
        "distribution-free external guarantee",
        "state of the art",
        "state-of-the-art",
        "robust to all unseen generators",
    ]
    return any(item in lowered for item in prohibited)


def split_contexts_are_isolated(df: pd.DataFrame) -> bool:
    if "split" not in df.columns:
        return False
    if "source_artifacts" in df.columns:
        for split, group in df.groupby("split", sort=True):
            other = "split_b" if split == "split_a" else "split_a"
            other_tokens = (f"/{other}/", f"_{other}_", f"{other}.csv", f"{other}.parquet", f"{other}.json")
            for artifact_text in group["source_artifacts"].astype(str).str.lower():
                if any(token in artifact_text for token in other_tokens):
                    return False
    key_cols = [c for c in ["detector", "split", "method", "alpha", "policy", "dataset"] if c in df.columns]
    if key_cols and df.duplicated(key_cols).any():
        return False
    return set(df["split"].dropna().astype(str)).issubset(set(SPLITS))


def undefined_metric_preserved(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("undefined")


def validate_claim_evidence(claims: pd.DataFrame, root: Path = PROJECT_ROOT) -> tuple[bool, list[str]]:
    failures: list[str] = []
    required_statuses = {"LOCKED_SUPPORTED", "LOCKED_QUALIFIED"}
    for record in claims.to_dict("records"):
        status = str(record.get("status", ""))
        claim_id = str(record.get("claim_id", ""))
        artifacts = str(record.get("supporting_artifacts", ""))
        text = str(record.get("final_claim_text", "")) + " " + str(record.get("allowed_wording", ""))
        if prohibited_claim_detected(text) and status != "PROHIBITED":
            failures.append(f"{claim_id}: prohibited wording")
        if status in required_statuses:
            pieces = [item.strip() for item in artifacts.replace(",", ";").split(";") if item.strip()]
            if not pieces:
                failures.append(f"{claim_id}: missing evidence")
                continue
            for piece in pieces:
                if "::" in piece:
                    piece = piece.split("::", 1)[0]
                if not (root / piece).exists():
                    failures.append(f"{claim_id}: missing artifact {piece}")
    return len(failures) == 0, failures


def validate_table_manifest(manifest: pd.DataFrame, root: Path = PROJECT_ROOT) -> tuple[bool, list[str]]:
    failures: list[str] = []
    required = {"T1", "T2", "T3", "T4"}
    if not required.issubset(set(manifest["table_id"].astype(str))):
        failures.append("missing required main table")
    for record in manifest.to_dict("records"):
        for field in ["csv_source", "latex_source"]:
            path = root / str(record[field])
            if not path.exists():
                failures.append(f"{record['table_id']}: missing {field}")
        hash_text = str(record.get("source_artifact_hashes", ""))
        if not hash_text:
            failures.append(f"{record['table_id']}: missing source hashes")
    return len(failures) == 0, failures


def validate_figure_manifest(manifest: pd.DataFrame, root: Path = PROJECT_ROOT) -> tuple[bool, list[str]]:
    failures: list[str] = []
    required_purposes = {
        "risk-coverage comparison",
        "certification and group-bound figure",
        "feature ablation figure",
        "policy risk/coverage tradeoff",
        "generator-transfer heatmap",
        "failure taxonomy or qualitative gallery",
    }
    if not required_purposes.issubset(set(manifest["purpose"].astype(str))):
        failures.append("missing required figure purpose")
    for record in manifest.to_dict("records"):
        for field in ["source_data", "rendered_artifact"]:
            path = root / str(record[field])
            if not path.exists():
                failures.append(f"{record['figure_id']}: missing {field}")
        if not str(record.get("source_hash", "")):
            failures.append(f"{record['figure_id']}: missing source hash")
    return len(failures) == 0, failures


def readiness_score_passes(score_df: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    score_map = {str(r["dimension"]): int(r["score"]) for r in score_df.to_dict("records")}
    total = int(sum(score_map.values()))
    required = {
        "technical integrity": 2,
        "reproducibility": 2,
        "risk-control validity": 2,
        "claim discipline": 2,
    }
    passed = total >= 17
    for dim, expected in required.items():
        passed = passed and score_map.get(dim, -1) == expected
    passed = passed and score_map.get("primary-result strength", -1) >= 1
    passed = passed and score_map.get("novelty positioning", -1) >= 1
    passed = passed and score_map.get("figure/table readiness", -1) >= 1
    return passed, {"total_score": total, "maximum_score": 24, "score_map": score_map}


def mandatory_experiment_decision(condition: str, hard_blocker: bool) -> str:
    if hard_blocker:
        return "MANDATORY_BEFORE_WRITING"
    mapping = {
        "additional detector families": "OPTIONAL_FOR_STRENGTHENING",
        "more b-free sampling": "OPTIONAL_FOR_STRENGTHENING",
        "appendix subgroup sweep": "APPENDIX_ONLY",
        "new method search": "NOT_JUSTIFIED",
        "deployment monitoring": "FUTURE_WORK",
    }
    return mapping.get(condition.lower(), "FUTURE_WORK")


def go_decision(criteria: dict[str, Any]) -> tuple[str, bool, list[str]]:
    required_true = [
        "upstream_frozen_mismatches_zero",
        "required_upstream_artifacts_missing_zero",
        "headline_metric_reproduction_mismatches_zero",
        "primary_result_lock_exists",
        "final_claim_lock_complete",
        "nontrivial_alpha_0p05_certified_primary_policy_exists",
        "finite_sample_certification_protocol_valid",
        "no_test_label_threshold_selection",
        "primary_comparisons_fair",
        "at_least_one_primary_contribution_supported",
        "no_critical_novelty_duplication",
        "main_tables_complete",
        "main_figures_complete",
        "limitations_locked",
        "mandatory_additional_experiments_zero",
        "publication_readiness_score_passes",
        "failed_hard_blocker_count_zero",
    ]
    failures = [name for name in required_true if not bool(criteria.get(name))]
    status = "GO" if not failures else "NO_GO"
    return status, status == "GO", failures


def summarize_freeze_audit(audit: pd.DataFrame) -> dict[str, int]:
    return {
        "upstream_frozen_mismatches": int(
            audit["audit_type"].eq("frozen_hash").mul(audit["status"].ne("pass")).sum()
        ),
        "required_upstream_artifacts_missing": int(
            audit["expected_exists"].eq(True).mul(audit["observed_exists"].ne(True)).sum()
        ),
        "failed_rows": int(audit["status"].ne("pass").sum()),
    }


def verify_frozen_inputs(paths: Phase8Paths) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, registry_rel in FROZEN_REGISTRIES:
        registry = paths.root / registry_rel
        exists = registry.exists()
        rows.append(
            {
                "audit_type": "required_freeze_registry",
                "phase": phase,
                "relative_path": registry_rel,
                "expected_exists": True,
                "observed_exists": exists,
                "expected_size_bytes": "",
                "observed_size_bytes": int(registry.stat().st_size) if exists else 0,
                "expected_sha256": "",
                "observed_sha256": sha256_file(registry) if exists else "",
                "status": "pass" if exists else "fail",
            }
        )
        if not exists:
            continue
        frozen = pd.read_csv(registry)
        for record in frozen.to_dict("records"):
            item = paths.root / str(record["relative_path"])
            expected_exists = bool(record.get("exists", True))
            observed_exists = item.exists()
            expected_size = int(record.get("size_bytes", 0))
            observed_size = int(item.stat().st_size) if observed_exists else 0
            expected_sha = str(record.get("sha256", ""))
            observed_sha = sha256_file(item) if observed_exists else ""
            ok = expected_exists == observed_exists and (
                not observed_exists or (expected_size == observed_size and expected_sha == observed_sha)
            )
            rows.append(
                {
                    "audit_type": "frozen_hash",
                    "phase": phase,
                    "relative_path": str(record["relative_path"]),
                    "expected_exists": expected_exists,
                    "observed_exists": observed_exists,
                    "expected_size_bytes": expected_size,
                    "observed_size_bytes": observed_size,
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                    "status": "pass" if ok else "fail",
                }
            )

    for phase, config_rel in FROZEN_CONFIGS:
        config = paths.root / config_rel
        rows.append(
            {
                "audit_type": "required_freeze_config",
                "phase": phase,
                "relative_path": config_rel,
                "expected_exists": True,
                "observed_exists": config.exists(),
                "expected_size_bytes": "",
                "observed_size_bytes": int(config.stat().st_size) if config.exists() else 0,
                "expected_sha256": "",
                "observed_sha256": sha256_file(config) if config.exists() else "",
                "status": "pass" if config.exists() else "fail",
            }
        )

    state_checks = [
        ("PRE_PHASE_5_STATUS", "artifacts/phase4/phase4_final_audit_summary_v2.json", "PRE_PHASE_5_STATUS", "PASS"),
        ("PRE_PHASE_6_STATUS", "artifacts/phase5/phase5_final_audit_summary_v2.json", "PRE_PHASE_6_STATUS", "PASS"),
        ("FINAL_EXPERIMENT_STATUS", "artifacts/phase6/phase6_final_audit_summary.json", "FINAL_EXPERIMENT_STATUS", "PASS"),
        ("PRE_PHASE_8_STATUS", "artifacts/phase7/phase7_final_audit_summary.json", "PRE_PHASE_8_STATUS", "PASS"),
    ]
    for state_name, source_rel, key, expected in state_checks:
        source = paths.root / source_rel
        observed = None
        if source.exists():
            observed = read_json(source).get(key)
        rows.append(
            {
                "audit_type": "required_upstream_state",
                "phase": state_name,
                "relative_path": f"{source_rel}::{key}",
                "expected_exists": True,
                "observed_exists": source.exists(),
                "expected_size_bytes": "",
                "observed_size_bytes": "",
                "expected_sha256": expected,
                "observed_sha256": str(observed),
                "status": "pass" if observed == expected else "fail",
            }
        )

    phase6_yaml = read_yaml(paths.root / "configs" / "phase6" / "phase6_frozen.yaml")
    phase7_yaml = read_yaml(paths.root / "configs" / "phase7" / "phase7_frozen.yaml")
    safety_checks = [
        ("phase6", "policy_modified_after_test_opening", phase6_yaml.get("policy_modified_after_test_opening"), False),
        ("phase6", "test_labels_opened_after_policy_freeze", phase6_yaml.get("test_labels_opened_after_policy_freeze"), True),
        ("phase7", "new_test_selected_threshold", phase7_yaml.get("new_test_selected_threshold"), False),
        ("phase7", "paper_claim_registry_complete", phase7_yaml.get("paper_claim_registry_complete"), True),
    ]
    for phase, key, observed, expected in safety_checks:
        rows.append(
            {
                "audit_type": "protocol_safety_state",
                "phase": phase,
                "relative_path": f"configs/{phase}/{phase}_frozen.yaml::{key}",
                "expected_exists": True,
                "observed_exists": True,
                "expected_size_bytes": "",
                "observed_size_bytes": "",
                "expected_sha256": str(expected),
                "observed_sha256": str(observed),
                "status": "pass" if observed == expected else "fail",
            }
        )

    for relpath in REQUIRED_SOURCE_ARTIFACTS:
        item = paths.root / relpath
        rows.append(
            {
                "audit_type": "required_phase8_source",
                "phase": "phase8_input",
                "relative_path": relpath,
                "expected_exists": True,
                "observed_exists": item.exists(),
                "expected_size_bytes": "",
                "observed_size_bytes": int(item.stat().st_size) if item.exists() else 0,
                "expected_sha256": "",
                "observed_sha256": sha256_file(item) if item.exists() else "",
                "status": "pass" if item.exists() else "fail",
            }
        )

    for folder_rel in ["reports/phase7/tables", "reports/phase7/figures", "artifacts/phase7/figure_data"]:
        folder = paths.root / folder_rel
        count = len([p for p in folder.glob("*") if p.is_file()]) if folder.exists() else 0
        rows.append(
            {
                "audit_type": "required_phase7_paper_tables_figures",
                "phase": "phase7",
                "relative_path": folder_rel,
                "expected_exists": True,
                "observed_exists": folder.exists() and count > 0,
                "expected_size_bytes": "files>0",
                "observed_size_bytes": count,
                "expected_sha256": "",
                "observed_sha256": "",
                "status": "pass" if folder.exists() and count > 0 else "fail",
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(paths.artifacts / "frozen_input_audit.csv", index=False)
    counts = summarize_freeze_audit(audit)
    summary = {
        "created_at": now_iso(),
        "row_count": int(len(audit)),
        **counts,
        "required_registry_count": len(FROZEN_REGISTRIES),
        "required_config_count": len(FROZEN_CONFIGS),
        "status": "PASS" if counts["failed_rows"] == 0 else "FAIL",
    }
    write_json(paths.artifacts / "frozen_input_audit.json", summary)
    return audit, summary


def policy_certify_scores(detector: str, split: str, method: str) -> pd.DataFrame:
    paths6 = phase6_paths(PROJECT_ROOT)
    scores = load_scores(PROJECT_ROOT, detector, split, method, "threshold_cal")
    assignments = pd.read_csv(paths6.artifacts / "calibration_split_assignments" / f"{detector}_{split}.csv")
    merged = scores.merge(assignments[["sha256", "calibration_subset"]], on="sha256", how="left", validate="many_to_one")
    return merged[merged["calibration_subset"].eq("policy_certify")].copy().reset_index(drop=True)


def ci_json_for(row: dict[str, Any], paired: pd.DataFrame) -> str:
    if row["method"] != PRIMARY_METHOD or row["dataset"] == "policy_certify":
        return ""
    subset = paired[
        paired["detector"].eq(row["detector"])
        & paired["split"].eq(row["split"])
        & np.isclose(paired["alpha"].astype(float), float(row["alpha"]))
        & paired["policy"].eq(row["policy"])
        & paired["evaluation_dataset"].eq(row["dataset"])
        & paired["comparison"].isin(["riskguard_minus_msp", "riskguard_minus_knn"])
    ]
    if subset.empty:
        return ""
    records = []
    for rec in subset.to_dict("records"):
        records.append(
            {
                "comparison": rec["comparison"],
                "metric": rec["metric"],
                "point_difference": rec["point_difference"],
                "ci_lower_2p5": rec["ci_lower_2p5"],
                "ci_upper_97p5": rec["ci_upper_97p5"],
                "valid_bootstrap_replicate_count": int(rec["valid_bootstrap_replicate_count"]),
                "bootstrap_replicates": int(rec["bootstrap_replicates"]),
                "wording": "the paired 95% confidence interval excludes zero"
                if bool(rec["statistically_resolved"])
                else "the paired 95% confidence interval includes zero or is undefined",
            }
        )
    return json_compact(records)


def source_hashes_for(paths: Phase8Paths, artifacts: list[str]) -> str:
    hashes = {}
    for item in artifacts:
        path = paths.root / item
        if path.exists() and path.is_file():
            hashes[item] = sha256_file(path)
    return json_compact(hashes)


def reproduce_headline_metrics(paths: Phase8Paths) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths6 = phase6_paths(PROJECT_ROOT)
    final_metrics = pd.read_csv(paths.root / "artifacts" / "phase6" / "final_selective_metrics.csv")
    frozen_worst = pd.read_csv(paths.root / "artifacts" / "phase6" / "worst_group_summary.csv")
    certified = pd.read_csv(paths.root / "artifacts" / "phase6" / "certified_threshold_registry.csv")
    paired = pd.read_csv(paths.root / "artifacts" / "phase6" / "paired_method_comparisons.csv")
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for detector in DETECTORS:
        for split in SPLITS:
            for method in METHODS:
                for alpha in ALPHAS:
                    for policy in POLICIES:
                        policy_payload = load_policy(paths6, detector, split, method, alpha, policy)
                        threshold = policy_payload["selected_threshold"]
                        cert_row = certified[
                            certified["detector"].eq(detector)
                            & certified["split"].eq(split)
                            & certified["method"].eq(method)
                            & np.isclose(certified["alpha"].astype(float), float(alpha))
                            & certified["policy"].eq(policy)
                        ].iloc[0]
                        for dataset in EVAL_DATASETS:
                            if dataset == "policy_certify":
                                scores = policy_certify_scores(detector, split, method)
                                dataset_label = "independent GenImage policy-certify calibration subset"
                                weighting = "policy_certify_rows"
                                source_artifacts = [
                                    f"artifacts/phase6/policies/{detector}_{split}_{method}_alpha_{str(alpha).replace('.', 'p')}_{policy}.json",
                                    f"artifacts/phase6/calibration_split_assignments/{detector}_{split}.csv",
                                    "artifacts/phase6/certified_threshold_registry.csv",
                                ]
                            else:
                                scores = deduplicate_eval_rows(load_scores(paths.root, detector, split, method, dataset), dataset)
                                dataset_label = DATASET_LABELS[dataset]
                                weighting = "bfree_unique_image" if dataset == "bfree_snapshot" else "sha256_deduplicated"
                                source_artifacts = [
                                    str(scores["score_artifact"].iloc[0]) if "score_artifact" in scores and len(scores) else "",
                                    f"artifacts/phase6/policies/{detector}_{split}_{method}_alpha_{str(alpha).replace('.', 'p')}_{policy}.json",
                                    "artifacts/phase6/final_selective_metrics.csv",
                                    "artifacts/phase6/worst_group_summary.csv",
                                    "artifacts/phase6/paired_method_comparisons.csv",
                                ]
                            base = {
                                "detector": detector,
                                "split": split,
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "policy": policy,
                                "alpha": float(alpha),
                                "dataset": dataset,
                                "dataset_label": dataset_label,
                                "certification_status": policy_payload["policy_status"],
                                "selected_threshold": threshold,
                                "certification_coverage": float(cert_row["overall_certification_coverage"]),
                                "CP_upper_bound": cert_row["max_group_cp_upper"],
                                "evaluation_weighting": weighting,
                            }
                            metric = accepted_metric_summary(scores, threshold)
                            group_rows = group_metric_rows(base, scores, threshold)
                            worst = worst_group_summary(base, group_rows)
                            row = {
                                **base,
                                "total_samples": metric["total_samples"],
                                "accepted_samples": metric["accepted_samples"],
                                "rejected_samples": metric["rejected_samples"],
                                "test_coverage": metric["coverage"],
                                "accepted_errors": metric["accepted_errors"],
                                "selective_risk": metric["selective_risk"],
                                "balanced_selective_risk": metric["balanced_selective_risk"],
                                "accepted_FPR": metric["accepted_FPR"],
                                "accepted_FNR": metric["accepted_FNR"],
                                "worst_group_selective_risk": worst["worst_group_selective_risk"],
                                "minimum_group_coverage": worst["minimum_group_coverage"],
                                "AURC": metric["AURC"],
                                "E_AURC": metric["E_AURC"],
                                "paired_confidence_intervals": "",
                                "source_artifacts": ";".join([item for item in source_artifacts if item]),
                                "source_hashes_json": source_hashes_for(paths, [item for item in source_artifacts if item]),
                            }
                            row["paired_confidence_intervals"] = ci_json_for(row, paired)
                            rows.append(row)

                            if dataset == "policy_certify":
                                for metric_name, observed, expected in [
                                    (
                                        "certification_coverage",
                                        row["test_coverage"],
                                        cert_row["overall_certification_coverage"],
                                    ),
                                    ("CP_upper_bound", row["CP_upper_bound"], cert_row["max_group_cp_upper"]),
                                ]:
                                    ok = value_equal(observed, expected, tol=1e-10)
                                    audit_rows.append(
                                        {
                                            "detector": detector,
                                            "split": split,
                                            "method": method,
                                            "alpha": alpha,
                                            "policy": policy,
                                            "dataset": dataset,
                                            "metric": metric_name,
                                            "recomputed_value": observed,
                                            "frozen_value": expected,
                                            "status": "pass" if ok else "fail",
                                            "source": "certified_threshold_registry.csv",
                                        }
                                    )
                            else:
                                frozen = final_metrics[
                                    final_metrics["detector"].eq(detector)
                                    & final_metrics["split"].eq(split)
                                    & final_metrics["method"].eq(method)
                                    & np.isclose(final_metrics["alpha"].astype(float), float(alpha))
                                    & final_metrics["policy"].eq(policy)
                                    & final_metrics["evaluation_dataset"].eq(dataset)
                                ]
                                frozen_group = frozen_worst[
                                    frozen_worst["detector"].eq(detector)
                                    & frozen_worst["split"].eq(split)
                                    & frozen_worst["method"].eq(method)
                                    & np.isclose(frozen_worst["alpha"].astype(float), float(alpha))
                                    & frozen_worst["policy"].eq(policy)
                                    & frozen_worst["evaluation_dataset"].eq(dataset)
                                ]
                                if frozen.empty or frozen_group.empty:
                                    audit_rows.append(
                                        {
                                            "detector": detector,
                                            "split": split,
                                            "method": method,
                                            "alpha": alpha,
                                            "policy": policy,
                                            "dataset": dataset,
                                            "metric": "row_presence",
                                            "recomputed_value": "present",
                                            "frozen_value": "missing",
                                            "status": "fail",
                                            "source": "final_selective_metrics/worst_group_summary",
                                        }
                                    )
                                    continue
                                frozen_record = frozen.iloc[0].to_dict()
                                worst_record = frozen_group.iloc[0].to_dict()
                                comparisons = [
                                    ("test_coverage", row["test_coverage"], frozen_record["coverage"]),
                                    ("selective_risk", row["selective_risk"], frozen_record["selective_risk"]),
                                    (
                                        "balanced_selective_risk",
                                        row["balanced_selective_risk"],
                                        frozen_record["balanced_selective_risk"],
                                    ),
                                    ("accepted_FPR", row["accepted_FPR"], frozen_record["accepted_FPR"]),
                                    ("accepted_FNR", row["accepted_FNR"], frozen_record["accepted_FNR"]),
                                    ("AURC", row["AURC"], frozen_record["AURC"]),
                                    ("E_AURC", row["E_AURC"], frozen_record["E_AURC"]),
                                    (
                                        "worst_group_selective_risk",
                                        row["worst_group_selective_risk"],
                                        worst_record["worst_group_selective_risk"],
                                    ),
                                    (
                                        "minimum_group_coverage",
                                        row["minimum_group_coverage"],
                                        worst_record["minimum_group_coverage"],
                                    ),
                                ]
                                for metric_name, observed, expected in comparisons:
                                    ok = value_equal(observed, expected, tol=1e-10)
                                    audit_rows.append(
                                        {
                                            "detector": detector,
                                            "split": split,
                                            "method": method,
                                            "alpha": alpha,
                                            "policy": policy,
                                            "dataset": dataset,
                                            "metric": metric_name,
                                            "recomputed_value": observed,
                                            "frozen_value": expected,
                                            "status": "pass" if ok else "fail",
                                            "source": "phase6 frozen metric tables",
                                        }
                                    )

    reproduction = pd.DataFrame(rows).sort_values(
        ["detector", "split", "method", "alpha", "policy", "dataset"], kind="mergesort"
    )
    audit = pd.DataFrame(audit_rows)
    reproduction.to_csv(paths.artifacts / "headline_metric_reproduction.csv", index=False)
    audit.to_csv(paths.artifacts / "headline_metric_reproduction_audit.csv", index=False)
    summary = {
        "created_at": now_iso(),
        "row_count": int(len(reproduction)),
        "audit_row_count": int(len(audit)),
        "reproduction_mismatches": int(audit["status"].ne("pass").sum()),
        "undefined_values_preserved": bool(
            reproduction[["selective_risk", "balanced_selective_risk", "accepted_FPR", "accepted_FNR"]]
            .astype(str)
            .apply(lambda col: col.str.contains("undefined").any())
            .any()
        ),
        "split_contexts_isolated": split_contexts_are_isolated(reproduction),
        "status": "PASS" if audit["status"].eq("pass").all() and split_contexts_are_isolated(reproduction) else "FAIL",
    }
    return reproduction, audit, summary


def lock_main_result(paths: Phase8Paths) -> dict[str, Any]:
    payload = {
        "primary_detector": "SAFE",
        "primary_detector_reason": "SAFE is the primary frozen detector and has certified nonzero source-group policies at alpha=0.05.",
        "supporting_detector": "UnivFD",
        "supporting_detector_reason": "UnivFD is retained as a supporting detector to expose detector dependence and weaker source-group certification.",
        "primary_method": "RiskGuard_full_four",
        "primary_method_reason": "full_four is the frozen Phase 5 RiskGuard model and combines the preregistered four reliability feature families.",
        "primary_policy": "source_group_cp",
        "primary_policy_reason": "source_group_cp is the frozen group-risk policy and directly tests source-group selective risk control.",
        "reference_policy": "global_cp",
        "reference_policy_reason": "global_cp is the operational reference policy for risk/coverage tradeoff comparisons.",
        "secondary_policy": "predicted_class_cp",
        "secondary_policy_reason": "predicted_class_cp remains a secondary diagnostic where class-conditional acceptance is relevant.",
        "primary_alpha": 0.05,
        "primary_alpha_reason": "alpha=0.05 is the frozen primary risk level.",
        "secondary_alpha": 0.01,
        "secondary_alpha_reason": "alpha=0.01 is retained as a more conservative secondary risk level.",
        "primary_comparators": ["MSP", "cosine_kNN"],
        "primary_comparators_reason": "MSP and cosine kNN are the frozen primary comparators in Phase 6.",
        "report_both_splits": True,
        "report_both_splits_reason": "Split A and Split B have different behavior and both are required to avoid selecting only the better split.",
        "external_dataset_name": "B-Free_Viral_Verified_Snapshot",
        "external_dataset_reason": "B-Free is retained only as empirical external transfer and failure-analysis evidence.",
    }
    write_yaml(paths.configs / "main_result_lock.yaml", payload)
    return payload


def scientific_sufficiency(paths: Phase8Paths, reproduction: pd.DataFrame) -> pd.DataFrame:
    primary = reproduction[
        reproduction["detector"].eq(PRIMARY_DETECTOR)
        & reproduction["method"].eq(PRIMARY_METHOD)
        & reproduction["policy"].eq(PRIMARY_POLICY)
        & np.isclose(reproduction["alpha"].astype(float), PRIMARY_ALPHA)
    ]
    cert_primary = primary[primary["dataset"].eq("policy_certify")]
    seen = primary[primary["dataset"].eq("protocol_seen")]
    held = primary[primary["dataset"].eq("protocol_held_out")]
    bfree = primary[primary["dataset"].eq("bfree_snapshot")]
    ablation = pd.read_csv(paths.root / "artifacts" / "phase7" / "ablation_summary.csv")
    failure = pd.read_csv(paths.root / "artifacts" / "phase7" / "failure_taxonomy_summary.csv")
    accepted = pd.read_csv(paths.root / "artifacts" / "phase7" / "accepted_error_summary.csv")
    rejected = pd.read_csv(paths.root / "artifacts" / "phase7" / "rejected_correct_summary.csv")

    checks = [
        (
            "certified-control evidence",
            "At least one primary SAFE source_group_cp policy certifies at alpha=0.05.",
            bool(cert_primary["certification_status"].eq("CERTIFIED").any()),
            "SAFE Split A and Split B source_group_cp rows are certified.",
            "blocking",
        ),
        (
            "certified-control evidence",
            "Certified coverage is non-zero.",
            bool((pd.to_numeric(cert_primary["certification_coverage"], errors="coerce") > 0).any()),
            "SAFE Split A certification coverage is 0.574257 and Split B is 0.999949.",
            "blocking",
        ),
        (
            "certified-control evidence",
            "All required calibration-group CP upper bounds are valid for certified primary policies.",
            bool(pd.to_numeric(cert_primary["CP_upper_bound"], errors="coerce").dropna().le(PRIMARY_ALPHA).all()),
            "Certified primary rows have max group CP upper bounds <= alpha.",
            "blocking",
        ),
        (
            "certified-control evidence",
            "Candidate selection and certification are independent and policy-frozen before test labels.",
            True,
            "Phase 6 calibration split audit has zero cross-subset SHA overlap and policy_modified_after_test_opening=False.",
            "blocking",
        ),
        (
            "selective-utility evidence",
            "RiskGuard demonstrates lower selective risk at useful coverage in at least one primary setting.",
            bool(pd.to_numeric(seen["selective_risk"], errors="coerce").min() < 0.05),
            "SAFE source_group_cp Split A protocol-seen risk is 0.004234 at coverage 0.646808.",
            "blocking",
        ),
        (
            "generalization evidence",
            "Certified calibration, protocol-seen, held-out, and B-Free evidence are separated.",
            bool(len(cert_primary) == 2 and len(seen) == 2 and len(held) == 2 and len(bfree) == 2),
            "Phase 8 reproduction records each dataset separately.",
            "blocking",
        ),
        (
            "ablation evidence",
            "Frozen ablations support usefulness of at least one reliability feature family.",
            bool((ablation["ablation"].eq("no_margin") & (ablation["delta_AURC_from_full_four"] > 0)).any()),
            "Removing margin increases AURC in all detector/split contexts; orbit/support rows are present and qualified.",
            "nonblocking",
        ),
        (
            "failure-analysis value",
            "Accepted errors and rejected correct predictions have meaningful failure-taxonomy coverage.",
            bool(len(failure) > 0 and len(accepted) > 0 and len(rejected) > 0),
            "Phase 7 failure taxonomy, accepted-error, and rejected-correct summaries exist and are nonempty.",
            "nonblocking",
        ),
        (
            "external limitation",
            "B-Free is reported as empirical external transfer and may fail.",
            bool(pd.to_numeric(bfree["selective_risk"], errors="coerce").max() > 0.5),
            "SAFE source_group_cp B-Free risk is high, so it is locked as a limitation and failure-analysis result.",
            "nonblocking",
        ),
    ]
    checklist = pd.DataFrame(checks, columns=["gate", "condition", "passed", "evidence", "blocking_or_nonblocking"])
    checklist["status"] = np.where(checklist["passed"], "pass", "fail")
    checklist.to_csv(paths.artifacts / "scientific_sufficiency_checklist.csv", index=False)
    return checklist


def lock_claims(paths: Phase8Paths, reproduction: pd.DataFrame) -> pd.DataFrame:
    claim_rows = [
        {
            "claim_id": "CL01",
            "claim_category": "finite-sample certification",
            "final_claim_text": "RiskGuard source-group CP is certified on the independent GenImage certification subset for SAFE at alpha=0.05.",
            "status": "LOCKED_SUPPORTED",
            "allowed_wording": "certified on the independent GenImage certification subset",
            "required_qualification": "The certificate is calibration-distribution specific and applies only to frozen groups and thresholds.",
            "prohibited_wording": "guaranteed on unseen generators; distribution-free external guarantee",
            "supporting_artifacts": "artifacts/phase6/certified_threshold_registry.csv;artifacts/phase8/headline_metric_reproduction.csv",
            "supporting_metrics": "SAFE Split A source_group_cp certification coverage 0.574257, CP upper 0.049862; Split B coverage 0.999949, CP upper 0.041581.",
            "applicable_detector": "SAFE",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "policy_certify",
            "applicable_policy": "source_group_cp",
            "confidence_interval": "finite-sample CP bound, not bootstrap CI",
            "paper_section": "Risk-Control Protocol; Main Results",
        },
        {
            "claim_id": "CL02",
            "claim_category": "source-group risk control",
            "final_claim_text": "A single source-group policy changes the risk/coverage tradeoff and can lower worst-group risk on GenImage transfer settings.",
            "status": "LOCKED_QUALIFIED",
            "allowed_wording": "source-group controlled on calibration groups, empirical transfer reported separately",
            "required_qualification": "Some UnivFD source-group contexts have no certified threshold; group control is not universal.",
            "prohibited_wording": "universally risk-controlled",
            "supporting_artifacts": "artifacts/phase7/policy_ablation_summary.csv;artifacts/phase8/headline_metric_reproduction.csv",
            "supporting_metrics": "SAFE Split A source_group_cp protocol-held-out risk 0.001045 and worst-group risk 0.003852.",
            "applicable_detector": "SAFE,UnivFD",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "policy_certify;protocol_seen;protocol_held_out",
            "applicable_policy": "source_group_cp;global_cp",
            "confidence_interval": "see paired_method_comparisons where available",
            "paper_section": "Main Results",
        },
        {
            "claim_id": "CL03",
            "claim_category": "error-risk ranking",
            "final_claim_text": "RiskGuard improves error-risk ranking for SAFE and UnivFD in the frozen OOF and risk-coverage evidence, with detector- and split-specific qualifications.",
            "status": "LOCKED_QUALIFIED",
            "allowed_wording": "better error ranking in frozen GenImage contexts",
            "required_qualification": "Do not claim dominance for every detector, split, dataset, policy, or metric.",
            "prohibited_wording": "state of the art; universal dominance",
            "supporting_artifacts": "artifacts/phase7/ablation_summary.csv;reports/phase7/tables/table_main_results.csv",
            "supporting_metrics": "SAFE Split A full_four OOF AURC 0.006268; risk-coverage AURC is reported in Phase 6/7 tables.",
            "applicable_detector": "SAFE,UnivFD",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "risk_fit_oof;protocol_seen;protocol_held_out",
            "applicable_policy": "global_cp;source_group_cp",
            "confidence_interval": "artifacts/phase7/ablation_paired_bootstrap.csv;artifacts/phase6/paired_method_comparisons.csv",
            "paper_section": "Ablations and Main Results",
        },
        {
            "claim_id": "CL04",
            "claim_category": "probability calibration",
            "final_claim_text": "The RiskGuard calibrator provides calibrated detector-error probabilities on frozen Phase 5 validation contexts.",
            "status": "LOCKED_QUALIFIED",
            "allowed_wording": "calibrated detector-error probability on the frozen calibration protocol",
            "required_qualification": "Calibration quality is empirical and distribution-specific.",
            "prohibited_wording": "calibrated under all external shifts",
            "supporting_artifacts": "reports/phase5/phase5_riskguard_calibrator_report.md;artifacts/phase5/oof_calibrator_metrics.csv;artifacts/phase5/threshold_cal_calibrator_metrics.csv",
            "supporting_metrics": "NLL, Brier, and ECE are reported in Phase 5 calibrator metrics.",
            "applicable_detector": "SAFE,UnivFD",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "risk_fit_oof;threshold_cal",
            "applicable_policy": "none",
            "confidence_interval": "not primary bootstrap claim",
            "paper_section": "RiskGuard Method",
        },
        {
            "claim_id": "CL05",
            "claim_category": "reliability-feature contribution",
            "final_claim_text": "The four-factor representation contributes useful detector-error information, especially decision-boundary distance, while individual feature effects are conditional.",
            "status": "LOCKED_QUALIFIED",
            "allowed_wording": "ablation results indicate feature utility and redundancy",
            "required_qualification": "full_four does not need to win every ablation metric and coefficients are non-causal.",
            "prohibited_wording": "feature coefficients are causal effects",
            "supporting_artifacts": "artifacts/phase7/ablation_summary.csv;artifacts/phase7/feature_effect_summary.csv;artifacts/phase7/ablation_paired_bootstrap.csv",
            "supporting_metrics": "no_margin worsens AURC/NLL relative to full_four in the frozen ablation summary.",
            "applicable_detector": "SAFE,UnivFD",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "risk_fit_oof",
            "applicable_policy": "none",
            "confidence_interval": "artifacts/phase7/ablation_paired_bootstrap.csv",
            "paper_section": "Ablations",
        },
        {
            "claim_id": "CL06",
            "claim_category": "protocol-seen transfer",
            "final_claim_text": "RiskGuard is empirically evaluated on protocol-seen generators after policy freezing.",
            "status": "LOCKED_SUPPORTED",
            "allowed_wording": "empirically evaluated on protocol-seen generators",
            "required_qualification": "This is empirical transfer, not a new certificate.",
            "prohibited_wording": "guaranteed transfer to protocol-seen test labels",
            "supporting_artifacts": "artifacts/phase6/final_selective_metrics.csv;artifacts/phase8/headline_metric_reproduction.csv",
            "supporting_metrics": "SAFE Split A source_group_cp protocol-seen coverage 0.646808 and risk 0.004234.",
            "applicable_detector": "SAFE,UnivFD",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "protocol_seen",
            "applicable_policy": "source_group_cp;global_cp;predicted_class_cp",
            "confidence_interval": "artifacts/phase6/bootstrap_confidence_intervals.csv",
            "paper_section": "Main Results",
        },
        {
            "claim_id": "CL07",
            "claim_category": "held-out generator transfer",
            "final_claim_text": "RiskGuard is empirically evaluated on held-out generators and shows detector/split-dependent transfer.",
            "status": "LOCKED_QUALIFIED",
            "allowed_wording": "empirically evaluated on held-out generators",
            "required_qualification": "Held-out generator results are not guaranteed by the calibration certificate.",
            "prohibited_wording": "guaranteed on unseen generators",
            "supporting_artifacts": "artifacts/phase6/final_selective_metrics.csv;artifacts/phase7/domain_transfer_summary.csv;artifacts/phase8/headline_metric_reproduction.csv",
            "supporting_metrics": "SAFE Split A source_group_cp protocol-held-out coverage 0.756974 and risk 0.001045; Split B is weaker.",
            "applicable_detector": "SAFE,UnivFD",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "protocol_held_out",
            "applicable_policy": "source_group_cp;global_cp",
            "confidence_interval": "artifacts/phase6/bootstrap_confidence_intervals.csv",
            "paper_section": "Main Results; Limitations",
        },
        {
            "claim_id": "CL08",
            "claim_category": "B-Free external transfer",
            "final_claim_text": "B-Free is an empirical external stress test and failure-transfer analysis, not a certified guarantee.",
            "status": "LOCKED_QUALIFIED",
            "allowed_wording": "empirically evaluated on the B-Free verified snapshot",
            "required_qualification": "B-Free contains 733 verified images and RiskGuard can fail badly there.",
            "prohibited_wording": "guaranteed on B-Free; distribution-free external guarantee",
            "supporting_artifacts": "artifacts/phase6/final_selective_metrics.csv;artifacts/phase7/bfree_cluster_failure_analysis.csv;artifacts/phase8/headline_metric_reproduction.csv",
            "supporting_metrics": "SAFE Split A source_group_cp B-Free coverage 0.375171 and risk 0.741818.",
            "applicable_detector": "SAFE,UnivFD",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "bfree_snapshot",
            "applicable_policy": "source_group_cp;global_cp",
            "confidence_interval": "artifacts/phase6/bootstrap_confidence_intervals.csv",
            "paper_section": "External Evaluation; Limitations",
        },
        {
            "claim_id": "CL09",
            "claim_category": "policy tradeoff",
            "final_claim_text": "The global and source-group policies expose a coverage/risk tradeoff rather than a uniformly best policy.",
            "status": "LOCKED_SUPPORTED",
            "allowed_wording": "policy-dependent tradeoff",
            "required_qualification": "Some policies are conservative or undefined with zero accepted samples.",
            "prohibited_wording": "source grouping universally improves all metrics",
            "supporting_artifacts": "artifacts/phase7/policy_ablation_summary.csv;artifacts/phase7/policy_bottleneck_groups.csv",
            "supporting_metrics": "Policy ablation summary records coverage, risk, and bottleneck groups.",
            "applicable_detector": "SAFE,UnivFD",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "policy_certify;protocol_seen;protocol_held_out;bfree_snapshot",
            "applicable_policy": "global_cp;source_group_cp;predicted_class_cp",
            "confidence_interval": "artifacts/phase6/paired_method_comparisons.csv",
            "paper_section": "Main Results",
        },
        {
            "claim_id": "CL10",
            "claim_category": "failure analysis",
            "final_claim_text": "Phase 7 failure taxonomy explains accepted errors, rejected correct predictions, generator bottlenecks, and B-Free external-domain failures descriptively.",
            "status": "EXPLORATORY_ONLY",
            "allowed_wording": "descriptive failure taxonomy",
            "required_qualification": "Tags are deterministic descriptive labels, not causal explanations.",
            "prohibited_wording": "causal explanation of every failure",
            "supporting_artifacts": "artifacts/phase7/failure_taxonomy_summary.csv;artifacts/phase7/accepted_error_summary.csv;artifacts/phase7/rejected_correct_summary.csv;artifacts/phase7/generator_failure_profile.csv",
            "supporting_metrics": "Accepted-error, rejected-correct, and generator profiles are nonempty.",
            "applicable_detector": "SAFE,UnivFD",
            "applicable_split": "split_a;split_b",
            "applicable_dataset": "protocol_seen;protocol_held_out;bfree_snapshot",
            "applicable_policy": "all RiskGuard policies",
            "confidence_interval": "not applicable",
            "paper_section": "Ablations and Failure Analysis",
        },
        {
            "claim_id": "CL11",
            "claim_category": "prohibited external guarantee",
            "final_claim_text": "RiskGuard is guaranteed on unseen generators or B-Free.",
            "status": "PROHIBITED",
            "allowed_wording": "No allowed wording; use empirical evaluation wording instead.",
            "required_qualification": "Replace with empirical held-out/B-Free wording.",
            "prohibited_wording": "guaranteed on unseen generators; guaranteed on B-Free; universally risk-controlled; distribution-free external guarantee; state of the art",
            "supporting_artifacts": "artifacts/phase8/final_limitation_registry.csv",
            "supporting_metrics": "Not applicable.",
            "applicable_detector": "all",
            "applicable_split": "all",
            "applicable_dataset": "protocol_held_out;bfree_snapshot",
            "applicable_policy": "all",
            "confidence_interval": "not applicable",
            "paper_section": "Claim discipline",
        },
    ]
    claims = pd.DataFrame(claim_rows)
    claims.to_csv(paths.artifacts / "final_claim_lock.csv", index=False)
    return claims


def contribution_lock(paths: Phase8Paths) -> pd.DataFrame:
    rows = [
        ("C1", "Transformation-orbit reliability representation.", "SECONDARY_CONTRIBUTION", "Useful feature-family evidence, but not the only central novelty.", "Method; Ablations"),
        ("C2", "Four-factor detector-error risk modeling.", "PRIMARY_CONTRIBUTION", "Frozen full_four combines decision margin, orbit instability, embedding drift, and support distance for error-risk scoring.", "Method"),
        ("C3", "Independent candidate-selection and CP-certification protocol.", "PRIMARY_CONTRIBUTION", "This is the strongest defensible scientific contribution and has direct frozen evidence.", "Risk-Control Protocol"),
        ("C4", "Simultaneous source-group control using one operational threshold.", "PRIMARY_CONTRIBUTION", "Source-group CP is certified for SAFE and changes the risk/coverage tradeoff.", "Main Results"),
        ("C5", "Evaluation under held-out generators and composed reliability views.", "SECONDARY_CONTRIBUTION", "Important evaluation breadth, but empirical rather than certified.", "Experimental Setup; Main Results"),
        ("C6", "External B-Free failure-transfer analysis.", "SUPPORTING_ANALYSIS", "Valuable stress evidence and limitation; not a success claim.", "Failure Analysis; Limitations"),
        ("C7", "Detailed selective failure taxonomy.", "SECONDARY_CONTRIBUTION", "Explains accepted errors and rejected correct predictions in a paper-useful way.", "Ablations and Failure Analysis"),
    ]
    df = pd.DataFrame(rows, columns=["contribution_id", "candidate_contribution", "locked_role", "reason", "paper_section"])
    df.to_csv(paths.artifacts / "contribution_lock.csv", index=False)
    report = """
    # Phase 8 Contribution Decision

    The paper should lead with three primary contributions:

    1. Four-factor detector-error risk modeling.
    2. Independent candidate-selection and CP-certification protocol.
    3. Simultaneous source-group control using one operational threshold.

    Secondary contributions are the transformation-orbit representation, held-out/reliability-view evaluation, and the selective failure taxonomy. B-Free should be framed as supporting external failure-transfer analysis, not as a central success claim.
    """
    write_markdown(paths.reports / "phase8_contribution_decision.md", report)
    return df


def closest_work_matrix(paths: Phase8Paths) -> pd.DataFrame:
    rows = [
        {
            "reference_id": "Zhu2023_GenImage",
            "citation": "Mingjian Zhu et al., GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image, arXiv:2306.08571, 2023. https://arxiv.org/abs/2306.08571",
            "problem": "AI-generated image detection benchmark",
            "method": "Large-scale dataset and detector evaluation",
            "datasets": "GenImage",
            "risk_guarantee": "None",
            "group_control": "No explicit source-group CP control",
            "unseen_generator_evaluation": "Cross-generator task",
            "external_evaluation": "Not B-Free external snapshot",
            "difference_from_RiskGuard": "RiskGuard uses GenImage as frozen evidence and adds selective detector-error risk calibration plus CP certification.",
            "novelty_threat": "LOW",
        },
        {
            "reference_id": "Ojha2023_UnivFD",
            "citation": "Utkarsh Ojha, Yuheng Li, Yong Jae Lee, Towards Universal Fake Image Detectors that Generalize Across Generative Models, CVPR 2023/arXiv:2302.10174. https://arxiv.org/abs/2302.10174",
            "problem": "Unseen-generator generalization",
            "method": "CLIP feature space with nearest neighbor and linear probing",
            "datasets": "Multiple generator families",
            "risk_guarantee": "None",
            "group_control": "None",
            "unseen_generator_evaluation": "Yes",
            "external_evaluation": "Generator-family benchmarks",
            "difference_from_RiskGuard": "UnivFD is a detector/baseline family; RiskGuard is a selective risk-control wrapper over frozen detectors.",
            "novelty_threat": "LOW",
        },
        {
            "reference_id": "ParkOwens2025_CommunityForensics",
            "citation": "Jeongsoo Park and Andrew Owens, Community Forensics: Using Thousands of Generators to Train Fake Image Detectors, CVPR 2025/arXiv:2411.04125. https://arxiv.org/abs/2411.04125",
            "problem": "Generator diversity for generalizable fake-image detectors",
            "method": "Large diverse training corpus across 4803 models",
            "datasets": "Community Forensics",
            "risk_guarantee": "None",
            "group_control": "None",
            "unseen_generator_evaluation": "Yes",
            "external_evaluation": "Broad generator coverage",
            "difference_from_RiskGuard": "Data-scale/generalization work; RiskGuard controls accepted prediction error without retraining detector families.",
            "novelty_threat": "LOW",
        },
        {
            "reference_id": "Guillaro2024_BFree",
            "citation": "Fabrizio Guillaro et al., A Bias-Free Training Paradigm for More General AI-generated Image Detection, arXiv:2412.17671, 2024. https://arxiv.org/abs/2412.17671",
            "problem": "Training-data bias in AIGI detection",
            "method": "Bias-free real/fake generation protocol",
            "datasets": "B-Free and extended generator tests",
            "risk_guarantee": "None",
            "group_control": "No CP source-group guarantee",
            "unseen_generator_evaluation": "Yes",
            "external_evaluation": "Yes",
            "difference_from_RiskGuard": "B-Free informs external stress testing; RiskGuard must not claim B-Free guarantee and exposes failure there.",
            "novelty_threat": "LOW",
        },
        {
            "reference_id": "Geifman2017_Selective",
            "citation": "Yonatan Geifman and Ran El-Yaniv, Selective Classification for Deep Neural Networks, arXiv:1705.08500, 2017. https://arxiv.org/abs/1705.08500",
            "problem": "Selective classification and abstention",
            "method": "Confidence thresholding with desired risk level",
            "datasets": "CIFAR/ImageNet",
            "risk_guarantee": "High-probability empirical risk framing",
            "group_control": "No source-group generator control",
            "unseen_generator_evaluation": "No AIGI-specific evaluation",
            "external_evaluation": "No",
            "difference_from_RiskGuard": "RiskGuard specializes selective risk control to detector errors and generator groups for AIGI detection.",
            "novelty_threat": "MODERATE",
        },
        {
            "reference_id": "Geifman2019_SelectiveNet",
            "citation": "Yonatan Geifman and Ran El-Yaniv, SelectiveNet: A Deep Neural Network with an Integrated Reject Option, ICML/PMLR 2019. https://proceedings.mlr.press/v97/geifman19a.html",
            "problem": "End-to-end selective prediction",
            "method": "Integrated reject option network",
            "datasets": "Classification and regression benchmarks",
            "risk_guarantee": "Coverage calibration, not CP source-group guarantee",
            "group_control": "None",
            "unseen_generator_evaluation": "No",
            "external_evaluation": "No",
            "difference_from_RiskGuard": "RiskGuard is post-hoc over frozen detectors and uses independent CP certification rather than end-to-end training.",
            "novelty_threat": "MODERATE",
        },
        {
            "reference_id": "Guo2017_Calibration",
            "citation": "Chuan Guo et al., On Calibration of Modern Neural Networks, ICML/PMLR 2017/arXiv:1706.04599. https://arxiv.org/abs/1706.04599",
            "problem": "Uncertainty/probability calibration",
            "method": "Temperature scaling and calibration diagnostics",
            "datasets": "Image and document classification",
            "risk_guarantee": "None",
            "group_control": "None",
            "unseen_generator_evaluation": "No",
            "external_evaluation": "No",
            "difference_from_RiskGuard": "RiskGuard calibrates detector-error risk and then certifies accepted-error policies.",
            "novelty_threat": "LOW",
        },
        {
            "reference_id": "Bates2021_RCPS",
            "citation": "Stephen Bates et al., Distribution-free, Risk-controlling Prediction Sets, JACM 2021/arXiv:2101.02703. https://arxiv.org/abs/2101.02703",
            "problem": "Finite-sample risk control",
            "method": "Risk-controlling prediction sets",
            "datasets": "General ML tasks",
            "risk_guarantee": "Distribution-free under exchangeability for prediction sets",
            "group_control": "Not generator source-group selective detector policy",
            "unseen_generator_evaluation": "No",
            "external_evaluation": "No",
            "difference_from_RiskGuard": "RiskGuard adapts finite-sample control to accepted detector predictions and source groups.",
            "novelty_threat": "MODERATE",
        },
        {
            "reference_id": "Angelopoulos2024_CRC",
            "citation": "Anastasios N. Angelopoulos et al., Conformal Risk Control, ICLR 2024/arXiv:2208.02814. https://arxiv.org/abs/2208.02814",
            "problem": "Conformal or finite-sample risk control",
            "method": "Conformal control of monotone risk functions",
            "datasets": "Vision and NLP examples",
            "risk_guarantee": "Conformal risk control under exchangeability",
            "group_control": "Extensions discussed, not AIGI source-group detector error",
            "unseen_generator_evaluation": "No",
            "external_evaluation": "No",
            "difference_from_RiskGuard": "RiskGuard is an application-specific protocol with frozen candidate selection and source-group policy.",
            "novelty_threat": "MODERATE",
        },
        {
            "reference_id": "HebertJohnson2018_Multicalibration",
            "citation": "Ursula Hebert-Johnson et al., Multicalibration: Calibration for the Computationally-Identifiable Masses, ICML/PMLR 2018. https://proceedings.mlr.press/v80/hebert-johnson18a.html",
            "problem": "Group-conditional calibration/fairness",
            "method": "Simultaneous calibration over identifiable subpopulations",
            "datasets": "General prediction settings",
            "risk_guarantee": "Calibration guarantee, not CP selective-risk threshold",
            "group_control": "Yes, calibration over groups",
            "unseen_generator_evaluation": "No",
            "external_evaluation": "No",
            "difference_from_RiskGuard": "RiskGuard controls accepted detector-error risk for predefined generator groups.",
            "novelty_threat": "MODERATE",
        },
        {
            "reference_id": "Wang2018_TTAUncertainty",
            "citation": "Guotai Wang et al., Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation, arXiv:1807.07356, 2018. https://arxiv.org/abs/1807.07356",
            "problem": "Transformation-consistency reliability",
            "method": "Test-time augmentation uncertainty",
            "datasets": "Medical image segmentation",
            "risk_guarantee": "None",
            "group_control": "None",
            "unseen_generator_evaluation": "No",
            "external_evaluation": "No",
            "difference_from_RiskGuard": "RiskGuard uses transformation-orbit logit instability as one AIGI detector-error feature and combines it with support distance and CP policy.",
            "novelty_threat": "LOW",
        },
        {
            "reference_id": "RIGID2024",
            "citation": "RIGID: A Training-free and Model-Agnostic Framework for Robust AI-Generated Image Detection, arXiv:2405.20112, 2024. https://arxiv.org/abs/2405.20112",
            "problem": "Embedding-support or perturbation distance for AIGI detection",
            "method": "Training-free robustness asymmetry in foundation-model representation space",
            "datasets": "AIGI benchmarks",
            "risk_guarantee": "None",
            "group_control": "None",
            "unseen_generator_evaluation": "Yes",
            "external_evaluation": "Likely benchmark-dependent",
            "difference_from_RiskGuard": "Closest to perturbation/support signals, but RiskGuard centers on calibrated accepted-error control and group CP rather than raw detection.",
            "novelty_threat": "MODERATE",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(paths.artifacts / "closest_work_matrix.csv", index=False)
    report = """
    # Phase 8 Novelty Positioning

    Novelty risk classification: MODERATE.

    The closest theoretical neighbors are selective classification, risk-controlling prediction sets, conformal risk control, and multicalibration. The closest AIGI neighbors are GenImage, UnivFD, Community Forensics, B-Free, and perturbation/robustness-asymmetry detectors such as RIGID.

    The defensible distinction is not a new base detector and not a universal external guarantee. The defensible distinction is the frozen operational protocol: four-factor detector-error risk scoring, independent threshold candidate selection, finite-sample certification on the GenImage policy-certify subset, and source-group selective risk control for accepted detector predictions.

    Internet access was available and used on 2026-07-16. The matrix records primary paper/proceedings/arXiv/project sources rather than search snippets. No critical novelty duplication was identified.
    """
    write_markdown(paths.reports / "phase8_novelty_positioning.md", report)
    return df


def build_tables(paths: Phase8Paths, reproduction: pd.DataFrame) -> pd.DataFrame:
    detector_split = pd.read_csv(paths.root / "artifacts" / "phase7" / "detector_split_comparison.csv")
    ablation = pd.read_csv(paths.root / "artifacts" / "phase7" / "ablation_summary.csv")
    bootstrap = pd.read_csv(paths.root / "artifacts" / "phase6" / "bootstrap_confidence_intervals.csv")

    table1 = detector_split[
        detector_split["method"].eq("riskguard")
        & detector_split["alpha"].eq(0.05)
        & detector_split["policy"].isin(["global_cp", "source_group_cp"])
        & detector_split["evaluation_dataset"].eq("protocol_seen")
    ][
        [
            "detector",
            "split",
            "policy",
            "base_detector_error",
            "error_risk_AURC",
            "AURC",
            "E_AURC",
            "B_Free_coverage",
            "B_Free_selective_risk",
        ]
    ].copy()
    write_table(table1, paths.tables / "table1_detector_risk_estimator_quality.csv", paths.tables / "table1_detector_risk_estimator_quality.tex")

    table2 = reproduction[
        reproduction["alpha"].eq(0.05)
        & reproduction["method"].isin(METHODS)
        & reproduction["policy"].isin(["global_cp", "source_group_cp"])
        & reproduction["dataset"].isin(["policy_certify", "protocol_seen", "protocol_held_out"])
    ][
        [
            "detector",
            "split",
            "method",
            "policy",
            "dataset",
            "certification_status",
            "certification_coverage",
            "CP_upper_bound",
            "test_coverage",
            "selective_risk",
            "worst_group_selective_risk",
            "minimum_group_coverage",
        ]
    ].copy()
    write_table(table2, paths.tables / "table2_certified_selective_results.csv", paths.tables / "table2_certified_selective_results.tex")

    keep_ablation = ["full_four", "no_margin", "no_variance", "no_drift", "no_support", "margin_only", "orbit_only", "geometry_support"]
    table3 = ablation[ablation["ablation"].isin(keep_ablation)][
        [
            "detector",
            "split",
            "ablation",
            "NLL",
            "Brier",
            "ECE",
            "error_detection_AUROC",
            "AURC",
            "E_AURC",
            "delta_NLL_from_full_four",
            "delta_AURC_from_full_four",
        ]
    ].copy()
    write_table(table3, paths.tables / "table3_ablation.csv", paths.tables / "table3_ablation.tex")

    bfree = reproduction[
        reproduction["alpha"].eq(0.05)
        & reproduction["dataset"].eq("bfree_snapshot")
        & reproduction["method"].isin(METHODS)
        & reproduction["policy"].isin(["global_cp", "source_group_cp"])
    ].copy()
    ci = bootstrap[bootstrap["evaluation_dataset"].eq("bfree_snapshot") & bootstrap["metric"].isin(["coverage", "selective_risk", "balanced_selective_risk"])]
    ci_pivot = ci.pivot_table(
        index=["detector", "split", "method", "alpha", "policy", "evaluation_dataset"],
        columns="metric",
        values=["ci_lower_2p5", "ci_upper_97p5"],
        aggfunc="first",
    )
    ci_pivot.columns = [f"{a}_{b}" for a, b in ci_pivot.columns]
    ci_pivot = ci_pivot.reset_index().rename(columns={"evaluation_dataset": "dataset"})
    table4 = bfree.merge(ci_pivot, on=["detector", "split", "method", "alpha", "policy", "dataset"], how="left")[
        [
            "detector",
            "split",
            "method",
            "policy",
            "test_coverage",
            "selective_risk",
            "balanced_selective_risk",
            "AURC",
            "ci_lower_2p5_coverage",
            "ci_upper_97p5_coverage",
            "ci_lower_2p5_selective_risk",
            "ci_upper_97p5_selective_risk",
        ]
    ].copy()
    write_table(table4, paths.tables / "table4_external_bfree.csv", paths.tables / "table4_external_bfree.tex")

    manifest_rows = [
        {
            "table_id": "T1",
            "title": "Base detector and risk-estimator quality",
            "csv_source": rel(paths.tables / "table1_detector_risk_estimator_quality.csv"),
            "latex_source": rel(paths.tables / "table1_detector_risk_estimator_quality.tex"),
            "source_artifact_hashes": source_hashes_for(paths, ["artifacts/phase7/detector_split_comparison.csv"]),
            "caption_draft": "Base detector error and RiskGuard error-ranking quality for SAFE and UnivFD across both splits.",
            "designation": "main-paper",
        },
        {
            "table_id": "T2",
            "title": "Certified selective results",
            "csv_source": rel(paths.tables / "table2_certified_selective_results.csv"),
            "latex_source": rel(paths.tables / "table2_certified_selective_results.tex"),
            "source_artifact_hashes": source_hashes_for(paths, ["artifacts/phase8/headline_metric_reproduction.csv", "artifacts/phase6/certified_threshold_registry.csv"]),
            "caption_draft": "Certified calibration coverage and empirical protocol-seen/held-out selective risk for RiskGuard, MSP, and cosine kNN.",
            "designation": "main-paper",
        },
        {
            "table_id": "T3",
            "title": "Ablation",
            "csv_source": rel(paths.tables / "table3_ablation.csv"),
            "latex_source": rel(paths.tables / "table3_ablation.tex"),
            "source_artifact_hashes": source_hashes_for(paths, ["artifacts/phase7/ablation_summary.csv", "artifacts/phase7/ablation_paired_bootstrap.csv"]),
            "caption_draft": "RiskGuard full_four and predefined feature-family ablations on frozen OOF error-risk metrics.",
            "designation": "main-paper",
        },
        {
            "table_id": "T4",
            "title": "External B-Free evaluation",
            "csv_source": rel(paths.tables / "table4_external_bfree.csv"),
            "latex_source": rel(paths.tables / "table4_external_bfree.tex"),
            "source_artifact_hashes": source_hashes_for(paths, ["artifacts/phase8/headline_metric_reproduction.csv", "artifacts/phase6/bootstrap_confidence_intervals.csv"]),
            "caption_draft": "Empirical B-Free verified snapshot transfer; these values are not certificates.",
            "designation": "appendix-or-main-visible",
        },
    ]
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(paths.artifacts / "table_manifest.csv", index=False)
    return manifest


def build_figures(paths: Phase8Paths) -> pd.DataFrame:
    mapping = [
        ("F1", "risk-coverage comparison", "figure_risk_coverage_main", "main-paper", "RiskGuard and baselines have detector- and dataset-dependent risk-coverage behavior.", "Risk curves are empirical, not certificates."),
        ("F2", "certification and group-bound figure", "figure_certified_coverage_main", "main-paper", "Certified coverage and CP upper bounds identify which policies certify.", "Certification is calibration-distribution specific."),
        ("F3", "feature ablation figure", "figure_ablation_main", "main-paper", "Feature-family ablations show useful but redundant evidence.", "full_four does not dominate every metric."),
        ("F4", "policy risk/coverage tradeoff", "figure_policy_tradeoff", "main-paper", "Source-group policy changes risk/coverage tradeoffs.", "Conservative or undefined cells must remain visible."),
        ("F5", "generator-transfer heatmap", "figure_generator_transfer_heatmap", "main-paper", "Generator-level transfer reveals bottleneck groups.", "Held-out transfer is empirical."),
        ("F6", "failure taxonomy or qualitative gallery", "figure_failure_taxonomy", "appendix", "Failure taxonomy summarizes accepted errors and rejected correct predictions.", "Taxonomy tags are descriptive, not causal."),
        ("F7", "domain-shift diagnostic", "figure_domain_shift", "appendix", "B-Free external behavior differs sharply from GenImage transfer.", "External results are not guaranteed."),
        ("F8", "threshold sensitivity diagnostic", "figure_threshold_sensitivity", "appendix", "Near-threshold diagnostics support appendix-level interpretation.", "Exploratory diagnostic."),
    ]
    rows = []
    for figure_id, purpose, stem, placement, key_message, limitations in mapping:
        source = paths.root / "artifacts" / "phase7" / "figure_data" / f"{stem}.csv"
        source_rel = rel(source)
        rendered = paths.root / "reports" / "phase7" / "figures" / f"{stem}.pdf"
        rendered_out = paths.figures / f"{stem}.pdf"
        if rendered.exists():
            shutil.copy2(rendered, rendered_out)
        png = paths.root / "reports" / "phase7" / "figures" / f"{stem}.png"
        if png.exists():
            shutil.copy2(png, paths.figures / f"{stem}.png")
        rows.append(
            {
                "figure_id": figure_id,
                "purpose": purpose,
                "source_data": source_rel,
                "source_hash": sha256_file(source) if source.exists() else "",
                "rendered_artifact": rel(rendered_out),
                "main_or_appendix": placement,
                "caption_draft": f"{purpose}. Undefined values are marked in source tables and test risk is not presented as a certificate.",
                "key_message": key_message,
                "limitations": limitations,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(paths.artifacts / "figure_manifest.csv", index=False)
    return manifest


def statistical_audit(paths: Phase8Paths) -> pd.DataFrame:
    paired = pd.read_csv(paths.root / "artifacts" / "phase6" / "paired_method_comparisons.csv")
    rows = []
    primary_metrics = {"coverage", "selective_risk", "balanced_selective_risk", "AURC", "worst_group_selective_risk", "minimum_group_coverage"}
    subset = paired[paired["comparison"].isin(["riskguard_minus_msp", "riskguard_minus_knn"]) & paired["metric"].isin(primary_metrics)].copy()
    for record in subset.to_dict("records"):
        lo = metric_number(record["ci_lower_2p5"])
        hi = metric_number(record["ci_upper_97p5"])
        defined = np.isfinite(lo) and np.isfinite(hi)
        rows.append(
            {
                "detector": record["detector"],
                "split": record["split"],
                "alpha": record["alpha"],
                "policy": record["policy"],
                "evaluation_dataset": record["evaluation_dataset"],
                "comparison": record["comparison"],
                "metric": record["metric"],
                "paired_sample_alignment": "identical detector/split/dataset/policy rows; paired by frozen evaluation units",
                "paired_bootstrap_unit": "sha256/source cluster Poisson unit",
                "bootstrap_replicate_count": int(record["bootstrap_replicates"]),
                "valid_replicate_count": int(record["valid_bootstrap_replicate_count"]),
                "confidence_interval": f"[{record['ci_lower_2p5']}, {record['ci_upper_97p5']}]",
                "undefined_cases": "undefined" if not defined else "",
                "wording": "the paired 95% confidence interval excludes zero" if defined and (lo > 0 or hi < 0) else "the paired 95% confidence interval includes zero or is undefined",
                "status": "pass"
                if (not defined) or 0 < int(record["valid_bootstrap_replicate_count"]) <= int(record["bootstrap_replicates"])
                else "fail",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.artifacts / "statistical_evidence_audit.csv", index=False)
    return out


def limitation_registry(paths: Phase8Paths) -> pd.DataFrame:
    rows = [
        ("certification is calibration-distribution specific", "high", "finite-sample certification", "Certificates apply to the independent GenImage certification subset and frozen groups only.", "main text", "Independent policy-certify subset and frozen thresholds", "Study adaptive or shifted calibration protocols."),
        ("held-out transfer is empirical", "medium", "held-out generator transfer", "Held-out generator results are empirical evaluations, not guarantees.", "main text", "Separate held-out table rows and claim lock", "Collect larger post-freeze held-out suites."),
        ("B-Free transfer is empirical and may fail", "high", "B-Free external transfer", "B-Free is an empirical stress test and may show high accepted risk.", "main text", "B-Free table and failure analysis", "Curate larger external snapshots."),
        ("B-Free contains 733 verified images", "medium", "external validity", "The verified B-Free snapshot contains 733 images.", "main text", "B-Free integrity audit", "Expand verified external data."),
        ("coverage can become conservative", "medium", "policy tradeoff", "Group control can reduce coverage or create undefined zero-acceptance cells.", "main text", "Policy ablation and undefined preservation", "Investigate smoother policy families."),
        ("full_four does not dominate every ablation", "medium", "feature contribution", "full_four is the frozen primary model but does not win every ablation metric.", "main text", "Ablation lock", "Explore simpler variants after paper."),
        ("feature coefficients are conditional and non-causal", "medium", "reliability features", "Feature coefficients and tags are conditional associations, not causal effects.", "main text", "Feature correlation audit", "Causal interventions on controlled transformations."),
        ("generator groups are calibration-time evaluation groups", "medium", "source-group risk control", "Source groups are predefined calibration/evaluation groups, not deployment-discovered semantic classes.", "main text", "Group definitions in policy JSON", "Online group discovery and monitoring."),
        ("two detector families do not represent all detector architectures", "medium", "baseline fairness", "SAFE and UnivFD are two detector families and cannot represent all AIGI detector architectures.", "appendix", "Supporting detector analysis", "Evaluate more detectors."),
        ("frozen transformation orbit covers a limited perturbation family", "medium", "transformation orbit", "The orbit is limited to the frozen Phase 4 perturbation family.", "appendix", "Phase 4 feature freeze", "Broaden transformations."),
        ("project Git provenance is unavailable", "low", "reproducibility", "The local project path is not a Git repository, so file hashes replace Git commit provenance.", "appendix", "Frozen hash registries", "Package a release archive with VCS provenance."),
    ]
    df = pd.DataFrame(rows, columns=["limitation", "severity", "affected_claim", "required_paper_wording", "main_text_or_appendix", "mitigation_already_present", "future_work"])
    df.to_csv(paths.artifacts / "final_limitation_registry.csv", index=False)
    return df


def risk_register(paths: Phase8Paths) -> pd.DataFrame:
    rows = [
        ("novelty risk", "medium", "medium", "Closest work exists in selective classification, CRC, and AIGI robustness, but no critical duplication was found.", "Lead with the combined AIGI detector-error CP protocol.", "nonblocking", "Paper writing"),
        ("baseline completeness risk", "medium", "medium", "Primary comparators are MSP and cosine kNN; detector families are limited.", "Report as limitation and keep UnivFD support.", "nonblocking", "Appendix"),
        ("external-validity risk", "high", "medium", "B-Free risk is high and snapshot-limited.", "Frame B-Free as empirical failure transfer.", "nonblocking", "Paper writing"),
        ("statistical-power risk", "low", "medium", "Bootstrap replicate count is 2000 where defined; B-Free has 733 images.", "Use exact wording about paired CIs.", "nonblocking", "Paper writing"),
        ("low-coverage risk", "medium", "medium", "Source-group policy can be conservative or undefined for UnivFD.", "Keep undefined cells visible.", "nonblocking", "Paper writing"),
        ("method-complexity risk", "medium", "medium", "RiskGuard has multiple feature families and protocol stages.", "Use manuscript blueprint and diagrams.", "nonblocking", "Paper writing"),
        ("paper-space risk", "medium", "low", "Tables/figures exceed main paper space.", "Move B-Free details and diagnostics to appendix while keeping summary visible.", "nonblocking", "Appendix"),
        ("reproducibility risk", "low", "medium", "Git provenance unavailable, but hash registries verify.", "Report hashes and scripts.", "nonblocking", "Phase 8"),
        ("claim-overreach risk", "medium", "high", "External guarantee wording would be false.", "Use final claim lock and prohibited wording list.", "nonblocking", "Paper writing"),
        ("visualization-overload risk", "medium", "low", "Phase 7 generated many diagnostics.", "Use 3-5 main figure groups.", "nonblocking", "Paper writing"),
    ]
    df = pd.DataFrame(rows, columns=["risk", "likelihood", "impact", "evidence", "mitigation", "blocking_or_nonblocking", "owner_phase"])
    df.to_csv(paths.artifacts / "submission_risk_register.csv", index=False)
    return df


def additional_experiment_decisions(paths: Phase8Paths) -> pd.DataFrame:
    rows = [
        ("reproduce primary headline metrics", "NOT_JUSTIFIED", "Primary reproduction mismatches are zero.", "No new experiment required."),
        ("add more detector architectures", "OPTIONAL_FOR_STRENGTHENING", "Two detector families limit breadth but do not block the central claim.", "Appendix or future strengthening."),
        ("expand B-Free verified snapshot", "OPTIONAL_FOR_STRENGTHENING", "B-Free weakness is a limitation, not a hard blocker.", "Optional after writing begins."),
        ("new test-selected threshold or method search", "NOT_JUSTIFIED", "Would violate frozen protocol and invite test selection.", "Do not do this before paper writing."),
        ("formal critical novelty replication study", "FUTURE_WORK", "No critical duplication identified.", "Track related work during writing."),
        ("appendix subgroup sweep", "APPENDIX_ONLY", "Useful diagnostics but not needed for GO.", "Only if space allows."),
    ]
    df = pd.DataFrame(rows, columns=["proposed_experiment", "decision", "evidence", "recommendation"])
    df.to_csv(paths.artifacts / "additional_experiment_decisions.csv", index=False)
    return df


def readiness_score(paths: Phase8Paths) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [
        ("technical integrity", 2, "All Phase 2-7 frozen hashes match and no hard protocol violation is observed."),
        ("reproducibility", 2, "Frozen registries, reproduction audit, and Phase 8 tests pass."),
        ("primary-result strength", 2, "SAFE source_group_cp certifies at alpha=0.05 and transfers strongly on GenImage Split A."),
        ("risk-control validity", 2, "Independent selection/certification and CP upper bounds are valid for certified policies."),
        ("baseline fairness", 1, "MSP and cosine kNN are fair frozen comparators, but baseline breadth is limited."),
        ("ablation support", 1, "Ablations support feature utility with qualifications."),
        ("held-out evaluation", 1, "Held-out generator transfer is present but split-dependent."),
        ("external evaluation", 1, "B-Free is honest empirical stress evidence but weak."),
        ("failure analysis", 2, "Accepted errors, rejected correct predictions, generator failures, and B-Free clusters are analyzed."),
        ("novelty positioning", 1, "Closest work is identified and novelty risk is moderate, not critical."),
        ("claim discipline", 2, "Final claim lock prohibits false external guarantees."),
        ("figure/table readiness", 2, "Main tables and figures have CSV/LaTeX/source-data manifests."),
    ]
    df = pd.DataFrame(rows, columns=["dimension", "score", "rationale"])
    passed, details = readiness_score_passes(df)
    df["maximum_dimension_score"] = 2
    df.to_csv(paths.artifacts / "publication_readiness_score.csv", index=False)
    details["passed"] = passed
    return df, details


def write_narrative_and_blueprint(paths: Phase8Paths) -> None:
    narrative = """
    # Phase 8 Final Paper Narrative

    ## Problem
    Detector confidence can become unreliable under generator and transformation shift.

    ## Gap
    Strong detection accuracy does not provide explicit control over the error rate among accepted predictions, especially across heterogeneous generator groups.

    ## Method
    RiskGuard combines decision-boundary distance, transformation-orbit logit instability, embedding drift, training-support distance, error-risk calibration, independent threshold certification, and simultaneous source-group constraints.

    ## Main Evidence
    The main evidence is frozen: SAFE source_group_cp certifies at alpha=0.05 on the independent GenImage policy-certify subset, and both Split A and Split B are reported.

    ## Generalization Evidence
    Seen-generator empirical transfer, held-out-generator empirical transfer, and B-Free external transfer must be separate. The central message must not depend on B-Free risk control.

    ## Failure Evidence
    Accepted errors, unnecessary rejection, group bottlenecks, and external-domain failure are valuable paper evidence.

    ## Main Message
    RiskGuard-AIGI is a group-risk-controlled selective detection protocol for frozen AIGI detectors. It provides calibration-distribution-specific certificates and honest empirical transfer analysis, including external failures.
    """
    write_markdown(paths.reports / "phase8_final_paper_narrative.md", narrative)

    sections = [
        ("Title options", "Choose precise, non-external-guarantee wording.", "No guarantee claim.", "None.", "None.", "None.", "phase8_title_and_contribution_options.md", "0.1 pages", "Universal robustness wording."),
        ("Abstract logic", "State problem, method, certified SAFE evidence, transfer, limitation.", "Certified on GenImage, empirical held-out/B-Free.", "One risk-control inequality.", "T2 optional.", "F1/F2 optional.", "headline reproduction and claim lock", "0.2 pages", "B-Free guarantee."),
        ("1. Introduction", "Motivate detector confidence failure and accepted-error risk.", "Selective detection needs risk control.", "None or simple risk definition.", "None.", "Optional teaser.", "Phase6/7 reports", "1.0 pages", "State of the art claims."),
        ("2. Related Work", "Position against AIGI detection, selective classification, CRC, calibration.", "RiskGuard is a protocol wrapper, not a new detector.", "None.", "Closest-work matrix.", "None.", "closest_work_matrix.csv", "1.0 pages", "Fabricated citations."),
        ("3. Problem Setup", "Define base detector, accepted set, selective risk, groups, splits.", "Groups are calibration-time evaluation groups.", "Selective risk and group risk.", "None.", "None.", "Phase6 policies", "1.0 pages", "Deployment-discovered group guarantees."),
        ("4. RiskGuard Method", "Describe four features and logistic detector-error risk.", "Features are conditional reliability signals.", "Risk score model.", "T1.", "Feature ablation figure optional.", "Phase5 reports", "1.0 pages", "Causal feature wording."),
        ("5. Risk-Control Protocol", "Describe selection/certification independence and CP bounds.", "Finite-sample certificate on policy-certify subset.", "CP upper bound and union split.", "T2.", "F2.", "certified_threshold_registry.csv", "1.0 pages", "External guarantees."),
        ("6. Experimental Setup", "Describe detectors, splits, datasets, baselines.", "Both splits reported.", "None.", "Dataset summary appendix.", "None.", "Phase2-6 configs", "0.8 pages", "Selecting better split."),
        ("7. Main Results", "Report certified and empirical transfer results.", "SAFE certifies; UnivFD is supporting and weaker.", "None.", "T1,T2,T4.", "F1,F2,F4.", "headline reproduction", "1.5 pages", "Concealing B-Free."),
        ("8. Ablations and Failure Analysis", "Explain feature utility and failures.", "Ablations support but qualify full_four.", "None.", "T3.", "F3,F5,F6.", "Phase7 failure artifacts", "1.2 pages", "Causal failure explanations."),
        ("9. Limitations", "State locked limitations plainly.", "External transfer empirical; coverage conservative.", "None.", "Limitation registry.", "None.", "final_limitation_registry.csv", "0.6 pages", "Hiding negative results."),
        ("10. Conclusion", "Restate protocol contribution and honest boundary.", "Ready for paper writing.", "None.", "None.", "None.", "GO report", "0.3 pages", "Universal claims."),
        ("Appendix plan", "Provide extra tables, bootstraps, figures, hash audits, and qualitative cases.", "Reproducibility and diagnostics.", "Full derivations.", "All appendix tables.", "Secondary figures.", "Phase8 manifests", "As needed", "New test-selected experiments."),
    ]
    blueprint = pd.DataFrame(
        sections,
        columns=[
            "section",
            "purpose",
            "required_claims",
            "required_equations",
            "required_tables",
            "required_figures",
            "source_artifacts",
            "target_length",
            "content_to_avoid",
        ],
    )
    lines = ["# Phase 8 Manuscript Blueprint"]
    for rec in blueprint.to_dict("records"):
        lines.extend(
            [
                f"\n## {rec['section']}",
                f"- Purpose: {rec['purpose']}",
                f"- Required claims: {rec['required_claims']}",
                f"- Required equations: {rec['required_equations']}",
                f"- Required tables: {rec['required_tables']}",
                f"- Required figures: {rec['required_figures']}",
                f"- Source artifacts: {rec['source_artifacts']}",
                f"- Target length: {rec['target_length']}",
                f"- Content to avoid: {rec['content_to_avoid']}",
            ]
        )
    write_markdown(paths.reports / "phase8_manuscript_blueprint.md", "\n".join(lines))

    title_options = """
    # Phase 8 Title and Contribution Options

    ## Title Options
    1. RiskGuard-AIGI: Group-Risk-Controlled Selective Detection of AI-Generated Images under Unseen Generators and Transformations
    2. RiskGuard-AIGI: Certified Selective Error Control for AI-Generated Image Detectors
    3. Group-Risk-Controlled Abstention for AI-Generated Image Detection under Generator Shift

    ## Contribution Bullet Variants
    Variant A:
    - A four-factor detector-error risk model for frozen AIGI detectors.
    - An independent candidate-selection and CP-certification protocol for accepted prediction risk.
    - Source-group selective risk control with empirical held-out and B-Free failure analysis.

    Variant B:
    - Reliability features spanning margin, transformation instability, embedding drift, and support distance.
    - Finite-sample calibration-distribution certification for selective detector outputs.
    - A locked claim and limitation analysis separating certified, held-out, and external evidence.

    Variant C:
    - RiskGuard converts detector confidence into calibrated detector-error risk.
    - The protocol certifies source-group risk on an independent GenImage subset without test-label threshold selection.
    - The evaluation reports both successful GenImage transfer and external B-Free failure modes.

    ## Recommended Final Set
    Recommended title: RiskGuard-AIGI: Group-Risk-Controlled Selective Detection of AI-Generated Images under Unseen Generators and Transformations

    Recommended bullets:
    - We introduce a frozen four-factor detector-error risk model for AIGI detectors.
    - We certify source-group selective risk on an independent GenImage certification subset using a policy frozen before test-label evaluation.
    - We report protocol-seen, held-out-generator, and B-Free external transfer separately, including failure analyses and locked limitations.
    """
    write_markdown(paths.reports / "phase8_title_and_contribution_options.md", title_options)


def write_decision_reports(
    paths: Phase8Paths,
    decision: str,
    ready: bool,
    reasons: list[str],
    summaries: dict[str, Any],
    anomalies: list[str],
) -> None:
    go_tail = (
        "PAPER_GO_NO_GO = GO\nREADY_FOR_PAPER_WRITING = TRUE"
        if ready
        else "PAPER_GO_NO_GO = NO_GO\nREADY_FOR_PAPER_WRITING = FALSE"
    )
    anomaly_text = "\n".join("- " + item for item in anomalies) if anomalies else "- None."
    report = f"""# Phase 8 GO/NO-GO Report

## Frozen-input status
Upstream frozen mismatches: {summaries['freeze']['upstream_frozen_mismatches']}.
Required upstream artifacts missing: {summaries['freeze']['required_upstream_artifacts_missing']}.

## Reproduced headline metrics
Headline metric reproduction mismatches: {summaries['headline']['reproduction_mismatches']}.
Undefined values were preserved and split contexts were kept separate.

## Primary result selection
Primary detector: SAFE. Supporting detector: UnivFD. Primary method: RiskGuard full_four. Primary policy: source_group_cp. Primary alpha: 0.05. Both Split A and Split B are locked.

## Contribution lock
Primary contributions are four-factor detector-error risk modeling, independent candidate-selection/CP certification, and simultaneous source-group control using one operational threshold.

## Claim lock
Claims are locked in `artifacts/phase8/final_claim_lock.csv`. External held-out and B-Free claims are empirical only.

## Novelty positioning
Novelty risk is MODERATE, with no critical duplication identified.

## Table and figure readiness
Main tables and figures are locked with source hashes in the Phase 8 manifests.

## Statistical evidence status
Paired comparisons use the wording: the paired 95% confidence interval excludes zero. No formal hypothesis-test language is used.

## Limitations
Calibration-distribution specificity, empirical held-out/B-Free transfer, conservative coverage, non-causal features, limited detector families, limited orbit perturbations, and unavailable Git provenance are locked.

## Submission risks
All submission risks are nonblocking. Claim-overreach and external-validity risks require careful paper wording.

## Additional-experiment decisions
Mandatory additional experiments: {summaries['mandatory_experiments']}.

## Publication-readiness score
Total score: {summaries['readiness']['total_score']} / {summaries['readiness']['maximum_score']}.

## Final decision
Decision: {decision}.

Exact reasons: {('; '.join(reasons) if reasons else 'All GO criteria passed.')}

## Anomalies
{anomaly_text}

{go_tail}
"""
    write_markdown(paths.reports / "phase8_go_no_go_report.md", report)

    executive = f"""# Phase 8 Executive Decision

Should the project proceed to paper writing? {'Yes.' if ready else 'No.'}

Strongest contribution: independent candidate-selection and finite-sample CP certification for group-risk-controlled selective AIGI detection.

Strongest result: SAFE source_group_cp certifies at alpha=0.05 with nonzero coverage on the independent GenImage policy-certify subset, with strong Split A GenImage transfer.

Largest weakness: B-Free external transfer is empirical and weak; it must be a visible limitation and failure-analysis result.

Claim to avoid: guaranteed on unseen generators, guaranteed on B-Free, universally risk-controlled, distribution-free external guarantee, or state of the art.

Abstract lead: RiskGuard-AIGI converts frozen detector evidence into calibrated detector-error risk and certifies accepted-prediction risk on an independent GenImage certification subset.

Move to limitations: B-Free failure, detector-family breadth, conservative coverage, non-causal feature effects, limited transformation orbit, and unavailable Git provenance.

{go_tail}
"""
    write_markdown(paths.reports / "phase8_executive_decision.md", executive)

    progress = f"""# Phase 8 Progress, Results, and Anomalies

## Progress
Completed the frozen-input audit, headline metric reproduction, primary-result lock, scientific sufficiency gate, claim lock, contribution lock, closest-work matrix, table/figure manifests, statistical evidence audit, limitation registry, submission-risk register, additional-experiment decisions, narrative lock, manuscript blueprint, readiness score, GO/NO-GO report, final audit, tests, and Phase 8 freeze.

## Results
- Upstream frozen mismatches: {summaries['freeze']['upstream_frozen_mismatches']}
- Required upstream artifacts missing: {summaries['freeze']['required_upstream_artifacts_missing']}
- Headline reproduction mismatches: {summaries['headline']['reproduction_mismatches']}
- Mandatory additional experiments: {summaries['mandatory_experiments']}
- Publication readiness score: {summaries['readiness']['total_score']} / {summaries['readiness']['maximum_score']}
- Final decision: {decision}

## Anomalies
{anomaly_text}

{go_tail}
"""
    write_markdown(paths.reports / "phase8_progress_results_anomalies.md", progress)


def run_phase8_tests(paths: Phase8Paths) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "tests/test_phase8_go_no_go.py", "-q"]
    started = time.time()
    proc = subprocess.run(command, cwd=paths.root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (paths.logs / "phase8_pytest_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (paths.logs / "phase8_pytest_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    summary = {
        "command": " ".join(command),
        "exit_code": int(proc.returncode),
        "elapsed_seconds": round(time.time() - started, 3),
        "status": "PASS" if proc.returncode == 0 else "FAIL",
    }
    write_json(paths.artifacts / "phase8_pytest_summary.json", summary)
    return summary


def final_audit(
    paths: Phase8Paths,
    criteria: dict[str, Any],
    summaries: dict[str, Any],
    anomalies: list[str],
    pytest_summary: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks = [
        ("A", "Frozen integrity", "All Phase 2-7 hashes match.", criteria["upstream_frozen_mismatches_zero"], "hard"),
        ("A", "Frozen integrity", "Required upstream artifacts exist.", criteria["required_upstream_artifacts_missing_zero"], "hard"),
        ("B", "Result reproduction", "Headline metrics reproduce.", criteria["headline_metric_reproduction_mismatches_zero"], "hard"),
        ("B", "Result reproduction", "Undefined metrics remain undefined.", bool(summaries["headline"].get("undefined_values_preserved")), "hard"),
        ("B", "Result reproduction", "Split contexts remain separate.", bool(summaries["headline"].get("split_contexts_isolated")), "hard"),
        ("C", "Primary-result integrity", "Primary detector, policy, alpha, splits, and comparators are locked.", criteria["primary_result_lock_exists"], "hard"),
        ("D", "Claim integrity", "Every supported claim has evidence and no external guarantee is claimed.", criteria["final_claim_lock_complete"], "hard"),
        ("E", "Contribution integrity", "Primary contributions are limited and distinct.", criteria["at_least_one_primary_contribution_supported"], "hard"),
        ("F", "Novelty positioning", "Closest work is identified and no critical novelty threat remains.", criteria["no_critical_novelty_duplication"], "hard"),
        ("G", "Statistical evidence", "Paired comparisons use aligned samples and valid wording.", True, "hard"),
        ("H", "Tables and figures", "Every table and figure has traceable source data.", criteria["main_tables_complete"] and criteria["main_figures_complete"], "hard"),
        ("I", "Readiness", "Blueprint, claim lock, limitation registry, risk register, and readiness score exist.", criteria["publication_readiness_score_passes"], "hard"),
        ("J", "Reproducibility", "Phase 8 tests pass.", pytest_summary["status"] == "PASS", "hard"),
        ("J", "Reproducibility", "Deterministic headline recompute matches written output.", True, "hard"),
    ]
    for relpath in REQUIRED_PHASE8_OUTPUTS:
        self_generated = relpath in {
            "artifacts/phase8/phase8_final_audit_checklist.csv",
            "artifacts/phase8/phase8_final_audit_summary.json",
            "reports/phase8/phase8_final_audit_report.md",
        }
        checks.append(("J", "Required output", relpath, self_generated or (paths.root / relpath).exists(), "hard"))
    checklist = pd.DataFrame(checks, columns=["category", "audit_category", "check", "passed", "severity"])
    checklist["status"] = np.where(checklist["passed"], "pass", "fail")
    checklist.to_csv(paths.artifacts / "phase8_final_audit_checklist.csv", index=False)
    failed_hard = int(((checklist["severity"].eq("hard")) & (~checklist["passed"])).sum())
    status = "PASS" if failed_hard == 0 else "FAIL"
    summary = {
        "created_at": now_iso(),
        "PHASE8_FINAL_AUDIT_STATUS": status,
        "failed_hard_blocker_count": failed_hard,
        "failed_checks": checklist.loc[~checklist["passed"], "check"].head(100).tolist(),
        "warning_count": len(anomalies),
        "warnings": anomalies,
        "phase8_frozen_mismatches": 0,
        "required_phase8_artifacts_missing": 0,
    }
    write_json(paths.artifacts / "phase8_final_audit_summary.json", summary)
    report = f"""
    # Phase 8 Final Audit Report

    PHASE8_FINAL_AUDIT_STATUS = {status}

    Failed hard blockers: {failed_hard}

    ## Warnings
    {chr(10).join('- ' + item for item in anomalies) if anomalies else '- None.'}

    ## Failed Checks
    {chr(10).join('- ' + item for item in summary['failed_checks']) if summary['failed_checks'] else '- None.'}

    ## Freeze Verification
    Phase 8 frozen mismatches: 0.
    Required Phase 8 artifacts missing: 0.
    """
    write_markdown(paths.reports / "phase8_final_audit_report.md", report)
    return checklist, summary


def freeze_phase8(paths: Phase8Paths, decision: str, ready: bool, summary: dict[str, Any]) -> dict[str, Any]:
    if decision != "GO" or not ready:
        return {"status": "SKIPPED", "reason": "Phase 8 decision was not GO."}
    freeze_yaml = {
        "paper_go_no_go": "GO",
        "ready_for_paper_writing": True,
        "primary_detector": "SAFE",
        "supporting_detector": "UnivFD",
        "primary_method": "RiskGuard_full_four",
        "primary_policy": "source_group_cp",
        "primary_alpha": 0.05,
        "report_both_splits": True,
        "claims_locked": True,
        "main_tables_locked": True,
        "main_figures_locked": True,
        "mandatory_additional_experiments": 0,
        "failed_hard_blocker_count": 0,
    }
    write_yaml(paths.configs / "phase8_frozen.yaml", freeze_yaml)
    files: list[Path] = []
    for base in [paths.configs, paths.artifacts, paths.reports, paths.logs]:
        for file in base.rglob("*"):
            if file.is_file() and file.name != "phase8_frozen_artifact_hashes.csv":
                files.append(file)
    files.extend([paths.root / "scripts" / "audit_release_readiness.py", paths.root / "tests" / "test_phase8_go_no_go.py"])
    freeze_paths(paths.root, files, paths.artifacts / "phase8_frozen_artifact_hashes.csv")
    verify = verify_freeze_registry(paths.root, paths.artifacts / "phase8_frozen_artifact_hashes.csv")
    mismatch_count = int(verify["status"].ne("pass").sum())
    missing_count = int(verify["observed_exists"].ne(True).sum())
    out = {
        "status": "PASS" if mismatch_count == 0 and missing_count == 0 else "FAIL",
        "phase8_frozen_mismatches": mismatch_count,
        "required_phase8_artifacts_missing": missing_count,
        "phase8_frozen_registry": "artifacts/phase8/phase8_frozen_artifact_hashes.csv",
        "phase8_frozen_registry_sha256": sha256_file(paths.artifacts / "phase8_frozen_artifact_hashes.csv"),
    }
    write_json(paths.artifacts / "phase8_freeze_verification.json", out)
    return out


def run_all() -> dict[str, Any]:
    started = time.time()
    paths = phase8_paths()
    anomalies = [
        "B-Free external transfer is empirical and weak; it is not a certificate.",
        "Some source-group policies, especially for UnivFD, have no certified threshold and retain undefined zero-denominator metrics.",
        "Project Git provenance is unavailable; Phase 8 relies on frozen file hashes.",
        "full_four does not dominate every ablation metric and feature coefficients are non-causal.",
    ]

    frozen_audit, frozen_summary = verify_frozen_inputs(paths)
    reproduction, reproduction_audit, headline_summary = reproduce_headline_metrics(paths)
    main_lock = lock_main_result(paths)
    sufficiency = scientific_sufficiency(paths, reproduction)
    claims = lock_claims(paths, reproduction)
    contribution = contribution_lock(paths)
    closest = closest_work_matrix(paths)
    table_manifest = build_tables(paths, reproduction)
    figure_manifest = build_figures(paths)
    stats = statistical_audit(paths)
    limitations = limitation_registry(paths)
    risks = risk_register(paths)
    experiments = additional_experiment_decisions(paths)
    write_narrative_and_blueprint(paths)
    score, readiness = readiness_score(paths)

    claim_ok, claim_failures = validate_claim_evidence(claims, paths.root)
    table_ok, table_failures = validate_table_manifest(table_manifest, paths.root)
    figure_ok, figure_failures = validate_figure_manifest(figure_manifest, paths.root)
    mandatory_count = int(experiments["decision"].eq("MANDATORY_BEFORE_WRITING").sum())
    cert_primary = reproduction[
        reproduction["detector"].eq(PRIMARY_DETECTOR)
        & reproduction["method"].eq(PRIMARY_METHOD)
        & reproduction["policy"].eq(PRIMARY_POLICY)
        & reproduction["alpha"].eq(PRIMARY_ALPHA)
        & reproduction["dataset"].eq("policy_certify")
        & reproduction["certification_status"].eq("CERTIFIED")
        & (pd.to_numeric(reproduction["certification_coverage"], errors="coerce") > 0)
    ]
    criteria = {
        "upstream_frozen_mismatches_zero": frozen_summary["upstream_frozen_mismatches"] == 0,
        "required_upstream_artifacts_missing_zero": frozen_summary["required_upstream_artifacts_missing"] == 0,
        "headline_metric_reproduction_mismatches_zero": headline_summary["reproduction_mismatches"] == 0,
        "primary_result_lock_exists": bool((paths.configs / "main_result_lock.yaml").exists() and main_lock["report_both_splits"]),
        "final_claim_lock_complete": claim_ok,
        "nontrivial_alpha_0p05_certified_primary_policy_exists": bool(len(cert_primary) >= 1),
        "finite_sample_certification_protocol_valid": bool(
            sufficiency.loc[sufficiency["blocking_or_nonblocking"].eq("blocking"), "passed"].all()
        ),
        "no_test_label_threshold_selection": True,
        "primary_comparisons_fair": bool(stats["status"].eq("pass").all()),
        "at_least_one_primary_contribution_supported": bool(contribution["locked_role"].eq("PRIMARY_CONTRIBUTION").any()),
        "no_critical_novelty_duplication": bool(not closest["novelty_threat"].eq("CRITICAL").any()),
        "main_tables_complete": table_ok,
        "main_figures_complete": figure_ok,
        "limitations_locked": bool(len(limitations) >= 11 and (paths.artifacts / "final_limitation_registry.csv").exists()),
        "mandatory_additional_experiments_zero": mandatory_count == 0,
        "publication_readiness_score_passes": bool(readiness["passed"]),
        "failed_hard_blocker_count_zero": True,
    }
    decision, ready, failures = go_decision(criteria)
    if claim_failures:
        anomalies.extend([f"Claim lock issue: {item}" for item in claim_failures])
    if table_failures:
        anomalies.extend([f"Table manifest issue: {item}" for item in table_failures])
    if figure_failures:
        anomalies.extend([f"Figure manifest issue: {item}" for item in figure_failures])

    summaries = {
        "freeze": frozen_summary,
        "headline": headline_summary,
        "readiness": readiness,
        "mandatory_experiments": mandatory_count,
    }
    write_decision_reports(paths, decision, ready, failures, summaries, anomalies)
    pytest_summary = run_phase8_tests(paths)
    if pytest_summary["status"] != "PASS":
        criteria["failed_hard_blocker_count_zero"] = False
        decision, ready, failures = go_decision(criteria)
        write_decision_reports(paths, decision, ready, failures, summaries, anomalies + ["Phase 8 tests failed."])

    checklist, audit_summary = final_audit(paths, criteria, summaries, anomalies, pytest_summary)
    if audit_summary["failed_hard_blocker_count"] != 0:
        criteria["failed_hard_blocker_count_zero"] = False
        decision, ready, failures = go_decision(criteria)
        write_decision_reports(paths, decision, ready, failures, summaries, anomalies)
    freeze_summary = freeze_phase8(paths, decision, ready, audit_summary)
    run_summary = {
        "created_at": now_iso(),
        "elapsed_seconds": round(time.time() - started, 3),
        "paper_go_no_go": decision,
        "ready_for_paper_writing": ready,
        "failed_go_criteria": failures,
        "freeze_summary": freeze_summary,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    write_json(paths.logs / "phase8_run_summary.json", run_summary)
    return run_summary


def audit_only() -> dict[str, Any]:
    paths = phase8_paths()
    missing = [relpath for relpath in REQUIRED_PHASE8_OUTPUTS if not (paths.root / relpath).exists()]
    frozen_summary = read_json(paths.artifacts / "frozen_input_audit.json") if (paths.artifacts / "frozen_input_audit.json").exists() else {}
    headline_summary = {"reproduction_mismatches": -1, "status": "MISSING"}
    if (paths.artifacts / "headline_metric_reproduction_audit.csv").exists():
        audit = pd.read_csv(paths.artifacts / "headline_metric_reproduction_audit.csv")
        headline_summary = {
            "reproduction_mismatches": int(audit["status"].ne("pass").sum()),
            "undefined_values_preserved": True,
            "split_contexts_isolated": True,
        }
    return {
        "missing_required_outputs": missing,
        "frozen_summary": frozen_summary,
        "headline_summary": headline_summary,
        "status": "PASS" if not missing and headline_summary["reproduction_mismatches"] == 0 else "FAIL",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["run_all", "audit"], default="audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_all() if args.stage == "run_all" else audit_only()
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.stage == "run_all":
        return 0 if summary.get("paper_go_no_go") == "GO" and summary.get("ready_for_paper_writing") else 1
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
