from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase8", ROOT / "scripts" / "audit_release_readiness.py")
phase8 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = phase8
SPEC.loader.exec_module(phase8)


def test_upstream_freeze_summary_detects_hash_mismatch() -> None:
    audit = pd.DataFrame(
        [
            {"audit_type": "frozen_hash", "status": "pass", "expected_exists": True, "observed_exists": True},
            {"audit_type": "frozen_hash", "status": "fail", "expected_exists": True, "observed_exists": True},
            {"audit_type": "required_phase8_source", "status": "fail", "expected_exists": True, "observed_exists": False},
        ]
    )
    summary = phase8.summarize_freeze_audit(audit)
    assert summary["upstream_frozen_mismatches"] == 1
    assert summary["required_upstream_artifacts_missing"] == 1


def test_headline_metric_undefined_preservation() -> None:
    assert phase8.undefined_metric_preserved("undefined_zero_denominator")
    assert phase8.value_equal("undefined_zero_denominator", "undefined_zero_denominator")
    assert not phase8.value_equal("undefined_zero_denominator", 0.0)


def test_claim_evidence_linkage_rejects_supported_claim_without_evidence() -> None:
    claims = pd.DataFrame(
        [
            {
                "claim_id": "X",
                "status": "LOCKED_SUPPORTED",
                "final_claim_text": "A supported claim.",
                "allowed_wording": "",
                "supporting_artifacts": "",
            }
        ]
    )
    ok, failures = phase8.validate_claim_evidence(claims, ROOT)
    assert not ok
    assert "missing evidence" in failures[0]


def test_prohibited_claim_detection_false_external_guarantees() -> None:
    assert phase8.prohibited_claim_detected("RiskGuard is guaranteed on unseen generators.")
    assert phase8.prohibited_claim_detected("RiskGuard is guaranteed on B-Free.")
    assert phase8.prohibited_claim_detected("This is a distribution-free external guarantee.")
    assert not phase8.prohibited_claim_detected("empirically evaluated on held-out generators")


def test_split_context_isolation_rejects_mixed_split_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "detector": "safe",
                "split": "split_a",
                "method": "riskguard",
                "alpha": 0.05,
                "policy": "source_group_cp",
                "dataset": "protocol_seen",
                "source_artifacts": "artifacts/phase5/scores/safe/split_a/protocol_seen.parquet",
            },
            {
                "detector": "safe",
                "split": "split_a",
                "method": "riskguard",
                "alpha": 0.05,
                "policy": "source_group_cp",
                "dataset": "protocol_held_out",
                "source_artifacts": "artifacts/phase5/scores/safe/split_b/protocol_held_out.parquet",
            },
        ]
    )
    assert not phase8.split_contexts_are_isolated(df)


def test_table_and_figure_source_consistency_synthetic_missing_main_table() -> None:
    manifest = pd.DataFrame(
        [
            {
                "table_id": "T1",
                "csv_source": "missing.csv",
                "latex_source": "missing.tex",
                "source_artifact_hashes": "{}",
            }
        ]
    )
    ok, failures = phase8.validate_table_manifest(manifest, ROOT)
    assert not ok
    assert any("missing required main table" in item for item in failures)


def test_readiness_score_calculation_below_go_threshold() -> None:
    df = pd.DataFrame(
        [
            ("technical integrity", 2),
            ("reproducibility", 2),
            ("primary-result strength", 1),
            ("risk-control validity", 2),
            ("baseline fairness", 1),
            ("ablation support", 1),
            ("held-out evaluation", 1),
            ("external evaluation", 0),
            ("failure analysis", 1),
            ("novelty positioning", 1),
            ("claim discipline", 2),
            ("figure/table readiness", 1),
        ],
        columns=["dimension", "score"],
    )
    passed, details = phase8.readiness_score_passes(df)
    assert not passed
    assert details["total_score"] == 15


def test_go_criteria_and_no_go_criteria() -> None:
    criteria = {
        "upstream_frozen_mismatches_zero": True,
        "required_upstream_artifacts_missing_zero": True,
        "headline_metric_reproduction_mismatches_zero": True,
        "primary_result_lock_exists": True,
        "final_claim_lock_complete": True,
        "nontrivial_alpha_0p05_certified_primary_policy_exists": True,
        "finite_sample_certification_protocol_valid": True,
        "no_test_label_threshold_selection": True,
        "primary_comparisons_fair": True,
        "at_least_one_primary_contribution_supported": True,
        "no_critical_novelty_duplication": True,
        "main_tables_complete": True,
        "main_figures_complete": True,
        "limitations_locked": True,
        "mandatory_additional_experiments_zero": True,
        "publication_readiness_score_passes": True,
        "failed_hard_blocker_count_zero": True,
    }
    decision, ready, failures = phase8.go_decision(criteria)
    assert decision == "GO"
    assert ready
    assert failures == []

    criteria["no_critical_novelty_duplication"] = False
    decision, ready, failures = phase8.go_decision(criteria)
    assert decision == "NO_GO"
    assert not ready
    assert "no_critical_novelty_duplication" in failures


def test_mandatory_experiment_logic() -> None:
    assert phase8.mandatory_experiment_decision("primary result cannot be reproduced", True) == "MANDATORY_BEFORE_WRITING"
    assert phase8.mandatory_experiment_decision("more B-Free sampling", False) == "OPTIONAL_FOR_STRENGTHENING"
    assert phase8.mandatory_experiment_decision("new method search", False) == "NOT_JUSTIFIED"


def test_determinism_stable_records_hash() -> None:
    records = [{"b": 2, "a": 1}, {"a": 3, "b": 4}]
    assert phase8.stable_records_hash(records) == phase8.stable_records_hash(records)
