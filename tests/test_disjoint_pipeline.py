"""End-to-end invariants for the disjoint protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "protocol_clean_v2"
MAN = ART / "manifests"


def require_clean_run() -> None:
    if not (ART / "certification" / "policy_freeze.json").exists():
        pytest.skip("clean protocol artifacts have not been materialized")


def shas(path: Path, parquet: bool = False) -> set[str]:
    df = pd.read_parquet(path, columns=["sha256"]) if parquet else pd.read_csv(path, usecols=["sha256"])
    return set(df.sha256.astype(str))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize("split", ["split_a", "split_b"])
def test_clean_base_cal_risk_fit_sha_disjoint(split: str) -> None:
    require_clean_run()
    assert shas(MAN / f"{split}_base_cal.csv").isdisjoint(shas(MAN / f"{split}_risk_fit.csv"))


@pytest.mark.parametrize("split", ["split_a", "split_b"])
def test_clean_base_cal_threshold_cal_sha_disjoint(split: str) -> None:
    require_clean_run()
    assert shas(MAN / f"{split}_base_cal.csv").isdisjoint(shas(MAN / f"{split}_threshold_cal.csv"))


@pytest.mark.parametrize("split", ["split_a", "split_b"])
def test_clean_risk_fit_threshold_cal_sha_disjoint(split: str) -> None:
    require_clean_run()
    assert shas(MAN / f"{split}_risk_fit.csv").isdisjoint(shas(MAN / f"{split}_threshold_cal.csv"))


@pytest.mark.parametrize("detector", ["safe", "univfd"])
@pytest.mark.parametrize("split", ["split_a", "split_b"])
def test_clean_policy_select_certify_sha_disjoint(detector: str, split: str) -> None:
    require_clean_run()
    df = pd.read_csv(MAN / f"{detector}_{split}_policy_assignment.csv")
    left = set(df.loc[df.calibration_subset.eq("policy_select"), "sha256"].astype(str))
    right = set(df.loc[df.calibration_subset.eq("policy_certify"), "sha256"].astype(str))
    assert left.isdisjoint(right)


@pytest.mark.parametrize("split", ["split_a", "split_b"])
def test_clean_no_evaluation_overlap(split: str) -> None:
    require_clean_run()
    development = set().union(*(shas(MAN / f"{split}_{p}.csv") for p in ["base_cal", "risk_fit", "threshold_cal"]))
    seen = shas(ART / "features" / "safe" / split / "protocol_seen.parquet", parquet=True)
    held = shas(ART / "features" / "safe" / split / "protocol_held_out.parquet", parquet=True)
    assert development.isdisjoint(seen)
    assert development.isdisjoint(held)
    assert seen.isdisjoint(held)


def test_clean_q_selected_only_from_base_cal() -> None:
    require_clean_run()
    q = pd.read_csv(ART / "base_thresholds.csv")
    assert q.labels_used.eq("base_cal_only").all()
    assert q.source_partition.eq("base_cal").all()
    assert q.decision_threshold.between(0, 1, inclusive="neither").all()


def test_clean_q_is_frozen_downstream() -> None:
    require_clean_run()
    q_hash = sha256_file(ART / "base_thresholds.csv")
    for path in (ART / "fitted_scorers").glob("*_riskguard.json"):
        model = json.loads(path.read_text())
        assert model["q_frozen_upstream"] is True
        assert model["q_registry_sha256"] == q_hash


def test_clean_scorer_training_reads_only_risk_fit() -> None:
    require_clean_run()
    for path in (ART / "fitted_scorers").glob("*.json"):
        model = json.loads(path.read_text())
        assert model["source_partition"] == "risk_fit"
        assert model["risk_fit_feature_sha256"] == sha256_file(
            ART / "features" / model["detector"] / model["split"] / "risk_fit.parquet"
        )


def test_clean_candidates_cannot_read_policy_certify_labels() -> None:
    require_clean_run()
    freeze = json.loads((ART / "policy_candidates" / "candidate_freeze.json").read_text())
    assert freeze["status"] == "FROZEN_BEFORE_CERTIFICATION"
    assert freeze["policy_certify_labels_used_for_candidates"] is False
    assert freeze["policy_certify_scores_used"] is False


def test_clean_certification_did_not_mutate_scorer_or_candidates() -> None:
    require_clean_run()
    freeze = json.loads((ART / "policy_candidates" / "candidate_freeze.json").read_text())
    assert {name: sha256_file(ROOT / name) for name in freeze["files"]} == freeze["files"]
    opening = json.loads((ART / "evaluation_opening_record.json").read_text())
    assert opening["opened_after_policy_freeze"] is True
    assert opening["scorer_hashes"] == {
        path.name: sha256_file(path) for path in sorted((ART / "fitted_scorers").glob("*_riskguard.json"))
    }


def test_clean_detector_split_outputs_have_provenance_hashes() -> None:
    require_clean_run()
    q = pd.read_csv(ART / "base_thresholds.csv")
    assert q.source_manifest_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    policies = pd.read_csv(ART / "certification" / "certification_registry.csv")
    assert policies.policy_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    assert set(zip(q.detector, q.split)) == {(d, s) for d in ["safe", "univfd"] for s in ["split_a", "split_b"]}


def test_clean_old_q_feature_cache_cannot_be_loaded() -> None:
    require_clean_run()
    q = pd.read_csv(ART / "base_thresholds.csv").set_index(["detector", "split"])
    for detector, split in q.index:
        expected = float(q.loc[(detector, split), "decision_threshold"])
        for partition in ["base_cal", "risk_fit", "threshold_cal", "protocol_seen", "protocol_held_out"]:
            path = ART / "features" / detector / split / f"{partition}.parquet"
            values = pd.read_parquet(path, columns=["q"])["q"]
            assert values.nunique() == 1
            assert float(values.iloc[0]) == expected
    provenance = json.loads((ART / "run_provenance.json").read_text())
    assert provenance["old_q_derived_inputs_used"] is False


def test_repaired_knn_cv_is_sha_grouped_without_leakage() -> None:
    require_clean_run()
    path = ART / "baselines" / "knn_sha_group_repair" / "sha_grouped_cv_audit.csv"
    audit = pd.read_csv(path)
    assert len(audit) == 40
    assert audit["cross_fold_SHA_groups"].eq(0).all()
    assert audit["status"].eq("PASS").all()


@pytest.mark.parametrize("detector", ["safe", "univfd"])
@pytest.mark.parametrize("split", ["split_a", "split_b"])
def test_repaired_knn_final_bank_has_one_row_per_sha(detector: str, split: str) -> None:
    require_clean_run()
    bank = pd.read_parquet(ART / "baselines" / f"{detector}_{split}_knn_bank.parquet", columns=["sha256"])
    model = json.loads((ART / "baselines" / f"{detector}_{split}_knn.json").read_text())
    assert bank.sha256.is_unique
    assert model["reference_configuration"] == "one_row_per_sha256"
    assert model["sha_grouped_cv"] is True
    assert model["cross_fold_SHA_groups"] == 0
    assert model["bank_sha256"] == sha256_file(ART / "baselines" / f"{detector}_{split}_knn_bank.parquet")


def test_repaired_knn_reference_sensitivity_passes() -> None:
    require_clean_run()
    decision = json.loads((ART / "baselines" / "knn_sha_group_repair" / "final_decision.json").read_text())
    assert decision["SHA_CV_LEAKAGE"] == 0
    assert decision["reference_bank_sensitivity_pass"] is True
    assert decision["status"] == "PASS"
