from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase7", ROOT / "scripts" / "audit_error_analysis.py")
phase7 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = phase7
SPEC.loader.exec_module(phase7)


def test_near_threshold_bins_cover_edges() -> None:
    assert phase7.near_threshold_bin(0.0) == "0-0.001"
    assert phase7.near_threshold_bin(0.001) == "0-0.001"
    assert phase7.near_threshold_bin(0.005) == "0.001-0.005"
    assert phase7.near_threshold_bin(0.01) == "0.005-0.01"
    assert phase7.near_threshold_bin(0.05) == "0.01-0.05"
    assert phase7.near_threshold_bin(0.051) == "greater_than_0.05"


def test_failure_tag_assignment_multiple_and_none() -> None:
    thresholds = {
        "margin_distance_p10": 0.2,
        "orbit_logit_variance_p90": 2.0,
        "embedding_drift_mean_p90": 0.4,
        "orbit_support_distance_max_p90": 0.8,
    }
    row = pd.Series(
        {
            "margin_distance": 0.1,
            "orbit_logit_variance": 3.0,
            "embedding_drift_mean": 0.1,
            "orbit_support_distance_max": 0.2,
        }
    )
    tags = phase7.threshold_tags(row, thresholds)
    assert "near_decision_boundary" in tags
    assert "orbit_logit_unstable" in tags
    assert "multiple_risk_factors" in tags

    quiet = pd.Series(
        {
            "margin_distance": 0.5,
            "orbit_logit_variance": 1.0,
            "embedding_drift_mean": 0.1,
            "orbit_support_distance_max": 0.2,
        }
    )
    assert phase7.threshold_tags(quiet, thresholds) == ["none_of_primary_risk_factors"]


def test_deterministic_case_selection_ties_and_duplicate_cluster() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "sha256": ["02", "01", "03", "04"],
            "risk_probability": [0.1, 0.1, 0.2, 0.3],
            "near_duplicate_group": ["g1", "g1", "g2", "g3"],
            "generator": ["adm", "adm", "glide", "sd14"],
        }
    )
    selected = phase7.deterministic_case_selection(
        df,
        category="accepted_error",
        n=3,
        sort_cols=["risk_probability"],
        ascending=[True],
    )
    assert selected["sha256"].tolist() == ["01", "03", "04"]
    assert selected["near_duplicate_group"].is_unique


def test_bootstrap_pairing_returns_replicate_count_for_negative_delta() -> None:
    diff = np.array([-1.0, -0.5, -0.25, 0.0], dtype=float)
    units = np.array(["a", "b", "c", "d"])
    lo, hi, valid = phase7.bootstrap_mean_delta(diff, units, replicates=100, seed=123)
    assert valid == 100
    assert lo < 0
    assert hi <= 0


def test_metric_helpers_handle_single_class_group() -> None:
    y = np.zeros(4, dtype=int)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    ids = np.array(["a", "b", "c", "d"])
    metrics = phase7.binary_metric_frame(y, p, ids)
    assert np.isnan(metrics["error_detection_AUROC"])
    assert np.isfinite(metrics["NLL"])
    assert np.isfinite(metrics["AURC"])
