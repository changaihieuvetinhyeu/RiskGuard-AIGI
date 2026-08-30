import math

import numpy as np
import pandas as pd
from scipy.stats import beta

from selective_detection.exact_binomial_bound import clopper_pearson_upper
from selective_detection.group_risk_certification import (
    accepted_metric_summary,
    bootstrap_rank_metric_draws,
    group_definitions_from_frame,
    policy_groups,
    threshold_scan,
    weighted_aurc_from_sorted_errors,
    weighted_source_group_extremes,
)


def test_phase6_cp_oracle_values():
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0
    assert clopper_pearson_upper(5, 5, 0.05) == 1.0
    assert math.isclose(clopper_pearson_upper(0, 10, 0.05), beta.ppf(0.95, 1, 10))
    assert math.isclose(clopper_pearson_upper(1, 10, 0.05), beta.ppf(0.95, 2, 9))
    assert math.isclose(clopper_pearson_upper(7, 1000, 1e-4), beta.ppf(0.9999, 8, 993))


def test_phase6_tie_safe_threshold_scan_does_not_split_equal_scores():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "risk_score": [0.1, 0.1, 0.2, 0.3],
            "base_error": [0, 1, 0, 0],
            "label": [0, 1, 1, 1],
            "generator": ["real", "g1", "g1", "g2"],
            "base_prediction": [0, 0, 1, 1],
        }
    )
    curve = threshold_scan(df, "global_cp", 0.6)
    assert curve["accepted_count"].tolist() == [2, 3, 4]
    assert curve.loc[curve["threshold"].eq(0.1), "accepted_errors"].iloc[0] == 1


def test_phase6_group_definitions_source_and_predicted_class():
    df = pd.DataFrame(
        {
            "label": [0, 1, 1],
            "generator": ["real", "adm", "glide"],
            "base_prediction": [0, 0, 1],
        }
    )
    assert policy_groups(df, "source_group_cp").tolist() == ["real_all", "adm", "glide"]
    assert policy_groups(df, "predicted_class_cp").tolist() == ["base_prediction_real", "base_prediction_real", "base_prediction_fake"]
    defs = group_definitions_from_frame(df, "source_group_cp")
    assert set(defs) == {"real_all", "adm", "glide"}


def test_phase6_metric_denominators_and_undefined_values():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "risk_score": [0.9, 0.8],
            "base_error": [1, 0],
            "label": [1, 1],
            "base_prediction": [0, 1],
        }
    )
    empty = accepted_metric_summary(df, None)
    assert empty["accepted_samples"] == 0
    assert empty["selective_risk"] == "undefined_zero_denominator"
    accepted = accepted_metric_summary(df, 1.0)
    assert accepted["accepted_samples"] == 2
    assert accepted["accepted_FPR"] == "undefined_zero_denominator"
    assert accepted["accepted_FNR"] == 0.5


def test_phase6_weighted_aurc_matches_simple_prefix_risk():
    errors = np.array([0.0, 1.0, 1.0])
    weights = np.array([1.0, 1.0, 1.0])
    expected = (0.0 / 1.0 + 1.0 / 2.0 + 2.0 / 3.0) / 3.0
    assert math.isclose(weighted_aurc_from_sorted_errors(errors, weights), expected)


def test_phase6_rank_bootstrap_returns_valid_draws():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "sha256": ["sha-a", "sha-b", "sha-c", "sha-d"],
            "source_id": ["s1", "s2", "s3", "s4"],
            "near_duplicate_group": ["s1", "s2", "s3", "s4"],
            "label": [0, 0, 1, 1],
            "generator": ["real", "real", "g1", "g1"],
            "risk_score": [0.1, 0.2, 0.3, 0.4],
            "base_error": [0, 1, 0, 1],
        }
    )
    aurc_draws, eaurc_draws = bootstrap_rank_metric_draws(df, "protocol_seen", 8, 123)
    assert len(aurc_draws) == 8
    assert np.isfinite(aurc_draws).all()
    assert np.isfinite(eaurc_draws).all()


def test_phase6_weighted_source_group_extremes():
    df = pd.DataFrame({"label": [0, 1, 1], "generator": ["real", "adm", "adm"]})
    labels = np.array([0, 1, 1])
    errors = np.array([0.0, 1.0, 0.0])
    accepted = np.array([True, True, False])
    weights = np.ones(3)
    result = weighted_source_group_extremes(labels, errors, accepted, weights, [("real_all", labels == 0), ("adm", labels == 1)])
    assert result["worst_group_selective_risk"] == 1.0
    assert result["minimum_group_coverage"] == 0.5
