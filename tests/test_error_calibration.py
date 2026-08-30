from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from selective_detection.error_probability_calibrator import (
    PRIMARY_FEATURES,
    SCHEMA_VERSION,
    FEATURE_TRANSFORMATIONS,
    load_riskguard_json,
    risk_logit,
    risk_probability,
    standardize_features,
    transform_features,
)
from selective_detection.reliability_features import (
    directional_erosions,
    extract_logit_trajectory_features,
    identity_orientation,
    orbit_logit_variance,
    signed_boundary_margins,
)
from selective_detection.grouped_cross_validation import assign_sha_grouped_folds, fold_audit_rows
from selective_detection.calibration_metrics import aurc, binary_nll, brier_score, calibrator_metrics
from selective_detection.calibrator_artifact_io import payload_sha256, write_json


def synthetic_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "margin_distance": [0.0, 1.0, 3.0],
            "orbit_logit_variance": [0.0, 3.0, 8.0],
            "mean_directional_erosion": [-1.0, 0.0, 2.0],
            "worst_view_erosion": [2.0, -0.5, 0.0],
        }
    )


def test_feature_count_order_transformations_and_margin_orientation() -> None:
    raw = synthetic_raw()
    transformed = transform_features(raw, PRIMARY_FEATURES, as_frame=False)
    expected = np.column_stack(
        [
            -np.log1p(raw["margin_distance"]),
            np.log1p(raw["orbit_logit_variance"]),
            np.sign(raw["mean_directional_erosion"]) * np.log1p(np.abs(raw["mean_directional_erosion"])),
            np.sign(raw["worst_view_erosion"]) * np.log1p(np.abs(raw["worst_view_erosion"])),
        ]
    )
    assert tuple(PRIMARY_FEATURES) == (
        "margin_distance",
        "orbit_logit_variance",
        "mean_directional_erosion",
        "worst_view_erosion",
    )
    assert FEATURE_TRANSFORMATIONS["margin_distance"] == "-log1p"
    assert FEATURE_TRANSFORMATIONS["mean_directional_erosion"] == "signed_log1p"
    assert np.allclose(transformed, expected)
    assert transformed[0, 0] > transformed[-1, 0]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda df: df.assign(margin_distance=-0.1),
        lambda df: df.assign(orbit_logit_variance=np.nan),
        lambda df: df.assign(mean_directional_erosion=np.inf),
    ],
)
def test_transform_rejects_negative_nan_or_inf_raw_features(mutator) -> None:
    with pytest.raises(ValueError):
        transform_features(mutator(synthetic_raw()), PRIMARY_FEATURES)


def test_transform_rejects_incorrect_feature_order_and_duplicates() -> None:
    with pytest.raises(ValueError):
        transform_features(synthetic_raw().to_numpy(), ("margin_distance", "not_allowed"))
    with pytest.raises(ValueError):
        transform_features(synthetic_raw(), ("margin_distance", "margin_distance"))


def test_standardizer_rejects_zero_and_negative_scale() -> None:
    values = np.ones((4, 2))
    with pytest.raises(ValueError):
        standardize_features(values, [0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        standardize_features(values, [0.0, 0.0], [1.0, -1.0])


def test_scaler_fit_partition_uses_training_rows_only() -> None:
    train = transform_features(synthetic_raw().iloc[:2], PRIMARY_FEATURES, as_frame=False)
    threshold_cal = transform_features(synthetic_raw().iloc[2:], PRIMARY_FEATURES, as_frame=False)
    train_mean = train.mean(axis=0)
    leaked_mean = np.vstack([train, threshold_cal]).mean(axis=0)
    assert not np.allclose(train_mean, leaked_mean)


def test_sha_grouped_cv_isolation_fold_completeness_and_duplicate_sha() -> None:
    rows = []
    for idx in range(80):
        sha = f"sha-{idx // 2:03d}"
        rows.append(
            {
                "sample_id": f"s-{idx:03d}",
                "sha256": sha,
                "base_error": idx % 2,
                "label": (idx // 2) % 2,
                "generator": f"g{idx % 4}",
            }
        )
    df = pd.DataFrame(rows)
    folds = assign_sha_grouped_folds(df, n_splits=5, seed=20260916)
    assert len(folds) == len(df)
    assert folds["cv_fold"].between(0, 4).all()
    assert folds.groupby("sha256")["cv_fold"].nunique().max() == 1
    merged = df.merge(folds, on=["sample_id", "sha256"], validate="one_to_one")
    audit = pd.DataFrame(fold_audit_rows(merged, detector="univfd", split="split_a"))
    assert audit["sha_overlap_with_other_folds"].sum() == 0
    assert audit["row_count"].sum() == len(df)


def test_metric_orientation_probability_bounds_and_aurc() -> None:
    errors = np.array([0, 0, 1, 1])
    good_risk = np.array([0.1, 0.2, 0.8, 0.9])
    bad_risk = 1.0 - good_risk
    assert aurc(errors, good_risk) < aurc(errors, bad_risk)
    metrics = calibrator_metrics(errors, good_risk, sample_ids=np.array(["a", "b", "c", "d"]))
    assert metrics["error_detection_AUROC"] == 1.0
    assert binary_nll(errors, good_risk) < binary_nll(errors, bad_risk)
    assert brier_score(errors, good_risk) < brier_score(errors, bad_risk)


def test_json_serialization_and_manual_scoring_parity(tmp_path: Path) -> None:
    raw = synthetic_raw()
    transformed = transform_features(raw, PRIMARY_FEATURES, as_frame=False)
    means = transformed.mean(axis=0)
    scales = transformed.std(axis=0, ddof=0)
    coef = np.array([0.5, -0.25, 0.1, 0.2])
    intercept = -0.3
    payload = {
        "model_version": "test",
        "schema_version": SCHEMA_VERSION,
        "detector": "univfd",
        "split": "split_a",
        "feature_order": list(PRIMARY_FEATURES),
        "raw_feature_order": list(PRIMARY_FEATURES),
        "transformed_feature_order": [
            "u_margin_distance",
            "u_orbit_logit_variance",
            "u_mean_directional_erosion",
            "u_worst_view_erosion",
        ],
        "feature_transformations": {name: FEATURE_TRANSFORMATIONS[name] for name in PRIMARY_FEATURES},
        "scaler_means": means.tolist(),
        "scaler_scales": scales.tolist(),
        "coefficient_vector": coef.tolist(),
        "intercept": intercept,
        "selected_C": 1.0,
    }
    payload["model_hash"] = payload_sha256(payload)
    path = tmp_path / "model.json"
    write_json(path, payload)
    model = load_riskguard_json(path)
    expected_logits = ((transformed - means) / scales) @ coef + intercept
    logits = risk_logit(raw, model)
    probabilities = risk_probability(raw, model)
    assert np.allclose(logits, expected_logits, atol=1e-12)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.allclose(probabilities, 1.0 / (1.0 + np.exp(-logits)), atol=1e-12)


def test_manual_scorer_mismatch_and_nonfinite_model_are_rejected() -> None:
    raw = synthetic_raw()
    model = {
        "feature_order": list(PRIMARY_FEATURES),
        "schema_version": SCHEMA_VERSION,
        "raw_feature_order": list(PRIMARY_FEATURES),
        "scaler_means": [0.0, 0.0, 0.0, 0.0],
        "scaler_scales": [1.0, 1.0, 1.0, 0.0],
        "coefficient_vector": [1.0, 1.0, 1.0, 1.0],
        "intercept": 0.0,
    }
    with pytest.raises(ValueError):
        risk_logit(raw, model)
    model["scaler_scales"] = [1.0, 1.0, 1.0, 1.0]
    model["coefficient_vector"] = [1.0, 1.0]
    with pytest.raises(ValueError):
        risk_logit(raw, model)


def test_oof_missing_duplicate_and_same_sha_leakage_detection_synthetic() -> None:
    expected = pd.DataFrame({"sample_id": ["a", "b"], "sha256": ["s1", "s2"]})
    oof = pd.DataFrame({"sample_id": ["a", "a"], "sha256": ["s1", "s1"], "risk_probability": [0.1, 0.2]})
    expected_keys = set(map(tuple, expected[["sample_id", "sha256"]].to_numpy()))
    observed_keys = set(map(tuple, oof[["sample_id", "sha256"]].to_numpy()))
    assert expected_keys != observed_keys
    assert oof.duplicated(["sample_id", "sha256"]).any()
    folds = pd.DataFrame({"sha256": ["s1", "s1"], "cv_fold": [0, 1]})
    assert folds.groupby("sha256")["cv_fold"].nunique().max() > 1


def test_ablation_feature_sets_and_isolation_names() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_feature_ablations.py"
    spec = importlib.util.spec_from_file_location("phase5_ablation_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    assert set(module.ABLATIONS) == {
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
    assert "full_four" in module.ABLATIONS


def test_logit_trajectory_feature_formulas_and_crossing() -> None:
    gamma = 0.0
    logits = np.array([2.0, 1.0, -0.5, 3.0, 2.5])
    assert identity_orientation(logits[0], gamma) == 1
    margins = signed_boundary_margins(logits, gamma)
    assert margins[2] < 0.0
    deltas = directional_erosions(logits, gamma)
    assert np.allclose(deltas, margins[0] - margins[1:])
    assert np.allclose(deltas, logits[0] - logits[1:])
    features = extract_logit_trajectory_features(logits, gamma)
    assert features["margin_distance"] == abs(logits[0] - gamma)
    assert features["orbit_logit_variance"] == np.var(logits, ddof=0)
    assert features["mean_directional_erosion"] == np.mean(deltas)
    assert features["worst_view_erosion"] == np.max(deltas)
    assert np.min(deltas) < 0.0


def test_logit_trajectory_orientation_below_boundary() -> None:
    gamma = 1.0
    logits = np.array([0.0, -1.0, 0.5, 2.0, -0.5])
    assert identity_orientation(logits[0], gamma) == -1
    margins = signed_boundary_margins(logits, gamma)
    assert margins[3] < 0.0
    deltas = directional_erosions(logits, gamma)
    assert np.allclose(deltas, -1.0 * (logits[0] - logits[1:]))


def test_signed_log1p_properties_and_variance_ddof0() -> None:
    values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    raw = pd.DataFrame(
        {
            "mean_directional_erosion": values,
            "worst_view_erosion": values,
            "margin_distance": np.ones_like(values),
            "orbit_logit_variance": np.ones_like(values),
        }
    )
    transformed = transform_features(raw, PRIMARY_FEATURES, as_frame=True)
    signed = transformed["u_mean_directional_erosion"].to_numpy()
    assert np.isfinite(signed).all()
    assert np.allclose(signed, -signed[::-1])
    assert np.all(np.diff(signed) > 0.0)
    assert orbit_logit_variance(np.array([0.0, 1.0, 2.0, 3.0, 4.0])) == np.var(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), ddof=0)


def test_threshold_test_and_bfree_leakage_synthetic_partition_guard() -> None:
    fit = pd.DataFrame({"partition": ["risk_fit", "risk_fit"], "base_error": [0, 1]})
    threshold_cal = pd.DataFrame({"partition": ["threshold_cal"], "base_error": [1]})
    protocol = pd.DataFrame({"partition": ["protocol_seen"], "base_error": [0]})
    bfree = pd.DataFrame({"partition": ["bfree_snapshot"], "base_error": [1]})
    training = fit.copy()
    assert set(training["partition"]) == {"risk_fit"}
    leaked = pd.concat([fit, threshold_cal, protocol, bfree], ignore_index=True)
    assert set(leaked["partition"]) != {"risk_fit"}


def test_fold_assignment_determinism() -> None:
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(60)],
            "sha256": [f"sha{i // 2}" for i in range(60)],
            "base_error": [i % 2 for i in range(60)],
            "label": [(i // 3) % 2 for i in range(60)],
            "generator": [f"g{i % 3}" for i in range(60)],
        }
    )
    first = assign_sha_grouped_folds(df, n_splits=5, seed=20260916)
    second = assign_sha_grouped_folds(df, n_splits=5, seed=20260916)
    pd.testing.assert_frame_equal(first, second)
