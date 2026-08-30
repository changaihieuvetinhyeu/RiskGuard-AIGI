"""Manual RiskGuard Phase 5 calibrator scorer.

The JSON model is the authoritative artifact.  These functions intentionally
avoid scikit-learn so Phase 6 can reproduce Phase 5 scores from the frozen
parameters alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "logit_trajectory_summary_v1"
LEGACY_SCHEMA_VERSION = "full_four_m_v_r_s_v1"

PRIMARY_FEATURES: tuple[str, ...] = (
    "margin_distance",
    "orbit_logit_variance",
    "mean_directional_erosion",
    "worst_view_erosion",
)

LEGACY_FULL_FOUR_FEATURES: tuple[str, ...] = (
    "margin_distance",
    "orbit_logit_variance",
    "embedding_drift_mean",
    "orbit_support_distance_max",
)

TRANSFORMED_FEATURE_NAMES: dict[str, str] = {
    "margin_distance": "u_margin_distance",
    "orbit_logit_variance": "u_orbit_logit_variance",
    "mean_directional_erosion": "u_mean_directional_erosion",
    "worst_view_erosion": "u_worst_view_erosion",
    "embedding_drift_mean": "u_embedding_drift_mean",
    "orbit_support_distance_max": "u_orbit_support_distance_max",
}

FEATURE_TRANSFORMATIONS: dict[str, str] = {
    "margin_distance": "-log1p",
    "orbit_logit_variance": "log1p",
    "mean_directional_erosion": "signed_log1p",
    "worst_view_erosion": "signed_log1p",
    "embedding_drift_mean": "log1p",
    "orbit_support_distance_max": "log1p",
}

SUPPORTED_FEATURES = (*PRIMARY_FEATURES, "embedding_drift_mean", "orbit_support_distance_max")

RAW_FEATURE_NEGATIVE_TOLERANCE = 1.0e-6


def transformed_feature_names(feature_order: Iterable[str] = PRIMARY_FEATURES) -> list[str]:
    """Return transformed feature names in the requested model order."""

    return [TRANSFORMED_FEATURE_NAMES[name] for name in feature_order]


def validate_feature_order(feature_order: Iterable[str]) -> tuple[str, ...]:
    """Validate that all requested features are Phase 5-approved features."""

    order = tuple(feature_order)
    missing = [name for name in order if name not in SUPPORTED_FEATURES]
    if missing:
        raise ValueError(f"unsupported Phase 5 feature(s): {missing}")
    if len(set(order)) != len(order):
        raise ValueError("feature_order contains duplicates")
    return order


def _raw_matrix(raw_features: pd.DataFrame | np.ndarray, feature_order: tuple[str, ...]) -> tuple[np.ndarray, Any]:
    if isinstance(raw_features, pd.DataFrame):
        missing = [name for name in feature_order if name not in raw_features.columns]
        if missing:
            raise ValueError(f"missing raw feature column(s): {missing}")
        return raw_features.loc[:, list(feature_order)].to_numpy(dtype=np.float64), raw_features.index
    values = np.asarray(raw_features, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(feature_order):
        raise ValueError(f"expected raw feature matrix with {len(feature_order)} columns, got {values.shape}")
    return values, None


def transform_features(
    raw_features: pd.DataFrame | np.ndarray,
    feature_order: Iterable[str] = PRIMARY_FEATURES,
    *,
    as_frame: bool | None = None,
) -> pd.DataFrame | np.ndarray:
    """Apply the fixed Phase 5 risk-oriented transformations.

    Formulas:
      margin_distance -> -log1p(margin_distance)
      all other primary features -> log1p(feature)
    """

    order = validate_feature_order(feature_order)
    values, index = _raw_matrix(raw_features, order)
    if not np.isfinite(values).all():
        raise ValueError("raw features contain NaN or Inf")
    for idx, name in enumerate(order):
        if name not in {"mean_directional_erosion", "worst_view_erosion"} and (values[:, idx] < -RAW_FEATURE_NEGATIVE_TOLERANCE).any():
            raise ValueError("raw features must be non-negative within numerical tolerance")

    columns: list[np.ndarray] = []
    for idx, name in enumerate(order):
        raw = values[:, idx]
        if FEATURE_TRANSFORMATIONS[name] == "signed_log1p":
            transformed = np.sign(raw) * np.log1p(np.abs(raw))
        else:
            transformed = np.log1p(raw)
        if FEATURE_TRANSFORMATIONS[name] == "-log1p":
            transformed = -transformed
        columns.append(transformed)
    out = np.column_stack(columns).astype(np.float64, copy=False)
    if not np.isfinite(out).all():
        raise ValueError("transformed features contain NaN or Inf")

    if as_frame is None:
        as_frame = isinstance(raw_features, pd.DataFrame)
    if as_frame:
        return pd.DataFrame(out, columns=transformed_feature_names(order), index=index)
    return out


def standardize_features(
    transformed_features: pd.DataFrame | np.ndarray,
    means: Iterable[float],
    scales: Iterable[float],
) -> np.ndarray:
    """Apply z-score standardization using frozen risk_fit statistics."""

    values = np.asarray(transformed_features, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    mu = np.asarray(list(means), dtype=np.float64)
    scale = np.asarray(list(scales), dtype=np.float64)
    if values.shape[1] != len(mu) or len(mu) != len(scale):
        raise ValueError("standardization shape mismatch")
    if not np.isfinite(mu).all() or not np.isfinite(scale).all():
        raise ValueError("scaler parameters contain NaN or Inf")
    if (scale <= 0.0).any():
        raise ValueError("scaler scales must be positive")
    return (values - mu) / scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    out = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def risk_logit(raw_features: pd.DataFrame | np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Score raw features with a frozen RiskGuard JSON payload."""

    schema = model.get("schema_version")
    if schema is not None and schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported scorer schema_version: {schema}")
    order = tuple(model.get("raw_feature_order", model["feature_order"]))
    expected_order = tuple(model.get("raw_feature_order", model.get("feature_order", ())))
    if order != expected_order:
        raise ValueError("model feature order is inconsistent")
    transformed = transform_features(raw_features, order, as_frame=False)
    means = model.get("scaler_means", model.get("scaler", {}).get("means"))
    scales = model.get("scaler_scales", model.get("scaler", {}).get("scales"))
    z = standardize_features(transformed, means, scales)
    coefs = np.asarray(model.get("coefficient_vector", model.get("coefficients")), dtype=np.float64)
    intercept = float(model["intercept"])
    if coefs.ndim != 1 or coefs.shape[0] != z.shape[1]:
        raise ValueError("coefficient shape mismatch")
    logits = z @ coefs + intercept
    if not np.isfinite(logits).all():
        raise ValueError("risk logits contain NaN or Inf")
    return logits


def risk_probability(raw_features: pd.DataFrame | np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Return calibrated error-risk probabilities from a frozen JSON model."""

    probs = _sigmoid(risk_logit(raw_features, model))
    if not np.isfinite(probs).all() or (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError("risk probabilities are outside [0, 1] or non-finite")
    return probs


def load_riskguard_json(path: str | Path) -> dict[str, Any]:
    """Load a RiskGuard model JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {
        "model_version",
        "detector",
        "split",
        "scaler_means",
        "scaler_scales",
        "intercept",
        "selected_C",
        "model_hash",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"model JSON is missing required field(s): {missing}")
    if payload.get("schema_version") not in {None, SCHEMA_VERSION}:
        raise ValueError(f"unsupported scorer schema_version: {payload.get('schema_version')}")
    if "raw_feature_order" not in payload and "feature_order" not in payload:
        raise ValueError("model JSON is missing raw_feature_order/feature_order")
    if "coefficient_vector" not in payload and "coefficients" not in payload:
        raise ValueError("model JSON is missing coefficient_vector/coefficients")
    validate_feature_order(payload.get("raw_feature_order", payload.get("feature_order")))
    return payload
