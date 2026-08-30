"""Phase 4 reliability feature formulas."""

from __future__ import annotations

import numpy as np


PRIMARY_FEATURES = (
    "margin_distance",
    "orbit_logit_variance",
    "mean_directional_erosion",
    "worst_view_erosion",
)

LEGACY_FULL_FOUR_FEATURES = (
    "margin_distance",
    "orbit_logit_variance",
    "embedding_drift_mean",
    "orbit_support_distance_max",
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def margin_distance(identity_logit: float, threshold_logit: float) -> float:
    """Distance between identity raw logit and the frozen raw-logit threshold."""
    value = abs(float(identity_logit) - float(threshold_logit))
    if not np.isfinite(value):
        raise ValueError("non-finite margin distance")
    return float(value)


def orbit_logit_variance(logits: np.ndarray) -> float:
    arr = np.asarray(logits, dtype=np.float64)
    if arr.shape != (5,):
        raise ValueError(f"expected exactly five orbit logits, got shape {arr.shape}")
    value = float(np.var(arr, ddof=0))
    if not np.isfinite(value):
        raise ValueError("non-finite orbit logit variance")
    return value


def identity_orientation(identity_logit: float, threshold_logit: float) -> int:
    """Orientation of the identity prediction relative to the frozen boundary."""
    z0 = float(identity_logit)
    gamma = float(threshold_logit)
    if not np.isfinite(z0) or not np.isfinite(gamma):
        raise ValueError("non-finite identity logit or threshold")
    return 1 if z0 >= gamma else -1


def signed_boundary_margins(logits: np.ndarray, threshold_logit: float) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    if arr.shape != (5,):
        raise ValueError(f"expected exactly five orbit logits, got shape {arr.shape}")
    if not np.isfinite(arr).all() or not np.isfinite(float(threshold_logit)):
        raise ValueError("non-finite orbit logit or threshold")
    c = identity_orientation(float(arr[0]), float(threshold_logit))
    return c * (arr - float(threshold_logit))


def directional_erosions(logits: np.ndarray, threshold_logit: float) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    margins = signed_boundary_margins(arr, threshold_logit)
    deltas = margins[0] - margins[1:]
    direct = identity_orientation(float(arr[0]), float(threshold_logit)) * (arr[0] - arr[1:])
    if not np.allclose(deltas, direct, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("directional erosion identity failed")
    return deltas.astype(np.float64, copy=False)


def mean_directional_erosion(logits: np.ndarray, threshold_logit: float) -> float:
    value = float(np.mean(directional_erosions(logits, threshold_logit)))
    if not np.isfinite(value):
        raise ValueError("non-finite mean directional erosion")
    return value


def worst_view_erosion(logits: np.ndarray, threshold_logit: float) -> float:
    value = float(np.max(directional_erosions(logits, threshold_logit)))
    if not np.isfinite(value):
        raise ValueError("non-finite worst-view erosion")
    return value


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a matrix")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("zero-norm or non-finite embedding encountered")
    return embeddings / norms[:, None]


def embedding_drift_mean(embeddings: np.ndarray) -> float:
    normed = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    if normed.shape[0] != 5:
        raise ValueError(f"expected exactly five orbit embeddings, got shape {normed.shape}")
    reference = normed[0]
    cosine = np.clip(normed[1:] @ reference, -1.0, 1.0)
    value = float(np.mean(1.0 - cosine))
    if not np.isfinite(value):
        raise ValueError("non-finite embedding drift")
    return value


def orbit_support_distance_max(view_support_distances: np.ndarray) -> float:
    arr = np.asarray(view_support_distances, dtype=np.float64)
    if arr.shape != (5,):
        raise ValueError(f"expected exactly five support distances, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite support distance")
    return float(np.max(arr))


def extract_reliability_features(
    logits: np.ndarray,
    embeddings: np.ndarray,
    view_support_distances: np.ndarray,
    threshold_logit: float,
) -> dict[str, float]:
    return extract_logit_trajectory_features(logits, threshold_logit)


def extract_logit_trajectory_features(logits: np.ndarray, threshold_logit: float) -> dict[str, float]:
    features = {
        "margin_distance": margin_distance(float(np.asarray(logits, dtype=np.float64)[0]), threshold_logit),
        "orbit_logit_variance": orbit_logit_variance(logits),
        "mean_directional_erosion": mean_directional_erosion(logits, threshold_logit),
        "worst_view_erosion": worst_view_erosion(logits, threshold_logit),
    }
    if tuple(features) != PRIMARY_FEATURES or not all(np.isfinite(value) for value in features.values()):
        raise ValueError(f"non-finite reliability features: {features}")
    return features


def extract_legacy_full_four_features(
    logits: np.ndarray,
    embeddings: np.ndarray,
    view_support_distances: np.ndarray,
    threshold_logit: float,
) -> dict[str, float]:
    features = {
        "margin_distance": margin_distance(float(np.asarray(logits, dtype=np.float64)[0]), threshold_logit),
        "orbit_logit_variance": orbit_logit_variance(logits),
        "embedding_drift_mean": embedding_drift_mean(embeddings),
        "orbit_support_distance_max": orbit_support_distance_max(view_support_distances),
    }
    if tuple(features) != LEGACY_FULL_FOUR_FEATURES or not all(np.isfinite(value) for value in features.values()):
        raise ValueError(f"non-finite legacy reliability features: {features}")
    return features


# Backward-compatible names from earlier scaffolding. Phase 4 code uses the
# explicit names above.
margin_uncertainty = margin_distance
orbit_variance = orbit_logit_variance
representation_drift = embedding_drift_mean
