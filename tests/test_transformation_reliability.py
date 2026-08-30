from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from selective_detection.transformation_orbit import (
    ORBIT_VERSION,
    VIEW_ORDER,
    apply_view,
    build_default_orbit,
    default_orbit_views,
    make_view_id,
    transform_chain_id,
    transformed_pixel_sha256,
)
from selective_detection.reliability_features import (
    LEGACY_FULL_FOUR_FEATURES,
    PRIMARY_FEATURES,
    embedding_drift_mean,
    margin_distance,
    orbit_logit_variance,
    orbit_support_distance_max,
)


def patterned_image(mode: str = "RGB", size: tuple[int, int] = (17, 19)) -> Image.Image:
    rng = np.random.default_rng(20260916)
    if mode == "L":
        arr = rng.integers(0, 255, size=(size[1], size[0]), dtype=np.uint8)
    elif mode == "RGBA":
        arr = rng.integers(0, 255, size=(size[1], size[0], 4), dtype=np.uint8)
    else:
        arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


def test_phase4_orbit_has_exact_five_views_in_order() -> None:
    views = default_orbit_views()
    assert tuple(view.view_name for view in views) == VIEW_ORDER
    assert VIEW_ORDER == (
        "identity",
        "jpeg_q75",
        "resize_075_restore",
        "gaussian_blur_sigma_05",
        "center_crop_090_restore",
    )
    assert len({transform_chain_id(view) for view in views}) == 5


def test_transformations_are_deterministic_for_odd_small_grayscale_and_rgba() -> None:
    for image in [patterned_image("RGB"), patterned_image("L", (9, 11)), patterned_image("RGBA", (13, 7))]:
        first = [transformed_pixel_sha256(view) for view in build_default_orbit(image)]
        second = [transformed_pixel_sha256(apply_view(image, view_name)) for view_name in VIEW_ORDER]
        assert first == second
        assert all(len(digest) == 64 for digest in first)
        assert all(view.mode == "RGB" for view in build_default_orbit(image))


def test_view_id_is_stable_and_parent_context_specific() -> None:
    view = default_orbit_views()[1]
    chain = transform_chain_id(view, ORBIT_VERSION)
    left = make_view_id("phase4::split_a::risk_fit::sample", "abc", chain)
    right = make_view_id("phase4::split_b::risk_fit::sample", "abc", chain)
    assert left == make_view_id("phase4::split_a::risk_fit::sample", "abc", chain)
    assert left != right


def test_reliability_formulas_match_phase4_spec() -> None:
    logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    embeddings = np.eye(5, dtype=np.float64)
    support = np.array([0.1, 0.2, 0.05, 0.3, 0.25], dtype=np.float64)
    assert margin_distance(-2.0, -1.5) == pytest.approx(0.5)
    assert orbit_logit_variance(logits) == pytest.approx(np.var(logits, ddof=0))
    assert embedding_drift_mean(embeddings) == pytest.approx(1.0)
    assert orbit_support_distance_max(support) == pytest.approx(0.3)
    assert PRIMARY_FEATURES == (
        "margin_distance",
        "orbit_logit_variance",
        "mean_directional_erosion",
        "worst_view_erosion",
    )
    assert LEGACY_FULL_FOUR_FEATURES == (
        "margin_distance",
        "orbit_logit_variance",
        "embedding_drift_mean",
        "orbit_support_distance_max",
    )


def test_reliability_rejects_missing_views_nan_inf_and_zero_norms() -> None:
    with pytest.raises(ValueError):
        orbit_logit_variance(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        orbit_support_distance_max(np.array([1.0, np.nan, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError):
        embedding_drift_mean(np.zeros((5, 3), dtype=np.float64))
    with pytest.raises(ValueError):
        embedding_drift_mean(np.ones((4, 3), dtype=np.float64))


def test_constant_logits_have_zero_population_variance() -> None:
    assert orbit_logit_variance(np.ones(5, dtype=np.float64) * 7.0) == pytest.approx(0.0)
