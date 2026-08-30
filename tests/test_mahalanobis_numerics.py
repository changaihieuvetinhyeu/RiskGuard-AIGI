import numpy as np

from selective_detection.selective_baselines import fit_mahalanobis, score_mahalanobis


def test_mahalanobis_distances_are_finite_and_nonnegative():
    rng = np.random.default_rng(7)
    x0 = rng.normal(0, 1, size=(20, 4))
    x1 = rng.normal(2, 1, size=(20, 4))
    x = np.vstack([x0, x1]).astype("float32")
    y = np.array([0] * 20 + [1] * 20)
    stats = fit_mahalanobis(x, y)
    scored = score_mahalanobis(x, stats)
    assert np.isfinite(scored["risk_score"]).all()
    assert (scored["risk_score"] >= 0).all()
