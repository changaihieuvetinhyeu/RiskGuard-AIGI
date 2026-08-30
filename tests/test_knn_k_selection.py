import numpy as np

from selective_detection.selective_baselines import select_knn_k_cv


def test_knn_k_selection_returns_candidate():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 5)).astype("float32")
    errors = np.array([0, 1] * 15)
    selected, cv = select_knn_k_cv(x, errors, np.array([str(i) for i in range(30)]), [1, 5], 3, 42, device="cpu")
    assert selected in {1, 5}
    assert set(cv["k"]) == {1, 5}
