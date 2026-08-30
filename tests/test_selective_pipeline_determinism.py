import numpy as np


def test_float32_score_tolerance_constant():
    left = np.array([0.1, 0.2], dtype="float32")
    right = left + np.array([1e-7, -1e-7], dtype="float32")
    assert np.allclose(left, right, atol=1e-6, rtol=0.0)
