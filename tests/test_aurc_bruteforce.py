import numpy as np

from selective_detection.selective_metrics import aurc


def test_aurc_matches_manual_prefix_average():
    errors = np.array([1, 0, 1])
    risks = np.array([0.3, 0.1, 0.2])
    expected = np.mean([0 / 1, 1 / 2, 2 / 3])
    assert np.isclose(aurc(errors, risks, np.array(["c", "a", "b"])), expected)
