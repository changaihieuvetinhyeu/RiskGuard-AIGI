import numpy as np

from selective_detection.selective_metrics import aurc, eaurc, optimal_aurc


def test_eaurc_is_aurc_minus_optimal_oracle():
    errors = np.array([1, 0, 1, 0])
    risks = np.array([0.4, 0.1, 0.3, 0.2])
    assert np.isclose(eaurc(errors, risks), aurc(errors, risks) - optimal_aurc(errors))
