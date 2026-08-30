import numpy as np

from selective_detection.selective_baselines import fit_temperature


def test_fit_temperature_is_positive_and_finite():
    logits = np.array([-3.0, -2.0, 2.0, 3.0])
    labels = np.array([0, 0, 1, 1])
    fitted = fit_temperature(logits, labels, 0.05, 20.0)
    assert fitted["success"]
    assert 0.05 <= fitted["temperature"] <= 20.0
