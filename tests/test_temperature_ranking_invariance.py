import numpy as np

from selective_detection.selective_baselines import msp_risk, sigmoid_prob, temp_msp_risk


def test_positive_temperature_preserves_binary_msp_ranking():
    logits = np.array([-3.0, -1.0, 0.2, 2.0, 5.0])
    base = np.argsort(msp_risk(sigmoid_prob(logits)))
    scaled = np.argsort(temp_msp_risk(logits, 3.0)[0])
    assert np.array_equal(base, scaled)
