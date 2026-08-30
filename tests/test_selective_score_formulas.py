import numpy as np

from selective_detection.selective_baselines import energy_risk, entropy_risk, msp_risk


def test_msp_entropy_energy_boundary_behavior():
    p = np.array([0.0, 0.5, 1.0])
    assert np.allclose(msp_risk(p), [0.0, 0.5, 0.0])
    entropy = entropy_risk(p)
    assert entropy[1] > 0.999999
    assert entropy[0] < 1e-9
    energy = energy_risk(np.array([-10.0, 0.0, 10.0]))
    assert energy[1] > energy[0]
    assert energy[1] > energy[2]
