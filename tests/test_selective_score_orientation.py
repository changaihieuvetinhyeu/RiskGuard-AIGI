import numpy as np

from selective_detection.selective_baselines import entropy_risk, msp_risk


def test_higher_score_is_more_uncertain_for_confidence_scores():
    probs = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    msp = msp_risk(probs)
    ent = entropy_risk(probs)
    assert msp[2] == msp.max()
    assert ent[2] == ent.max()
    assert np.isclose(msp[0], 0.01)
    assert np.isclose(msp[-1], 0.01)
