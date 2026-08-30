import numpy as np

from selective_detection.selective_thresholds import select_global_threshold


def test_no_feasible_threshold_when_every_prefix_too_risky():
    risks = np.array([0.1, 0.2, 0.3])
    errors = np.array([1, 1, 1])
    result, _ = select_global_threshold(risks, errors, alpha=0.01, delta=0.05)
    assert result.selection_status == "no_feasible_nonempty_threshold"
    assert result.coverage == 0.0


def test_selects_nonempty_low_risk_prefix():
    risks = np.array([0.1, 0.2, 0.3, 0.4])
    errors = np.array([0, 0, 0, 1])
    result, _ = select_global_threshold(risks, errors, alpha=0.95, delta=0.05)
    assert result.selection_status == "selected"
    assert result.accepted_count >= 1
