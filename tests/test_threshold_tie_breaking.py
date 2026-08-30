import numpy as np

from selective_detection.selective_thresholds import select_global_threshold


def test_tie_breaking_is_deterministic_for_identical_risks():
    risks = np.array([0.2, 0.2, 0.2, 0.2])
    errors = np.array([0, 0, 0, 0])
    ids = np.array(["d", "c", "b", "a"])
    left, _ = select_global_threshold(risks, errors, ids, alpha=0.99, delta=0.05)
    right, _ = select_global_threshold(risks, errors, ids[::-1], alpha=0.99, delta=0.05)
    assert left.threshold == right.threshold == 0.2
    assert left.accepted_count == right.accepted_count == 4
