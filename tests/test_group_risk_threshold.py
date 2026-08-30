from selective_detection.group_risk_threshold import clopper_pearson_upper, select_group_controlled_threshold


def test_cp_upper_handles_zero_accepted():
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0


def test_cp_upper_is_one_when_all_accepted_samples_are_errors():
    assert clopper_pearson_upper(3, 3, 0.05) == 1.0


def test_threshold_selects_none_when_group_bound_cannot_be_met():
    result = select_group_controlled_threshold(
        risks=[0.1, 0.2, 0.3, 0.4],
        errors=[1, 1, 1, 1],
        groups=["clean", "clean", "blur", "blur"],
        alpha=0.05,
        delta=0.05,
    )
    assert result.tau is None
    assert result.coverage == 0.0


def test_threshold_finds_feasible_low_risk_prefix():
    result = select_group_controlled_threshold(
        risks=[0.01, 0.02, 0.90, 0.01, 0.02, 0.90],
        errors=[0, 0, 1, 0, 0, 1],
        groups=["clean", "clean", "clean", "blur", "blur", "blur"],
        alpha=0.90,
        delta=0.05,
    )
    assert result.tau == 0.02
    assert result.coverage == 4 / 6
