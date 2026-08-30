from selective_detection.dataset_manifest_validation import REQUIRED_COLUMNS, validate_manifest_rows


def make_row(sample_id, split, chain, severity, is_external=False):
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "sample_id": sample_id,
            "riskguard_split": split,
            "transformation_chain": chain,
            "severity_tuple": severity,
            "is_external_test": str(is_external).lower(),
        }
    )
    return row


def test_external_samples_cannot_enter_train_or_calibration():
    rows = [make_row("external_1", "calibration", "clean", "0", is_external=True)]
    result = validate_manifest_rows(rows)
    assert not result.ok
    assert result.external_trainlike_rows == ("external_1",)


def test_calibration_and_test_transformation_keys_must_not_overlap():
    rows = [
        make_row("cal_1", "calibration", "resize->jpeg", "medium"),
        make_row("test_1", "unseen_test", "resize->jpeg", "medium"),
    ]
    result = validate_manifest_rows(rows)
    assert not result.ok
    assert result.overlapping_transformations == ("resize->jpeg::medium",)


def test_distinct_transformations_pass_protocol_check():
    rows = [
        make_row("cal_1", "calibration", "resize->jpeg", "medium"),
        make_row("test_1", "unseen_test", "crop->resize->webp", "high"),
    ]
    result = validate_manifest_rows(rows)
    assert result.ok
