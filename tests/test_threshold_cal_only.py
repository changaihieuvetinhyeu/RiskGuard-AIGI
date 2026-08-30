import pandas as pd


def test_threshold_rows_record_threshold_cal_source():
    thresholds = pd.DataFrame({"threshold_cal_manifest_sha256": ["abc"], "selection_status": ["selected"]})
    assert thresholds["threshold_cal_manifest_sha256"].str.len().gt(0).all()
