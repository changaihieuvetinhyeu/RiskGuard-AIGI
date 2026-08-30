import pandas as pd

from selective_detection.selective_metrics import sha256_deduplicate


def test_sha256_dedup_keeps_lexicographically_smallest_sample_id():
    df = pd.DataFrame({"sha256": ["x", "x", "y"], "sample_id": ["b", "a", "c"], "label": [0, 0, 1]})
    kept, mapping = sha256_deduplicate(df)
    assert kept["sample_id"].tolist() == ["a", "c"]
    assert mapping[mapping["alias_sample_id"] == "a"]["is_canonical"].iloc[0]
    assert not mapping[mapping["alias_sample_id"] == "b"]["is_canonical"].iloc[0]
