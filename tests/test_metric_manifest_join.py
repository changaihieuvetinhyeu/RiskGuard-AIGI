import pandas as pd


def test_metric_manifest_join_one_to_one_synthetic():
    manifest = pd.DataFrame({"sample_id": ["a", "b"], "label": [0, 1]})
    scores = pd.DataFrame({"sample_id": ["a", "b"], "risk_score": [0.1, 0.2]})
    merged = manifest.merge(scores, on="sample_id", validate="one_to_one")
    assert len(merged) == len(manifest)
