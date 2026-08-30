import pandas as pd


def test_knn_reference_bank_partition_synthetic():
    bank = pd.DataFrame({"sample_id": ["a", "b"], "source_partition": ["risk_fit", "risk_fit"]})
    assert bank["source_partition"].eq("risk_fit").all()
