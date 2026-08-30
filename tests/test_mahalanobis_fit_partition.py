import pandas as pd


def test_fit_registry_partition_rule_synthetic():
    registry = pd.DataFrame({"component": ["temperature scalar"], "source_partition": ["risk_fit"]})
    assert set(registry["source_partition"]) == {"risk_fit"}
