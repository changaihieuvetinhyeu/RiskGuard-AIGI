import numpy as np
import pandas as pd

from selective_detection.selective_bootstrap import stratified_unit_bootstrap


def test_bfree_bootstrap_resamples_source_id_units():
    df = pd.DataFrame({"source_id": ["s1", "s1", "s2", "s2"], "label": [0, 0, 1, 1], "value": [0.0, 1.0, 2.0, 3.0]})
    draws = stratified_unit_bootstrap(df, "source_id", ["label"], lambda frame: frame["value"].mean(), 5, 1)
    assert len(draws) == 5
    assert np.isfinite(draws).all()
