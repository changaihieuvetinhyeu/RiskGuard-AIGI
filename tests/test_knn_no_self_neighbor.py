import numpy as np

from selective_detection.selective_baselines import exact_knn_distance


def test_knn_removes_self_neighbor_when_ids_match():
    x = np.eye(4, dtype="float32")
    scored = exact_knn_distance(x, x, 1, bank_ids=np.array(["a", "b", "c", "d"]), query_ids=np.array(["a", "b", "c", "d"]), device="cpu")
    assert np.all(scored["neighbor_index_1"] != np.arange(4))
