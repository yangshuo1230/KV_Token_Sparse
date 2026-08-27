import numpy as np

from kvstudy.semantic.metrics import coverage, pair_metrics


def test_pair_metrics_identical():
    p = np.array([0.1, 0.2, 0.7])
    result = pair_metrics(p, p, 2)
    assert result["jaccard"] == 1
    assert np.isclose(result["js_similarity"], 1)
    assert np.isclose(result["spearman"], 1)


def test_coverage():
    route = np.array([0.1, 0.8, 0.1])
    targets = np.array([[0.2, 0.7, 0.1], [0.0, 0.9, 0.1]])
    assert np.isclose(coverage(route, targets, 1), 0.8)
