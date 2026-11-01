import pytest

from eval.metrics import hit_at, mrr, ndcg_at, recall_at, unique_in_order


def test_unique_in_order():
    assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_metrics():
    ranked, gold = ["x", "a", "y", "b"], {"a", "b"}
    assert recall_at(ranked, gold, 2) == 0.5
    assert recall_at(ranked, gold, 4) == 1.0
    assert hit_at(ranked, gold, 1) == 0.0 and hit_at(ranked, gold, 2) == 1.0
    assert mrr(ranked, gold) == 0.5
    assert ndcg_at(["a", "b"], gold, 10) == pytest.approx(1.0)
    assert 0 < ndcg_at(ranked, gold, 10) < 1
