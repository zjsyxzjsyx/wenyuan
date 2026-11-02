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


def test_paired_bootstrap_detects_consistent_gain_only():
    from eval.compare import paired_bootstrap
    a = [0.0, 1.0] * 50
    delta, lo, hi = paired_bootstrap(a, [x + 0.1 for x in a], n=500)
    assert round(delta, 6) == 0.1 and lo > 0                       # every question improves: significant
    delta, lo, hi = paired_bootstrap(a, a[1:] + a[:1], n=500)        # same scores, shuffled: no effect
    assert lo <= 0 <= hi
