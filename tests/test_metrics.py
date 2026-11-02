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


def test_evidence_matches_overlapping_text_whatever_the_chunk_boundaries():
    from eval.metrics import evidence_ranks, overlaps
    passage = "Binary search halves the problem at each step, so it runs in logarithmic time on sorted data."
    inside = "Recall that binary search halves the problem at each step, so it runs in logarithmic time."
    assert overlaps(inside, passage) and not overlaps("Selection sort repeatedly finds the minimum.", passage)
    note = "滑动窗口适合求最长无重复子串，用哈希表记录每个字符上次出现的位置，左指针跳到重复位置之后。"
    assert overlaps("## 核心思路\n滑动窗口适合求最长无重复子串，用哈希表记录每个字符上次出现的位置", note)
    assert not overlaps("最长无重复子串可以用滑动窗口", note)   # a rewording is not the same evidence
    assert evidence_ranks(["unrelated text about sorting algorithms here", inside], [passage, "missing evidence text here"]) == [1, None]


def test_evidence_metrics_ignore_the_context_header():
    from eval.run import body_of, evidence_metrics
    assert body_of("CS50x 2024 > Week 3 · Lecture 3 > Binary Search\n\nHalve it.") == "Halve it."
    assert body_of("No header here.\n\nSecond paragraph.") == "No header here.\n\nSecond paragraph."
    scores = evidence_metrics(["a b c d e f g h"], ["x y z w v u t", "a b c d e f g h i"])
    assert (scores["ev_hit@5"], scores["ev_mrr"], scores["ev_ranks"]) == (1.0, 0.5, [1])
    assert evidence_metrics(None, ["anything"]) == {}


def test_private_note_urls_never_reach_the_results_file():
    from eval.run import redact
    assert redact("https://cs50.harvard.edu/x/2024/notes/3/") == "https://cs50.harvard.edu/x/2024/notes/3/"
    hidden = redact("private://7/周记_示例.md")
    assert hidden.startswith("private:") and "周记" not in hidden and hidden == redact("private://7/周记_示例.md")
