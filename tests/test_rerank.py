import pytest

from app.retrieval.rerank import aggregate_max, rerank, split_for_rerank
from test_fixed_token import CharTokenizer

tok = CharTokenizer()


def test_split_keeps_short_docs_whole_and_windows_long_ones():
    windows, owner = split_for_rerank(tok, ["abc", "abcdefghij"], max_tokens=4, overlap_tokens=1)
    assert windows == ["abc", "abcd", "defg", "ghij"]
    assert owner == [0, 1, 1, 1]


def test_split_rejects_bad_params():
    with pytest.raises(ValueError):
        split_for_rerank(tok, ["a"], max_tokens=0)
    with pytest.raises(ValueError):
        split_for_rerank(tok, ["a"], overlap_tokens=-1)


def test_aggregate_takes_best_window():
    assert aggregate_max([0.1, 0.9, 0.3, 0.2], [0, 1, 1, 2], 3) == [0.1, 0.9, 0.2]


def test_rerank_maps_window_scores_back_to_documents():
    class FakeModels:
        def rerank(self, query, windows, model=None):
            return [1.0 if "x" in w else 0.0 for w in windows]

    scores = rerank(FakeModels(), tok, "q", ["aaaa", "aaaaaaax"], max_tokens=4)
    assert scores == [0.0, 1.0]
