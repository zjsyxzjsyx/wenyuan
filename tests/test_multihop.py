from app.query import multihop as mh
from app.query.understand import Understanding


def hit(url):
    return {"url": url, "chunk_id": url, "library": "redis", "text": f"text of {url}", "heading_path": url}


class Scripted:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def chat_json(self, messages, temperature=0.0):
        self.calls += 1
        return self.replies.pop(0) if self.replies else {}


U = Understanding("Cache a FastAPI endpoint in Redis", "fastapi redis cache")
FIRST = [hit("a1"), hit("a2"), hit("a3")]


def fake_search(monkeypatch, results):
    searched = []

    def search(mc, models, coll, dense, bm25, libs, top_k, owner):
        searched.append((dense, libs))
        return results[dense]
    monkeypatch.setattr(mh, "_search", search)
    return searched


def test_interleave_takes_each_list_in_turn_and_each_page_once():
    lists = [[hit("a"), hit("b"), hit("c")], [hit("x"), hit("a"), hit("y")]]
    assert [h["url"] for h in mh.interleave(lists, 10)] == ["a", "x", "b", "c", "y"]
    assert len(mh.interleave(lists, 2)) == 2


def test_decompose_searches_each_sub_query_in_its_library(monkeypatch):
    searched = fake_search(monkeypatch, {"fastapi caching": [hit("f1")], "redis set with ttl": [hit("r1")]})
    llm = Scripted({"queries": [{"query": "fastapi caching", "library": "fastapi"},
                                {"query": "redis set with ttl", "library": "redis"}]})
    r = mh.expand(U, FIRST, llm, None, "c", ["fastapi", "redis"], ["fastapi", "redis"], mode="decompose", mc=object())
    assert searched == [("fastapi caching", ["fastapi"]), ("redis set with ttl", ["redis"])]
    assert [h["url"] for h in r.hits][:3] == ["a1", "f1", "r1"]          # each sub-query's best page up front
    assert r.llm_calls == 1


def test_a_single_sub_query_keeps_the_first_search(monkeypatch):
    searched = fake_search(monkeypatch, {})
    r = mh.expand(U, FIRST, Scripted({"queries": [{"query": "cache fastapi in redis", "library": "redis"}]}),
                  None, "c", ["redis"], None, mode="decompose", mc=object())
    assert r.hits is FIRST and searched == []


def test_iterative_stops_when_enough_or_repeating(monkeypatch):
    fake_search(monkeypatch, {"redis ttl": [hit("r1")]})
    llm = Scripted({"enough": False, "query": "redis ttl", "library": "redis"},
                   {"enough": False, "query": "Redis  TTL", "library": "redis"})   # same query again
    r = mh.expand(U, FIRST, llm, None, "c", ["redis"], None, mode="iterative", mc=object())
    assert [h.query for h in r.hops] == [U.standalone_question, "redis ttl"] and llm.calls == 2
    done = mh.expand(U, FIRST, Scripted({"enough": True}), None, "c", ["redis"], None, mode="iterative", mc=object())
    assert done.hits is FIRST and len(done.hops) == 1


def test_iterative_respects_the_token_budget(monkeypatch):
    fake_search(monkeypatch, {})
    llm = Scripted({"enough": False, "query": "more"})
    r = mh.expand(U, FIRST, llm, None, "c", ["redis"], None, mode="iterative", budget_tokens=10, mc=object())
    assert llm.calls == 0 and r.hits is FIRST
