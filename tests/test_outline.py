from types import SimpleNamespace

from app.query import outline as o
from app.query.understand import Understanding


class FakeLLM:
    def chat_json(self, messages, temperature=0.0):
        self.prompt = messages[0]["content"]
        return {"search_query_en": "Redis persistence", "candidate_apis": ["BGSAVE"]}


def test_topic_docs_list_guides_before_reference_pages(monkeypatch):
    hits = [{"doc_id": "persist-cmd", "doc_type": "reference"}, {"doc_id": "persistence", "doc_type": "guide"},
            {"doc_id": "persist-cmd", "doc_type": "reference"}, {"doc_id": "bgsave", "doc_type": "reference"},
            {"doc_id": "replication", "doc_type": "guide"}]
    seen = {}
    monkeypatch.setattr(o.milvus_store, "hybrid_search",
                        lambda mc, coll, vec, text, top_k, libraries: seen.update(text=text) or hits)
    llm = FakeLLM()
    models = SimpleNamespace(embed=lambda texts: [[0.0]])
    docs = o.topic_docs(llm, None, models, "c", "redis", "持久化", max_docs=3)
    assert docs == ["persistence", "replication", "persist-cmd"]
    assert "系统学习 Redis 的「持久化」" in llm.prompt          # asked as a learning goal, not a lookup
    assert seen["text"].startswith("BGSAVE")                     # BM25 gets the inferred API names


def test_section_ids_must_name_a_section():
    class DB:
        def get(self, model, key):
            return None
    for bad in ("", "abc", "abc:", "abc:x", ":3"):
        assert o.get_section(DB(), bad) is None


def test_private_sections_are_hidden_from_other_users():
    doc = SimpleNamespace(library="private", owner_id=1)
    assert o._visible(doc, 1) and not o._visible(doc, 2) and not o._visible(None, 1)
    assert o._visible(SimpleNamespace(library="redis", owner_id=None), 2)


def test_course_documents_are_visible_only_inside_their_course():
    doc = SimpleNamespace(library="cs50", owner_id=None)
    assert o._visible(doc, 2, ["cs50"]) and not o._visible(doc, 2, ["algo"]) and not o._visible(doc, 2, [])
    assert o._visible(doc, 2)                                     # no course limit (eval, CLI)
    note = SimpleNamespace(library="private", owner_id=2)
    assert o._visible(note, 2, []) and not o._visible(note, 3, ["cs50"])
