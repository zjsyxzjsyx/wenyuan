from types import SimpleNamespace

from app.generation.answer import REFUSAL, answer, check_citations
from app.retrieval.context import Source, _subtree_text, build_sources
from test_fixed_token import CharTokenizer

SOURCES = [Source("S1", "u1", "FLUSHALL", "Redis > FLUSHALL", "Deletes all keys.", 17),
           Source("S2", "u2", "FLUSHDB", "Redis > FLUSHDB", "Deletes keys of the current db.", 31)]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat_json(self, messages, temperature=0.0):
        return self.reply


def test_check_citations_drops_unknown_ids():
    text, valid, dropped = check_citations("Use FLUSHALL [S1][S9]. Or FLUSHDB [S2].", SOURCES)
    assert text == "Use FLUSHALL [S1]. Or FLUSHDB [S2]."
    assert valid == ["S1", "S2"] and dropped == ["S9"]


def test_answer_with_valid_citations():
    a = answer(FakeLLM({"found": True, "answer": "用 FLUSHALL [S1]。", "citations": ["S1"]}), "q", SOURCES)
    assert a.found and a.citations[0]["url"] == "u1"


def test_answer_without_any_valid_citation_becomes_refusal():
    a = answer(FakeLLM({"found": True, "answer": "Just restart it [S7].", "citations": ["S7"]}), "q", SOURCES)
    assert not a.found and a.text == REFUSAL and a.dropped_citations == ["S7"]


def test_answer_refuses_without_sources():
    assert answer(FakeLLM({}), "q", []).text == REFUSAL


def test_subtree_text_keeps_child_headings_in_order():
    s = lambda idx, parent, level, heading, text: SimpleNamespace(idx=idx, parent_idx=parent, level=level,
                                                                  heading=heading, text=text)
    sections = {1: s(1, None, 2, "Top", "intro"), 2: s(2, 1, 3, "A", "a text"), 3: s(3, 1, 3, "B", "b text")}
    assert _subtree_text(sections, {1: [2, 3]}, 1) == "intro\n\n### A\n\na text\n\n### B\n\nb text"


def test_build_sources_respects_budget_and_truncates_first():
    hit = {"url": "u", "heading_path": "h"}
    blocks = [(hit, "t", "x" * 50), (hit, "t", "y" * 30), (hit, "t", "z" * 10)]
    sources = build_sources(CharTokenizer(), blocks, budget=40)
    assert [s.sid for s in sources] == ["S1"] and sources[0].tokens == 40
    sources = build_sources(CharTokenizer(), blocks[1:], budget=40)
    assert [s.text[0] for s in sources] == ["y", "z"]
