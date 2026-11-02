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


def test_expand_steps_down_from_oversized_parent():
    from app.retrieval.context import _children, expand
    s = lambda idx, parent, level, heading, text: SimpleNamespace(idx=idx, parent_idx=parent, level=level,
                                                                  heading=heading, text=text)
    sections = {0: s(0, None, 1, "Page", "p" * 30), 1: s(1, 0, 2, "FLUSHALL", "f" * 20), 2: s(2, 0, 2, "Other", "o" * 50)}
    children = _children(sections)
    hit = {"doc_id": "d", "parent_index": 0, "section_index": 1, "chunk_id": "c", "text": "chunk"}
    assert expand(CharTokenizer(), hit, sections, children, 1000)[0] == ("d", 0)             # whole parent fits
    key, text = expand(CharTokenizer(), hit, sections, children, 40)
    assert key == ("d", 1) and text == "f" * 20                                               # its own section
    assert expand(CharTokenizer(), hit, sections, children, 10) == (("d", "chunk", "c"), "chunk")


def test_citations_field_counts_when_text_has_no_markers():
    a = answer(FakeLLM({"found": True, "answer": "用 dropna(subset=...)。", "citations": ["S2", "S8"]}), "q", SOURCES)
    assert a.found and a.text == "用 dropna(subset=...)。 [S2]"
    assert [c["sid"] for c in a.citations] == ["S2"] and a.dropped_citations == ["S8"]



class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def chat_json(self, messages, temperature=0.0):
        self.prompts.append(messages[0]["content"])
        return self.reply


def test_claims_split_on_sentences_not_on_code_dots():
    from app.generation.verify import split_claims
    claims = split_claims("用 `df.dropna(subset=[\"a\"])` 删除。Python 3.14 也支持 [S1][S2]. 好的 [S3]")
    assert [c.cites for c in claims] == [[], ["S1", "S2"], ["S3"]]
    assert claims[0].text.startswith("用 `df.dropna(")


def test_unsupported_claims_are_removed_and_citations_narrowed():
    from app.generation.answer import Answer, apply_verification
    from app.generation.verify import verify
    text = "用 FLUSHALL 删除所有 key [S1]。它会顺便重启服务器 [S2]。"
    llm = ScriptedLLM({"claims": [{"id": "C1", "verdict": "supported"}, {"id": "C2", "verdict": "unsupported"}]})
    v = verify(llm, text, SOURCES)
    assert v.counts == {"supported": 1, "partial": 0, "unsupported": 1}
    assert "Deletes keys of the current db." in llm.prompts[0]                 # cited sources are shown
    before = Answer(True, text, [{"sid": "S1"}, {"sid": "S2"}])
    after = apply_verification(before, v)
    assert after.text == "用 FLUSHALL 删除所有 key [S1]。" and [c["sid"] for c in after.citations] == ["S1"]
    assert after.verification["removed"] == ["它会顺便重启服务器 [S2]。"]


def test_answer_with_no_supported_claim_becomes_refusal_and_skipped_claims_are_kept():
    from app.generation.answer import Answer, apply_verification
    from app.generation.verify import verify
    v = verify(ScriptedLLM({"claims": [{"id": "C1", "verdict": "unsupported"}]}), "瞎编的 [S1]。", SOURCES)
    assert apply_verification(Answer(True, "瞎编的 [S1]。", [{"sid": "S1"}]), v).text == REFUSAL
    kept = verify(ScriptedLLM({"claims": []}), "没被判的一句 [S1]。", SOURCES)    # checker said nothing
    assert kept.removed == [] and kept.counts["supported"] == 1
