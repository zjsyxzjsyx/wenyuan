from types import SimpleNamespace

from app.generation.answer import REFUSAL, answer, check_citations
from app.retrieval.context import Source, _subtree_text, build_sources
from test_fixed_token import CharTokenizer

SOURCES = [Source("S1", "u1", "FLUSHALL", "Redis > FLUSHALL", "Deletes all keys.", 17),
           Source("S2", "u2", "FLUSHDB", "Redis > FLUSHDB", "Deletes keys of the current db.", 31)]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat_text(self, messages, temperature=0.0):
        return self.reply


def test_check_citations_drops_unknown_ids():
    text, valid, dropped = check_citations("Use FLUSHALL [S1][S9]. Or FLUSHDB [S2].", SOURCES)
    assert text == "Use FLUSHALL [S1]. Or FLUSHDB [S2]."
    assert valid == ["S1", "S2"] and dropped == ["S9"]


def test_answer_with_valid_citations():
    a = answer(FakeLLM("用 FLUSHALL [S1]。\nSOURCES: S1"), "q", SOURCES)
    assert a.found and a.text == "用 FLUSHALL [S1]。" and a.citations[0]["url"] == "u1"


def test_answer_without_any_valid_citation_becomes_refusal():
    a = answer(FakeLLM("Just restart it [S7].\nSOURCES: S7"), "q", SOURCES)
    assert not a.found and a.text == REFUSAL and a.dropped_citations == ["S7"]


def test_answer_refuses_without_sources():
    assert answer(FakeLLM(""), "q", []).text == REFUSAL


def test_code_with_double_quotes_survives_and_not_found_keeps_the_models_reason():
    reply = 'Put shared parts in layout.html, then start each page with `{% extends "layout.html" %}` [S1].\n\nSOURCES: S1'
    a = answer(FakeLLM(reply), "q", SOURCES)
    assert a.found and a.text.endswith('`{% extends "layout.html" %}` [S1].')
    refused = answer(FakeLLM("NOT FOUND: 资料只讲了 FLUSHALL，没有讲备份。"), "q", SOURCES)
    assert not refused.found and refused.text == "资料只讲了 FLUSHALL，没有讲备份。"


def test_refusals_are_recognised_wherever_the_model_puts_them():
    late = "Use a service account [S1].\nNOT FOUND: Information on fine-tuning is not in the sources. [S1]\nSOURCES: S1, S2"
    assert not answer(FakeLLM(late), "q", SOURCES).found
    for prose in ("The sources provided do not contain information on CSRF in Flask. [S1][S2]\nSOURCES: S1, S2",
                  "提供的资料中没有关于如何使用 Unity 创建 3D 游戏的信息。 [S1]\nSOURCES: S1"):
        a = answer(FakeLLM(prose), "q", SOURCES)
        assert not a.found and "[S" not in a.text
    code = 'It prints both [S1].\n```\nFound\nNot found: try again\n```\nSOURCES: S1'
    assert answer(FakeLLM(code), "q", SOURCES).found                 # program output, not a refusal


def test_an_answer_that_loses_most_of_its_claims_becomes_a_refusal():
    from app.generation.answer import Answer, apply_verification
    from app.generation.verify import verify
    text = "用 FLUSHALL [S1]。还会重启服务器 [S2]。并且清空磁盘 [S2]。"
    two_of_three = ScriptedLLM({"claims": [{"id": "C1", "verdict": "supported"}, {"id": "C2", "verdict": "unsupported"},
                                           {"id": "C3", "verdict": "unsupported"}]})
    after = apply_verification(Answer(True, text, [{"sid": "S1"}, {"sid": "S2"}]), verify(two_of_three, text, SOURCES))
    assert not after.found and after.text == REFUSAL


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
    a = answer(FakeLLM("用 dropna(subset=...)。\n**SOURCES:** S2, S8"), "q", SOURCES)
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


def test_claims_do_not_split_inside_code_or_on_a_question_mark_mid_sentence():
    from app.generation.verify import split_claims
    scratch = ("Drag the 'forever' block and place the 'if <touching (mouse pointer)? > then meow' block inside it. "
               "This checks it continuously [S1].")
    assert [c.text for c in split_claims(scratch)] == [
        "Drag the 'forever' block and place the 'if <touching (mouse pointer)? > then meow' block inside it.",
        "This checks it continuously [S1]."]
    code = "Run `sqlite3 x.db`. Then type:\n```\n.tables\n.schema users\n```\nIt lists them, e.g. this one. Done? Yes."
    assert [c.text for c in split_claims(code)] == [
        "Run `sqlite3 x.db`.", "Then type:\n```\n.tables\n.schema users\n```\nIt lists them, e.g. this one.",
        "Done?", "Yes."]


def test_list_numbers_do_not_end_a_claim_but_a_number_ending_a_sentence_does():
    from app.generation.verify import split_claims
    text = "Here's how you can do it: 1. Download the ZIP [S1]. 2. Unzip it [S1]. The result is 42. Done."
    assert [c.text for c in split_claims(text)] == [
        "Here's how you can do it: 1. Download the ZIP [S1].", "2. Unzip it [S1].", "The result is 42.", "Done."]


def test_every_sources_line_is_taken_out_of_the_answer():
    a = answer(FakeLLM("Use FLUSHALL [S1].\nSOURCES: S1\n\nIt deletes every key.\nSOURCES: S1, S2"), "q", SOURCES)
    assert "SOURCES" not in a.text and [c["sid"] for c in a.citations] == ["S1", "S2"]


def test_every_source_is_shown_to_the_checker_not_only_the_cited_ones():
    from app.generation.verify import verify
    llm = ScriptedLLM({"claims": [{"id": "C1", "verdict": "supported"}]})
    verify(llm, "FLUSHDB 只删除当前库 [S1]。", SOURCES)
    assert "Deletes keys of the current db." in llm.prompts[0]          # S2, never cited


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


def test_a_source_links_to_its_own_section_when_the_original_says_where_it_is():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.db.models import Base, Document, Section
    from app.retrieval.context import parent_blocks
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(Document(id="d", library="cs50", url="https://cs50/notes/3/", title="Lecture 3", content_hash="h"))
        db.add_all([Section(doc_id="d", idx=0, parent_idx=None, level=1, heading="Lecture 3", path=["Lecture 3"],
                            text="p" * 30, url="https://cs50/notes/3/"),
                    Section(doc_id="d", idx=1, parent_idx=0, level=2, heading="Binary Search", path=["Lecture 3", "Binary Search"],
                            text="b" * 20, url="https://cs50/notes/3/#binary-search"),
                    Section(doc_id="d", idx=2, parent_idx=0, level=2, heading="Notes", path=["Lecture 3", "Notes"],
                            text="n" * 50)])
        db.commit()
        hit = lambda sec, cid: {"doc_id": "d", "url": "https://cs50/notes/3/", "parent_index": 0, "section_index": sec,
                                "chunk_id": cid, "text": "chunk", "heading_path": "Lecture 3"}
        blocks = parent_blocks(db, [hit(1, "c1"), hit(2, "c2")], CharTokenizer(), max_block_tokens=40)
    (first, _, text), (second, _, _) = blocks
    assert text == "b" * 20 and first["url"].endswith("#binary-search") and first["section_id"] == "d:1"
    assert second["url"] == "https://cs50/notes/3/" and second["section_id"] == "d:2"   # no anchor: the page
    src = build_sources(CharTokenizer(), blocks)[0]
    assert (src.url, src.section_id) == ("https://cs50/notes/3/#binary-search", "d:1")
