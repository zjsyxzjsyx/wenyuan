"""`ask`: a short answer grounded in numbered sources, with citations checked by code.

The model may only use the sources it is given and must cite them as [S1], [S2]...
Code then drops citations to sources that do not exist (a fabricated [S9]); an
answer left with no valid citation is turned into a refusal, so nothing reaches
the learner unsupported. Citations count from both the inline markers and the
closing "SOURCES:" line: Qwen3-8B, answering in Chinese, often lists its sources
only there, and reading only the markers turned correct answers into refusals.
Checking that each cited source really supports its sentence is app.generation.verify.

The answer is plain text, not JSON. In JSON mode, a model writing code with an
unescaped double quote (`{% extends "layout.html" %}`) ends the JSON string at that
quote: the answer was cut there, its citations were lost, and code turned it into a
refusal (3 of the 9 refusals of answerable questions in the M3 answer evaluation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.llm import LLM
from app.retrieval.context import Source
from app.tracing import spans

PROMPT = """You answer a student's question using their course materials and notes.

---Rules---
1. Use only the sources below. Write only what they say: do not add steps, numbers or details from
   your own knowledge.
2. Answer concisely in the student's language, but answer every part of the question. Keep code,
   commands and names exactly as written in the sources.
3. After every sentence that states a fact, cite its source as [S1], [S2], ... (one or more).
4. Write plain text (code in backticks or fenced blocks). End with one line listing every source you used:
SOURCES: S1, S3
5. If the sources do not contain the answer, write only this one line, with no answer and no SOURCES line:
NOT FOUND: <one short sentence, in the student's language, on what is missing>

---Sources---
{sources}

---Question---
{question}"""

REFUSAL = "资料里没有找到能回答这个问题的内容。"
_CITE = re.compile(r"\[(S\d+)\]")
_NOT_FOUND = re.compile(r"^[ \t*]*NOT[ _]FOUND\**[ \t]*[:：][ \t]*(.*)$", re.I | re.M)
_SOURCES_LINE = re.compile(r"^[ \t*]*SOURCES?\**[ \t]*[:：](.*)$", re.I | re.M)
_FENCE = re.compile(r"```.*?```", re.S)
# A sentence that only says the materials lack something. Qwen3-8B sometimes refuses in prose
# ("The sources provided do not contain information on ...") instead of the NOT FOUND line.
_LACKS = re.compile(r"\b(?:sources?|materials?|notes?|context)\b[^.]{0,40}?\b(?:do|does)(?: not|n['’]t)\s+"
                    r"(?:contain|include|mention|cover|provide|discuss|explain|say)|\bno information (?:about|on)\b"
                    r"|(?:资料|来源|材料|笔记|课程)[^。]{0,20}?(?:没有|未|不包含|没提到|没有提到|没有涉及)", re.I)
# Verification that removes more than this share of an answer's claims leaves fragments of an
# answer the model should not have given; it becomes a refusal. M3: the three out-of-course
# questions answered this way lost 10 of 11, 12 of 13 and 10 of 11 claims; on 100 answerable
# questions, no fully correct answer lost more than half (a 0.5 cutoff would have refused one).
MAX_REMOVED_SHARE = 0.6


def says_not_found(text: str) -> bool:
    """Every sentence of the text says the materials do not cover the question."""
    from app.generation.verify import split_claims    # verify does not import this module back
    claims = [c for c in split_claims(_CITE.sub("", text)) if c.text.strip()]
    return bool(claims) and all(_LACKS.search(c.text) for c in claims)


def parse_reply(reply: str) -> tuple[bool, str, list[str]]:
    """(found, answer text, listed source ids) from the model's plain-text reply. A NOT FOUND line
    anywhere (outside code) is a refusal: the model sometimes writes a sentence first, or adds a
    SOURCES line after it."""
    refused = _NOT_FOUND.search(_FENCE.sub("", reply))
    if refused:
        return False, _CITE.sub("", refused.group(1)).strip(), []
    # Every SOURCES line goes, not only the last: a second one left in the text became a claim of
    # its own ("SOURCES: S1, S3") that the checker and the judge both found nowhere (M3).
    listed = [sid for m in _SOURCES_LINE.finditer(reply) for sid in re.findall(r"S\d+", m.group(1))]
    return True, _SOURCES_LINE.sub("", reply).strip(), list(dict.fromkeys(listed))


@dataclass
class Answer:
    found: bool
    text: str
    citations: list[dict] = field(default_factory=list)   # {"sid", "url", "title", "heading_path", "section_id"}
    dropped_citations: list[str] = field(default_factory=list)
    verification: dict | None = None    # claim verdict counts and removed sentences (app.generation.verify)


def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.sid}] {s.title} > {s.heading_path}\n{s.text}" for s in sources)


def check_citations(text: str, sources: list[Source]) -> tuple[str, list[str], list[str]]:
    """Remove citation markers that point at no source. Returns (text, valid ids, dropped ids)."""
    known = {s.sid for s in sources}
    cited = list(dict.fromkeys(_CITE.findall(text)))
    dropped = [c for c in cited if c not in known]
    text = _CITE.sub(lambda m: m.group(0) if m.group(1) in known else "", text)
    return re.sub(r"[ \t]+([，。.,;；])", r"\1", text).strip(), [c for c in cited if c in known], dropped


def merge_citations(text: str, listed, sources: list[Source]) -> tuple[str, list[str], list[str]]:
    """check_citations plus the ids in the "citations" field; listed ids missing from the
    text are appended at its end so the learner still sees where the answer came from."""
    text, valid, dropped = check_citations(text, sources)
    known = {s.sid for s in sources}
    listed = [str(c).strip("[] ") for c in listed] if isinstance(listed, list) else []
    extra = [c for c in dict.fromkeys(listed) if c in known and c not in valid]
    dropped += [c for c in dict.fromkeys(listed) if c not in known and c not in dropped]
    if extra:
        text = f"{text} " + "".join(f"[{c}]" for c in extra)
    return text, valid + extra, dropped


def answer(llm: LLM, question: str, sources: list[Source]) -> Answer:
    if not sources:
        return Answer(False, REFUSAL)
    with spans.span("llm", "answer", sources=len(sources)) as s:
        reply = llm.chat_text([{"role": "user", "content": PROMPT.format(sources=format_sources(sources),
                                                                         question=question)}])
        said_found, body, listed = parse_reply(reply)
        if said_found and says_not_found(body):
            said_found, body, listed = False, _CITE.sub("", body).strip(), []
        text, valid, dropped = merge_citations(body, listed, sources)
        found = said_found and bool(valid)
        s.set(found=found, citations=len(valid), dropped_citations=dropped)
    if not found:
        return Answer(False, text if not said_found and text else REFUSAL, dropped_citations=dropped)
    by_id = {src.sid: src for src in sources}
    return Answer(True, text, [{"sid": c, "url": by_id[c].url, "title": by_id[c].title,
                                "heading_path": by_id[c].heading_path, "section_id": by_id[c].section_id}
                               for c in valid], dropped)


def apply_verification(result: Answer, verified) -> Answer:
    """The answer after claim-level checking: unsupported sentences removed, citations narrowed
    to those still in the text (all of them if the kept sentences carry no markers). An answer that
    loses more than MAX_REMOVED_SHARE of its claims, or keeps none, becomes a refusal."""
    record = {"counts": verified.counts, "removed": verified.removed}
    if not verified.removed:
        return Answer(result.found, result.text, result.citations, result.dropped_citations, record)
    claims = sum(verified.counts.values())
    if not verified.text.strip() or len(verified.removed) > MAX_REMOVED_SHARE * claims:
        return Answer(False, REFUSAL, [], result.dropped_citations, record)
    still = set(_CITE.findall(verified.text))
    citations = [c for c in result.citations if c["sid"] in still] or result.citations
    return Answer(True, verified.text, citations, result.dropped_citations, record)
