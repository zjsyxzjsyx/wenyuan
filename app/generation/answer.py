"""`ask`: a short answer grounded in numbered sources, with citations checked by code.

The model may only use the sources it is given and must cite them as [S1], [S2]...
Code then drops citations to sources that do not exist (a fabricated [S9]); an
answer left with no valid citation is turned into a refusal, so nothing reaches
the learner unsupported. Citations count from both the inline markers and the
"citations" field: Qwen3-8B, answering in Chinese, often fills only the field,
and reading only the markers turned correct answers into refusals. Checking that each cited source really supports its
sentence (claim-level verification) is a separate step planned for W4.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.llm import LLM
from app.retrieval.context import Source
from app.tracing import spans

PROMPT = """You answer a learner's question about technical documentation.

---Rules---
1. Use only the sources below. If they do not contain the answer, set "found" to false and say so briefly.
2. Answer concisely in the learner's language; keep code, commands and API names as written in the sources.
3. After every sentence that states a fact, cite its source as [S1], [S2], ... (one or more).
4. Output one JSON object: {{"found": true|false, "answer": "...", "citations": ["S1", ...]}} and nothing else.

---Sources---
{sources}

---Question---
{question}"""

REFUSAL = "资料里没有找到能回答这个问题的内容。"
_CITE = re.compile(r"\[(S\d+)\]")


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
        raw = llm.chat_json([{"role": "user", "content": PROMPT.format(sources=format_sources(sources),
                                                                       question=question)}])
        text, valid, dropped = merge_citations(str(raw.get("answer", "")), raw.get("citations"), sources)
        found = bool(raw.get("found")) and bool(valid)
        s.set(found=found, citations=len(valid), dropped_citations=dropped)
    if not found:
        return Answer(False, text if raw.get("found") is False and text else REFUSAL, dropped_citations=dropped)
    by_id = {src.sid: src for src in sources}
    return Answer(True, text, [{"sid": c, "url": by_id[c].url, "title": by_id[c].title,
                                "heading_path": by_id[c].heading_path, "section_id": by_id[c].section_id}
                               for c in valid], dropped)


def apply_verification(result: Answer, verified) -> Answer:
    """The answer after claim-level checking: unsupported sentences removed, citations narrowed
    to those still in the text (all of them if the kept sentences carry no markers)."""
    record = {"counts": verified.counts, "removed": verified.removed}
    if not verified.removed:
        return Answer(result.found, result.text, result.citations, result.dropped_citations, record)
    if not verified.text.strip():
        return Answer(False, REFUSAL, [], result.dropped_citations, record)
    still = set(_CITE.findall(verified.text))
    citations = [c for c in result.citations if c["sid"] in still] or result.citations
    return Answer(True, verified.text, citations, result.dropped_citations, record)
