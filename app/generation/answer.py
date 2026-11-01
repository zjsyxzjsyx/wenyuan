"""`ask`: a short answer grounded in numbered sources, with citations checked by code.

The model may only use the sources it is given and must cite them as [S1], [S2]...
Code then drops citations to sources that do not exist (a fabricated [S9]); an
answer left with no valid citation is turned into a refusal, so nothing reaches
the learner unsupported. Checking that each cited source really supports its
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
    citations: list[dict] = field(default_factory=list)   # {"sid", "url", "title", "heading_path"}
    dropped_citations: list[str] = field(default_factory=list)


def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.sid}] {s.title} > {s.heading_path}\n{s.text}" for s in sources)


def check_citations(text: str, sources: list[Source]) -> tuple[str, list[str], list[str]]:
    """Remove citation markers that point at no source. Returns (text, valid ids, dropped ids)."""
    known = {s.sid for s in sources}
    cited = list(dict.fromkeys(_CITE.findall(text)))
    dropped = [c for c in cited if c not in known]
    text = _CITE.sub(lambda m: m.group(0) if m.group(1) in known else "", text)
    return re.sub(r"[ \t]+([，。.,;；])", r"\1", text).strip(), [c for c in cited if c in known], dropped


def answer(llm: LLM, question: str, sources: list[Source]) -> Answer:
    if not sources:
        return Answer(False, REFUSAL)
    with spans.span("llm", "answer", sources=len(sources)) as s:
        raw = llm.chat_json([{"role": "user", "content": PROMPT.format(sources=format_sources(sources),
                                                                       question=question)}])
        text, valid, dropped = check_citations(str(raw.get("answer", "")), sources)
        found = bool(raw.get("found")) and bool(valid)
        s.set(found=found, citations=len(valid), dropped_citations=dropped)
    if not found:
        return Answer(False, text if raw.get("found") is False and text else REFUSAL, dropped_citations=dropped)
    by_id = {src.sid: src for src in sources}
    return Answer(True, text, [{"sid": c, "url": by_id[c].url, "title": by_id[c].title,
                                "heading_path": by_id[c].heading_path} for c in valid], dropped)
