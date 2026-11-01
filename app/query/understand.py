# Prompt adapted from HKUDS/LightRAG (MIT License, see third_party/LICENSES/LightRAG-MIT.txt),
# commit aaebfe1bb8d5, lightrag/prompt.py PROMPTS["keywords_extraction"].
# Modifications: one call now also rewrites the question with the conversation history
# (LightRAG never shows history to keyword extraction, operate.py:4658), routes to
# libraries, and adds `candidate_apis`. LightRAG forbids inferring terms not in the query;
# that rule is kept for `keywords` but deliberately lifted for `candidate_apis`, because
# the commonest retrieval failure here is a question that never names the API
# ("delete everything in Redis" -> FLUSHALL).
"""Query understanding: one structured LLM call before retrieval."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.models.llm import LLM
from app.tracing import spans

PROMPT = """---Role---
You prepare a learner's question for retrieval over official technical documentation (written in English).

---Goal---
Return one JSON object with exactly these keys:
- "standalone_question": the question rewritten so it can be understood without the conversation history (resolve "it", "that command", ...). Keep the learner's language.
- "search_query_en": a concise English search query for the documentation.
- "keywords": up to 8 terms that appear in, or are directly stated by, the question and history (API names, error messages, option names, concepts). Do not invent anything here.
- "candidate_apis": up to 5 exact names of commands, functions, classes, config options or SQL statements that most likely answer the question, even if the learner did not name them (e.g. "delete everything in redis" -> ["FLUSHALL", "FLUSHDB"]). Use the documentation's spelling. Empty if unsure.
- "libraries": the subset of {libraries} the question is about; empty if it could be any.
- "needs_retrieval": false only for greetings or questions about the conversation itself.

---Rules---
1. Output only the JSON object: no markdown fences, no comments, no text before or after.
2. Keep lists short and high-signal; no duplicates.
3. For empty or meaningless input return empty strings and lists and "needs_retrieval": false.

---Conversation history (oldest first)---
{history}

---Question---
{question}"""


@dataclass
class Understanding:
    standalone_question: str
    search_query_en: str
    keywords: list[str] = field(default_factory=list)
    candidate_apis: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    needs_retrieval: bool = True

    def bm25_text(self) -> str:
        """Lexical query: exact API names first, they are what BM25 is good at matching."""
        return " ".join(self.candidate_apis + self.keywords + [self.search_query_en])


def _strings(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(v).strip() for v in value if str(v).strip()))[:limit]


def understand(llm: LLM, question: str, history: list[dict] | None = None,
               libraries: list[str] | None = None) -> Understanding:
    turns = "\n".join(f"{t['role']}: {t['content']}" for t in (history or [])[-6:]) or "(none)"
    known = sorted(libraries or [])
    prompt = PROMPT.format(libraries=json.dumps(known), history=turns, question=question)
    with spans.span("llm", "query_understanding") as s:
        raw = llm.chat_json([{"role": "user", "content": prompt}])
        result = Understanding(
            standalone_question=str(raw.get("standalone_question") or question),
            search_query_en=str(raw.get("search_query_en") or raw.get("standalone_question") or question),
            keywords=_strings(raw.get("keywords"), 8),
            candidate_apis=_strings(raw.get("candidate_apis"), 5),
            # Only libraries we actually index; a hallucinated id would filter everything out.
            libraries=[lib for lib in _strings(raw.get("libraries"), 5) if lib in known],
            needs_retrieval=bool(raw.get("needs_retrieval", True)),
        )
        s.set(candidate_apis=result.candidate_apis, libraries=result.libraries)
    return result
