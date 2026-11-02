"""HyDE (Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels"):
search with a passage the LLM imagines the answer to be, not only with the question.

In the SO set 9% of questions have no gold page among the top 100 chunks, typically
because the answer is a feature the question never names ("run a thread in the
background" is answered by lifespan events). Query understanding already guesses API
names for BM25 (`candidate_apis`); a hypothetical documentation passage gives the dense
side the same vocabulary, written the way the docs would put it. The passage may be
wrong; it is only a query, never shown or cited.
"""
from __future__ import annotations

from app.models.llm import LLM
from app.tracing import spans

PROMPT = """Write the passage of the official {library} documentation that would answer the question below.
Name the specific features, functions, commands, options or settings the solution uses, the way the
documentation would describe them. English, 60-120 words, no code longer than one line.

Question:
{question}

Return one JSON object: {{"passage": "..."}}"""


def hypothetical_passage(llm: LLM, question: str, library: str = "") -> str:
    with spans.span("stage", "hyde", library=library) as s:
        raw = llm.chat_json([{"role": "user", "content": PROMPT.format(
            library=library or "software", question=question)}])
        passage = str(raw.get("passage") or "").strip()
        s.set(chars=len(passage))
        return passage
