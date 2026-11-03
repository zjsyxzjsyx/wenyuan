"""Listwise page rerank by an LLM.

The candidate pool already holds most answers (86% of SO gold pages are among the
pages of the top 100 chunks, eval R0) but only about half reach the top 5. The bge cross-encoder
made that worse once query understanding was on (E3/E6): it scores how much a chunk
looks like the question, and favours API reference pages dense with the question's
words. Many misses are answered by a feature the question never names ("run a thread
in the background" -> lifespan events), which takes knowing the library, so here an
LLM sees the question and one excerpt per candidate page and orders the pages itself.
"""
from __future__ import annotations

from typing import Any

from app.models.llm import LLM
from app.tokenizer import TiktokenTokenizer

# The question comes first and again last: if a server ever cuts a long prompt, it drops the
# beginning (Ollama keeps the last tokens), and a model that never sees the question ranks blind.
PROMPT = """You are ranking passages (course materials, documentation or a learner's own notes) for a
question about {library}.

Question:
{question}

Candidates, each as [id] where it is, then an excerpt:
{pages}

Which candidates would most help someone write the correct answer? The solution may rely on a
feature or command the question does not mention, so think about what the fix actually is
before matching words.

Question (again): {question}

Return one JSON object: {{"ranking": [ids, most useful first]}} with the {keep} most useful ids."""


def _pages(hits: list[dict[str, Any]], unit: str = "page") -> list[list[dict[str, Any]]]:
    """Hits grouped by page, or by section ("section": the page title, its top heading and the one
    below, so a course lecture of over 100,000 characters becomes its subsections; grouping by the
    top heading alone did nothing, since every CS50 file has a single H1), in order of their best hit."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        key = hit["url"] if unit == "page" else f"{hit['url']}#{' > '.join(hit['heading_path'].split(' > ')[:3])}"
        groups.setdefault(key, []).append(hit)
    return list(groups.values())


_TOKENIZER: TiktokenTokenizer | None = None


def _tokenizer() -> TiktokenTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = TiktokenTokenizer()
    return _TOKENIZER


def _describe(group: list[dict[str, Any]], excerpt_chars: int, second_chars: int, more_headings: int) -> str:
    """A page as the model sees it: where the best hit is, its opening, a second matching section and
    the names of other sections that matched. The best hit alone often missed the part that answers."""
    best = group[0]
    lines = [best["heading_path"], best["text"][:excerpt_chars]]
    others = [h for h in group[1:] if h["heading_path"] != best["heading_path"]]
    if second_chars and others:
        lines.append(f"... {others[0]['heading_path'].rsplit(' > ', 1)[-1]}: {others[0]['text'][:second_chars]}")
    names = list(dict.fromkeys(h["heading_path"].rsplit(" > ", 1)[-1] for h in others[1:]))[:more_headings]
    if names:
        lines.append("Also matched: " + " | ".join(names))
    return "\n".join(lines)


def rerank_pages(llm: LLM, question: str, hits: list[dict[str, Any]], library: str = "", top_pages: int = 20,
                 keep: int = 10, excerpt_chars: int = 300, second_chars: int = 150, more_headings: int = 3,
                 max_tokens: int = 5000, unit: str = "page") -> list[dict[str, Any]]:
    """`hits` reordered so the pages (or sections, `unit="section"`) the LLM ranks come first;
    the ones it leaves out keep their order after them.

    `max_tokens` keeps the candidate listing inside a local model's context (8,192 tokens, which
    the instructions, the question twice and the reply share). It is counted in tokens, not
    characters: Chinese notes run 1-1.5 characters per token, so a character budget meant for
    English let prompts reach three times the context (M2). Excerpts shrink first; if the
    listing still does not fit, the last candidates are dropped."""
    pages = _pages(hits, unit)
    head = pages[:top_pages]
    if len(head) < 2:
        return hits
    tokens = _tokenizer()
    while True:
        listing = "\n\n".join(f"[{i + 1}] {_describe(group, excerpt_chars, second_chars, more_headings)}"
                              for i, group in enumerate(head))
        if len(tokens.encode(listing)) <= max_tokens or len(head) <= 2:
            break
        if excerpt_chars > 80:
            excerpt_chars, second_chars = int(excerpt_chars * 0.75), int(second_chars * 0.75)
        else:
            head = head[:-1]
    raw = llm.chat_json([{"role": "user", "content": PROMPT.format(
        library=library or "a software library", question=question, pages=listing, keep=min(keep, len(head)))}])
    order: list[int] = []
    for item in raw.get("ranking") or []:
        try:
            i = int(item) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(head) and i not in order:
            order.append(i)
    ranked = [head[i] for i in order] + [g for i, g in enumerate(head) if i not in order] + pages[len(head):]
    return [hit for group in ranked for hit in group]
