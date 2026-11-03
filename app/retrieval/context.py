"""Small-to-big context assembly under a token budget.

Retrieval ranks small chunks; the LLM gets each hit's top-level section (its
parent block) in full, read from MySQL, so an answer is not cut off mid-section,
unless that section is too big, in which case it steps down (see `expand`).
Blocks are added in rank order, a parent hit twice is added once, and a block
that does not fit the remaining budget is skipped (the first one is truncated
instead, so there is always some evidence). A trimmed take on LightRAG's
`_apply_token_truncation`: one budget instead of separate entity/relation/chunk ones.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section
from app.tokenizer import Tokenizer


@dataclass
class Source:
    sid: str            # "S1", "S2", ... as cited by the model
    url: str
    title: str
    heading_path: str
    text: str
    tokens: int
    section_id: str = ""     # "<doc_id>:<idx>" of the section the text comes from (GET /v1/sections/{id})


def _subtree_text(sections: dict[int, Section], children: dict[int, list[int]], root: int) -> str:
    parts, stack = [], [root]
    while stack:
        section = sections[stack.pop()]
        if section.text:
            heading = f"{'#' * max(section.level, 1)} {section.heading}\n\n" if section.idx != root else ""
            parts.append(heading + section.text)
        stack.extend(reversed(children.get(section.idx, [])))
    return "\n\n".join(parts)


def _children(sections: dict[int, Section]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = defaultdict(list)
    for s in sorted(sections.values(), key=lambda s: s.idx):
        if s.parent_idx is not None:
            children[s.parent_idx].append(s.idx)
    return children


def expand(tokenizer: Tokenizer, hit: dict, sections: dict[int, Section], children: dict[int, list[int]],
           max_block_tokens: int) -> tuple[tuple, str]:
    """(dedup key, text) of the largest unit around a hit that fits `max_block_tokens`:
    the top-level section, else the hit's own section, else the chunk itself.

    Many pages have a single H1, so the top-level section is the whole page (up to
    110k characters). Taken whole, one such block filled the entire budget and pushed
    out the page that held the answer (FLUSHALL); stepping down keeps blocks small.
    """
    doc = hit["doc_id"]
    for root in dict.fromkeys((hit["parent_index"], hit.get("section_index", -1))):
        if root in sections:
            text = _subtree_text(sections, children, root)
            if len(tokenizer.encode(text)) <= max_block_tokens:
                return (doc, root), text
    return (doc, "chunk", hit["chunk_id"]), hit["text"]


def parent_blocks(db: DBSession, hits: list[dict], tokenizer: Tokenizer,
                  max_block_tokens: int = 1500) -> list[tuple[dict, str, str]]:
    """(hit, document title, block text) for each distinct block, in hit order."""
    doc_ids = {h["doc_id"] for h in hits}
    titles = dict(db.execute(select(Document.id, Document.title).where(Document.id.in_(doc_ids))).all())
    by_doc: dict[str, dict[int, Section]] = defaultdict(dict)
    for section in db.scalars(select(Section).where(Section.doc_id.in_(doc_ids))):
        by_doc[section.doc_id][section.idx] = section
    children = {doc: _children(sections) for doc, sections in by_doc.items()}
    blocks, seen = [], set()
    for hit in hits:
        sections = by_doc.get(hit["doc_id"], {})
        key, text = expand(tokenizer, hit, sections, children.get(hit["doc_id"], {}), max_block_tokens)
        if key not in seen:
            seen.add(key)
            # The section the block stands for: its root, or the hit's own section when only the chunk fit.
            idx = key[1] if key[1] != "chunk" else hit.get("section_index")
            section = sections.get(idx)
            blocks.append(({**hit, "url": (section.url if section is not None and section.url else hit["url"]),
                            "section_id": f"{hit['doc_id']}:{idx}" if section is not None else ""},
                           titles.get(hit["doc_id"], hit["url"]), text))
    return blocks


def build_sources(tokenizer: Tokenizer, blocks: list[tuple[dict, str, str]], budget: int = 6000) -> list[Source]:
    sources: list[Source] = []
    used = 0
    for hit, title, text in blocks:
        tokens = tokenizer.encode(text)
        if used + len(tokens) > budget:
            if sources:
                continue
            tokens = tokens[:budget]
            text = tokenizer.decode(tokens)
        sources.append(Source(f"S{len(sources) + 1}", hit["url"], title, hit.get("heading_path", ""), text, len(tokens),
                              hit.get("section_id", "")))
        used += len(tokens)
    return sources
