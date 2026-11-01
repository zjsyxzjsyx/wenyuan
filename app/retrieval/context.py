"""Small-to-big context assembly under a token budget.

Retrieval ranks small chunks; the LLM gets each hit's top-level section (its
parent block) in full, read from MySQL, so an answer is not cut off mid-section.
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


def _subtree_text(sections: dict[int, Section], children: dict[int, list[int]], root: int) -> str:
    parts, stack = [], [root]
    while stack:
        section = sections[stack.pop()]
        if section.text:
            heading = f"{'#' * max(section.level, 1)} {section.heading}\n\n" if section.idx != root else ""
            parts.append(heading + section.text)
        stack.extend(reversed(children.get(section.idx, [])))
    return "\n\n".join(parts)


def parent_blocks(db: DBSession, hits: list[dict]) -> list[tuple[dict, str, str]]:
    """(hit, document title, parent block text) for each distinct parent, in hit order."""
    doc_ids = {h["doc_id"] for h in hits}
    titles = dict(db.execute(select(Document.id, Document.title).where(Document.id.in_(doc_ids))).all())
    by_doc: dict[str, dict[int, Section]] = defaultdict(dict)
    for section in db.scalars(select(Section).where(Section.doc_id.in_(doc_ids))):
        by_doc[section.doc_id][section.idx] = section
    blocks, seen = [], set()
    for hit in hits:
        key = (hit["doc_id"], hit["parent_index"])
        if key in seen:
            continue
        seen.add(key)
        sections = by_doc.get(hit["doc_id"], {})
        if hit["parent_index"] in sections:
            children: dict[int, list[int]] = defaultdict(list)
            for s in sorted(sections.values(), key=lambda s: s.idx):
                if s.parent_idx is not None:
                    children[s.parent_idx].append(s.idx)
            text = _subtree_text(sections, children, hit["parent_index"])
        else:  # fixed-window chunks have no section; use the chunk itself
            text = hit["text"]
        blocks.append((hit, titles.get(hit["doc_id"], hit["url"]), text))
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
        sources.append(Source(f"S{len(sources) + 1}", hit["url"], title, hit.get("heading_path", ""), text, len(tokens)))
        used += len(tokens)
    return sources
