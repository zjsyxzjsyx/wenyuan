"""Section-aware chunking: one child chunk per section (split further only when too long).

Each chunk records its parent: the top-level section (H2, or the page intro) that
contains it, which is what small-to-big retrieval hands to the LLM. With
`context_header` on, the chunk text starts with "Site > Page > Section" so the
embedding knows where an otherwise ambiguous paragraph comes from (E4 toggles it).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ingest.chunking.fixed_token import chunk_by_token_size
from app.ingest.sections import Section
from app.tokenizer import Tokenizer


@dataclass
class Chunk:
    chunk_id: str
    library: str
    url: str
    section_index: int
    parent_index: int            # top-level section used as the parent block
    heading_path: str
    text: str                    # what gets embedded and indexed
    tokens: int


def top_level(sections: list[Section], index: int) -> int:
    while sections[index].parent_index is not None:
        index = sections[index].parent_index
    return index


def chunk_sections(
    tokenizer: Tokenizer,
    sections: list[Section],
    *,
    library: str,
    site_name: str,
    url: str,
    doc_id: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    context_header: bool = True,
    merge_sections: bool = False,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    units = merge_small_sections(sections, tokenizer, max_tokens - 32) if merge_sections else sections
    for section in units:
        if not section.text:
            continue
        heading_path = " > ".join(section.path)
        header = f"{site_name} > {heading_path}\n\n" if context_header else ""
        budget = max_tokens - len(tokenizer.encode(header))
        pieces = chunk_by_token_size(tokenizer, section.text, chunk_token_size=budget,
                                     chunk_overlap_token_size=min(overlap_tokens, budget // 4),
                                     split_by_character="\n\n")
        for piece_no, piece in enumerate(_merge_small(pieces, budget)):
            text = header + piece
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:{section.index}:{piece_no}",
                library=library, url=url, section_index=section.index,
                parent_index=top_level(sections, section.index), heading_path=heading_path,
                text=text, tokens=len(tokenizer.encode(text)),
            ))
    return chunks


def merge_small_sections(sections: list[Section], tokenizer: Tokenizer, budget: int) -> list[Section]:
    """Fold consecutive sections that share a top-level parent into one, while they fit the budget.

    Bad case behind this (E1 on Redis): a short command page such as TYPE becomes
    five tiny chunks (syntax, return value, examples...) that each carry too little
    meaning to rank, while fixed windows kept the page whole. Merged sections keep
    the path of their first member, minus its own heading when siblings join it.
    """
    merged: list[Section] = []
    joined = False   # whether merged[-1] already absorbed a sibling
    size = 0
    for section in sections:
        if not section.text:
            continue
        own = f"{'#' * max(section.level, 1)} {section.heading}\n\n{section.text}"
        tokens = len(tokenizer.encode(own))
        last = merged[-1] if merged else None
        if last and top_level(sections, last.index) == top_level(sections, section.index) and size + tokens <= budget:
            if not joined:  # first join: label the first member's text with its own heading
                last.text = f"{'#' * max(last.level, 1)} {last.heading}\n\n{last.text}"
                last.path = last.path[:-1] or last.path
                joined = True
            last.text += "\n\n" + own
            size += tokens
        else:
            merged.append(Section(index=section.index, level=section.level, heading=section.heading,
                                  path=list(section.path), text=section.text, parent_index=section.parent_index))
            joined = False
            size = tokens
    return merged


def _merge_small(pieces: list[dict], budget: int) -> list[str]:
    """Re-join consecutive paragraphs while they fit, so a section is not cut at every blank line."""
    merged: list[str] = []
    size = 0
    for piece in pieces:
        if merged and size + piece["tokens"] <= budget:
            merged[-1] += "\n\n" + piece["content"]
            size += piece["tokens"]
        else:
            merged.append(piece["content"])
            size = piece["tokens"]
    return [m for m in merged if m.strip()]


def chunk_fixed(
    tokenizer: Tokenizer, markdown: str, *, library: str, url: str, doc_id: str,
    max_tokens: int = 512, overlap_tokens: int = 64,
) -> list[Chunk]:
    """E0 baseline: fixed token windows over the whole page, no structure."""
    return [
        Chunk(chunk_id=f"{doc_id}:f:{p['chunk_order_index']}", library=library, url=url,
              section_index=-1, parent_index=-1, heading_path="", text=p["content"], tokens=p["tokens"])
        for p in chunk_by_token_size(tokenizer, markdown, chunk_token_size=max_tokens,
                                     chunk_overlap_token_size=overlap_tokens)
    ]
