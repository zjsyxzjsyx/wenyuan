# Adapted from HKUDS/LightRAG (MIT License, see third_party/LICENSES/LightRAG-MIT.txt),
# commit aaebfe1bb8d5, file lightrag/chunker/token_size.py.
# Modifications: removed source-span tracking and the legacy/file-chunker dual entry
# points; replaced lightrag.utils / lightrag.exceptions with local equivalents.
"""Fixed-size token-window chunking: the E0 baseline and the fallback for oversized sections.

Splits text into windows of at most ``chunk_token_size`` tokens, with
``chunk_overlap_token_size`` tokens shared by neighbouring windows. When
``split_by_character`` is given, the text is first cut on that delimiter; each
piece is kept whole if it fits, otherwise windowed (or rejected when
``split_by_character_only`` is set).
"""
from __future__ import annotations

from typing import Any

from app.tokenizer import Tokenizer


class ChunkTokenLimitExceededError(ValueError):
    def __init__(self, chunk_tokens: int, chunk_token_limit: int, chunk_preview: str) -> None:
        super().__init__(
            f"chunk has {chunk_tokens} tokens, limit is {chunk_token_limit}: {chunk_preview!r}"
        )
        self.chunk_tokens = chunk_tokens
        self.chunk_token_limit = chunk_token_limit


def _window_step(chunk_token_size: int, chunk_overlap_token_size: int) -> int:
    # overlap >= size would make range() empty (silently dropping text) or raise on a zero step.
    if chunk_overlap_token_size >= chunk_token_size:
        raise ValueError(
            f"chunk_overlap_token_size ({chunk_overlap_token_size}) must be < "
            f"chunk_token_size ({chunk_token_size})"
        )
    return chunk_token_size - chunk_overlap_token_size


def _windows(
    tokenizer: Tokenizer, tokens: list[int], chunk_token_size: int, chunk_overlap_token_size: int
) -> list[tuple[int, str]]:
    step = _window_step(chunk_token_size, chunk_overlap_token_size)
    return [
        (min(chunk_token_size, len(tokens) - start), tokenizer.decode(tokens[start:start + chunk_token_size]))
        for start in range(0, len(tokens), step)
    ]


def chunk_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    chunk_overlap_token_size: int = 100,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
) -> list[dict[str, Any]]:
    """Return chunks as ``{"tokens", "content", "chunk_order_index"}`` dicts, in order."""
    # Checked before branching so a negative overlap is rejected even when no window is ever built.
    if chunk_overlap_token_size < 0:
        raise ValueError(f"chunk_overlap_token_size must be non-negative, got {chunk_overlap_token_size}")

    pieces: list[tuple[int, str]] = []
    if split_by_character:
        for segment in content.split(split_by_character):
            tokens = tokenizer.encode(segment)
            if len(tokens) <= chunk_token_size:
                pieces.append((len(tokens), segment))
            elif split_by_character_only:
                raise ChunkTokenLimitExceededError(len(tokens), chunk_token_size, segment[:120])
            else:
                pieces.extend(_windows(tokenizer, tokens, chunk_token_size, chunk_overlap_token_size))
    else:
        pieces = _windows(tokenizer, tokenizer.encode(content), chunk_token_size, chunk_overlap_token_size)

    return [
        {"tokens": count, "content": text.strip(), "chunk_order_index": index}
        for index, (count, text) in enumerate(pieces)
    ]
