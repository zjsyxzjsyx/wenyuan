# Window splitting and score aggregation adapted from HKUDS/LightRAG (MIT License, see
# third_party/LICENSES/LightRAG-MIT.txt), commit aaebfe1bb8d5, file lightrag/rerank.py
# (chunk_documents_for_rerank, aggregate_chunk_scores).
# Modifications: dropped the async wrappers, the character-count fallback and the remote
# provider clients (Cohere/Jina/Ali); scoring goes to the local GPU model service.
"""Cross-encoder rerank, safe for documents longer than the reranker's input window.

A long document (e.g. a parent section in small-to-big) is cut into overlapping
token windows, every window is scored against the query, and the document keeps
its best window score: the relevant paragraph may sit anywhere in it.
"""
from __future__ import annotations

from app.models.local import ModelServer
from app.tokenizer import Tokenizer


def split_for_rerank(tokenizer: Tokenizer, documents: list[str], max_tokens: int = 480,
                     overlap_tokens: int = 32) -> tuple[list[str], list[int]]:
    """Return (windows, owner) where owner[i] is the index of the document window i came from."""
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens must be non-negative, got {overlap_tokens}")
    # overlap >= window would never advance; halving keeps the step at about half a window.
    overlap_tokens = min(overlap_tokens, max_tokens // 2) if overlap_tokens >= max_tokens else overlap_tokens
    windows, owner = [], []
    for index, doc in enumerate(documents):
        tokens = tokenizer.encode(doc)
        if len(tokens) <= max_tokens:
            windows.append(doc)
            owner.append(index)
            continue
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            windows.append(tokenizer.decode(tokens[start:end]))
            owner.append(index)
            if end >= len(tokens):
                break
            start = end - overlap_tokens
    return windows, owner


def aggregate_max(scores: list[float], owner: list[int], n_documents: int) -> list[float]:
    best = [float("-inf")] * n_documents
    for score, index in zip(scores, owner, strict=True):
        best[index] = max(best[index], score)
    return best


def rerank(models: ModelServer, tokenizer: Tokenizer, query: str, documents: list[str],
           max_tokens: int = 480, model: str | None = None) -> list[float]:
    """One score per document, in input order."""
    if not documents:
        return []
    windows, owner = split_for_rerank(tokenizer, documents, max_tokens=max_tokens)
    return aggregate_max(models.rerank(query, windows, model=model), owner, len(documents))
