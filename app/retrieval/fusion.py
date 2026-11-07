"""Application-level reciprocal rank fusion, for lists that come from separate searches.

Milvus fuses dense and BM25 inside one search; lists from separate searches (in eval.run, a
course and a student's notes searched with different queries) are fused here with the same formula.
"""
from __future__ import annotations

from typing import Any


def rrf(lists: list[list[dict[str, Any]]], k: int = 60, limit: int | None = None,
        weights: list[float] | None = None) -> list[dict[str, Any]]:
    """score(item) = sum over lists of weight / (k + rank), rank from 1; items keyed by chunk_id."""
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    for hits, weight in zip(lists, weights or [1.0] * len(lists), strict=True):
        for rank, hit in enumerate(hits, start=1):
            scores[hit["chunk_id"]] = scores.get(hit["chunk_id"], 0.0) + weight / (k + rank)
            items.setdefault(hit["chunk_id"], hit)
    ranked = sorted(scores, key=lambda cid: -scores[cid])[:limit]
    return [{**items[cid], "score": scores[cid]} for cid in ranked]
