"""Page-level retrieval metrics: a retrieved chunk counts as relevant when its page is a gold page."""
from __future__ import annotations

import math


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(gold.intersection(ranked[:k])) / len(gold)


def hit_at(ranked: list[str], gold: set[str], k: int) -> float:
    return float(any(x in gold for x in ranked[:k]))


def mrr(ranked: list[str], gold: set[str]) -> float:
    return next((1 / (i + 1) for i, x in enumerate(ranked) if x in gold), 0.0)


def ndcg_at(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))] if ordered else 0.0
