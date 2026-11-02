"""Retrieval metrics: page level (a chunk counts when its page is a gold page) and evidence level (below)."""
from __future__ import annotations

import math
import re


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


# Evidence-level matching: a retrieved chunk counts when it shares enough text with the gold
# evidence (a passage, not a page id), so the gold survives changes to chunking and finer
# relevance than "the right page" can be scored. Units are English words and single CJK
# characters; overlap is measured on 5-grams relative to the smaller side, so a short excerpt
# inside a long chunk and a chunk inside a long passage both count.
_UNIT = re.compile(r"[a-z0-9_]+|[一-鿿]")


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    units = _UNIT.findall(text.lower())
    if len(units) < n:
        return {tuple(units)} if units else set()
    return {tuple(units[i:i + n]) for i in range(len(units) - n + 1)}


def overlaps(chunk: str, evidence: str, threshold: float = 0.5) -> bool:
    a, b = shingles(chunk), shingles(evidence)
    return bool(a and b) and len(a & b) / min(len(a), len(b)) >= threshold


def evidence_ranks(chunks: list[str], evidence: list[str]) -> list[int | None]:
    """For each evidence passage, the 0-based rank of the first chunk that overlaps it (None: not found)."""
    found: list[int | None] = []
    for passage in evidence:
        found.append(next((i for i, chunk in enumerate(chunks) if overlaps(chunk, passage)), None))
    return found
