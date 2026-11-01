"""Near-duplicate pages, e.g. the same build guide once per Linux distribution.

Pages are compared only with their siblings (same parent URL path): near-copies
in docs sites are almost always siblings, and this avoids an all-pairs scan.
Within a group of near-copies the first URL in sorted order is kept.
"""
from __future__ import annotations

import re
from collections import defaultdict

_WORD = re.compile(r"\w+")


def shingles(text: str, k: int = 5) -> set[int]:
    words = _WORD.findall(text.lower())
    return {hash(" ".join(words[i:i + k])) for i in range(max(len(words) - k + 1, 1))}


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def parent_path(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[0]


def near_duplicates(pages: dict[str, str], threshold: float = 0.85) -> dict[str, str]:
    """Map each dropped URL to the kept URL it duplicates."""
    groups: dict[str, list[str]] = defaultdict(list)
    for url in sorted(pages):
        groups[parent_path(url)].append(url)
    dropped: dict[str, str] = {}
    for urls in groups.values():
        kept: list[tuple[str, set[int]]] = []
        for url in urls:
            sig = shingles(pages[url])
            match = next((k for k, ks in kept if jaccard(sig, ks) >= threshold), None)
            if match:
                dropped[url] = match
            else:
                kept.append((url, sig))
    return dropped
