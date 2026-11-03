"""Paired comparison of two experiment runs on the same questions.

`python -m eval.compare experiments/A.json experiments/B.json [--metric recall@5 mrr]`
Both runs score the same questions, so the per-question differences are paired:
resampling questions (not runs) gives a 95% interval for B - A that accounts for
some questions being easy for every method. An interval that excludes 0 is
reported as significant.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

METRICS = ("recall@5", "recall@10", "mrr", "ndcg@10")


def paired_bootstrap(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """(mean of b - a, 2.5th percentile, 97.5th percentile) over `n` resamples of the questions."""
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return mean(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def compare(path_a: Path, path_b: Path, metrics=METRICS, n: int = 10000) -> dict[str, dict]:
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (path_a, path_b))
    qa, qb = ({q["id"]: q for q in r["questions"]} for r in (a, b))
    shared = sorted(qa.keys() & qb.keys())
    out = {}
    for m in metrics:
        ids = [i for i in shared if m in qa[i] and m in qb[i]]     # e.g. evidence scores exist only where gold text does
        if not ids:
            continue
        delta, lo, hi = paired_bootstrap([qa[i][m] for i in ids], [qb[i][m] for i in ids], n=n)
        out[m] = {"a": round(mean(qa[i][m] for i in ids), 4), "b": round(mean(qb[i][m] for i in ids), 4),
                  "delta": round(delta, 4), "ci95": [round(lo, 4), round(hi, 4)], "significant": lo > 0 or hi < 0,
                  "n": len(ids)}
    return {"n": len(shared), "a": a["exp"], "b": b["exp"], "metrics": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--metric", nargs="+", default=list(METRICS))
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(compare(args.a, args.b, args.metric, args.resamples), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
