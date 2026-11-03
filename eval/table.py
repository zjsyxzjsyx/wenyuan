"""Several runs side by side against the first: overall and per question type, with paired intervals.

`python -m eval.table experiments/base.json experiments/v1.json ... [--metric ev_hit@5] [--by qtype]`
Each later run is compared with the first on the questions both scored (eval.compare's paired
bootstrap); a difference whose 95% interval excludes 0 is marked with *. Questions without the
metric (e.g. CS50's own questions that name a whole file instead of an excerpt) are left out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from eval.compare import paired_bootstrap


def load(path: Path) -> tuple[str, dict[str, dict]]:
    run = json.loads(path.read_text(encoding="utf-8"))
    return run["exp"], {q["id"]: q for q in run["questions"]}


def cell(base: dict, other: dict, ids: list[str], metric: str) -> str:
    ids = [i for i in ids if metric in base.get(i, {}) and metric in other.get(i, {})]
    if not ids:
        return "-"
    value = mean(other[i][metric] for i in ids)
    if other is base:
        return f"{value:.3f}"
    delta, lo, hi = paired_bootstrap([base[i][metric] for i in ids], [other[i][metric] for i in ids], n=4000)
    return f"{value:.3f} ({delta:+.3f}{'*' if lo > 0 or hi < 0 else ''})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+", help="the first is the baseline")
    parser.add_argument("--metric", default="ev_hit@5")
    parser.add_argument("--by", default="qtype", help="per-question field to split by")
    args = parser.parse_args()
    runs = [load(p) for p in args.runs]
    base = runs[0][1]
    groups = {"all": sorted(base)}
    for qid, q in sorted(base.items()):
        if args.by in q:
            groups.setdefault(q[args.by], []).append(qid)
    width = max(len(name) for name, _ in runs)
    print(f"{args.metric:{width}} | " + " | ".join(f"{g} (n={len(ids)})" for g, ids in groups.items()))
    for name, questions in runs:
        print(f"{name:{width}} | " + " | ".join(cell(base, questions, ids, args.metric) for ids in groups.values()))


if __name__ == "__main__":
    main()
