"""Release gate: fail when a new retrieval run is significantly worse than the baseline.

`python -m eval.gate experiments/<new run>.json [--baseline retrieval] [--metric recall@5 mrr]`
The baseline is a name from eval/baselines.json (or a path). Both runs must score
the same questions; the comparison is the paired bootstrap of eval.compare. A
metric fails when the whole 95% interval of (new - baseline) lies below
-tolerance, i.e. a drop that is both real and larger than the tolerance; noise and
small drops pass. Exit code 1 on failure, so it can stop a release script.

The evals need a GPU, the indexed corpus and the local models, which hosted CI
does not have; CI runs the unit tests, and this gate runs before a release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.compare import compare

BASELINES = Path("eval/baselines.json")


def resolve(baseline: str) -> Path:
    known = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {}
    return Path(known.get(baseline, baseline))


def gate(baseline: Path, candidate: Path, metrics: list[str], tolerance: float, resamples: int = 4000) -> dict:
    result = compare(baseline, candidate, metrics, resamples)
    failures = [m for m, v in result["metrics"].items() if v["ci95"][1] < -tolerance]
    return {**result, "tolerance": tolerance, "failed": failures, "passed": not failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", default="retrieval", help="name in eval/baselines.json, or a path")
    parser.add_argument("--metric", nargs="+", default=["recall@5", "recall@10", "mrr"])
    parser.add_argument("--tolerance", type=float, default=0.0, help="drop allowed even when significant")
    args = parser.parse_args()
    result = gate(resolve(args.baseline), args.candidate, args.metric, args.tolerance)
    for m, v in result["metrics"].items():
        mark = "FAIL" if m in result["failed"] else "ok"
        print(f"{mark:4} {m:10} {v['a']:.3f} -> {v['b']:.3f}  delta {v['delta']*100:+.1f}  "
              f"CI [{v['ci95'][0]*100:+.1f}, {v['ci95'][1]*100:+.1f}]")
    print("PASSED" if result["passed"] else f"FAILED: significantly worse on {', '.join(result['failed'])}")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
