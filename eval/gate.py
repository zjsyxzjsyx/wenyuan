"""Release gate: fail when a new retrieval run is significantly worse than its baseline.

`python -m eval.gate experiments/<new run>.json [--baseline course] [--metric ev_hit@5] [--tolerance 0.02]`
eval/baselines.json names a baseline run and the metrics that matter for it, per eval set: the v2
course, notes and CS50 sets are scored on evidence (ev_*), v0.7's Stack Overflow sets on pages.
Without --baseline, the one whose run used the same dataset (and the same --owner) as the new run
is chosen. Both runs must score the same questions; the comparison is the paired bootstrap of
eval.compare, on the questions that have the metric. A metric fails when the whole 95% interval of
(new - baseline) lies below -tolerance, i.e. a drop that is both real and larger than the
tolerance; noise and small drops pass. Exit code 1 on failure, so it can stop a release script.

The evals need a GPU, the indexed corpus and the local models, which hosted CI does not have;
CI runs the unit tests, and this gate runs before a release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.compare import compare

BASELINES = Path("eval/baselines.json")


def known() -> dict[str, dict]:
    raw = json.loads(BASELINES.read_text(encoding="utf-8")) if BASELINES.exists() else {}
    # v0.7 entries were bare paths scored on pages
    return {name: entry if isinstance(entry, dict) else {"run": entry, "metrics": ["recall@5", "recall@10", "mrr"]}
            for name, entry in raw.items()}


def _setting(path: Path) -> tuple[str, str | None]:
    run = json.loads(path.read_text(encoding="utf-8"))
    return Path(run.get("dataset", "")).as_posix(), run.get("owner")


def resolve(candidate: Path, baseline: str | None) -> tuple[Path, list[str]]:
    """(baseline run, metrics) by name, by path, or by matching the candidate's dataset and owner."""
    entries = known()
    if baseline in entries:
        return Path(entries[baseline]["run"]), entries[baseline]["metrics"]
    if baseline:
        return Path(baseline), ["recall@5", "recall@10", "mrr"]
    wanted = _setting(candidate)
    for entry in entries.values():
        if Path(entry["run"]).exists() and _setting(Path(entry["run"])) == wanted:
            return Path(entry["run"]), entry["metrics"]
    raise SystemExit(f"no baseline in {BASELINES} for dataset {wanted[0]} (owner {wanted[1]}); pass --baseline")


def gate(baseline: Path, candidate: Path, metrics: list[str], tolerance: float, resamples: int = 4000) -> dict:
    result = compare(baseline, candidate, metrics, resamples)
    failures = [m for m, v in result["metrics"].items() if v["ci95"][1] < -tolerance]
    return {**result, "tolerance": tolerance, "failed": failures, "passed": not failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", help="name in eval/baselines.json, or a path (default: same dataset)")
    parser.add_argument("--metric", nargs="+", help="default: the baseline's metrics")
    parser.add_argument("--tolerance", type=float, default=0.0, help="drop allowed even when significant")
    args = parser.parse_args()
    baseline, metrics = resolve(args.candidate, args.baseline)
    result = gate(baseline, args.candidate, args.metric or metrics, args.tolerance)
    print(f"baseline {baseline} ({result['n']} questions)")
    for m, v in result["metrics"].items():
        mark = "FAIL" if m in result["failed"] else "ok"
        print(f"{mark:4} {m:12} {v['a']:.3f} -> {v['b']:.3f}  delta {v['delta']*100:+.1f}  "
              f"CI [{v['ci95'][0]*100:+.1f}, {v['ci95'][1]*100:+.1f}]  (n={v['n']})")
    print("PASSED" if result["passed"] else f"FAILED: significantly worse on {', '.join(result['failed'])}")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
