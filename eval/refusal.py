"""Refusal test: does the system say the materials do not cover a question when they don't, and still
answer when they do?

`python -m eval.refusal --exp M3-refusal [--absent data/eval/cs50_absent_v1.jsonl]
                        [--answerable data/eval/cs50_gen_v1.jsonl --n 40]`
The served ask chain (claim verification on) answers the course's unanswerable questions and a sample
of answerable ones from the same course. An answer counts as a refusal when it is not `found`.
Reported: the refusal rate on unanswerable questions (want high) and false refusals on answerable
ones (want low), each split by question language, since half the unanswerable ones are Chinese.

Only the local model runs, so nothing leaves the machine; one question at a time behind
eval.memory_guard. Question texts and answers go to .cache/eval_refusal/ (they derive from the course,
which stays out of the repository); experiments/ gets ids and outcomes.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import random
import re
import time
from pathlib import Path

from app.db.models import engine
from app.models.llm import LLM
from app.models.local import ModelServer
from app.query import ask as chain
from app.tracing import spans
from eval.memory_guard import wait_for_memory

OUT_DIR = Path("experiments")
WORK = Path(".cache/eval_refusal")
_CJK = re.compile(r"[一-鿿]")


def language(text: str) -> str:
    return "zh" if _CJK.search(text) else "en"


def outcome(row: dict, answer, expected: str) -> dict:
    return {"id": row["id"], "expected": expected, "language": language(row["question"]),
            "refused": not answer.found, "citations": len(answer.citations)}


def rates(items: list[dict]) -> dict:
    """Refusal rate overall and per language, as {group: [share, n]}."""
    out = {}
    for name, group in [("all", items), *[(lang, [i for i in items if i["language"] == lang]) for lang in ("zh", "en")]]:
        if group:
            out[name] = [round(sum(i["refused"] for i in group) / len(group), 4), len(group)]
    return out


def summarize(results: list[dict]) -> dict:
    done = [r for r in results if "error" not in r]
    return {"n": len(results), "errors": len(results) - len(done),
            "refused_when_absent": rates([r for r in done if r["expected"] == "refuse"]),
            "refused_when_answerable": rates([r for r in done if r["expected"] == "answer"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--absent", type=Path, default=Path("data/eval/cs50_absent_v1.jsonl"))
    parser.add_argument("--answerable", type=Path, default=Path("data/eval/cs50_gen_v1.jsonl"))
    parser.add_argument("--n", type=int, default=40, help="answerable questions to sample as the control")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    load = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    answerable = load(args.answerable)
    random.Random(args.seed).shuffle(answerable)
    rows = [(r, "refuse") for r in load(args.absent)] + [(r, "answer") for r in answerable[:args.n]]
    spans.configure(engine())
    llm, models = LLM(), ModelServer()
    results, kept = [], []
    for i, (row, expected) in enumerate(rows, 1):
        wait_for_memory()
        try:
            r = contextvars.copy_context().run(chain.ask, row["question"], libraries=[row["library"]], llm=llm,
                                               models=models)
            results.append(outcome(row, r.answer, expected))
            kept.append({"id": row["id"], "question": row["question"], "answer": r.answer.text})
        except Exception as exc:  # noqa: BLE001 - recorded per question, left out of the rates
            results.append({"id": row["id"], "expected": expected, "error": f"{type(exc).__name__}: {exc}"[:300]})
        if i % 10 == 0:
            print(f"{i}/{len(rows)}", json.dumps(summarize(results)), flush=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"{args.exp}-{stamp}.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept),
                                                    encoding="utf-8")
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{args.exp}-{stamp}.json"
    summary = summarize(results)
    path.write_text(json.dumps({"exp": args.exp, "absent": args.absent.as_posix(), "answerable": args.answerable.as_posix(),
                                "summary": summary, "questions": results}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("saved", path, "| texts:", WORK)


if __name__ == "__main__":
    main()
