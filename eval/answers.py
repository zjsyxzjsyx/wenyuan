"""Answer-level evaluation (E8): what claim-level verification does to the answers.

`python -m eval.answers --exp W4-E8 --n 100`
For each sampled question the chain answers once (verification off); the answer is
then verified, giving a "before" and an "after" from the same generation. A judge
model, separate from the verifier, scores both:

- faithfulness: the judge's own claim-by-claim verdicts (same claim split), so the
  share of unsupported claims before and after can be compared, and the verifier's
  removals checked against the judge (precision: removed claims the judge also calls
  unsupported; recall: judge-unsupported claims that were removed);
- correctness against the Stack Overflow accepted answer: 2 same solution, 1 partly,
  0 wrong or off-topic; refusals are counted separately.

The judge defaults to the answering model with thinking on: cheaper than an API
and different from the no-think verifier, but the same model, so it shares its
blind spots. Set JUDGE_* (app.config) to a stronger model before quoting numbers.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from app.config import get_settings
from app.db.models import engine
from app.generation.answer import apply_verification
from app.generation.verify import verify
from app.ingest.sites import SITES
from app.models.llm import LLM
from app.models.local import ModelServer
from app.query import ask as chain
from app.tracing import spans
from eval.metrics import percentile
from eval.run import DATASET, load_labels, query_text

OUT_DIR = Path("experiments")

CORRECTNESS = """You grade an answer to a programming question against a reference answer written by an expert
(the accepted Stack Overflow answer).

Score:
- 2: the candidate gives the same solution or facts as the reference on the main point.
- 1: partly right: the right direction but missing the key step, or mixed with something wrong.
- 0: wrong, off-topic, or contradicts the reference.
The reference may be longer and cover more; do not penalize a short answer that gets the main point right.

Output one JSON object: {{"score": 0|1|2, "reason": "<one sentence>"}}

---Question---
{question}

---Reference answer---
{reference}

---Candidate answer---
{candidate}"""


def judge_llm() -> LLM:
    s = get_settings()
    if s.judge_base_url:
        return LLM(base_url=s.judge_base_url, api_key=s.judge_api_key or "", model=s.judge_model)
    return LLM(think=True)


def sample(n: int, seed: int) -> list[dict]:
    labels = load_labels(DATASET)
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if labels.get(r["id"]) != "not_applicable" and r["library"] in SITES and r.get("answer_text")]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def correctness(judge: LLM, row: dict, text: str) -> int:
    raw = judge.chat_json([{"role": "user", "content": CORRECTNESS.format(
        question=query_text(row, 1500), reference=row["answer_text"][:3000], candidate=text)}])
    try:
        return max(0, min(2, int(raw.get("score", 0))))
    except (TypeError, ValueError):
        return 0


def one(row: dict, llm: LLM, judge: LLM, models: ModelServer) -> dict:
    """One question; a failure is recorded instead of ending the run."""
    try:
        return _one(row, llm, judge, models)
    except Exception as exc:  # noqa: BLE001 - reported per question and left out of the summary
        return {"id": row["id"], "library": row["library"], "error": f"{type(exc).__name__}: {exc}"[:300]}


def _one(row: dict, llm: LLM, judge: LLM, models: ModelServer) -> dict:
    r = chain.ask(query_text(row, 1500), libraries=[row["library"]], llm=llm, models=models,
                  cfg=chain.AskConfig(verify=False))    # verified below, so before and after share one answer
    before = r.answer
    record = {"id": row["id"], "library": row["library"], "found_before": before.found}
    if not before.found:
        return {**record, "found_after": False}
    began = time.perf_counter()
    verified = verify(llm, before.text, r.sources)
    record["verify_ms"] = round((time.perf_counter() - began) * 1000)
    after = apply_verification(before, verified)
    judged = verify(judge, before.text, r.sources)              # the judge's own claim verdicts
    by_text = {c.text: c.verdict for c in judged.claims}
    removed = set(verified.removed)
    claims = [{"text": c.text, "verifier": c.verdict, "judge": by_text.get(c.text, "supported"),
               "removed": c.text in removed} for c in verified.claims]
    record.update(found_after=after.found, claims=claims,
                  answer_before=before.text, answer_after=after.text,
                  correct_before=correctness(judge, row, before.text))
    record["correct_after"] = (record["correct_before"] if after.text == before.text
                               else correctness(judge, row, after.text) if after.found else None)
    return record


def summarize(rows: list[dict]) -> dict:
    errors = sum("error" in r for r in rows)
    rows = [r for r in rows if "error" not in r]
    answered = [r for r in rows if r["found_before"]]
    claims = [c for r in answered for c in r["claims"]]
    kept = [c for c in claims if not c["removed"]]
    bad = [c for c in claims if c["judge"] == "unsupported"]
    removed = [c for c in claims if c["removed"]]

    def share(items, pred):
        return round(sum(map(pred, items)) / len(items), 4) if items else None

    def correct_rate(key: str, found_key: str) -> float:
        # Over all questions: a refusal scores 0, as a learner left without an answer.
        return round(mean((r.get(key) or 0) == 2 if r.get(found_key) else 0 for r in rows), 4)
    return {
        "n": len(rows), "errors": errors, "answered_before": len(answered), "answered_after": sum(r["found_after"] for r in rows),
        "claims": len(claims), "verifier_verdicts": dict(Counter(c["verifier"] for c in claims)),
        "judge_verdicts": dict(Counter(c["judge"] for c in claims)),
        "unsupported_claims_before": share(claims, lambda c: c["judge"] == "unsupported"),
        "unsupported_claims_after": share(kept, lambda c: c["judge"] == "unsupported"),
        "answers_with_unsupported_before": share(answered, lambda r: any(c["judge"] == "unsupported" for c in r["claims"])),
        "answers_with_unsupported_after": share(
            [r for r in answered if r["found_after"]],
            lambda r: any(c["judge"] == "unsupported" and not c["removed"] for c in r["claims"])),
        "removal_precision": share(removed, lambda c: c["judge"] == "unsupported"),
        "removal_recall": share(bad, lambda c: c["removed"]),
        "correct2_before": correct_rate("correct_before", "found_before"),
        "correct2_after": correct_rate("correct_after", "found_after"),
        "mean_score_before": round(mean(r["correct_before"] for r in answered), 3) if answered else None,
        "verify_ms_p50": percentile([r["verify_ms"] for r in answered], 0.5) if answered else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2, help="parallel questions (match the LLM's slots)")
    args = parser.parse_args()
    spans.configure(engine())
    llm, judge, models = LLM(), judge_llm(), ModelServer()
    rows = sample(args.n, args.seed)
    done = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, one, row, llm, judge, models) for row in rows]
        for i, f in enumerate(futures, 1):
            done.append(f.result())
            if i % 10 == 0:
                print(f"{i}/{len(rows)}", json.dumps(summarize(done), ensure_ascii=False), flush=True)
    result = {"exp": args.exp, "judge": {"model": judge.model, "think": judge.think}, "summary": summarize(done),
              "questions": done}
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{args.exp}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("exp", "judge", "summary")}, ensure_ascii=False, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
