"""Answer-level evaluation (E8): what claim-level verification does to the answers.

`python -m eval.answers --exp W4-E8 --n 100`                                    (v0.7: Stack Overflow)
`python -m eval.answers --exp M3-answers --n 100 --dataset data/eval/cs50_gen_v1.jsonl`   (v2: the course)
For each sampled question the chain answers once (verification off); the answer is
then verified, giving a "before" and an "after" from the same generation. A judge
model, separate from the verifier, scores both:

- faithfulness: the judge's own claim-by-claim verdicts (same claim split), so the
  share of unsupported claims before and after can be compared, and the verifier's
  removals checked against the judge (precision: removed claims the judge also calls
  unsupported; recall: judge-unsupported claims that were removed);
- correctness against the question's reference answer (the accepted Stack Overflow answer,
  or the one written with a generated question): 2 same solution, 1 partly, 0 wrong or
  off-topic; refusals are counted separately;
- whether the answer cites a gold page (anchors and video timestamps aside), for sets that
  name gold pages;
- what the judge's unsupported claims are: asked again with only those claims, it sorts each into
  supported after all, contradicts the sources, outside the sources but correct, or outside and
  wrong. Only the second and the last mislead a student ("harmful"); M3 read all 46 unsupported
  claims of one run by hand: about 13 were in the sources after all and 15 were correct general
  knowledge, so "unsupported" alone overstated the problem several times over.

Every question goes through eval.memory_guard first, one at a time by default: on the
development laptop the local model and Docker together can exhaust memory (M2).
A notes set (--owner) would send the student's notes to the judge; that needs
--notes-to-judge, given only when the notes' owner agreed.

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
from urllib.parse import urlsplit

from app.config import get_settings
from app.db.models import engine
from app.generation.answer import apply_verification, format_sources
from app.generation.verify import verify
from app.ingest.sites import SITES
from app.models.llm import LLM
from app.models.local import ModelServer
from app.query import ask as chain
from app.tracing import spans
from eval.memory_guard import wait_for_memory
from eval.metrics import percentile
from eval.run import DATASET, load_labels, owner_id_of, query_text

OUT_DIR = Path("experiments")
WORK = Path(".cache/eval_answers")     # the full run, answer texts included (not committed)

CORRECTNESS = """You grade an answer to a programming question against a reference answer
(an expert's, or one written from the course material the question is about).

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


KINDS = """A checker could not find the claims below in the sources an answer was written from.
Look again, then put each claim into one kind:
- "supported": on a closer look the sources do state it (paraphrase is fine).
- "contradicts": the sources say something different.
- "outside_correct": the sources do not cover it, but it is correct, standard knowledge for the subject.
- "outside_wrong": the sources do not cover it, and it is wrong or misleading.

Output one JSON object: {{"claims": [{{"id": "C1", "kind": "..."}}, ...]}}

---Sources---
{sources}

---Claims---
{claims}"""
KIND_NAMES = ("supported", "contradicts", "outside_correct", "outside_wrong")
HARMFUL = ("contradicts", "outside_wrong")


def classify_unsupported(judge: LLM, texts: list[str], sources) -> dict[str, str]:
    """Claim text -> kind for the judge's unsupported claims; "unclassified" if the judge skips one."""
    if not texts:
        return {}
    listing = "\n".join(f"[C{i}] {t}" for i, t in enumerate(texts, 1))
    raw = judge.chat_json([{"role": "user", "content": KINDS.format(sources=format_sources(sources), claims=listing)}])
    kinds = {str(v.get("id")): str(v.get("kind", "")).lower() for v in raw.get("claims", []) if isinstance(v, dict)}
    return {t: kinds.get(f"C{i}") if kinds.get(f"C{i}") in KIND_NAMES else "unclassified"
            for i, t in enumerate(texts, 1)}


def judge_llm() -> LLM:
    s = get_settings()
    if s.judge_base_url:
        return LLM(base_url=s.judge_base_url, api_key=s.judge_api_key or "", model=s.judge_model)
    return LLM(think=True)


def sample(n: int, seed: int, dataset: Path = DATASET) -> list[dict]:
    labels = load_labels(dataset)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
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


def page_of(url: str) -> str:
    """A URL without its anchor and query: a citation to a notes section or a video second still names its page."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/")


def cites_gold(citations: list[dict], gold_urls: list[str]) -> bool:
    gold = {page_of(u) for u in gold_urls}
    return any(page_of(c.get("url", "")) in gold for c in citations)


def one(row: dict, llm: LLM, judge: LLM, models: ModelServer, owner_id: int | None = None) -> dict:
    """One question; a failure is recorded instead of ending the run."""
    wait_for_memory()
    try:
        return _one(row, llm, judge, models, owner_id)
    except Exception as exc:  # noqa: BLE001 - reported per question and left out of the summary
        return {"id": row["id"], "library": row["library"], "error": f"{type(exc).__name__}: {exc}"[:300]}


def _one(row: dict, llm: LLM, judge: LLM, models: ModelServer, owner_id: int | None = None) -> dict:
    r = chain.ask(query_text(row, 1500), libraries=[row["library"]], llm=llm, models=models, owner_id=owner_id,
                  cfg=chain.AskConfig(verify=False))    # verified below, so before and after share one answer
    before = r.answer
    record = {"id": row["id"], "library": row["library"], "found_before": before.found,
              **({"qtype": row["qtype"]} if "qtype" in row else {})}
    if not before.found:
        return {**record, "found_after": False}
    began = time.perf_counter()
    verified = verify(llm, before.text, r.sources)
    record["verify_ms"] = round((time.perf_counter() - began) * 1000)
    after = apply_verification(before, verified)
    judged = verify(judge, before.text, r.sources)              # the judge's own claim verdicts
    by_text = {c.text: c.verdict for c in judged.claims}
    kinds = classify_unsupported(judge, [c.text for c in judged.claims if c.verdict == "unsupported"], r.sources)
    removed = set(verified.removed)
    claims = [{"text": c.text, "verifier": c.verdict, "judge": by_text.get(c.text, "supported"),
               "removed": c.text in removed,
               **({"kind": kinds[c.text]} if c.text in kinds else {})} for c in verified.claims]
    record.update(found_after=after.found, claims=claims,
                  answer_before=before.text, answer_after=after.text,
                  **({"cites_gold": cites_gold(after.citations, row["gold_urls"])} if row.get("gold_urls") else {}),
                  correct_before=correctness(judge, row, before.text))
    record["correct_after"] = (record["correct_before"] if after.text == before.text
                               else correctness(judge, row, after.text) if after.found else None)
    return record


def without_texts(result: dict) -> dict:
    """The run without answer and claim texts, for experiments/ (committed). The texts restate the
    course, whose license keeps it out of the repository, or Stack Overflow answers; they stay in
    .cache/eval_answers/. Every summary number can be recomputed from what is left."""
    def strip(q: dict) -> dict:
        q = {k: v for k, v in q.items() if k not in ("answer_before", "answer_after")}
        if "claims" in q:
            q["claims"] = [{k: v for k, v in c.items() if k != "text"} for c in q["claims"]]
        return q
    return {**result, "questions": [strip(q) for q in result["questions"]]}


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
        "cites_gold": share([r for r in answered if "cites_gold" in r], lambda r: r["cites_gold"]),
        # The judge's unsupported claims by kind (runs before this field have none); "harmful" are
        # those that contradict the sources or are wrong outright.
        "unsupported_kinds_before": dict(Counter(c.get("kind", "unclassified") for c in bad)),
        "unsupported_kinds_after": dict(Counter(c.get("kind", "unclassified") for c in bad if not c["removed"])),
        "harmful_claims_before": share(claims, lambda c: c.get("kind") in HARMFUL),
        "harmful_claims_after": share(kept, lambda c: c.get("kind") in HARMFUL),
        "answers_with_harmful_before": share(answered, lambda r: any(c.get("kind") in HARMFUL for c in r["claims"])),
        "answers_with_harmful_after": share(
            [r for r in answered if r["found_after"]],
            lambda r: any(c.get("kind") in HARMFUL and not c["removed"] for c in r["claims"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="parallel questions (match the LLM's slots: 1)")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--owner", help="search this account's notes too (a notes set)")
    parser.add_argument("--notes-to-judge", action="store_true",
                        help="allow sending the owner's notes to an API judge (only with the owner's agreement)")
    args = parser.parse_args()
    if args.owner and get_settings().judge_base_url and not args.notes_to_judge:
        raise SystemExit("this would send the owner's notes to the API judge; pass --notes-to-judge if they agreed")
    spans.configure(engine())
    llm, judge, models = LLM(), judge_llm(), ModelServer()
    owner_id = owner_id_of(args.owner)
    rows = sample(args.n, args.seed, args.dataset)
    done = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, one, row, llm, judge, models, owner_id) for row in rows]
        for i, f in enumerate(futures, 1):
            done.append(f.result())
            if i % 10 == 0:
                print(f"{i}/{len(rows)}", json.dumps(summarize(done), ensure_ascii=False), flush=True)
    result = {"exp": args.exp, "dataset": args.dataset.as_posix(), "owner": args.owner,
              "judge": {"model": judge.model, "think": judge.think}, "summary": summarize(done),
              "by_qtype": {t: summarize([r for r in done if r.get("qtype") == t])
                           for t in sorted({r["qtype"] for r in done if "qtype" in r})},
              "questions": done}
    name = f"{args.exp}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / name).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(without_texts(result), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("exp", "judge", "summary")}, ensure_ascii=False, indent=1))
    print("saved", path, "| with answer texts:", WORK / name)


if __name__ == "__main__":
    main()
