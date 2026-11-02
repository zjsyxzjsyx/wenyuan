"""Judged retrieval: are the top 5 pages useful, whichever pages the SO answer happened to link?

`python -m eval.judge_retrieval experiments/<run>.json [--n 200] [--seed 0]`
Gold pages in the SO set are the pages the accepted answer linked, usually one of
several that would do: a sample of misses (R0 run) found retrieved pages such as SQLite's
"foreign keys" for "add a foreign key to an existing table" scored wrong because the
answer linked the FAQ. Here an independent judge (JUDGE_*, the same DeepSeek setup as
eval.answers) reads the question, the accepted answer as reference and each top-5
excerpt (saved by eval.run under .cache/eval_evidence/), and says whether the excerpt
gives what the solution needs. A question counts as answered when a gold page is in
the top 5 or the judge accepts one. The judge sees only public SO and docs text.

The same seed draws the same questions, so two runs can be compared question by question.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from app.models.llm import LLM
from eval.answers import judge_llm
from eval.run import DATASET, EVIDENCE_DIR, OUT_DIR, query_text

PROMPT = """A developer asked this question about {library}:

{question}

The accepted answer on Stack Overflow (reference for what the solution is):
{answer}

Below are excerpts from documentation pages a search engine returned. For each one, decide
whether it gives what the solution needs: the feature, command, option or explanation the
answer relies on, so that someone reading it could write a correct answer.
- "yes": it does (wording may differ from the reference; another valid solution also counts)
- "partial": related and helpful, but the key part is missing
- "no": not useful for this question

{pages}

Return one JSON object: {{"verdicts": {{"1": "yes|partial|no", ...}}}} with one entry per excerpt."""


def judge_one(judge: LLM, row: dict, pages: list[dict]) -> dict[str, str]:
    listing = "\n\n".join(f"[{i + 1}] {p['heading_path']}\n{p['text']}" for i, p in enumerate(pages))
    raw = judge.chat_json([{"role": "user", "content": PROMPT.format(
        library=row["library"], question=query_text(row, 1500), answer=row.get("answer_text", "")[:2500],
        pages=listing)}])
    verdicts = raw.get("verdicts") or {}
    return {p["url"]: str(verdicts.get(str(i + 1), "no")).lower() for i, p in enumerate(pages)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="an eval.run result whose evidence file exists")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    evidence = {e["id"]: e["pages"] for e in map(json.loads, (EVIDENCE_DIR / f"{args.run.stem}.jsonl")
                                                  .read_text(encoding="utf-8").splitlines())}
    rows = {r["id"]: r for r in map(json.loads, Path(run.get("dataset", DATASET)).read_text(encoding="utf-8")
                                    .splitlines())}
    ids = sorted(q["id"] for q in run["questions"])
    sample = random.Random(args.seed).sample(ids, min(args.n, len(ids)))
    judge = judge_llm()

    def one(qid: str) -> dict:
        row, pages = rows[qid], evidence[qid][:args.k]
        verdicts = judge_one(judge, row, pages)
        gold = set(row["gold_urls"])
        return {"id": qid, "library": row["library"],
                "gold_hit": any(p["url"] in gold for p in pages),
                "judged_hit": any(v == "yes" for v in verdicts.values()),
                "gold_judged_yes": [verdicts[p["url"]] for p in pages if p["url"] in gold],
                "verdicts": [verdicts[p["url"]] for p in pages]}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda q: contextvars.copy_context().run(one, q), sample))
    for r in results:
        r["answered"] = r["gold_hit"] or r["judged_hit"]
    gold_verdicts = [v for r in results for v in r["gold_judged_yes"]]
    summary = {
        "n": len(results), "k": args.k, "judge": judge.model,
        "gold_hit": round(mean(r["gold_hit"] for r in results), 4),
        "judged_hit": round(mean(r["judged_hit"] for r in results), 4),
        "answered": round(mean(r["answered"] for r in results), 4),
        # how often the judge accepts a gold page: its own recall, a check on the judge
        "judge_accepts_gold": round(mean(v == "yes" for v in gold_verdicts), 4) if gold_verdicts else None,
    }
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"judged-{args.run.stem}-n{len(results)}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"run": str(args.run), "seed": args.seed, "summary": summary, "questions": results},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
