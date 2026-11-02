"""E9: do multi-hop strategies find both pages of a cross-library question?

`python -m eval.multihop --exp W4-E9 [--modes single decompose iterative]`
On eval/datasets/crosslib_v1.jsonl (eval.build_multihop). For each mode the merged
hit list is collapsed to pages; reported: both@k (both gold pages in the top k
pages), recall@k (share of the two found), LLM calls and latency per question.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from app.db.models import engine
from app.models.llm import LLM
from app.models.local import ModelServer
from app.query import multihop
from app.tracing import spans
from eval.metrics import percentile, unique_in_order

DATASET = Path("eval/datasets/crosslib_v1.jsonl")
OUT_DIR = Path("experiments")


class FixedEmbedding(ModelServer):
    """A model-server client pinned to one embedding model and query prompt (E7 runs)."""

    def __init__(self, model: str, prompt_name: str | None) -> None:
        super().__init__()
        self.model, self.prompt_name = model, prompt_name

    def embed(self, texts, model=None, batch=64, prompt_name=None):
        return super().embed(texts, model=self.model, batch=batch, prompt_name=self.prompt_name)


def score(ranked: list[str], gold: set[str], k: int) -> tuple[float, float]:
    found = gold & set(ranked[:k])
    return float(found == gold), len(found) / len(gold)


def one(row: dict, mode: str, llm: LLM, models: ModelServer, collection: str) -> dict:
    began = time.perf_counter()
    r = multihop.retrieve(row["question"], llm, models, collection, libraries=row["libraries"], mode=mode)
    ranked = unique_in_order([h["url"] for h in r.hits])
    gold = set(row["gold_urls"])
    out = {"id": row["id"], "mode": mode, "llm_calls": r.llm_calls, "rounds": len(r.hops),
           "queries": [h.query for h in r.hops], "ms": round((time.perf_counter() - began) * 1000)}
    for k in (5, 10):
        out[f"both@{k}"], out[f"recall@{k}"] = score(ranked, gold, k)
    return out


def summarize(rows: list[dict]) -> dict:
    keys = ("both@5", "both@10", "recall@5", "recall@10", "llm_calls", "rounds")
    return {"n": len(rows), **{k: round(mean(r[k] for r in rows), 4) for k in keys},
            "ms_p50": percentile([r["ms"] for r in rows], 0.5)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--modes", nargs="+", default=["single", "decompose", "iterative"])
    parser.add_argument("--collection", default="chunks_struct_merge")
    parser.add_argument("--embed-model", help="e.g. Qwen/Qwen3-Embedding-0.6B (match the collection)")
    parser.add_argument("--query-prompt")
    args = parser.parse_args()
    spans.configure(engine())
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    models = FixedEmbedding(args.embed_model, args.query_prompt) if args.embed_model else ModelServer()
    llm, results = LLM(), {}
    for mode in args.modes:
        with ThreadPoolExecutor(max_workers=2) as pool:
            done = list(pool.map(lambda row: contextvars.copy_context().run(one, row, mode, llm, models,
                                                                            args.collection), rows))
        results[mode] = {"summary": summarize(done), "questions": done}
        print(mode, json.dumps(results[mode]["summary"]), flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{args.exp}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"exp": args.exp, "dataset": str(DATASET), "modes": results}, ensure_ascii=False,
                               indent=1), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
