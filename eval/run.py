"""Run a retrieval experiment over an eval set and save the scores.

`python -m eval.run --exp E2 --collection chunks_struct_merge --mode hybrid [--rerank 50] [--understand] [--summaries]`
Chunks are retrieved, collapsed to their pages in rank order, and scored against
the gold pages. Results (config, metrics, per-question ranks) go to
experiments/<exp>-<timestamp>.json so every run stays comparable.
"""
from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

from app.db.models import engine
from app.ingest.sites import SITES
from app.models.llm import LLM
from app.models.local import ModelServer
from app.ingest.summarize import COLLECTION as SUMMARIES
from app.query.understand import understand
from app.retrieval import milvus_store
from app.retrieval.fusion import best_per_block, rrf
from app.retrieval.rerank import rerank
from app.tokenizer import TiktokenTokenizer
from app.tracing import spans
from eval.metrics import hit_at, mrr, ndcg_at, percentile, recall_at, unique_in_order

DATASET = Path("eval/datasets/so_v1.jsonl")
OUT_DIR = Path("experiments")
QUERY_CACHE = Path(".cache/eval_query_vectors.jsonl")
MODES = ("dense", "bm25", "hybrid")


@dataclass
class Config:
    exp: str
    collection: str
    top_k: int = 30                     # chunks retrieved before collapsing to pages
    body_chars: int = 1500              # question body chars in the dense query
    bm25_body_chars: int | None = None  # ... in the BM25 query (None: same as body_chars)
    mode: str = "dense"
    rerank_top: int = 0                 # rerank the first N chunks (0: off)
    rerank_body_chars: int = 500
    understand: bool = False            # LLM query understanding before retrieval (E6)
    summaries: bool = False             # also search section summaries/questions, fuse by RRF (E5)
    summary_weight: float = 1.0         # RRF weight of the summary list (chunks: 1.0)
    dense_query: str = "rewritten"      # with --understand: embed the LLM rewrite or the raw question
    llm_workers: int = 4                # parallel LLM calls (match the server's parallel slots)
    dataset: Path = DATASET
    exclude_labels: frozenset[str] = field(default_factory=lambda: frozenset({"not_applicable"}))


def query_text(row: dict, body_chars: int) -> str:
    return f"{row['question']}\n\n{row['body'][:body_chars]}".strip()


def embed_queries(models: ModelServer, texts: list[str], model: str = "BAAI/bge-m3") -> list[list[float]]:
    """Query vectors, computed once per (model, text) and reused by every experiment.

    fp16 inference pads each batch to its longest text, so the same query embeds
    slightly differently depending on which other queries share its batch (min
    cosine 0.9995 between batch-64 and one-at-a-time). That was enough to move
    Recall@5 by ~0.01 between two runs of the same index, i.e. the size of the
    effects being measured. Caching makes runs reproducible.
    """
    cache: dict[str, list[float]] = {}
    if QUERY_CACHE.exists():
        for line in QUERY_CACHE.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["key"]] = row["vector"]
    key = lambda text: hashlib.sha1(f"{model}\n{text}".encode()).hexdigest()
    missing = [t for t in dict.fromkeys(texts) if key(t) not in cache]
    if missing:
        QUERY_CACHE.parent.mkdir(exist_ok=True)
        with QUERY_CACHE.open("a", encoding="utf-8") as f:
            for text, vector in zip(missing, models.embed(missing, model=model)):
                cache[key(text)] = vector
                f.write(json.dumps({"key": key(text), "vector": vector}) + "\n")
    return [cache[key(t)] for t in texts]


def load_labels(dataset: Path) -> dict[str, str]:
    """Audit labels from <dataset>_audit.jsonl; questions without one are 'unaudited'."""
    audit = dataset.with_name(dataset.stem + "_audit.jsonl")
    if not audit.exists():
        return {}
    return {row["id"]: row["label"] for row in map(json.loads, audit.read_text(encoding="utf-8").splitlines())}


def retrieve(mc, collection: str, mode: str, vector: list[float], text: str, top_k: int, libraries: list[str],
             summaries: bool = False, summary_weight: float = 1.0):
    if mode == "dense":
        hits = milvus_store.dense_search(mc, collection, vector, top_k=top_k, libraries=libraries)
    elif mode == "bm25":
        hits = milvus_store.bm25_search(mc, collection, text, top_k=top_k, libraries=libraries)
    else:
        hits = milvus_store.hybrid_search(mc, collection, vector, text, top_k=top_k, libraries=libraries)
    if not summaries:
        return hits
    extra = best_per_block(milvus_store.dense_search(mc, SUMMARIES, vector, top_k=top_k, libraries=libraries))
    return rrf([hits, extra], limit=top_k, weights=[1.0, summary_weight])


def prepare_queries(cfg: Config, rows: list[dict], models: ModelServer) -> tuple[list[list[float]], list[str]]:
    """Dense vector and BM25 text per question, with or without LLM query understanding."""
    bm25_chars = cfg.body_chars if cfg.bm25_body_chars is None else cfg.bm25_body_chars
    if not cfg.understand:
        with spans.span("stage", "embed_queries", n=len(rows)):
            return (embed_queries(models, [query_text(r, cfg.body_chars) for r in rows]),
                    [query_text(r, bm25_chars) for r in rows])
    llm = LLM()
    with spans.span("stage", "understand_queries", n=len(rows), workers=cfg.llm_workers):
        # Library routing is not scored here (each question's library is given), so offer only its own.
        # Each call runs in a copy of this context so its span stays under this stage in the trace.
        def one(row: dict):
            return contextvars.copy_context().run(
                understand, llm, query_text(row, cfg.body_chars), libraries=[row["library"]])
        with ThreadPoolExecutor(max_workers=cfg.llm_workers) as pool:
            parsed = list(pool.map(one, rows))
    # The rewrite drops the body's error text and code, which the dense model can use;
    # "raw" keeps the question for dense and gives the LLM's API names to BM25 only.
    dense_texts = ([query_text(r, cfg.body_chars) for r in rows] if cfg.dense_query == "raw"
                   else [u.search_query_en for u in parsed])
    with spans.span("stage", "embed_queries", n=len(rows)):
        vectors = embed_queries(models, dense_texts)
    return vectors, [u.bm25_text() for u in parsed]


def run(cfg: Config) -> dict:
    with spans.span("eval", cfg.exp, collection=cfg.collection, mode=cfg.mode, rerank_top=cfg.rerank_top,
                    understand=cfg.understand, summaries=cfg.summaries) as root:
        result = _run(cfg)
        root.set(**result["overall"])
        result["trace_id"] = root.trace_id
    return result


def _run(cfg: Config) -> dict:
    labels = load_labels(cfg.dataset)
    rows = [json.loads(line) for line in cfg.dataset.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["label"] = labels.get(row["id"], "unaudited")
    excluded = sum(row["label"] in cfg.exclude_labels for row in rows)
    rows = [row for row in rows if row["label"] not in cfg.exclude_labels and row["library"] in SITES]
    models, mc, tokenizer = ModelServer(), milvus_store.client(), TiktokenTokenizer()
    vectors, texts = prepare_queries(cfg, rows, models)
    per_question, latencies = [], []
    for row, vector, text in zip(rows, vectors, texts):
        with spans.span("retrieval", f"{cfg.mode}_search", question=row["id"]) as s:
            began = time.perf_counter()  # timed inside the span so the trace writes are not counted
            hits = retrieve(mc, cfg.collection, cfg.mode, vector, text, cfg.top_k, [row["library"]], cfg.summaries,
                            cfg.summary_weight)
            if cfg.rerank_top:
                with spans.span("retrieval", "rerank", n=min(cfg.rerank_top, len(hits))):
                    head = hits[:cfg.rerank_top]
                    scores = rerank(models, tokenizer, query_text(row, cfg.rerank_body_chars), [h["text"] for h in head])
                    hits = [h for _, h in sorted(zip(scores, head), key=lambda p: -p[0])] + hits[cfg.rerank_top:]
            latencies.append((time.perf_counter() - began) * 1000)
            s.set(hits=len(hits))
        ranked = unique_in_order([h["url"] for h in hits])
        gold = set(row["gold_urls"])
        per_question.append({
            "id": row["id"], "library": row["library"], "type": row["type"], "label": row["label"],
            "recall@5": recall_at(ranked, gold, 5), "recall@10": recall_at(ranked, gold, 10),
            "hit@5": hit_at(ranked, gold, 5), "mrr": mrr(ranked, gold), "ndcg@10": ndcg_at(ranked, gold, 10),
            "ranked": ranked[:10], "gold": sorted(gold),
        })

    def summarize(items: list[dict]) -> dict:
        keys = ("recall@5", "recall@10", "hit@5", "mrr", "ndcg@10")
        return {"n": len(items), **{k: round(mean(q[k] for q in items), 4) for k in keys}} if items else {"n": 0}

    config = {k: (sorted(v) if isinstance(v, frozenset) else str(v) if isinstance(v, Path) else v)
              for k, v in asdict(cfg).items()}
    result = {
        **config, "excluded": excluded,
        "overall": summarize(per_question),
        "by_library": {lib: summarize([q for q in per_question if q["library"] == lib])
                       for lib in sorted({q["library"] for q in per_question})},
        "by_label": {label: summarize([q for q in per_question if q["label"] == label])
                     for label in sorted({q["label"] for q in per_question})},
        "search_ms": {"p50": round(percentile(latencies, 0.5), 1), "p95": round(percentile(latencies, 0.95), 1)},
        "questions": per_question,
    }
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{cfg.exp}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    result["saved_to"] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--top-k", type=int, default=30, help="chunks retrieved before collapsing to pages")
    parser.add_argument("--body-chars", type=int, default=1500)
    parser.add_argument("--bm25-body-chars", type=int, help="question body chars in the BM25 query (default: --body-chars)")
    parser.add_argument("--mode", choices=MODES, default="dense")
    parser.add_argument("--rerank", type=int, default=0, metavar="N", help="rerank the first N chunks (0 = off)")
    parser.add_argument("--rerank-body-chars", type=int, default=500)
    parser.add_argument("--understand", action="store_true", help="LLM query understanding first (needs LLM_API_KEY)")
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--summaries", action="store_true", help="fuse in section summary/question hits (E5)")
    parser.add_argument("--summary-weight", type=float, default=1.0)
    parser.add_argument("--dense-query", choices=("rewritten", "raw"), default="rewritten",
                        help="with --understand: what the dense side embeds")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--exclude-labels", nargs="*", default=["not_applicable"],
                        help="audit labels left out of the scores (pass none to keep every question)")
    args = parser.parse_args()
    spans.configure(engine())
    result = run(Config(
        exp=args.exp, collection=args.collection, top_k=args.top_k, body_chars=args.body_chars,
        bm25_body_chars=args.bm25_body_chars, mode=args.mode, rerank_top=args.rerank,
        rerank_body_chars=args.rerank_body_chars, understand=args.understand, llm_workers=args.llm_workers,
        summaries=args.summaries, summary_weight=args.summary_weight, dense_query=args.dense_query,
        dataset=args.dataset,
        exclude_labels=frozenset(args.exclude_labels)))
    print(json.dumps({k: result[k] for k in ("exp", "collection", "excluded", "overall", "by_library", "by_label",
                                             "search_ms", "saved_to", "trace_id")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
