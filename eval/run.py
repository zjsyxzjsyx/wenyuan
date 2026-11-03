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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean

from app.db.models import engine
from app.ingest.sites import SITES
from app.models.llm import LLM
from app.models.local import ModelServer
from app.ingest.summarize import COLLECTION as SUMMARIES
from app.query import multihop
from app.query.hyde import hypothetical_passage
from app.query.understand import understand
from app.retrieval import milvus_store
from app.retrieval.fusion import best_per_block, rrf
from app.retrieval.llm_rerank import rerank_pages
from app.retrieval.rerank import rerank
from app.tokenizer import TiktokenTokenizer
from app.tracing import spans
from eval.memory_guard import wait_for_memory
from eval.metrics import evidence_ranks, hit_at, mrr, ndcg_at, percentile, recall_at, unique_in_order

DATASET = Path("eval/datasets/so_v1.jsonl")
OUT_DIR = Path("experiments")
QUERY_CACHE = Path(".cache/eval_query_vectors.jsonl")
EVIDENCE_DIR = Path(".cache/eval_evidence")   # top pages' excerpts per question, for eval.judge_retrieval
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
    llm_rerank: int = 0                 # LLM reorders the first N pages (0: off)
    owner: str | None = None            # account whose notes are searched too (as a student's questions are)
    llm_rerank_plain: bool = False      # R1's page view: the best hit's first 400 characters only
    llm_rerank_unit: str = "page"       # what the LLM ranks: "page" or "section" (top-level section of a page)
    hyde: str = ""                      # "dense" | "both": add an LLM-imagined answer passage to the query
    hyde_scope: str = "all"             # "shared": HyDE on the course side only; notes are searched with the question
    generate_only: bool = False         # run (and cache) the query-side LLM calls, then stop
    understand: bool = False            # LLM query understanding before retrieval (E6)
    summaries: bool = False             # also search section summaries/questions, fuse by RRF (E5)
    summary_weight: float = 1.0         # RRF weight of the summary list (chunks: 1.0)
    dense_query: str = "rewritten"      # with --understand: embed the LLM rewrite or the raw question
    history: bool = True                # multi-turn rows: give the conversation so far to understanding
    question_field: str = "question"    # e.g. "original_question" for the multi-turn upper bound
    multihop: str = ""                  # "decompose" | "iterative": app.query.multihop instead of one search (E9)
    embed_model: str = "BAAI/bge-m3"    # must match the collection's vectors (E7: Qwen/Qwen3-Embedding-0.6B)
    query_prompt: str | None = None     # the embedding model's query prompt name, e.g. "query" for Qwen3
    llm_workers: int = 4                # parallel LLM calls (match the server's parallel slots)
    dataset: Path = DATASET
    exclude_labels: frozenset[str] = field(default_factory=lambda: frozenset({"not_applicable"}))


def query_text(row: dict, body_chars: int, field: str = "question") -> str:
    return f"{row[field]}\n\n{row.get('body', '')[:body_chars]}".strip()


def embed_queries(models: ModelServer, texts: list[str], model: str = "BAAI/bge-m3",
                  prompt_name: str | None = None) -> list[list[float]]:
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
    tag = f"{model}|{prompt_name}" if prompt_name else model
    key = lambda text: hashlib.sha1(f"{tag}\n{text}".encode()).hexdigest()
    missing = [t for t in dict.fromkeys(texts) if key(t) not in cache]
    if missing:
        QUERY_CACHE.parent.mkdir(exist_ok=True)
        with QUERY_CACHE.open("a", encoding="utf-8") as f:
            for text, vector in zip(missing, models.embed(missing, model=model, prompt_name=prompt_name)):
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
             summaries: bool = False, summary_weight: float = 1.0, owner_id: int | None = None):
    if mode == "dense":
        hits = milvus_store.dense_search(mc, collection, vector, top_k=top_k, libraries=libraries, owner_id=owner_id)
    elif mode == "bm25":
        hits = milvus_store.bm25_search(mc, collection, text, top_k=top_k, libraries=libraries, owner_id=owner_id)
    else:
        hits = milvus_store.hybrid_search(mc, collection, vector, text, top_k=top_k, libraries=libraries,
                                          owner_id=owner_id)
    if not summaries:
        return hits
    extra = best_per_block(milvus_store.dense_search(mc, SUMMARIES, vector, top_k=top_k, libraries=libraries))
    return rrf([hits, extra], limit=top_k, weights=[1.0, summary_weight])


def hyde_passages(cfg: Config, rows: list[dict]) -> list[str]:
    llm = LLM()
    with spans.span("stage", "hyde_queries", n=len(rows), workers=cfg.llm_workers):
        def one(row: dict) -> str:
            wait_for_memory()
            return contextvars.copy_context().run(
                hypothetical_passage, llm, query_text(row, 1000, cfg.question_field), row["library"])
        with ThreadPoolExecutor(max_workers=cfg.llm_workers) as pool:
            return list(pool.map(one, rows))


def prepare_queries(cfg: Config, rows: list[dict], models: ModelServer) -> tuple[list[list[float]], list[str]] | None:
    """Dense vector and BM25 text per question, with or without LLM query understanding and HyDE.
    With `generate_only`, stops after the LLM calls (they are cached) and returns None."""
    bm25_chars = cfg.body_chars if cfg.bm25_body_chars is None else cfg.bm25_body_chars
    raw = [query_text(r, cfg.body_chars, cfg.question_field) for r in rows]
    if not cfg.understand:
        dense_texts, bm25_texts = raw, [query_text(r, bm25_chars, cfg.question_field) for r in rows]
    else:
        llm = LLM()
        with spans.span("stage", "understand_queries", n=len(rows), workers=cfg.llm_workers):
            # Library routing is not scored here (each question's library is given), so offer only its own.
            # Each call runs in a copy of this context so its span stays under this stage in the trace.
            def one(row: dict):
                wait_for_memory()
                history = row.get("history") if cfg.history else None
                return contextvars.copy_context().run(
                    understand, llm, query_text(row, cfg.body_chars, cfg.question_field), history,
                    libraries=[row["library"]])
            with ThreadPoolExecutor(max_workers=cfg.llm_workers) as pool:
                parsed = list(pool.map(one, rows))
        # The rewrite drops the body's error text and code, which the dense model can use;
        # "raw" keeps the question for dense and gives the LLM's API names to BM25 only.
        dense_texts = raw if cfg.dense_query == "raw" else [u.search_query_en for u in parsed]
        bm25_texts = [u.bm25_text() for u in parsed]
    if cfg.hyde:
        passages = hyde_passages(cfg, rows)
        dense_texts = [f"{t}\n\n{p}" for t, p in zip(dense_texts, passages)]
        if cfg.hyde == "both":
            bm25_texts = [f"{t} {p}" for t, p in zip(bm25_texts, passages)]
    if cfg.generate_only:
        return None
    with spans.span("stage", "embed_queries", n=len(rows)):
        vectors = embed_queries(models, dense_texts, cfg.embed_model, cfg.query_prompt)
    return vectors, bm25_texts


def owner_id_of(name: str | None) -> int | None:
    if not name:
        return None
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from app.db.models import User
    with Session(engine()) as db:
        found = db.scalar(select(User.id).where(User.name == name))
    if found is None:
        raise SystemExit(f"no account named {name!r}")
    return found


def redact(url: str) -> str:
    """Results are committed, so a private note is named by a hash, never by its file name."""
    return f"private:{hashlib.sha1(url.encode()).hexdigest()[:12]}" if url.startswith("private://") else url


def body_of(chunk_text: str) -> str:
    """A chunk without its context header ("Site > Page > Section"), which no gold passage contains."""
    head, sep, rest = chunk_text.partition("\n\n")
    return rest if sep and " > " in head else chunk_text


def evidence_metrics(evidence: list[str] | None, chunks: list[str]) -> dict:
    """Evidence-level scores over the ranked chunks; empty when the question has no gold evidence."""
    if not evidence:
        return {}
    ranks = evidence_ranks(chunks, evidence)
    within = lambda k: sum(r is not None and r < k for r in ranks) / len(ranks)
    first = min((r for r in ranks if r is not None), default=None)
    return {"ev_recall@5": within(5), "ev_recall@10": within(10), "ev_hit@5": float(first is not None and first < 5),
            "ev_mrr": 1 / (first + 1) if first is not None else 0.0, "ev_ranks": ranks}


def run(cfg: Config) -> dict:
    with spans.span("eval", cfg.exp, collection=cfg.collection, mode=cfg.mode, rerank_top=cfg.rerank_top,
                    understand=cfg.understand, summaries=cfg.summaries) as root:
        result = _run(cfg)
        root.set(**result.get("overall", {}))
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
    owner_id = owner_id_of(cfg.owner)
    if cfg.multihop:
        # The whole multi-hop chain (understanding, sub-queries, searches) per question, timed end to end.
        llm = LLM()

        def hop(row: dict):
            began = time.perf_counter()
            r = multihop.retrieve(query_text(row, cfg.body_chars, cfg.question_field), llm, models, cfg.collection,
                                  libraries=[row["library"]], history=row.get("history") if cfg.history else None,
                                  mode=cfg.multihop, top_k=cfg.top_k, mc=mc, owner_id=owner_id)
            return r.hits, (time.perf_counter() - began) * 1000
        with ThreadPoolExecutor(max_workers=cfg.llm_workers) as pool:
            hopped = list(pool.map(lambda row: contextvars.copy_context().run(hop, row), rows))
        vectors, texts = [None] * len(rows), [None] * len(rows)
    else:
        prepared = prepare_queries(cfg, rows, models)
        if prepared is None:
            return {"exp": cfg.exp, "generated_only": len(rows)}
        vectors, texts = prepared
    split = bool(cfg.hyde and cfg.hyde_scope == "shared" and owner_id is not None and not cfg.multihop)
    if split:
        # The HyDE passage is English: it helps a Chinese question find English course text and
        # is noise for the student's Chinese notes (M2), so the two sides get different queries.
        plain_vectors, plain_texts = prepare_queries(replace(cfg, hyde=""), rows, models)
    all_hits, latencies = [], []
    for i, (row, vector, text) in enumerate(zip(rows, vectors, texts)):
        with spans.span("retrieval", f"{cfg.mode}_search", question=row["id"]) as s:
            began = time.perf_counter()  # timed inside the span so the trace writes are not counted
            if cfg.multihop:
                hits, elapsed = hopped[i]
                began -= elapsed / 1000
            else:
                hits = retrieve(mc, cfg.collection, cfg.mode, vector, text, cfg.top_k, [row["library"]],
                                cfg.summaries, cfg.summary_weight, None if split else owner_id)
                if split:
                    notes = retrieve(mc, cfg.collection, cfg.mode, plain_vectors[i], plain_texts[i], cfg.top_k, [],
                                     owner_id=owner_id)
                    hits = rrf([hits, notes], limit=cfg.top_k)
            if cfg.rerank_top:
                with spans.span("retrieval", "rerank", n=min(cfg.rerank_top, len(hits))):
                    head = hits[:cfg.rerank_top]
                    scores = rerank(models, tokenizer, query_text(row, cfg.rerank_body_chars), [h["text"] for h in head])
                    hits = [h for _, h in sorted(zip(scores, head), key=lambda p: -p[0])] + hits[cfg.rerank_top:]
            latencies.append((time.perf_counter() - began) * 1000)
            s.set(hits=len(hits))
        all_hits.append(hits)
    rerank_ms: list[float] = []
    if cfg.llm_rerank:
        llm = LLM()

        def reorder(row: dict, hits: list[dict]) -> list[dict]:
            wait_for_memory()
            with spans.span("retrieval", "llm_rerank", question=row["id"], pages=cfg.llm_rerank):
                began = time.perf_counter()
                view = dict(excerpt_chars=400, second_chars=0, more_headings=0) if cfg.llm_rerank_plain else {}
                out = rerank_pages(llm, query_text(row, 1000, cfg.question_field), hits, row["library"],
                                   top_pages=cfg.llm_rerank, unit=cfg.llm_rerank_unit, **view)
                rerank_ms.append((time.perf_counter() - began) * 1000)
                return out
        with ThreadPoolExecutor(max_workers=cfg.llm_workers) as pool:
            all_hits = list(pool.map(lambda pair: contextvars.copy_context().run(reorder, *pair), zip(rows, all_hits)))
    per_question, evidence = [], []
    for row, hits in zip(rows, all_hits):
        best = {}
        for h in hits:
            best.setdefault(h["url"], h)
        evidence.append({"id": row["id"], "pages": [{"url": u, "heading_path": h["heading_path"], "text": h["text"][:1500]}
                                                    for u, h in list(best.items())[:10]]})
        ranked = unique_in_order([h["url"] for h in hits])
        gold = set(row["gold_urls"])
        found = evidence_metrics(row.get("gold_evidence"), [body_of(h["text"]) for h in hits[:30]])
        per_question.append({
            "id": row["id"], "library": row["library"], "type": row["type"], "label": row["label"],
            **({"qtype": row["qtype"]} if "qtype" in row else {}), **found,
            "recall@5": recall_at(ranked, gold, 5), "recall@10": recall_at(ranked, gold, 10),
            "recall@20": recall_at(ranked, gold, 20),
            # every page the top_k chunks came from: the most any reordering of this pool could reach
            "recall@pool": recall_at(ranked, gold, len(ranked)), "pool": len(ranked),
            "hit@5": hit_at(ranked, gold, 5), "mrr": mrr(ranked, gold), "ndcg@10": ndcg_at(ranked, gold, 10),
            "ranked": [redact(u) for u in ranked[:30]], "gold": sorted(redact(u) for u in gold),
        })

    def summarize(items: list[dict]) -> dict:
        keys = ("recall@5", "recall@10", "recall@20", "recall@pool", "pool", "hit@5", "mrr", "ndcg@10",
                "ev_recall@5", "ev_recall@10", "ev_hit@5", "ev_mrr")
        return ({"n": len(items), **{k: round(mean(q[k] for q in items), 4) for k in keys if all(k in q for q in items)}}
                if items else {"n": 0})

    config = {k: (sorted(v) if isinstance(v, frozenset) else str(v) if isinstance(v, Path) else v)
              for k, v in asdict(cfg).items()}
    result = {
        **config, "excluded": excluded,
        "overall": summarize(per_question),
        "by_library": {lib: summarize([q for q in per_question if q["library"] == lib])
                       for lib in sorted({q["library"] for q in per_question})},
        "by_label": {label: summarize([q for q in per_question if q["label"] == label])
                     for label in sorted({q["label"] for q in per_question})},
        "by_qtype": {t: summarize([q for q in per_question if q.get("qtype") == t])
                     for t in sorted({q["qtype"] for q in per_question if "qtype" in q})},
        "search_ms": {"p50": round(percentile(latencies, 0.5), 1), "p95": round(percentile(latencies, 0.95), 1)},
        **({"llm_rerank_ms": {"p50": round(percentile(rerank_ms, 0.5)), "p95": round(percentile(rerank_ms, 0.95))}}
           if rerank_ms else {}),
        "questions": per_question,
    }
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{cfg.exp}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    result["saved_to"] = str(path)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / f"{path.stem}.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in evidence), encoding="utf-8")
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
    parser.add_argument("--llm-rerank", type=int, default=0, metavar="N", help="LLM reorders the first N pages (0 = off)")
    parser.add_argument("--llm-rerank-plain", action="store_true", help="show the reranker only the best hit's opening")
    parser.add_argument("--llm-rerank-unit", choices=("page", "section"), default="page")
    parser.add_argument("--hyde", choices=("dense", "both"), help="add a hypothetical answer passage to the dense (or both) queries")
    parser.add_argument("--hyde-scope", choices=("all", "shared"), default="all",
                        help="shared: HyDE only for the course side; with --owner, notes are searched with the plain query")
    parser.add_argument("--generate-only", action="store_true",
                        help="only run and cache the query-side LLM calls (no model server needed)")
    parser.add_argument("--understand", action="store_true", help="LLM query understanding first (needs LLM_API_KEY)")
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--summaries", action="store_true", help="fuse in section summary/question hits (E5)")
    parser.add_argument("--summary-weight", type=float, default=1.0)
    parser.add_argument("--multihop", choices=("decompose", "iterative"), help="multi-hop retrieval (E9)")
    parser.add_argument("--embed-model", default="BAAI/bge-m3", help="query embedding model (match the collection)")
    parser.add_argument("--query-prompt", help="the embedding model's query prompt name (Qwen3: query)")
    parser.add_argument("--no-history", action="store_true", help="multi-turn rows: drop the conversation so far")
    parser.add_argument("--question-field", default="question",
                        help="row field to ask (multi-turn upper bound: original_question)")
    parser.add_argument("--dense-query", choices=("rewritten", "raw"), default="rewritten",
                        help="with --understand: what the dense side embeds")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--owner", help="also search this account's notes")
    parser.add_argument("--exclude-labels", nargs="*", default=["not_applicable"],
                        help="audit labels left out of the scores (pass none to keep every question)")
    args = parser.parse_args()
    spans.configure(engine())
    result = run(Config(
        exp=args.exp, collection=args.collection, top_k=args.top_k, body_chars=args.body_chars,
        bm25_body_chars=args.bm25_body_chars, mode=args.mode, rerank_top=args.rerank,
        rerank_body_chars=args.rerank_body_chars, llm_rerank=args.llm_rerank, llm_rerank_plain=args.llm_rerank_plain, llm_rerank_unit=args.llm_rerank_unit, hyde=args.hyde or "", hyde_scope=args.hyde_scope,
        generate_only=args.generate_only, understand=args.understand, llm_workers=args.llm_workers,
        summaries=args.summaries, summary_weight=args.summary_weight, dense_query=args.dense_query,
        history=not args.no_history, question_field=args.question_field, multihop=args.multihop or "",
        embed_model=args.embed_model, query_prompt=args.query_prompt,
        dataset=args.dataset, owner=args.owner,
        exclude_labels=frozenset(args.exclude_labels)))
    print(json.dumps({k: result[k] for k in ("exp", "collection", "excluded", "overall", "by_library", "by_label",
                                             "search_ms", "saved_to", "trace_id") if k in result},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
