"""The `ask` chain end to end: understand -> retrieve -> rerank -> small-to-big context -> cited answer.

`python -m app.query.ask "How do I delete everything in Redis?" --library redis`
Every stage is a span under one trace, so a slow or wrong answer can be traced to
the stage that caused it. The HTTP API and the MCP server call `ask()`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

from sqlalchemy.orm import Session as DBSession

from app.db.models import engine
from app.generation.answer import REFUSAL, Answer, answer
from app.ingest.summarize import COLLECTION as SUMMARIES
from app.ingest.sites import SITES
from app.models.llm import LLM
from app.models.local import ModelServer
from app.query.understand import Understanding, understand
from app.retrieval import milvus_store
from app.retrieval.context import build_sources, parent_blocks
from app.retrieval.fusion import best_per_block, rrf
from app.retrieval.rerank import rerank
from app.tokenizer import TiktokenTokenizer
from app.tracing import spans

COLLECTION = "chunks_struct_merge"


@dataclass
class AskConfig:
    collection: str = COLLECTION
    # Defaults are the best measured setting (W2, 1184 SO questions): hybrid search with the
    # LLM's candidate API names on the BM25 side. Reranking on top of that was significantly
    # worse (Recall@5 -2.4, +440 ms): the reranker prefers terse API reference pages.
    mode: str = "hybrid"         # dense | hybrid
    top_k: int = 50
    rerank_top: int = 0          # rerank the first N hits; 0: off
    summaries: bool = False      # also search section summaries/questions (E5)
    budget: int = 6000           # context tokens for the answer


@dataclass
class AskResult:
    answer: Answer
    understanding: Understanding | None
    trace_id: str = ""
    retrieved: list[dict] = field(default_factory=list)   # {"url", "heading_path", "score"} in final order


def ask(question: str, history: list[dict] | None = None, libraries: list[str] | None = None,
        cfg: AskConfig | None = None, llm: LLM | None = None, models: ModelServer | None = None) -> AskResult:
    cfg, llm, models = cfg or AskConfig(), llm or LLM(), models or ModelServer()
    tokenizer = TiktokenTokenizer()
    with spans.span("request", "ask", mode=cfg.mode, rerank_top=cfg.rerank_top) as root:
        u = understand(llm, question, history, libraries=libraries or sorted(SITES))
        if not u.needs_retrieval:
            return AskResult(Answer(False, REFUSAL), u, root.trace_id)
        # The router may narrow the libraries; an empty answer means "could be any" of those offered.
        scope = u.libraries or libraries or None
        with spans.span("retrieval", f"{cfg.mode}_search", libraries=scope) as s:
            # The standalone question keeps the learner's details (error text, code) and has its
            # references resolved; exact names go to BM25 through bm25_text().
            vector = models.embed([u.standalone_question])[0]
            mc = milvus_store.client()
            if cfg.mode == "hybrid":
                hits = milvus_store.hybrid_search(mc, cfg.collection, vector, u.bm25_text(), top_k=cfg.top_k,
                                                  libraries=scope)
            else:
                hits = milvus_store.dense_search(mc, cfg.collection, vector, top_k=cfg.top_k, libraries=scope)
            if cfg.summaries:
                extra = milvus_store.dense_search(mc, SUMMARIES, vector, top_k=cfg.top_k, libraries=scope)
                hits = rrf([hits, best_per_block(extra)], limit=cfg.top_k)
            s.set(hits=len(hits))
        if cfg.rerank_top and hits:
            with spans.span("retrieval", "rerank", n=min(cfg.rerank_top, len(hits))):
                head = hits[:cfg.rerank_top]
                scores = rerank(models, tokenizer, u.standalone_question, [h["text"] for h in head])
                head = [{**h, "score": sc} for sc, h in sorted(zip(scores, head), key=lambda p: -p[0])]
                hits = head + hits[cfg.rerank_top:]
        with spans.span("context", "small_to_big", budget=cfg.budget) as s:
            with DBSession(engine()) as db:
                sources = build_sources(tokenizer, parent_blocks(db, hits, tokenizer), budget=cfg.budget)
            s.set(sources=len(sources), tokens=sum(src.tokens for src in sources))
        result = answer(llm, u.standalone_question, sources)
        root.set(found=result.found, citations=len(result.citations))
        return AskResult(result, u, root.trace_id,
                         [{"url": h["url"], "heading_path": h.get("heading_path", ""), "score": round(h["score"], 4)}
                          for h in hits[:10]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--library", action="append", help="restrict to these libraries (repeatable)")
    parser.add_argument("--mode", choices=("dense", "hybrid"), default="hybrid")
    parser.add_argument("--rerank", type=int, default=0)
    parser.add_argument("--summaries", action="store_true")
    args = parser.parse_args()
    spans.configure(engine())
    r = ask(args.question, libraries=args.library, cfg=AskConfig(mode=args.mode, rerank_top=args.rerank, summaries=args.summaries))
    print(json.dumps({"understanding": asdict(r.understanding) if r.understanding else None,
                      "answer": asdict(r.answer), "retrieved": r.retrieved[:5], "trace_id": r.trace_id},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
