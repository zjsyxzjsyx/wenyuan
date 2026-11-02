"""Cross-library questions (E9): answering needs one page from each of two libraries.

`python -m eval.build_multihop --n 120`  ->  eval/datasets/crosslib_v1.jsonl
A learner's harder questions span libraries ("cache a FastAPI endpoint in Redis",
"route Celery tasks through RabbitMQ"). For a pair of libraries used together, a
guide section of library A is drawn, the closest section of library B is found by
dense search, and a writer model (the JUDGE_* model, not the answering model)
writes a question that needs both, or declines when the two do not combine
naturally. The gold is both pages; the metric is whether retrieval finds both.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section, engine
from app.ingest.pipeline import doc_type
from app.ingest.sites import SITES
from app.models.local import ModelServer
from app.retrieval import milvus_store
from app.retrieval.context import _children, _subtree_text
from eval.answers import judge_llm

OUT = Path("eval/datasets/crosslib_v1.jsonl")
COLLECTION = "chunks_struct_merge"
PAIRS = [("fastapi", "redis"), ("django", "postgresql"), ("celery", "rabbitmq"), ("celery", "redis"),
         ("flask", "sqlite"), ("docker", "nginx"), ("kubernetes", "docker"), ("pytorch", "numpy"),
         ("pandas", "numpy"), ("django", "celery"), ("fastapi", "docker"), ("python", "sqlite")]

PROMPT = """You write evaluation questions for a documentation search system.

Below are two documentation sections from different projects that are often used together.
Write ONE question a learner might really ask whose answer needs information from BOTH sections
(for example, how to use the feature of project A together with project B). The question must not be
answerable from either section alone, and should not quote the section titles verbatim.
If the two sections do not combine into a natural question, set "skip" to true.

Output one JSON object: {{"skip": false, "question": "...", "needs_a": "<what it takes from A>", "needs_b": "<what it takes from B>"}}

---Section A ({lib_a})---
{text_a}

---Section B ({lib_b})---
{text_b}"""


def guide_blocks(db: DBSession, library: str) -> list[tuple[Document, str]]:
    docs = [d for d in db.scalars(select(Document).where(Document.library == library)) if doc_type(d.url) == "guide"]
    blocks = []
    for doc in docs:
        tree = {s.idx: s for s in db.scalars(select(Section).where(Section.doc_id == doc.id))}
        for root in (i for i, s in tree.items() if s.parent_idx is None and s.level > 0):
            text = _subtree_text(tree, _children(tree), root)
            if 500 <= len(text) <= 6000:
                blocks.append((doc, f"{tree[root].heading}\n{text}"))
    return blocks


def make(writer, models: ModelServer, mc, db_blocks: dict, pair: tuple[str, str], rng: random.Random) -> dict | None:
    lib_a, lib_b = pair
    doc_a, text_a = rng.choice(db_blocks[lib_a])
    hits = milvus_store.dense_search(mc, COLLECTION, models.embed([text_a[:1500]])[0], top_k=10, libraries=[lib_b])
    guides = [h for h in hits if h["doc_type"] == "guide"] or hits
    if not guides:
        return None
    hit = guides[0]
    with DBSession(engine()) as db:
        doc_b = db.get(Document, hit["doc_id"])
        tree = {s.idx: s for s in db.scalars(select(Section).where(Section.doc_id == hit["doc_id"]))}
    text_b = _subtree_text(tree, _children(tree), hit["parent_index"]) if hit["parent_index"] in tree else hit["text"]
    raw = writer.chat_json([{"role": "user", "content": PROMPT.format(
        lib_a=SITES[lib_a].name, lib_b=SITES[lib_b].name, text_a=text_a[:3000], text_b=text_b[:3000])}])
    question = str(raw.get("question") or "").strip()
    if raw.get("skip") or not question or doc_a.url == doc_b.url:
        return None
    return {"id": f"xl-{doc_a.id[:8]}-{doc_b.id[:8]}", "type": "crosslib", "question": question, "body": "",
            "libraries": [lib_a, lib_b], "library": lib_a, "gold_urls": [doc_a.url, doc_b.url],
            "needs": [raw.get("needs_a"), raw.get("needs_b")]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120, help="attempts; declined pairs are dropped")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    with DBSession(engine()) as db:
        blocks = {lib: guide_blocks(db, lib) for lib in {l for pair in PAIRS for l in pair}}
    pairs = [p for p in PAIRS if blocks[p[0]]]
    rng = random.Random(args.seed)
    jobs = [(rng.choice(pairs), random.Random(args.seed * 1000 + i)) for i in range(args.n)]
    writer, models, mc = judge_llm(), ModelServer(), milvus_store.client()

    def one(job):
        pair, local_rng = job
        try:
            return contextvars.copy_context().run(make, writer, models, mc, blocks, pair, local_rng)
        except Exception as exc:  # noqa: BLE001 - a failed attempt is just not used
            print("skipped:", type(exc).__name__, exc)
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for r in pool.map(one, jobs) if r]
    rows = list({r["id"]: r for r in rows}.values())
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} cross-library questions from {args.n} attempts -> {OUT}")


if __name__ == "__main__":
    main()
