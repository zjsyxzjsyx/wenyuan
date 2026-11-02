"""Build one index variant synchronously, in this process.

`python -m app.ingest.build redis fastapi --variant struct_merge`
Handy for small libraries and tests; large builds go through Celery
(`python -m app.workers.submit ...`), which runs the same pipeline steps.
The Milvus collection `chunks_<variant>` is rebuilt from scratch.
"""
from __future__ import annotations

import argparse
import time

from sqlalchemy.orm import Session as DBSession

from app.cache.answers import invalidate_now
from app.db.models import create_tables, engine
from app.ingest import pipeline
from app.ingest.sites import SITES
from app.models.local import ModelServer
from app.retrieval import milvus_store
from app.tokenizer import TiktokenTokenizer


def build(libraries: list[str], variant: str) -> dict[str, int]:
    tokenizer, models, mc = TiktokenTokenizer(), ModelServer(), milvus_store.client()
    collection = f"chunks_{variant}"
    milvus_store.create_collection(mc, collection, drop=True)
    create_tables()
    stats = {"pages": 0, "near_dup": 0, "chunks": 0}
    began = time.perf_counter()
    with DBSession(engine()) as db:
        for library in libraries:
            pages, duplicates = pipeline.prepare(library)
            stats["near_dup"] += len(duplicates)
            kept = [url for url in pages if url not in duplicates]
            pipeline.drop_stale_documents(db, library, kept)
            for url in kept:
                doc_id, rows = pipeline.process_page(db, tokenizer, pages[url], variant)
                vectors = models.embed([r["text"] for r in rows]) if rows else []
                pipeline.index_chunks(mc, collection, doc_id, rows, vectors)
                stats["pages"] += 1
                stats["chunks"] += len(rows)
            db.commit()
            invalidate_now(library)
    stats["seconds"] = round(time.perf_counter() - began)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("libraries", nargs="+", choices=sorted(SITES))
    parser.add_argument("--variant", choices=pipeline.VARIANTS, default="struct_merge")
    args = parser.parse_args()
    print(build(args.libraries, args.variant))


if __name__ == "__main__":
    main()
