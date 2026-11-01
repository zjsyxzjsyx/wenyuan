"""Ingestion as Celery tasks: build_library fans out one process_page -> embed_and_index chain per page.

Tasks are safe to retry: documents are upserted and a page's chunks are replaced
by doc_id (see app.ingest.pipeline). The trace started by the submitter travels in
a `trace` argument, so the dashboard shows one tree across all workers.
"""
from __future__ import annotations

from functools import lru_cache

from celery import chain, group
from sqlalchemy.orm import Session as DBSession

from app.db.models import create_tables, engine
from app.ingest import pipeline
from app.ingest.extract import Page
from app.models.local import ModelServer
from app.retrieval import milvus_store
from app.tokenizer import TiktokenTokenizer
from app.tracing import spans
from app.workers.celery_app import app

spans.configure(engine())
create_tables()
RETRY = dict(autoretry_for=(Exception,), retry_backoff=True, retry_backoff_max=60, max_retries=3)


@lru_cache
def _tokenizer() -> TiktokenTokenizer:
    return TiktokenTokenizer()


@lru_cache
def _models() -> ModelServer:
    return ModelServer()


@lru_cache
def _milvus():
    return milvus_store.client()


@app.task(**RETRY)
def build_library(library: str, variant: str, trace: dict | None = None) -> dict:
    """Extract and de-duplicate a library, then queue one chain per kept page. Returns the group id."""
    with spans.attach(trace or {}), spans.span("ingest", f"build_library {library}", variant=variant) as s:
        pages, duplicates = pipeline.prepare(library)
        kept = [url for url in pages if url not in duplicates]
        with DBSession(engine()) as db:
            pipeline.drop_stale_documents(db, library, kept)
            db.commit()
        carrier = spans.inject()
        jobs = group(
            chain(process_page.s(vars(pages[url]), variant, carrier),
                  embed_and_index.s(f"chunks_{variant}", carrier))
            for url in kept
        ).apply_async()
        jobs.save()
        s.set(pages=len(kept), near_dup=len(duplicates))
        return {"library": library, "pages": len(kept), "near_dup": len(duplicates), "group_id": jobs.id}


@app.task(**RETRY)
def process_page(page: dict, variant: str, trace: dict | None = None) -> dict:
    with spans.attach(trace or {}), spans.span("ingest", "process_page", url=page["url"]) as s:
        with DBSession(engine()) as db:
            doc_id, rows = pipeline.process_page(db, _tokenizer(), Page(**page), variant)
            db.commit()
        s.set(chunks=len(rows))
        return {"doc_id": doc_id, "rows": rows}


@app.task(**RETRY)
def embed_and_index(result: dict, collection: str, trace: dict | None = None) -> int:
    with spans.attach(trace or {}), spans.span("ingest", "embed_and_index", doc_id=result["doc_id"]) as s:
        rows = result["rows"]
        vectors = _models().embed([r["text"] for r in rows]) if rows else []
        pipeline.index_chunks(_milvus(), collection, result["doc_id"], rows, vectors)
        s.set(chunks=len(rows))
        return len(rows)
