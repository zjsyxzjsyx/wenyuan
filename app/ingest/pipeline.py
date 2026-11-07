"""Ingestion steps shared by the course loader (app.ingest.cs50), teachers' uploads
(app.ingest.materials) and students' notes (app.ingest.private).

    page -> sections (MySQL) -> chunks -> embed -> Milvus

Every step is keyed by url/doc_id so it can be retried: documents are upserted in
MySQL and a document's chunks are deleted from Milvus before its new ones go in.
v0.7 also fed crawled and DevDocs pages through here, from a build script and Celery
tasks; those went with v0.7's public libraries (see `git tag v0.7.0`).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict

from pymilvus import MilvusClient
from sqlalchemy import delete
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section
from app.ingest.chunking.structural import Chunk, chunk_fixed, chunk_sections
from app.ingest.page import Page
from app.ingest.sections import split_sections
from app.ingest.sites import SITES
from app.retrieval import milvus_store
from app.tokenizer import Tokenizer

VARIANTS = ("fixed", "struct", "struct_nohdr", "struct_merge")

# API reference pages (one per function/command/class) vs guides and tutorials. The E3 bad
# cases showed the reranker favouring dense reference chunks over the tutorial an answer cites.
_REFERENCE = re.compile(r"/(reference|api|generated|c3ref|commands|ref|man)/|/sql-[a-z]+$|/git-[a-z-]+$")


def doc_id_for(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def doc_type(url: str) -> str:
    return "reference" if _REFERENCE.search(url) else "guide"


def drop_stale_documents(db: DBSession, library: str, kept_urls: list[str]) -> None:
    """Documents from earlier builds that are now excluded must not linger as citable sources."""
    db.execute(delete(Document).where(Document.library == library,
                                      Document.id.not_in([doc_id_for(u) for u in kept_urls])))


def save_document(db: DBSession, page: Page, sections) -> str:
    doc_id = doc_id_for(page.url)
    content_hash = hashlib.sha256(page.markdown.encode()).hexdigest()
    existing = db.get(Document, doc_id)
    if not (existing and existing.content_hash == content_hash):
        db.execute(delete(Section).where(Section.doc_id == doc_id))
        db.merge(Document(id=doc_id, library=page.library, url=page.url, title=page.title[:512],
                          content_hash=content_hash))
        db.flush()
        db.add_all(Section(doc_id=doc_id, idx=s.index, parent_idx=s.parent_index, level=s.level,
                           heading=s.heading[:512], path=s.path, text=s.text) for s in sections)
    return doc_id


def chunk_page(tokenizer: Tokenizer, page: Page, sections, variant: str, site_name: str | None = None) -> list[Chunk]:
    """`site_name` heads every chunk's context; it defaults to the site's name ("My notes" for notes)."""
    doc_id = doc_id_for(page.url)
    if variant == "fixed":
        return chunk_fixed(tokenizer, page.markdown, library=page.library, url=page.url, doc_id=doc_id)
    site_name = site_name or (SITES[page.library].name if page.library in SITES else "My notes")
    return chunk_sections(tokenizer, sections, library=page.library, site_name=site_name,
                          url=page.url, doc_id=doc_id, context_header=variant != "struct_nohdr",
                          merge_sections=variant == "struct_merge")


def process_page(db: DBSession, tokenizer: Tokenizer, page: Page, variant: str,
                 site_name: str | None = None) -> tuple[str, list[dict]]:
    """CPU half of a page: sections into MySQL, chunks out as plain dicts (they cross a queue)."""
    sections = split_sections(page.markdown, page.title)
    doc_id = save_document(db, page, sections)
    rows = []
    for chunk in chunk_page(tokenizer, page, sections, variant, site_name):
        row = asdict(chunk)
        row.update(doc_id=doc_id, doc_type=doc_type(page.url))
        rows.append(row)
    return doc_id, rows


def index_chunks(mc: MilvusClient, collection: str, doc_id: str, rows: list[dict],
                 vectors: list[list[float]]) -> None:
    """GPU half: replace this document's chunks. Delete-then-insert keeps retries idempotent."""
    mc.delete(collection, filter=f'doc_id == "{doc_id}"')
    if rows:
        milvus_store.insert_rows(mc, collection, [{**r, "dense": v} for r, v in zip(rows, vectors, strict=True)])
