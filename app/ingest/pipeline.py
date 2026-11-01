"""Ingestion steps shared by the synchronous build script and the Celery tasks.

    source pages -> extract -> drop near-duplicates -> sections (MySQL) -> chunks -> embed -> Milvus

Every step is keyed by url/doc_id so it can be retried: documents are upserted in
MySQL and a document's chunks are deleted from Milvus before its new ones go in.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict

from pymilvus import MilvusClient
from sqlalchemy import delete
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section
from app.ingest import devdocs
from app.ingest.chunking.structural import Chunk, chunk_fixed, chunk_sections
from app.ingest.crawl import cached_pages, page_path
from app.ingest.dedup import near_duplicates
from app.ingest.extract import Page, extract
from app.ingest.sections import split_sections
from app.ingest.sites import SITES, Site
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


def source_urls(site: Site) -> list[str]:
    if site.source == "devdocs":
        return list(devdocs.pages(site.library))
    return [url for url in cached_pages(site.library) if site.wants(url)]


def load_html(site: Site, url: str) -> str:
    if site.source == "devdocs":
        return devdocs.pages(site.library)[url]
    return page_path(site.library, url).read_text(encoding="utf-8")


def load_page(library: str, url: str) -> Page | None:
    site = SITES[library]
    page = extract(site, url, load_html(site, url))
    return page if page and page.markdown else None


def prepare(library: str) -> tuple[dict[str, Page], dict[str, str]]:
    """Extract every wanted page of a library; return (pages, near-duplicate url -> kept url)."""
    site = SITES[library]
    pages = {url: page for url in source_urls(site) if (page := load_page(library, url))}
    return pages, near_duplicates({url: p.markdown for url, p in pages.items()})


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


def chunk_page(tokenizer: Tokenizer, page: Page, sections, variant: str) -> list[Chunk]:
    doc_id = doc_id_for(page.url)
    if variant == "fixed":
        return chunk_fixed(tokenizer, page.markdown, library=page.library, url=page.url, doc_id=doc_id)
    return chunk_sections(tokenizer, sections, library=page.library, site_name=SITES[page.library].name,
                          url=page.url, doc_id=doc_id, context_header=variant != "struct_nohdr",
                          merge_sections=variant == "struct_merge")


def process_page(db: DBSession, tokenizer: Tokenizer, page: Page, variant: str) -> tuple[str, list[dict]]:
    """CPU half of a page: sections into MySQL, chunks out as plain dicts (they cross a queue)."""
    sections = split_sections(page.markdown, page.title)
    doc_id = save_document(db, page, sections)
    rows = []
    for chunk in chunk_page(tokenizer, page, sections, variant):
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
