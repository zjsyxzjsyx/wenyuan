"""Users' own notes: the public ingestion path (sections -> chunks -> vectors), tagged with an owner.

A note is stored as library "private" at the URL private://<owner>/<filename>, so
uploading the same filename again replaces it (same doc_id, delete-then-insert),
and two users' files never collide. Notes are small (a few KB of Markdown), so they
are indexed inline rather than through the Celery pipeline built for whole sites.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

from pymilvus import MilvusClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section
from app.ingest import pipeline
from app.ingest.extract import Page
from app.models.local import ModelServer
from app.retrieval.milvus_store import PRIVATE
from app.tokenizer import Tokenizer

EXTENSIONS = {".md", ".markdown", ".txt"}
MAX_BYTES = 1_000_000
VARIANT = "struct_merge"


class NoteError(ValueError):
    pass


def note_url(owner_id: int, filename: str) -> str:
    return f"private://{owner_id}/{filename}"


def clean_filename(filename: str) -> str:
    name = PurePosixPath(filename.replace("\\", "/")).name.strip()
    if not name or PurePosixPath(name).suffix.lower() not in EXTENSIONS:
        raise NoteError(f"expected a {'/'.join(sorted(EXTENSIONS))} file name, got {filename!r}")
    return name


def title_of(markdown: str, filename: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown, flags=re.M)
    return match.group(1).strip() if match else PurePosixPath(filename).stem


def add_note(db: DBSession, mc: MilvusClient, models: ModelServer, tokenizer: Tokenizer, collection: str,
             owner_id: int, filename: str, markdown: str) -> dict:
    filename = clean_filename(filename)
    if len(markdown.encode()) > MAX_BYTES:
        raise NoteError(f"note is larger than {MAX_BYTES // 1000} KB")
    if not markdown.strip():
        raise NoteError("note is empty")
    page = Page(url=note_url(owner_id, filename), library=PRIVATE, title=title_of(markdown, filename),
                markdown=markdown)
    doc_id, rows = pipeline.process_page(db, tokenizer, page, VARIANT)
    db.get(Document, doc_id).owner_id = owner_id
    db.commit()
    rows = [{**r, "owner_id": owner_id} for r in rows]
    vectors = models.embed([r["text"] for r in rows]) if rows else []
    pipeline.index_chunks(mc, collection, doc_id, rows, vectors)
    return {"doc_id": doc_id, "filename": filename, "title": page.title, "chunks": len(rows)}


def list_notes(db: DBSession, owner_id: int) -> list[dict]:
    docs = db.scalars(select(Document).where(Document.owner_id == owner_id, Document.library == PRIVATE)
                      .order_by(Document.url))
    return [{"doc_id": d.id, "filename": d.url.rsplit("/", 1)[-1], "title": d.title,
             "updated_at": d.updated_at.isoformat() if d.updated_at else None} for d in docs]


def delete_note(db: DBSession, mc: MilvusClient, collection: str, owner_id: int, doc_id: str) -> bool:
    """False if there is no such note of this owner (someone else's id looks exactly like a missing one)."""
    doc = db.get(Document, doc_id)
    if doc is None or doc.owner_id != owner_id or doc.library != PRIVATE:
        return False
    mc.delete(collection, filter=f'doc_id == "{doc_id}"')
    db.execute(delete(Section).where(Section.doc_id == doc_id))
    db.delete(doc)
    db.commit()
    return True
