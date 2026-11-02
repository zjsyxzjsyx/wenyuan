"""A course's materials uploaded by its teacher: the notes path (sections -> chunks -> vectors),
stored under the course's library id so everyone enrolled searches them.

An uploaded file lives at course://<course>/<filename>; uploading the same name again replaces it.
Files brought in by an importer (app.ingest.cs50) keep their own URLs and are listed and deleted
the same way, by document id within the course.
"""
from __future__ import annotations

from pymilvus import MilvusClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section
from app.ingest import pipeline
from app.ingest.extract import Page
from app.ingest.private import MAX_BYTES, VARIANT, NoteError, clean_filename, title_of
from app.models.local import ModelServer
from app.tokenizer import Tokenizer


def material_url(course_id: str, filename: str) -> str:
    return f"course://{course_id}/{filename}"


def add_material(db: DBSession, mc: MilvusClient, models: ModelServer, tokenizer: Tokenizer, collection: str,
                 course_id: str, course_name: str, filename: str, markdown: str) -> dict:
    filename = clean_filename(filename)
    if len(markdown.encode()) > MAX_BYTES:
        raise NoteError(f"file is larger than {MAX_BYTES // 1000} KB")
    if not markdown.strip():
        raise NoteError("file is empty")
    page = Page(url=material_url(course_id, filename), library=course_id, title=title_of(markdown, filename),
                markdown=markdown)
    doc_id, rows = pipeline.process_page(db, tokenizer, page, VARIANT, site_name=course_name)
    db.commit()
    for row in rows:
        row["doc_type"] = "material"
    pipeline.index_chunks(mc, collection, doc_id, rows, models.embed([r["text"] for r in rows]) if rows else [])
    return {"doc_id": doc_id, "filename": filename, "title": page.title, "chunks": len(rows)}


def list_materials(db: DBSession, course_id: str) -> list[dict]:
    docs = db.scalars(select(Document).where(Document.library == course_id, Document.owner_id.is_(None))
                      .order_by(Document.title))
    return [{"doc_id": d.id, "title": d.title, "url": d.url,
             "updated_at": d.updated_at.isoformat() if d.updated_at else None} for d in docs]


def delete_material(db: DBSession, mc: MilvusClient, collection: str, course_id: str, doc_id: str) -> bool:
    """False when the document is not this course's (another course's id looks exactly like a missing one)."""
    doc = db.get(Document, doc_id)
    if doc is None or doc.library != course_id or doc.owner_id is not None:
        return False
    mc.delete(collection, filter=f'doc_id == "{doc_id}"')
    db.execute(delete(Section).where(Section.doc_id == doc_id))
    db.delete(doc)
    db.commit()
    return True
