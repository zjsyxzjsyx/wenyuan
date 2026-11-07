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
from app.ingest.page import Page
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


SKIP_DIRS = {".git", ".venv", "node_modules", "site-packages", ".cache", ".pytest_cache", "__pycache__"}


FOLDER_EXTENSIONS = {".md", ".markdown"}   # a folder holds code projects' .txt files too; take notes only


def note_names(roots: list, exclude: tuple[str, ...] = ()) -> dict:
    """Note name per file under `roots` (recursive). Files at a folder's top level keep their own name
    (as single-folder imports always did); files in subfolders are named by their path in the folder
    ("mcp__README.md"), and names that still clash across folders get the folder's name in front.
    A file is skipped when any part of its path in the folder matches an `exclude` glob."""
    from fnmatch import fnmatch
    found = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if (path.is_file() and path.suffix.lower() in FOLDER_EXTENSIONS and not SKIP_DIRS.intersection(rel.parts[:-1])
                    and not any(fnmatch(part, pattern) for part in rel.parts for pattern in exclude)):
                found.append((root, path, "__".join(rel.parts)))
    counts: dict[str, int] = {}
    for _, _, name in found:
        counts[name] = counts.get(name, 0) + 1
    return {path: (f"{root.name}__{name}" if counts[name] > 1 else name) for root, path, name in found}


def main() -> None:
    """`python -m app.ingest.private <folder>... --user <name>`: add every note under the folders
    (same name replaces; names that already exist are listed first)."""
    import argparse
    from pathlib import Path

    from app.cache.answers import invalidate_now, private_scope
    from app.db.models import User, create_tables, engine
    from app.query.ask import COLLECTION
    from app.retrieval import milvus_store
    from app.tokenizer import TiktokenTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--user", required=True, help="account name that owns the notes")
    parser.add_argument("--exclude", nargs="*", default=[], help="glob patterns of files or folders to leave out")
    args = parser.parse_args()
    create_tables()
    with DBSession(engine()) as db:
        owner = db.scalar(select(User.id).where(User.name == args.user))
        existing = {n["filename"] for n in list_notes(db, owner)} if owner is not None else set()
    if owner is None:
        raise SystemExit(f"no account named {args.user!r}")
    names = note_names(args.folders, tuple(args.exclude))
    if replaced := sorted(existing.intersection(names.values())):
        print(f"replacing {len(replaced)} existing notes of the same name: {replaced[:10]}")
    mc, models, tokenizer = milvus_store.client(), ModelServer(), TiktokenTokenizer()
    milvus_store.ensure_owner_field(mc, COLLECTION)
    added = chunks = 0
    for path, name in names.items():
        with DBSession(engine()) as db:
            try:
                note = add_note(db, mc, models, tokenizer, COLLECTION, owner, name, path.read_text(encoding="utf-8"))
            except (NoteError, UnicodeDecodeError) as exc:
                print(f"skipped {path}: {exc}")
                continue
        added += 1
        chunks += note["chunks"]
    invalidate_now(private_scope(owner))       # cached answers for this user may now be stale
    print(f"{added} of {len(names)} notes, {chunks} chunks -> {args.user}")


if __name__ == "__main__":
    main()
