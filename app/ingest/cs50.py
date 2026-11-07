"""CS50x 2024 course materials as a course library (v2 classroom scenario, design doc section 0).

`python -m app.ingest.cs50 [--folder data/raw/cs50/source_markdown_files]`
The files come from the Hugging Face dataset dev-jonathanb/cs50-educational-rag (CC BY-NC-SA 4.0,
kept out of the repository): per week, the lecture notes, a lecture transcript, the problem set
and its exercises, each with front matter giving the original link, title, type and week. The
link becomes the page URL, so citations point at the real CS50 page (or the lecture video).

Cleaning before the shared pipeline (sections -> chunks -> vectors):
- headings are links ("## [Binary Search](https://...#binary-search)"): keep the text, or every
  heading path would carry a URL. The link itself goes to the section's `url`, so a citation can
  open the exact place: an anchor on the notes page, or the second of the lecture video it starts at
  (`--links-only` writes these links for an already indexed course: MySQL only, no embedding);
- the notes open with a table of contents of anchor links, the transcripts with a timestamp list:
  dropped, they would only become chunks that match everything weakly;
- transcripts put each timestamp on a line of its own: dropped.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.cache.answers import invalidate_now
from app.db.models import Section, create_tables, engine
from app.ingest import pipeline
from app.ingest.page import Page
from app.models.local import ModelServer
from app.query.ask import COLLECTION
from app.retrieval import milvus_store
from app.tokenizer import TiktokenTokenizer
from app.tracing import spans

LIBRARY = "cs50"
FOLDER = Path("data/raw/cs50/source_markdown_files")
DOC_TYPES = {"notes": "notes", "transcription": "transcript", "problemset": "problemset", "exercise": "exercise"}

_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_HEADING_LINK = re.compile(r"^(#+[ \t]+)\[(.+?)\]\(([^)]*)\)[ \t]*$", re.M)
_TOC_LINE = re.compile(r"^[ \t]*- \[[^\]]+\]\(#[^)]*\)[ \t]*\n", re.M)
_TRANSCRIPT_TOC = re.compile(r"^TABLE OF CONTENTS:[ \t]*\n(?:[ \t]*\n)*(?:\d{1,2}(?::\d{2}){1,2} - .*\n)+", re.M)
_TIMESTAMP_LINE = re.compile(r"^\d{1,2}(?::\d{2}){1,2}[ \t]*\n", re.M)


def clean(body: str) -> str:
    body = _TRANSCRIPT_TOC.sub("", body)
    body = _TOC_LINE.sub("", body)
    body = _TIMESTAMP_LINE.sub("", body)
    body = _HEADING_LINK.sub(r"\1\2", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def heading_links(text: str) -> list[tuple[str, str]]:
    """(heading text, link) for every linked heading of a course file, in order."""
    return [(m.group(2).strip(), m.group(3).strip()) for m in _HEADING_LINK.finditer(text.replace("\r\n", "\n"))]


def section_links(sections, links: list[tuple[str, str]]) -> dict[int, str]:
    """Section index -> link, pairing headings in order (a lecture can have two segments of the same name)."""
    out, pending = {}, list(links)
    for section in sorted(sections, key=lambda s: s.idx):
        found = next((i for i, (heading, _) in enumerate(pending) if heading == section.heading), None)
        if found is not None:
            out[section.idx] = pending.pop(found)[1]
    return out


def link_sections(db: DBSession, page: Page, raw: str) -> int:
    """Write each section's link from the original file; returns how many sections got one."""
    links = section_links(db.scalars(select(Section).where(Section.doc_id == pipeline.doc_id_for(page.url))),
                          heading_links(raw))
    for section in db.scalars(select(Section).where(Section.doc_id == pipeline.doc_id_for(page.url))):
        section.url = links.get(section.idx)
    db.commit()
    return len(links)


def load(text: str) -> tuple[Page, str]:
    """(page, doc_type) of one course file."""
    text = text.replace("\r\n", "\n")
    match = _FRONT.match(text)
    if not match:
        raise ValueError("no front matter")
    meta = yaml.safe_load(match.group(1))
    week = str(meta.get("week/lecture", "")).strip()
    title = str(meta["title"]).strip()
    title = f"Week {week.upper() if week.isalpha() else week} · {title}" if week else title
    return (Page(url=meta["link"], library=LIBRARY, title=title, markdown=clean(text[match.end():])),
            DOC_TYPES.get(meta.get("type"), "notes"))


def build(folder: Path = FOLDER, collection: str = COLLECTION) -> Counter:
    create_tables()
    raws = [p.read_text(encoding="utf-8") for p in sorted(folder.rglob("*.md"))]
    pages = [load(raw) for raw in raws]
    urls = Counter(page.url for page, _ in pages)
    if duplicated := [u for u, n in urls.items() if n > 1]:
        raise ValueError(f"several files share a link: {duplicated}")
    mc, models, tokenizer = milvus_store.client(), ModelServer(), TiktokenTokenizer()
    milvus_store.create_collection(mc, collection)
    chunks: Counter = Counter()
    with spans.span("ingest", "cs50", files=len(pages), collection=collection) as s, DBSession(engine()) as db:
        pipeline.drop_stale_documents(db, LIBRARY, list(urls))
        for (page, kind), raw in zip(pages, raws):
            doc_id, rows = pipeline.process_page(db, tokenizer, page, "struct_merge")
            db.commit()
            link_sections(db, page, raw)
            for row in rows:
                row["doc_type"] = kind
            pipeline.index_chunks(mc, collection, doc_id, rows, models.embed([r["text"] for r in rows]) if rows else [])
            chunks[kind] += len(rows)
        s.set(chunks=sum(chunks.values()), **{f"chunks_{k}": v for k, v in chunks.items()})
    invalidate_now(LIBRARY)        # answers cached before this build may cite stale text
    return chunks


def relink(folder: Path = FOLDER) -> int:
    """Section links for an indexed course, from its files: MySQL only."""
    create_tables()
    with DBSession(engine()) as db:
        return sum(link_sections(db, load(raw)[0], raw)
                   for raw in (p.read_text(encoding="utf-8") for p in sorted(folder.rglob("*.md"))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=FOLDER)
    parser.add_argument("--links-only", action="store_true", help="only write section links (no embedding, no Milvus)")
    args = parser.parse_args()
    if args.links_only:
        print(f"{LIBRARY}: {relink(args.folder)} sections linked")
        return
    spans.configure(engine())
    chunks = build(args.folder)
    print(f"{LIBRARY}: {sum(chunks.values())} chunks into {COLLECTION}", dict(chunks))


if __name__ == "__main__":
    main()
