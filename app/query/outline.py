"""`outline` (a library's section trees, optionally around a topic) and `get_section` (one section in full).

These serve planning rather than answering: a tutor agent asks "what is there to
learn about Redis persistence?", gets the relevant pages with their headings (and
the E5 summary of each top-level block where one exists), then fetches whole
sections to teach from.

A topic is located the way `ask` locates a question: LLM query understanding, then
hybrid search. Dense search alone failed on short topics: "持久化" (persistence)
ranked the PERSIST command first and missed the persistence guide entirely. The
topic is phrased as a learning goal, so the model reads it as a field to study, not
a command to look up, and guide pages are listed before API reference pages.

Section ids are "<doc_id>:<idx>". Private notes are only visible to their owner;
anyone else gets "not found", exactly as for an id that does not exist.
"""
from __future__ import annotations

from collections import defaultdict

from pymilvus import MilvusClient
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section
from app.ingest.summarize import COLLECTION as SUMMARIES
from app.ingest.sites import SITES
from app.models.llm import LLM
from app.models.local import ModelServer
from app.query.understand import understand
from app.retrieval import milvus_store
from app.retrieval.context import _children, _subtree_text
from app.retrieval.milvus_store import PRIVATE

MAX_LEVEL = 3          # deeper headings make an outline unreadable


def _visible(doc: Document | None, owner_id: int | None) -> bool:
    return doc is not None and (doc.library != PRIVATE or doc.owner_id == owner_id)


def _summaries(mc: MilvusClient, blocks: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
    if not blocks or not mc.has_collection(SUMMARIES):
        return {}
    ids = [f"{doc}:{idx}:s" for doc, idx in blocks]
    rows = mc.query(SUMMARIES, filter=f"chunk_id in {ids!r}", output_fields=["doc_id", "parent_index", "text"])
    return {(r["doc_id"], r["parent_index"]): r["text"] for r in rows}


def topic_docs(llm: LLM, mc: MilvusClient, models: ModelServer, collection: str, library: str, topic: str,
               max_docs: int) -> list[str]:
    """Doc ids for a topic, best first, guide pages before reference pages."""
    goal = f"我想系统学习 {SITES[library].name} 的「{topic}」，有哪些相关章节？"
    u = understand(llm, goal, libraries=[library])
    hits = milvus_store.hybrid_search(mc, collection, models.embed([u.search_query_en])[0], u.bm25_text(),
                                      top_k=40, libraries=[library])
    first = {}
    for h in hits:
        first.setdefault(h["doc_id"], h["doc_type"])
    return sorted(first, key=lambda d: first[d] != "guide")[:max_docs]   # stable: rank kept within each kind


def outline(db: DBSession, mc: MilvusClient, models: ModelServer, collection: str, library: str,
            topic: str | None = None, owner_id: int | None = None, max_docs: int = 6,
            llm: LLM | None = None) -> list[dict]:
    if library == PRIVATE:
        docs = list(db.scalars(select(Document).where(Document.library == PRIVATE, Document.owner_id == owner_id)
                               .order_by(Document.url)))
    elif topic:
        order = topic_docs(llm or LLM(), mc, models, collection, library, topic, max_docs)
        found = {d.id: d for d in db.scalars(select(Document).where(Document.id.in_(order)))}
        docs = [found[i] for i in order if i in found]
    else:
        docs = list(db.scalars(select(Document).where(Document.library == library).order_by(Document.url)
                               .limit(max_docs)))
    docs = docs[:max_docs] if library != PRIVATE else docs
    sections: dict[str, list[Section]] = defaultdict(list)
    for s in db.scalars(select(Section).where(Section.doc_id.in_([d.id for d in docs]), Section.level <= MAX_LEVEL)
                        .order_by(Section.doc_id, Section.idx)):
        sections[s.doc_id].append(s)
    tops = [(s.doc_id, s.idx) for d in docs for s in sections[d.id] if s.parent_idx is None and s.level > 0]
    summary = _summaries(mc, tops)
    return [{"doc_id": d.id, "title": d.title, "url": d.url,
             "sections": [{"section_id": f"{d.id}:{s.idx}", "level": s.level, "heading": s.heading,
                           **({"summary": summary[(d.id, s.idx)]} if (d.id, s.idx) in summary else {})}
                          for s in sections[d.id] if s.level > 0 or s.text.strip()]}
            for d in docs]


def get_section(db: DBSession, section_id: str, owner_id: int | None = None, max_chars: int = 20000) -> dict | None:
    doc_id, _, idx = section_id.rpartition(":")
    if not doc_id or not idx.isdigit():
        return None
    doc = db.get(Document, doc_id)
    if not _visible(doc, owner_id):
        return None
    tree = {s.idx: s for s in db.scalars(select(Section).where(Section.doc_id == doc_id))}
    if int(idx) not in tree:
        return None
    section = tree[int(idx)]
    text = _subtree_text(tree, _children(tree), section.idx)
    return {"section_id": section_id, "title": doc.title, "url": doc.url, "heading_path": " > ".join(section.path),
            "text": text[:max_chars], "truncated": len(text) > max_chars}
