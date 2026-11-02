"""E5: section summaries and "questions this section answers", indexed beside the chunks.

`python -m app.ingest.summarize --libraries redis fastapi [--limit 50]`
Learners ask in their own words ("how do I wipe everything"), the docs describe
the mechanism ("FLUSHALL deletes all the keys of all the existing databases").
For each parent block of a guide page the LLM writes a short summary and the
questions a learner would ask that the block answers; each becomes its own row in
a separate collection, so a question can match a question. Rows point at the same
(doc_id, parent_index) as the chunks, so a hit expands to the same parent block.

Only guide pages: API reference pages already name the API in every line, which
is what the summaries are meant to add. Calls go through the LLM cache, so a rerun
after an interruption resumes for free.
"""
from __future__ import annotations

import argparse
import contextvars
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section, engine
from app.ingest.pipeline import doc_type
from app.ingest.sites import SITES
from app.models.llm import LLM, LLMError
from app.models.local import ModelServer
from app.retrieval import milvus_store
from app.retrieval.context import _subtree_text
from app.tokenizer import TiktokenTokenizer
from app.tracing import spans

COLLECTION = "summaries_struct_merge"
MIN_CHARS = 300          # blocks shorter than this are already one chunk; a summary adds nothing
MAX_TOKENS = 2500        # block text sent to the model (keeps prompt + output inside an 8k context)

PROMPT = """Below is one section of the {site} documentation.

---Task---
Return one JSON object:
- "summary": 2-3 sentences in English saying what this section explains, naming the key commands, functions or options.
- "questions": 3 to 5 different questions a learner might ask that this section answers. Phrase them the way a learner would, often without knowing the exact API name (e.g. "how do I delete every key?" rather than "what does FLUSHALL do?").
Output only the JSON object.

---Section: {heading_path}---
{text}"""


@dataclass
class Block:
    library: str
    doc_id: str
    url: str
    parent_index: int
    heading_path: str
    text: str


def guide_blocks(libraries: list[str]) -> list[Block]:
    """Parent blocks (top-level sections) of guide pages, in a stable order."""
    blocks = []
    with DBSession(engine()) as db:
        docs = db.execute(select(Document.id, Document.url, Document.library, Document.title)
                          .where(Document.library.in_(libraries)).order_by(Document.url)).all()
        docs = [d for d in docs if doc_type(d.url) == "guide"]
        sections: dict[str, dict[int, Section]] = defaultdict(dict)
        for s in db.scalars(select(Section).where(Section.doc_id.in_([d.id for d in docs]))):
            sections[s.doc_id][s.idx] = s
    for doc in docs:
        tree = sections[doc.id]
        children: dict[int, list[int]] = defaultdict(list)
        for s in sorted(tree.values(), key=lambda s: s.idx):
            if s.parent_idx is not None:
                children[s.parent_idx].append(s.idx)
        for root in sorted(i for i, s in tree.items() if s.parent_idx is None):
            text = _subtree_text(tree, children, root)
            if len(text) >= MIN_CHARS:
                path = " > ".join(tree[root].path) or doc.title
                blocks.append(Block(doc.library, doc.id, doc.url, root, path, text))
    return blocks


def summarize(llm: LLM, tokenizer, block: Block) -> dict | None:
    text = tokenizer.decode(tokenizer.encode(block.text)[:MAX_TOKENS])
    prompt = PROMPT.format(site=SITES[block.library].name, heading_path=block.heading_path, text=text)
    with spans.span("llm", "section_summary", doc_id=block.doc_id, parent_index=block.parent_index):
        try:
            raw = llm.chat_json([{"role": "user", "content": prompt}])
        except (LLMError, ValueError):   # unparseable output: skip this block, keep the run going
            return None
    summary = str(raw.get("summary") or "").strip()
    questions = [str(q).strip() for q in raw.get("questions") or [] if str(q).strip()][:5]
    return {"summary": summary, "questions": questions} if summary or questions else None


def rows_for(block: Block, result: dict) -> list[dict]:
    texts = [("s", result["summary"])] if result["summary"] else []
    texts += [(f"q{i}", q) for i, q in enumerate(result["questions"])]
    return [{"chunk_id": f"{block.doc_id}:{block.parent_index}:{tag}", "library": block.library,
             "doc_id": block.doc_id, "doc_type": "guide", "url": block.url, "section_index": block.parent_index,
             "parent_index": block.parent_index, "heading_path": block.heading_path, "text": text}
            for tag, text in texts]


def build(libraries: list[str], limit: int | None = None, workers: int = 4) -> dict:
    blocks = guide_blocks(libraries)[:limit]
    llm, models, tokenizer, mc = LLM(), ModelServer(), TiktokenTokenizer(), milvus_store.client()
    milvus_store.create_collection(mc, COLLECTION)
    with spans.span("ingest", "section_summaries", libraries=libraries, blocks=len(blocks)) as root:
        def one(block: Block):
            return contextvars.copy_context().run(summarize, llm, tokenizer, block)
        done = failed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Embed and write per batch of blocks so an interrupted run keeps what it finished.
            for start in range(0, len(blocks), 64):
                batch = blocks[start:start + 64]
                rows = []
                for block, result in zip(batch, pool.map(one, batch)):
                    if result is None:
                        failed += 1
                    else:
                        rows += rows_for(block, result)
                if rows:
                    vectors = models.embed([r["text"] for r in rows])
                    # chunk_id is deterministic and LLM replies are cached, so a rerun overwrites in place.
                    mc.upsert(COLLECTION, [{**r, "dense": v} for r, v in zip(rows, vectors, strict=True)])
                done += len(batch)
                print(f"{done}/{len(blocks)} blocks, {failed} failed", flush=True)
        root.set(failed=failed)
    return {"blocks": len(blocks), "failed": failed, "trace_id": root.trace_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libraries", nargs="+", default=sorted(SITES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    spans.configure(engine())
    print(build(args.libraries, args.limit, args.workers))


if __name__ == "__main__":
    main()
