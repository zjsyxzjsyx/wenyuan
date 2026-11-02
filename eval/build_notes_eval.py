"""Questions about one student's own notes, asked over "course + my notes" (design doc 0.4).

`python -m eval.build_notes_eval --user koyo [--recall 60 --concept 60 --mixed 40]`
-> data/eval/notes_<user>_v1.jsonl (data/ stays out of the repository: these are personal notes).

Written by JUDGE_* (DeepSeek); the notes' author agreed to send these notes out for this purpose
(the HR note was excluded at import). Rows carry the course library, and eval.run is given
`--owner <user>`, so each question is searched the way the student would search: the course and
their own notes together, never anyone else's.

Types (all in Chinese, as the student writes):
- recall: looking up what they wrote ("我之前接雨水那题是怎么做的？").
- concept: a question about what a note explains, without mentioning notes; the note competes with
  the English course material.
- mixed: needs the note and a course passage found by vector search; skipped when they do not connect.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section, User, engine
from app.models.llm import LLM, LLMError
from app.models.local import ModelServer
from app.query.ask import COLLECTION
from app.retrieval import milvus_store
from eval.answers import judge_llm
from eval.build_course_eval import windows
from eval.run import body_of

OUT_DIR = Path("data/eval")

ASK = """Below is a passage from a learner's own study notes ("{title}", {where}).

{task}
The passage alone must be enough to answer it. Also give the answer in one or two sentences,
based on the passage. Write both in Chinese. If the passage holds nothing worth asking about (a
list of links, a table of contents, a date), return {{"skip": true}}.

Passage:
{text}

Return one JSON object: {{"question": "...", "answer": "..."}}"""

TASKS = {
    "recall": ("Write ONE question the learner might type later to find or recall what they wrote here: how "
               "they solved a problem, a mistake they noted, a summary or comparison they made. Casual chat "
               "style; it may or may not say “我之前” or “我的笔记”. "
               "Do not copy the heading word for word."),
    "concept": ("Write ONE question about the concept or technique this passage explains, as a student would "
                "ask a tutor. Do not mention notes, and do not copy the passage's headings."),
}

MIXED = """Passage A is from a learner's own study notes ("{title}", {where_a}); passage B is from a course
they are taking ({where_b}).

Write ONE question, in Chinese, the learner might ask that needs BOTH passages to answer well: for
example checking their note against the course, or connecting the two. If the passages do not
connect naturally, return {{"skip": true}}. Also give the answer in one or two sentences, in Chinese.

Passage A:
{text_a}

Passage B:
{text_b}

Return one JSON object: {{"question": "...", "answer": "..."}}"""


def note_passages(owner_id: int) -> list[dict]:
    out = []
    with DBSession(engine()) as db:
        docs = {d.id: d for d in db.scalars(select(Document).where(Document.owner_id == owner_id))}
        for s in db.scalars(select(Section).where(Section.doc_id.in_(list(docs))).order_by(Section.doc_id, Section.idx)):
            if len(s.text) < 300:
                continue
            for i, text in enumerate(windows(s.text, 2000)):
                if len(text) >= 250:
                    doc = docs[s.doc_id]
                    out.append({"url": doc.url, "title": doc.title, "where": " > ".join(s.path),
                                "key": f"{s.doc_id}:{s.idx}:{i}", "text": text})
    return out


def ask_one(llm: LLM, qtype: str, p: dict) -> dict | None:
    raw = llm.chat_json([{"role": "user", "content": ASK.format(
        title=p["title"], where=p["where"], task=TASKS[qtype], text=p["text"])}])
    question = str(raw.get("question") or "").strip()
    if raw.get("skip") or not question:
        return None
    return {"qtype": qtype, "question": question, "answer_text": str(raw.get("answer") or ""),
            "gold_urls": [p["url"]], "gold_evidence": [p["text"]], "source": [p["key"]]}


def mixed_one(llm: LLM, mc, models: ModelServer, library: str, p: dict) -> dict | None:
    hits = milvus_store.dense_search(mc, COLLECTION, models.embed([p["text"][:1500]])[0], top_k=5, libraries=[library])
    course = next((h for h in hits if len(body_of(h["text"])) >= 300), None)
    if course is None:
        return None
    b = body_of(course["text"])
    raw = llm.chat_json([{"role": "user", "content": MIXED.format(
        title=p["title"], where_a=p["where"], text_a=p["text"], where_b=course["heading_path"], text_b=b)}])
    question = str(raw.get("question") or "").strip()
    if raw.get("skip") or not question:
        return None
    return {"qtype": "mixed", "question": question, "answer_text": str(raw.get("answer") or ""),
            "gold_urls": sorted({p["url"], course["url"]}), "gold_evidence": [p["text"], b],
            "source": [p["key"], course["chunk_id"]]}


def safe(fn, *args):
    """One question's failure (a malformed JSON reply, a timeout) skips that question, not the run."""
    try:
        return fn(*args)
    except (LLMError, ValueError, httpx.HTTPError) as exc:
        print("skipped one:", type(exc).__name__, str(exc)[:80])
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--library", default="cs50", help="the course searched alongside the notes")
    parser.add_argument("--recall", type=int, default=60)
    parser.add_argument("--concept", type=int, default=60)
    parser.add_argument("--mixed", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with DBSession(engine()) as db:
        owner = db.scalar(select(User.id).where(User.name == args.user))
    if owner is None:
        raise SystemExit(f"no account named {args.user!r}")
    rng, llm = random.Random(args.seed), judge_llm()
    pool = note_passages(owner)
    rng.shuffle(pool)
    print(len(pool), "note passages from", len({p["url"] for p in pool}), "notes")

    def run_all(fn, items):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            return [r for r in ex.map(lambda p: contextvars.copy_context().run(safe, fn, p), items) if r]

    rows, start = [], 0
    for qtype in ("recall", "concept"):
        n = getattr(args, qtype)
        picked, start = pool[start:start + round(n * 1.3)], start + round(n * 1.3)
        made = run_all(lambda p, t=qtype: ask_one(llm, t, p), picked)
        rows += made[:n]
        print(qtype, f"{len(made)} usable of {len(picked)}")
    mc, models = milvus_store.client(), ModelServer()
    picked = pool[start:start + round(args.mixed * 2)]
    made = run_all(lambda p: mixed_one(llm, mc, models, args.library, p), picked)
    rows += made[:args.mixed]
    print("mixed", f"{len(made)} usable of {len(picked)}")
    for i, row in enumerate(rows):
        row.update(id=f"note-{row['qtype']}-{i + 1:03d}", library=args.library, type="generated", body="",
                   owner=args.user)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"notes_{args.user}_v1.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(len(rows), "questions ->", out)


if __name__ == "__main__":
    main()
