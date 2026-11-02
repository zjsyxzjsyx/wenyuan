"""Private-library eval: does an owner find their own notes, and does nobody else ever see them?

`python -m eval.private_eval --notes .cache/private_notes [--exp W4-private]`
The notes are a learner's own (here: exported study notes), so everything derived
from them stays out of the repository: the notes, the generated questions and the
per-question results go to .cache/private_eval/; experiments/ only gets counts and
rates. For the same reason questions are written by the local model, never an API
(routing rule: private content stays local), which means the writer and the
answering model are the same; the numbers are reported with that caveat.

Per question, for the owner: is the note among the top-5 retrieved pages, and did a
private source reach the answer context. For a second user asking the same question:
the number of private hits, which must be zero.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.api.auth import create_user
from app.db.models import User, create_tables, engine
from app.ingest import private as notes
from app.models.llm import LLM
from app.models.local import ModelServer
from app.query import ask as chain
from app.retrieval import milvus_store
from app.tokenizer import TiktokenTokenizer
from app.tracing import spans
from eval.metrics import unique_in_order

WORK = Path(".cache/private_eval")
OUT_DIR = Path("experiments")

PROMPT = """Below is one section of a learner's own study notes.

Write {n} questions the learner might later ask to recall what THIS section says, in Chinese,
the way they would type them in a chat. Each must be answerable from this section; prefer what is
specific to these notes (their own mistakes, examples, comparisons, conclusions) over textbook facts.
Do not copy the section heading verbatim.

Output one JSON object: {{"questions": ["...", "..."]}}

---Note: {title}---
## {heading}
{text}"""


def user_id(name: str) -> int:
    create_tables()
    with DBSession(engine()) as db:
        found = db.scalar(select(User).where(User.name == name))
        if found:
            return found.id
    create_user(name)
    with DBSession(engine()) as db:
        return db.scalar(select(User).where(User.name == name)).id


def sections(markdown: str) -> list[tuple[str, str]]:
    """(heading, text) of each ## section with enough text to ask about."""
    parts = re.split(r"^##\s+(.+)$", markdown, flags=re.M)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)
            if len(parts[i + 1].strip()) >= 200]


def build(owner: int, folder: Path, llm: LLM, per_section: int = 2) -> list[dict]:
    mc, models, tokenizer = milvus_store.client(), ModelServer(), TiktokenTokenizer()
    rows = []
    for path in sorted(folder.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        with DBSession(engine()) as db:
            note = notes.add_note(db, mc, models, tokenizer, chain.COLLECTION, owner, path.name, markdown)
        for heading, text in sections(markdown):
            raw = llm.chat_json([{"role": "user", "content": PROMPT.format(
                n=per_section, title=note["title"], heading=heading, text=text[:3000])}])
            for q in [str(q).strip() for q in raw.get("questions") or [] if str(q).strip()][:per_section]:
                rows.append({"id": f"pv-{len(rows) + 1:03d}", "question": q, "note": path.name,
                             "url": notes.note_url(owner, path.name), "heading": heading})
    return rows


def evaluate(rows: list[dict], owner: int, stranger: int, llm: LLM) -> list[dict]:
    models, results = ModelServer(), []
    for row in rows:
        mine = chain.search(row["question"], llm=llm, models=models, owner_id=owner)
        theirs = chain.search(row["question"], llm=llm, models=models, owner_id=stranger)
        pages = unique_in_order([r["url"] for r in mine.retrieved])
        results.append({
            "id": row["id"], "note": row["note"],
            "hit@5": float(row["url"] in pages[:5]), "mrr": next((1 / (i + 1) for i, u in enumerate(pages)
                                                                  if u == row["url"]), 0.0),
            "private_in_context": any(chain.is_private(s) for s in mine.sources),
            "stranger_private_hits": sum(chain.is_private(s) for s in theirs.sources)
                                     + sum(u.startswith("private://") for u in
                                           (r["url"] for r in theirs.retrieved)),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=Path, default=Path(".cache/private_notes"))
    parser.add_argument("--exp", default="W4-private")
    parser.add_argument("--per-section", type=int, default=2)
    args = parser.parse_args()
    spans.configure(engine())
    WORK.mkdir(parents=True, exist_ok=True)
    owner, stranger = user_id("private-eval-owner"), user_id("private-eval-stranger")
    llm = LLM()                                   # the local model: these notes never go to an API
    dataset = WORK / "private_v1.jsonl"
    rows = build(owner, args.notes, llm, args.per_section)
    dataset.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    results = evaluate(rows, owner, stranger, llm)
    (WORK / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results),
                                       encoding="utf-8")
    summary = {"n": len(results), "notes": len({r["note"] for r in results}),
               "owner_hit@5": round(mean(r["hit@5"] for r in results), 4),
               "owner_mrr": round(mean(r["mrr"] for r in results), 4),
               "owner_private_in_context": round(mean(r["private_in_context"] for r in results), 4),
               "stranger_private_hits": sum(r["stranger_private_hits"] for r in results),
               "question_writer": f"{llm.model} (same as the answering model)"}
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{args.exp}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"exp": args.exp, "summary": summary,
                                "questions": [{k: r[k] for k in ("id", "hit@5", "mrr", "private_in_context",
                                                                 "stranger_private_hits")} for r in results]},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved", path, "| questions and per-note results:", WORK)


if __name__ == "__main__":
    main()
