"""Course question set written by an independent model from the course files (main eval set, design doc 0.4).

`python -m eval.build_course_eval [--library cs50] [--paraphrase 100 --chinese 80 --cross 60 --absent 40]`
-> data/eval/<library>_gen_v1.jsonl, and the unanswerable questions in <library>_absent_v1.jsonl.

The questions come from JUDGE_* (DeepSeek), not from the local model that serves retrieval, so the
system is never scored on questions it wrote itself. The course is public material. Each question is
written from one passage: a section of the course's section tree, or a ~2,500-character window of a
long lecture-transcript section. That passage is the gold evidence: a retrieved chunk counts when it
shares enough text with it (eval.metrics.overlaps), whatever the chunking.

Types, chosen to be harder than the dataset's own questions (which reuse the files' wording):
- paraphrase: asked by a student who has not read the passage: a goal or a symptom in everyday
  words, without its terms. Questions sharing over 60% of their content words with the passage
  are dropped (a question written from a text tends to copy it, which flatters BM25).
- chinese: the same kind of question asked in Chinese about the English course.
- cross: needs two passages from different weeks (the second one found by vector search).
- absent: plausible questions the course does not answer; no gold, kept apart for refusal tests.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Document, Section, engine
from app.ingest.cs50 import FOLDER, load
from app.models.llm import LLM, LLMError
from app.models.local import ModelServer
from app.query.ask import COLLECTION
from app.retrieval import milvus_store
from eval.answers import judge_llm
from eval.run import body_of

OUT_DIR = Path("data/eval")
WINDOW = 2500
QUOTAS = {"notes": 0.35, "transcript": 0.35, "practice": 0.30}     # practice: problem sets and exercises

ASK = """You write evaluation questions for a search engine over a course's materials.

Below is a passage from {course} ({kind}: {where}).

Write ONE question {who} whose answer is in this passage. Ask it the way a student who has NOT read
the passage would: describe what they want to do, or the problem or error they ran into, in
everyday words. Do not use the passage's distinctive terms (function names, jargon, its headings)
unless a beginner would already know them. The passage alone must be enough to answer it.
Also give the answer in one or two sentences, based on the passage.
If the passage holds nothing a student would ask about (logistics, small talk, applause),
return {{"skip": true}}.

Passage:
{text}

Return one JSON object: {{"question": "...", "answer": "..."}}"""

WHO = {"paraphrase": "a student might ask in English",
       "chinese": ("a Chinese-speaking student might type in Chinese, casually, as in a chat (common "
                   "technical words may stay in English)")}

CROSS = """You write evaluation questions for a search engine over a course's materials.

Below are two passages from different weeks of {course}.

Write ONE question a student might ask that needs BOTH passages to answer well: for example
comparing the two ideas, or applying the earlier one in the later context. Ask it in everyday
words, without copying either passage's headings. If the passages do not connect naturally,
return {{"skip": true}}.

Passage A ({where_a}):
{text_a}

Passage B ({where_b}):
{text_b}

Return one JSON object: {{"question": "...", "answer": "..."}}"""

ABSENT = """Here is the outline of {course}: each line is a file and its headings.

{outline}

Write {n} questions a student taking this course might plausibly ask a course assistant that
these materials do NOT answer: tools, services, languages or topics the course never covers, or
details beyond it. Make about half of them English and half Chinese. Avoid anything the outline
suggests is covered.

Return one JSON object: {{"questions": ["...", "..."]}}"""

_STOP = set("""about above after again also always another because before being below between both
could does doing done each even every from have having here into just like make many more most much
must need only other over same should since some such than that their them then there these they
this those through under until very want what when where which while will with within without would
your yours code program""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9_]+", text.lower()) if len(w) >= 4 and w not in _STOP}


def lexical_overlap(question: str, passage: str) -> float:
    words = content_words(question)
    return len(words & content_words(passage)) / len(words) if words else 0.0


def windows(text: str, size: int = WINDOW) -> list[str]:
    """A long text cut at paragraph breaks into pieces of about `size` characters."""
    if len(text) <= size * 1.3:
        return [text]
    pieces, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) > size:
            pieces.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    return pieces + ([current] if current else [])


def week_of(title: str) -> str:
    match = re.match(r"Week (\S+) ·", title)
    return match.group(1) if match else ""


def passages(library: str) -> list[dict]:
    kinds = {}
    for path in FOLDER.rglob("*.md"):
        page, kind = load(path.read_text(encoding="utf-8"))
        kinds[page.url] = "practice" if kind in ("problemset", "exercise") else kind
    out = []
    with DBSession(engine()) as db:
        docs = {d.id: d for d in db.scalars(select(Document).where(Document.library == library))}
        for s in db.scalars(select(Section).where(Section.doc_id.in_(list(docs))).order_by(Section.doc_id, Section.idx)):
            doc = docs[s.doc_id]
            if len(s.text) < 500:
                continue
            for i, text in enumerate(windows(s.text)):
                if len(text) >= 400:
                    out.append({"url": doc.url, "title": doc.title, "week": week_of(doc.title),
                                "kind": kinds.get(doc.url, "notes"), "where": " > ".join(s.path),
                                "key": f"{s.doc_id}:{s.idx}:{i}", "text": text})
    return out


def sample(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """`n` passages across kinds by QUOTAS, different passages each time it is called."""
    chosen = []
    for kind, share in QUOTAS.items():
        group = [p for p in pool if p["kind"] == kind]
        chosen += rng.sample(group, min(len(group), round(n * share)))
    for p in chosen:
        pool.remove(p)
    return chosen


def ask_one(llm: LLM, course: str, qtype: str, p: dict) -> dict | None:
    raw = llm.chat_json([{"role": "user", "content": ASK.format(
        course=course, kind=p["kind"], where=p["where"], who=WHO[qtype], text=p["text"])}])
    question = str(raw.get("question") or "").strip()
    if raw.get("skip") or not question:
        return None
    overlap = lexical_overlap(question, p["text"])
    if qtype == "paraphrase" and overlap > 0.6:
        return None
    return {"qtype": qtype, "question": question, "answer_text": str(raw.get("answer") or ""),
            "gold_urls": [p["url"]], "gold_evidence": [p["text"]], "lexical_overlap": round(overlap, 3),
            "source": [p["key"]]}


def partner(mc, models: ModelServer, collection: str, library: str, p: dict, urls_by_week: dict) -> dict | None:
    """A passage from another week that vector search finds closest to `p`."""
    hits = milvus_store.dense_search(mc, collection, models.embed([p["text"][:1500]])[0], top_k=30, libraries=[library])
    for h in hits:
        week = urls_by_week.get(h["url"])
        if week is not None and week != p["week"] and len(body_of(h["text"])) >= 300:
            return {"url": h["url"], "where": h["heading_path"], "key": h["chunk_id"], "text": body_of(h["text"])}
    return None


def cross_one(llm: LLM, course: str, a: dict, b: dict) -> dict | None:
    raw = llm.chat_json([{"role": "user", "content": CROSS.format(
        course=course, where_a=a["where"], text_a=a["text"], where_b=b["where"], text_b=b["text"])}])
    question = str(raw.get("question") or "").strip()
    if raw.get("skip") or not question:
        return None
    return {"qtype": "cross", "question": question, "answer_text": str(raw.get("answer") or ""),
            "gold_urls": sorted({a["url"], b["url"]}), "gold_evidence": [a["text"], b["text"]],
            "source": [a["key"], b["key"]]}


def safe(fn, *args):
    """One question's failure (a malformed JSON reply, a timeout) skips that question, not the run."""
    try:
        return fn(*args)
    except (LLMError, ValueError, httpx.HTTPError) as exc:
        print("skipped one:", type(exc).__name__, str(exc)[:80])
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="cs50")
    parser.add_argument("--course", default="Harvard's CS50x 2024 (introduction to computer science)")
    parser.add_argument("--paraphrase", type=int, default=100)
    parser.add_argument("--chinese", type=int, default=80)
    parser.add_argument("--cross", type=int, default=60)
    parser.add_argument("--absent", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rng, llm = random.Random(args.seed), judge_llm()
    pool = passages(args.library)
    print(f"{len(pool)} passages:", {k: sum(p["kind"] == k for p in pool) for k in QUOTAS})

    def run_all(fn, items):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            return list(ex.map(lambda item: contextvars.copy_context().run(safe, fn, *item), items))

    rows = []
    for qtype in ("paraphrase", "chinese"):
        # ask for a third more than needed: some passages are skipped or too close in wording
        picked = sample(pool, round(getattr(args, qtype) * 1.35), rng)
        made = [r for r in run_all(lambda p, t=qtype: ask_one(llm, args.course, t, p), [(p,) for p in picked]) if r]
        rows += made[:getattr(args, qtype)]
        print(qtype, f"{len(made)} usable of {len(picked)}")
    mc, models = milvus_store.client(), ModelServer()
    urls_by_week = {p["url"]: p["week"] for p in passages(args.library)}
    firsts = [p for p in sample(pool, round(args.cross * 1.6), rng) if p["kind"] != "practice" or rng.random() < 0.5]
    pairs = [(a, b) for a in firsts if (b := partner(mc, models, COLLECTION, args.library, a, urls_by_week))]
    made = [r for r in run_all(lambda a, b: cross_one(llm, args.course, a, b), pairs) if r]
    rows += made[:args.cross]
    print("cross", f"{len(made)} usable of {len(pairs)} pairs")
    for i, row in enumerate(rows):
        row.update(id=f"gen-{row['qtype']}-{i + 1:03d}", library=args.library, type="generated", body="")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.library}_gen_v1.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    with DBSession(engine()) as db:
        outline = "\n".join(f"- {d.title}: " + "; ".join(
            s.heading for s in db.scalars(select(Section).where(Section.doc_id == d.id, Section.level == 2)))
            for d in db.scalars(select(Document).where(Document.library == args.library).order_by(Document.title)))
    raw = llm.chat_json([{"role": "user", "content": ABSENT.format(course=args.course, outline=outline[:20000],
                                                                   n=args.absent)}])
    absent = [{"id": f"gen-absent-{i + 1:03d}", "library": args.library, "type": "generated", "qtype": "absent",
               "question": str(q).strip(), "body": "", "gold_urls": []}
              for i, q in enumerate(raw.get("questions") or []) if str(q).strip()]
    (OUT_DIR / f"{args.library}_absent_v1.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in absent), encoding="utf-8")
    print(len(rows), "questions ->", out, "|", len(absent), "unanswerable ->", OUT_DIR / f"{args.library}_absent_v1.jsonl")


if __name__ == "__main__":
    main()
