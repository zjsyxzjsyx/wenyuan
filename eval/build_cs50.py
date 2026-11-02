"""The CS50 dataset's own 97 questions as an eval set (the calibration set, design doc 0.4).

`python -m eval.build_cs50` -> data/eval/cs50_v1.jsonl (under data/, so it stays out of the
repository with the rest of the CC BY-NC-SA material).
The questions were written by a model from the course files and rated by a person (question
quality 4-5); each names the file it was written from, which becomes the gold page through that
file's front-matter link, plus the excerpt it used ("chunk") or the whole file ("document").
Query style and persona are kept so results can be split by how the question was asked.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from app.ingest.cs50 import FOLDER, load

SOURCE = Path("data/raw/cs50/data/qa_evaluation_dataset.csv")
OUT = Path("data/eval/cs50_v1.jsonl")


def main() -> None:
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    links = {"/" + p.relative_to(FOLDER).as_posix(): load(p.read_text(encoding="utf-8"))[0].url
             for p in FOLDER.rglob("*.md")}
    rows = []
    with SOURCE.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            path = r["reference_context_path"].strip()
            if path not in links:
                raise SystemExit(f"question {r['id']}: unknown file {path}")
            rows.append({"id": f"cs50-{int(r['id']):03d}", "library": "cs50", "type": "real",
                         "question": r["user_input"].strip(), "body": "", "gold_urls": [links[path]],
                         "answer_text": r["reference"], "gold_context_type": r["reference_context_type"],
                         # an excerpt ("chunk" questions) is evidence; a whole file would only repeat the page id
                         **({"gold_evidence": [r["reference_contexts"]]} if r["reference_context_type"] == "chunk" else {}),
                         "query_style": r["query_style"], "query_length": r["query_length"],
                         "persona": r["persona_name"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(len(rows), "questions ->", OUT)


if __name__ == "__main__":
    main()
