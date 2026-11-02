"""Synthetic two-turn eval set (E6 multi-turn): does conversation history get used?

`python -m eval.build_multiturn --n 200`  ->  eval/datasets/so_v1_multiturn.jsonl
Each real Stack Overflow question (its title) becomes a short conversation: a first
turn that introduces the topic, then a follow-up that asks the original question
with its key subject replaced by a reference ("it", "that command"), so the follow-up
alone does not say what it is about. The gold pages stay those of the original.

Retrieval is then scored three ways (eval.run): the original title (upper bound),
the follow-up alone, and the follow-up with the first turn as history.

Caveat: the follow-ups are written by an LLM, and the default writer is the same
local model that later resolves them, which flatters the with-history score; the
original title is kept as the reference point, and a row whose follow-up still
names the subject is dropped.
"""
from __future__ import annotations

import argparse
import contextvars
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.config import get_settings
from app.ingest.sites import SITES
from app.models.llm import LLM
from eval.run import DATASET, load_labels

OUT = Path("eval/datasets/so_v1_multiturn.jsonl")

PROMPT = """Turn a learner's programming question into a two-turn conversation with a tutor.

- "first_turn": a short, natural question the learner asks first that introduces the subject
  (the library, command, function or concept the original question is about), without asking the original question.
- "assistant_reply": one or two sentences the tutor might answer to the first turn.
- "followup": the original question asked as a follow-up. Replace its most specific technical subject (the
  command, function, class, option, data type or concept it is about, not merely the library or product name)
  with a reference such as "it", "that", "this command", "那个", "它", so that the follow-up alone does not say
  what it is about. Keep everything else the original asks.
- "subject": the words you replaced.
Write the conversation in the same language as the original question.

Output one JSON object with exactly these keys.

---Library---
{library}

---Original question---
{question}"""


def build_row(llm: LLM, row: dict) -> dict | None:
    raw = llm.chat_json([{"role": "user", "content": PROMPT.format(library=SITES[row["library"]].name,
                                                                    question=row["question"])}])
    first, reply, followup = (str(raw.get(k) or "").strip() for k in ("first_turn", "assistant_reply", "followup"))
    subject = str(raw.get("subject") or "").strip()
    if not (first and followup) or not subject:
        return None
    # The follow-up must not still name its subject, or it tests nothing; and hiding only the library
    # name tests nothing either, since retrieval is told the library (71 of 200 did, before the prompt
    # asked for the most specific subject).
    library = {row["library"].lower(), SITES[row["library"]].name.lower()}
    if re.search(re.escape(subject), followup, re.I) or subject.lower() in library:
        return None
    return {"id": row["id"] + "-mt", "library": row["library"], "type": "multiturn", "question": followup, "body": "",
            "history": [{"role": "user", "content": first}, {"role": "assistant", "content": reply}],
            "original_question": row["question"], "subject": subject, "gold_urls": row["gold_urls"],
            "source": row["source"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    labels = load_labels(DATASET)
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if labels.get(r["id"]) != "not_applicable" and r["library"] in SITES]
    random.Random(args.seed).shuffle(rows)
    s = get_settings()
    writer = (LLM(base_url=s.judge_base_url, api_key=s.judge_api_key or "", model=s.judge_model)
              if s.judge_base_url else LLM())
    built: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda r: contextvars.copy_context().run(build_row, writer, r), rows[:args.n]):
            if result:
                built.append(result)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in built), encoding="utf-8")
    print(f"{len(built)} of {args.n} questions became two-turn conversations -> {OUT}")


if __name__ == "__main__":
    main()
