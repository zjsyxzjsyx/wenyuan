"""Claim-level citation check (E8): does each sentence's cited source actually say it?

`answer.check_citations` only catches citations to sources that do not exist. A
sentence can cite a real source that says something else, or nothing about it.
Here the answer is split into claims (sentences, each with the sources it cites),
and one LLM call judges every claim against the text of its sources. Unsupported
claims are removed rather than rewritten (a rewrite is new, unchecked text); an
answer left with no claims becomes a refusal.

Every claim is checked against all the sources the answer was written from, not only
the ones it cites. Answers cite about 2 of 11 sources and most sentences carry no
[S#] of their own (models cite once at the end of a paragraph); checked against the
cited sources only, true sentences grounded in an uncited source were removed, and
verification cost 9 points of correct answers (M3: 70.7% -> 61.6%). Against all
sources it cost none, and still removed 11 of the 16 claims the judge found in no
source. The question it answers is "is this in the materials?", not "is the
citation exact?".

Sentences end at 。！？ or at . ! ? followed by whitespace and something that can start a
sentence; code spans are never split, so `df.dropna()` or a Scratch block like
`if <touching (mouse pointer)?> then` stays one claim.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.models.llm import LLM
from app.retrieval.context import Source
from app.tracing import spans

PROMPT = """You check an answer against its sources, claim by claim.

For each claim decide:
- "supported": the cited sources state it (paraphrase is fine).
- "partial": the sources state part of it, or it adds a detail they do not contain.
- "unsupported": the sources do not state it, or contradict it.
Judge only against the source text below, not against your own knowledge.

Output one JSON object: {{"claims": [{{"id": "C1", "verdict": "supported|partial|unsupported"}}, ...]}}

---Sources---
{sources}

---Claims---
{claims}"""

VERDICTS = ("supported", "partial", "unsupported")
_CITE = re.compile(r"\[(S\d+)\]")
# Code is masked before looking for sentence ends. A CJK full stop always ends a sentence; . ! ?
# only before whitespace and a character that can start one (not a lowercase letter or a
# continuation like > ) , ;), so "3.14", "e.g. this" and "mouse pointer)? > then" do not split.
# Trailing [S#] markers stay with their sentence.
_CODE = re.compile(r"```.*?```|`[^`\n]+`", re.S)
_END = re.compile(r"[。！？]|[.!?](?=\s*$|\s+[^\sa-z>)\]},;:|])")
_TRAILING_CITES = re.compile(r"(?:\s*\[S\d+\])*")


def _is_list_number(masked: str, dot: int) -> bool:
    """The "." of a list item number ("steps: 1. Download", "... [S1]. 2. Unzip"): one or two digits
    at the start of a line or after the end of a sentence or a colon. M3: these split off as claims
    of their own ("Here's how you can do it: 1."), which no source supports."""
    k = dot
    while k > 0 and masked[k - 1].isdigit():
        k -= 1
    if not 1 <= dot - k <= 2:
        return False
    j = k
    while j > 0 and masked[j - 1].isspace():
        j -= 1
    return j == 0 or "\n" in masked[j:k] or masked[j - 1] in ".。:：]"


def _sentences(text: str) -> list[str]:
    masked = _CODE.sub(lambda m: "X" * len(m.group()), text)
    pieces, start = [], 0
    for m in _END.finditer(masked):
        if m.start() < start:          # inside the markers already taken by the previous sentence
            continue
        if masked[m.start()] == "." and _is_list_number(masked, m.start()):
            continue
        end = _TRAILING_CITES.match(masked, m.end()).end()
        pieces.append(text[start:end])
        start = end
    pieces.append(text[start:])
    return pieces


@dataclass
class Claim:
    cid: str
    text: str
    cites: list[str]
    verdict: str = "supported"


@dataclass
class Verification:
    text: str
    claims: list[Claim] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {v: sum(c.verdict == v for c in self.claims) for v in VERDICTS}


def split_claims(text: str) -> list[Claim]:
    claims = []
    for piece in _sentences(text):
        sentence = piece.strip()
        if len(_CITE.sub("", sentence).strip(" 。.，,；;:：")) < 2:   # stray markers or punctuation
            if claims and sentence:
                claims[-1].text += " " + sentence
                claims[-1].cites += _CITE.findall(sentence)
            continue
        claims.append(Claim(f"C{len(claims) + 1}", sentence, list(dict.fromkeys(_CITE.findall(sentence)))))
    return claims


def verify(llm: LLM, text: str, sources: list[Source]) -> Verification:
    claims = split_claims(text)
    if not claims:
        return Verification(text)
    shown = sources              # every source, not only the cited ones (see the module docstring)
    listing = "\n".join(f"[{c.cid}] (cites {', '.join(c.cites) or 'all sources'}) {_CITE.sub('', c.text).strip()}"
                        for c in claims)
    prompt = PROMPT.format(sources="\n\n".join(f"[{s.sid}] {s.title} > {s.heading_path}\n{s.text}" for s in shown),
                           claims=listing)
    with spans.span("llm", "verify_claims", claims=len(claims)) as s:
        raw = llm.chat_json([{"role": "user", "content": prompt}])
        verdicts = {str(v.get("id")): str(v.get("verdict", "")).lower() for v in raw.get("claims", [])
                    if isinstance(v, dict)}
        for claim in claims:
            # A claim the checker skipped is kept: removal needs a positive "unsupported".
            claim.verdict = verdicts.get(claim.cid) if verdicts.get(claim.cid) in VERDICTS else "supported"
        kept = [c for c in claims if c.verdict != "unsupported"]
        result = Verification(" ".join(c.text for c in kept), claims,
                              [c.text for c in claims if c.verdict == "unsupported"])
        s.set(**result.counts, removed=len(result.removed))
    return result


def to_json(v: Verification) -> str:
    return json.dumps({"counts": v.counts, "removed": v.removed}, ensure_ascii=False)
