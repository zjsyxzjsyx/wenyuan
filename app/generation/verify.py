"""Claim-level citation check (E8): does each sentence's cited source actually say it?

`answer.check_citations` only catches citations to sources that do not exist. A
sentence can cite a real source that says something else, or nothing about it.
Here the answer is split into claims (sentences, each with the sources it cites),
and one LLM call judges every claim against the text of its sources. Unsupported
claims are removed rather than rewritten (a rewrite is new, unchecked text); an
answer left with no claims becomes a refusal.

A sentence without its own [S#] (models often cite once at the end of a
paragraph) is checked against all sources the answer cites.
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
# A sentence ends at 。！？!? or at a period followed by whitespace or the end, so the dots in
# `df.dropna()` or "3.14" do not split; trailing [S#] markers stay with their sentence.
_SENTENCE = re.compile(r".+?(?:[。！？!?]|\.(?=\s|$)|$)(?:\s*\[S\d+\])*", re.S)


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
    for piece in _SENTENCE.findall(text):
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
    by_id = {s.sid: s for s in sources}
    cited_anywhere = list(dict.fromkeys(c for claim in claims for c in claim.cites if c in by_id))
    shown = [by_id[sid] for sid in cited_anywhere] or sources
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
