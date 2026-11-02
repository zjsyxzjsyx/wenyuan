"""Multi-hop retrieval (E9): questions whose answer is spread over pages, often across libraries.

Two bounded strategies on top of the single-pass search (`ask._search`):

- decompose: one LLM call splits the question into at most 3 self-contained
  sub-queries, each with the library to search; their hits are fused by RRF.
- iterative: search once; the LLM looks at what was found (headings and short
  excerpts) and either says it is enough or names what is missing with one more
  query; at most `max_rounds` searches and `budget_tokens` of prompts, after which
  it answers with what it has.

Both only ever add retrieval rounds: the first round is the normal search, so a
single-hop question pays one extra LLM call and keeps its results.

The rounds' hits are interleaved (one from each list in turn, a page taken once),
not fused by RRF. RRF lost the point of splitting: a page with many sections ranks
in several lists and outscores the page that one sub-query found first. On 66
cross-library questions both pages were in the top 10 for 0.29 with RRF and 0.44
interleaved (single pass: 0.30; either page in some list's top 5: 0.49).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.ingest.sites import SITES
from app.models.llm import LLM
from app.models.local import ModelServer
from app.query.understand import Understanding, understand
from app.retrieval import milvus_store
from app.tokenizer import TiktokenTokenizer
from app.tracing import spans

DECOMPOSE = """Split a learner's question about technical documentation into search queries.

Return at most 3 self-contained English search queries, one per piece of information the answer needs,
each with the one library to search, chosen from {libraries}. A question that needs only one lookup gets
one query. Output one JSON object: {{"queries": [{{"query": "...", "library": "..."}}]}}

---Question---
{question}"""

ENOUGH = """You decide whether the documentation found so far is enough to answer a learner's question.

If something the answer needs is missing, give ONE English search query for it and the library to search,
chosen from {libraries}. Output one JSON object:
{{"enough": true|false, "missing": "<what is missing, or empty>", "query": "<search query, or empty>", "library": "<library>"}}

---Question---
{question}

---Found so far---
{found}"""


@dataclass
class Hop:
    query: str
    library: list[str] | None
    hits: int


@dataclass
class MultiHopResult:
    understanding: Understanding
    hits: list[dict]
    hops: list[Hop] = field(default_factory=list)
    llm_calls: int = 1
    lists: list[list[dict]] = field(default_factory=list)   # hits of each hop, before fusion


def _search(mc, models: ModelServer, collection: str, dense_text: str, bm25_text: str, libraries, top_k: int,
            owner_id: int | None) -> list[dict]:
    return milvus_store.hybrid_search(mc, collection, models.embed([dense_text])[0], bm25_text, top_k=top_k,
                                      libraries=libraries, owner_id=owner_id)


def interleave(lists: list[list[dict]], limit: int) -> list[dict]:
    """Round-robin over the lists, best first, each page once: every sub-query keeps its best pages
    near the top instead of being outvoted by pages that show up in several lists."""
    out, seen = [], set()
    for rank in range(max(map(len, lists), default=0)):
        for hits in lists:
            if rank < len(hits) and hits[rank]["url"] not in seen:
                seen.add(hits[rank]["url"])
                out.append(hits[rank])
    return out[:limit]


def _library(name, offered: list[str]) -> list[str] | None:
    return [name] if isinstance(name, str) and name in offered else None


def _digest(hits: list[dict], max_pages: int = 8, chars: int = 160) -> str:
    seen, lines = set(), []
    for h in hits:
        if h["url"] in seen:
            continue
        seen.add(h["url"])
        text = " ".join(h["text"].split())[:chars]
        lines.append(f"- [{h['library']}] {h.get('heading_path') or h['url']}: {text}")
        if len(lines) == max_pages:
            break
    return "\n".join(lines) or "(nothing)"


def expand(u: Understanding, first: list[dict], llm: LLM, models: ModelServer, collection: str,
           offered: list[str], scope: list[str] | None, mode: str = "decompose", top_k: int = 50,
           max_rounds: int = 3, budget_tokens: int = 6000, owner_id: int | None = None,
           mc=None) -> MultiHopResult:
    """More search rounds after a first one; `first` is that round's hits for `u`."""
    mc = mc or milvus_store.client()
    with spans.span("retrieval", f"multihop_{mode}") as span:
        lists, hops, calls = [first], [Hop(u.standalone_question, scope, len(first))], 0   # LLM calls made here
        if mode == "decompose":
            raw = llm.chat_json([{"role": "user", "content": DECOMPOSE.format(
                libraries=offered, question=u.standalone_question)}])
            calls += 1
            queries = [q for q in (raw.get("queries") or [])[:3]
                       if isinstance(q, dict) and str(q.get("query", "")).strip()]
            # One sub-query means a single-hop question: keep the first search as it is. Interleaving
            # it with a near-duplicate search cost single-hop SO questions 2 points of Recall@5 (gate).
            if len(queries) > 1:
                for q in queries:
                    text, lib = str(q["query"]).strip(), _library(q.get("library"), offered) or scope
                    lists.append(_search(mc, models, collection, text, text, lib, top_k, owner_id))
                    hops.append(Hop(text, lib, len(lists[-1])))
        elif mode == "iterative":
            tokenizer, used = TiktokenTokenizer(), 0
            while len(lists) < max_rounds:
                prompt = ENOUGH.format(libraries=offered, question=u.standalone_question,
                                       found=_digest(interleave(lists, limit=top_k)))
                used += len(tokenizer.encode(prompt))
                if used > budget_tokens:
                    break
                raw = llm.chat_json([{"role": "user", "content": prompt}])
                calls += 1
                text = str(raw.get("query") or "").strip()
                if raw.get("enough", True) is True or not text:
                    break
                if " ".join(text.lower().split()) in {" ".join(h.query.lower().split()) for h in hops}:
                    break   # asking again for what was already searched adds nothing but noise
                lib = _library(raw.get("library"), offered) or scope
                lists.append(_search(mc, models, collection, text, text, lib, top_k, owner_id))
                hops.append(Hop(text, lib, len(lists[-1])))
        hits = interleave(lists, limit=top_k) if len(lists) > 1 else first
        span.set(rounds=len(hops), llm_calls=calls, queries=[h.query for h in hops])
    return MultiHopResult(u, hits, hops, calls, lists)


def retrieve(question: str, llm: LLM, models: ModelServer, collection: str, libraries: list[str] | None = None,
             history: list[dict] | None = None, mode: str = "iterative", top_k: int = 50, max_rounds: int = 3,
             budget_tokens: int = 6000, owner_id: int | None = None, mc=None) -> MultiHopResult:
    """Understanding, a first search, then `expand` (mode "single": the first search only)."""
    mc = mc or milvus_store.client()
    offered = libraries or sorted(SITES)
    u = understand(llm, question, history, libraries=offered)
    scope = u.libraries or libraries or None
    first = _search(mc, models, collection, u.standalone_question, u.bm25_text(), scope, top_k, owner_id)
    if mode == "single":
        return MultiHopResult(u, first, [Hop(u.standalone_question, scope, len(first))], 1, [first])
    r = expand(u, first, llm, models, collection, offered, scope, mode, top_k, max_rounds, budget_tokens,
               owner_id, mc)
    r.llm_calls += 1   # the understanding call
    return r
