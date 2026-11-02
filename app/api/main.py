"""FastAPI entry point: `uvicorn app.api.main:app`.

POST /v1/search  sources for a question, no generation (what a tutor agent quotes from)
POST /v1/ask     cited answer; server-sent events by default ("stream": false for one JSON body)
GET  /v1/libraries
POST /v1/notes, GET /v1/notes, DELETE /v1/notes/{doc_id}   the caller's private notes
GET  /dashboard  local trace dashboard (this machine only)
Every request needs `Authorization: Bearer <key>` (`python -m app.api.auth create <name>`).

Query requests pass Redis guards before any work: a per-user token bucket (429 +
Retry-After), then the answer cache (app.cache.answers: a hit, or the result of an
identical request already running, costs no GPU), then the GPU gate
(app.cache.limits: queue position while waiting, 503 + Retry-After when the wait
would exceed its limit). Responses say which happened in "cached": hit | shared | miss.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from redis.asyncio import BlockingConnectionPool, Redis

from app.api import dashboard, sse
from app.api.auth import Caller, current_user
from app.cache.answers import AnswerCache, private_scope
from app.cache.limits import Gate, GateBusy, RateLimiter, Ticket
from app.config import get_settings
from app.db.models import create_tables, engine
from app.ingest import private as notes
from app.ingest.sites import SITES
from app.models.llm import LLM, LLMError
from app.models.local import ModelServer
from app.query import ask as chain
from app.retrieval import milvus_store
from app.tokenizer import TiktokenTokenizer
from app.tracing import spans


@dataclass
class Guards:
    limiter: RateLimiter
    gate: Gate
    cache: AnswerCache

    @classmethod
    def from_redis(cls, redis: Redis) -> Guards:
        s = get_settings()
        return cls(RateLimiter(redis, s.rate_burst, s.rate_per_minute),
                   Gate(redis, slots=s.gate_slots, per_user=s.gate_per_user, max_wait_s=s.gate_max_wait_s),
                   AnswerCache(redis))


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    spans.configure(engine())
    # A bounded pool: under a burst, requests wait for a free connection instead of opening
    # dozens at once (which the Docker port forward on Windows refused in the W3 load test).
    redis = Redis(connection_pool=BlockingConnectionPool.from_url(get_settings().redis_url, max_connections=64,
                                                                  timeout=10))
    app.state.guards = Guards.from_redis(redis)
    # One client each for the whole process: pooled connections, and no eval disk cache
    # (a served answer must come from the model or the Redis answer cache, never an eval run).
    s = get_settings()
    llm = LLM(use_cache=False)
    local = (LLM(base_url=s.local_llm_base_url, model=s.local_llm_model, api_key="local", use_cache=False)
             if s.local_llm_base_url else llm)
    app.state.clients = {"llm": llm, "local_llm": local, "models": ModelServer()}
    milvus_store.ensure_owner_field(milvus_store.client(), chain.COLLECTION)
    yield
    await redis.aclose()


app = FastAPI(title="wenyuan", lifespan=lifespan)
app.include_router(dashboard.router)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    libraries: list[str] | None = None
    include_private: bool = True        # also search the caller's own notes
    stream: bool = True

    @field_validator("libraries")
    @classmethod
    def known_libraries(cls, value: list[str] | None) -> list[str] | None:
        unknown = sorted(set(value or []) - SITES.keys())
        if unknown:
            raise ValueError(f"unknown libraries: {unknown}")
        return value or None


def error_payload(exc: Exception) -> dict:
    """Failures are named so the client can tell 'try again later' from 'model down' from 'bug'."""
    if isinstance(exc, GateBusy):
        return {"type": "busy", "message": str(exc), "position": exc.position, "retry_after_s": exc.retry_after_s}
    if isinstance(exc, LLMError):
        return {"type": "llm_error", "message": str(exc)}
    if isinstance(exc, httpx.HTTPError):
        return {"type": "upstream_unavailable", "message": f"{type(exc).__name__}: {exc}"}
    return {"type": "internal_error", "message": type(exc).__name__}


def guards(request: Request) -> Guards:
    return request.app.state.guards


def clients(request: Request) -> dict:
    return getattr(request.app.state, "clients", {})


def rate_limited(route: str):
    async def check(user: Caller = Depends(current_user), g: Guards = Depends(guards)) -> Caller:
        decision = await g.limiter.hit(user.id, route)
        if not decision.allowed:
            raise HTTPException(429, {"type": "rate_limited", "retry_after_s": round(decision.retry_after_s, 1)},
                                headers={"Retry-After": str(max(1, round(decision.retry_after_s)))})
        return user
    return check


def _history(req: QueryRequest) -> list[dict]:
    return [t.model_dump() for t in req.history]


def _owner(req: QueryRequest, user: Caller) -> int | None:
    return user.id if req.include_private else None


def search_payload(r: chain.SearchResult) -> dict:
    return {"understanding": asdict(r.understanding),
            "sources": [{**chain.source_view(s), "text": s.text} for s in r.sources], "trace_id": r.trace_id}


def ask_payload(r: chain.AskResult) -> dict:
    return {"answer": asdict(r.answer), "trace_id": r.trace_id}


async def _from_cache(cache: AnswerCache, key: str) -> tuple[dict | None, bool]:
    """(cached or shared result, whether this request leads the computation)."""
    if (hit := await cache.get(key)) is not None:
        await cache.count("hit")
        return {**hit, "cached": "hit"}, False
    if await cache.claim(key):
        return None, True
    if (shared := await cache.wait(key)) is not None:
        await cache.count("shared")
        return {**shared, "cached": "shared"}, False
    return None, False   # the leader failed: compute without the lock


async def _store(cache: AnswerCache, key: str, result: dict) -> dict:
    await cache.put(key, result)
    await cache.count("miss")
    return {**result, "cached": "miss"}


async def _respond(g: Guards, key: str, ticket: Ticket, run, payload):
    """Non-streaming path: cache, then a GPU slot, then the chain in a thread; failures map to status codes."""
    found, leader = await _from_cache(g.cache, key)
    if found is not None:
        return found
    try:
        try:
            async for _ in ticket.wait():
                pass
        except GateBusy as exc:
            raise HTTPException(503, error_payload(exc), headers={"Retry-After": str(exc.retry_after_s)}) from exc
        try:
            result = payload(await run_in_threadpool(run, chain._noop))
        except Exception as exc:
            raise HTTPException(503 if isinstance(exc, (LLMError, httpx.HTTPError)) else 500,
                                error_payload(exc)) from exc
        finally:
            await ticket.release()
        return await _store(g.cache, key, result)
    finally:
        if leader:
            await g.cache.release(key)


async def _stream_events(g: Guards, key: str, ticket: Ticket, run):
    found, leader = await _from_cache(g.cache, key)
    if found is not None:
        yield sse.format_event("done", found)
        return
    try:
        async for chunk in sse.stream(run, error_payload, ticket, on_done=lambda r: _store(g.cache, key, r)):
            yield chunk
    finally:
        if leader:
            await g.cache.release(key)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/libraries")
def libraries(_: Caller = Depends(current_user)) -> list[dict]:
    return [{"id": lib, "name": site.name} for lib, site in sorted(SITES.items())]


@app.post("/v1/search")
async def search(req: QueryRequest, user: Caller = Depends(rate_limited("search")), g: Guards = Depends(guards),
                 c: dict = Depends(clients)):
    ticket = g.gate.ticket(user.id)

    def run(emit):
        with spans.span("http", "POST /v1/search", user_id=user.id, queued_s=round(ticket.waited_s, 3)):
            return chain.search(req.question, _history(req), req.libraries, on_event=emit,
                                owner_id=_owner(req, user), **c)
    key = await g.cache.key("search", req.question, req.libraries, _history(req), _owner(req, user))
    return await _respond(g, key, ticket, run, search_payload)


@app.post("/v1/ask")
async def ask(req: QueryRequest, user: Caller = Depends(rate_limited("ask")), g: Guards = Depends(guards),
              c: dict = Depends(clients)):
    ticket = g.gate.ticket(user.id)

    def run(emit):
        # queued_s: time spent waiting for a GPU slot, so a trace separates queueing from model time.
        with spans.span("http", "POST /v1/ask", user_id=user.id, queued_s=round(ticket.waited_s, 3)):
            return chain.ask(req.question, _history(req), req.libraries, on_event=emit,
                             owner_id=_owner(req, user), **c)
    key = await g.cache.key("ask", req.question, req.libraries, _history(req), _owner(req, user))
    if not req.stream:
        return await _respond(g, key, ticket, run, ask_payload)
    return StreamingResponse(_stream_events(g, key, ticket, lambda emit: ask_payload(run(emit))),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class NoteRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=notes.MAX_BYTES)


def _db():
    from sqlalchemy.orm import Session as DBSession
    return DBSession(engine())


@app.post("/v1/notes")
async def add_note(req: NoteRequest, user: Caller = Depends(rate_limited("notes")), g: Guards = Depends(guards),
                   c: dict = Depends(clients)):
    def run():
        with spans.span("http", "POST /v1/notes", user_id=user.id), _db() as db:
            return notes.add_note(db, milvus_store.client(), c["models"], TiktokenTokenizer(), chain.COLLECTION,
                                  user.id, req.filename, req.content)
    try:
        result = await run_in_threadpool(run)
    except notes.NoteError as exc:
        raise HTTPException(422, {"type": "bad_note", "message": str(exc)}) from exc
    await g.cache.invalidate(private_scope(user.id))
    return result


@app.get("/v1/notes")
async def list_notes(user: Caller = Depends(current_user)):
    def run():
        with _db() as db:
            return notes.list_notes(db, user.id)
    return await run_in_threadpool(run)


@app.delete("/v1/notes/{doc_id}")
async def delete_note(doc_id: str, user: Caller = Depends(current_user), g: Guards = Depends(guards)):
    def run():
        with _db() as db:
            return notes.delete_note(db, milvus_store.client(), chain.COLLECTION, user.id, doc_id)
    if not await run_in_threadpool(run):
        raise HTTPException(404, {"type": "not_found"})
    await g.cache.invalidate(private_scope(user.id))
    return {"deleted": doc_id}
