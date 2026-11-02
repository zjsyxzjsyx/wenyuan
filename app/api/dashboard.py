"""Local trace dashboard: the JSON under /dashboard/api, shown in the workbench's 看板 tab (GET /ui#trace).

Read-only, and answered only for requests from this machine (traces hold user ids
and what the model inferred from their questions). Adapted from Koyo's dashboard
(zhiqiu.dashboard / zhiqiu.traces): a request is a span tree; the page lists recent
requests with a per-stage breakdown, opens one as a waterfall, and summarizes
p50/p95 per stage, so "slow" can be attributed to queueing, the LLM, or retrieval.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from app.db.models import engine
from app.tracing.spans import spans_table as T

router = APIRouter()
LOCAL = {"127.0.0.1", "::1", "localhost", "testclient"}

# Child spans reported as stages, by span name.
STAGES = {"query_understanding": "understand", "hybrid_search": "retrieve", "dense_search": "retrieve",
          "rerank": "rerank", "small_to_big": "context", "answer": "generate"}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def local_only(request: Request) -> None:
    if request.client is None or request.client.host not in LOCAL:
        raise HTTPException(403, "the dashboard is only served to this machine")


def _rows(where, limit: int | None = None, order=T.c.started_at_ns.desc()) -> list[dict[str, Any]]:
    stmt = select(T).where(where).order_by(order)
    with engine().connect() as conn:
        rows = conn.execute(stmt.limit(limit) if limit else stmt).mappings().all()
    return [{**r, "attrs": json.loads(r["attrs_json"] or "{}")} for r in rows]


def requests_summary(limit: int = 50) -> list[dict[str, Any]]:
    """Recent API requests (root spans of kind http) with their stage durations."""
    roots = _rows((T.c.kind == "http") & T.c.parent_span_id.is_(None), limit)
    stages: dict[str, dict[str, int]] = defaultdict(dict)
    for span in _rows(T.c.trace_id.in_([r["trace_id"] for r in roots]) & T.c.name.in_(list(STAGES))):
        if span["duration_ms"] is not None:
            stage = STAGES[span["name"]]
            stages[span["trace_id"]][stage] = stages[span["trace_id"]].get(stage, 0) + span["duration_ms"]
    return [{"trace_id": r["trace_id"], "name": r["name"], "started_ns": r["started_at_ns"],
             "duration_ms": r["duration_ms"], "status": r["status"], "error": r["error_kind"],
             "user_id": r["attrs"].get("user_id"), "queued_ms": round(1000 * (r["attrs"].get("queued_s") or 0)),
             "stages": stages.get(r["trace_id"], {})} for r in roots]


def stats(requests: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in requests if r["duration_ms"] is not None]

    def pct(values: list[float]) -> dict[str, float | None]:
        return {"p50": percentile(values, 0.5) if values else None, "p95": percentile(values, 0.95) if values else None}
    by_stage = {stage: pct([r["stages"][stage] for r in done if stage in r["stages"]])
                for stage in dict.fromkeys(STAGES.values())}
    return {"requests": len(requests), "errors": sum(r["status"] == "error" for r in requests),
            "total_ms": pct([r["duration_ms"] for r in done]), "queued_ms": pct([r["queued_ms"] for r in done]),
            "stages_ms": by_stage}


def trace_detail(trace_id: str) -> list[dict[str, Any]]:
    spans = _rows(T.c.trace_id == trace_id, order=T.c.started_at_ns)
    return [{k: s[k] for k in ("span_id", "parent_span_id", "kind", "name", "started_at_ns", "duration_ms",
                               "status", "error_kind", "attrs")} for s in spans]


@router.get("/dashboard", dependencies=[Depends(local_only)])
def page() -> RedirectResponse:
    return RedirectResponse("/ui#trace")


@router.get("/dashboard/api/summary", dependencies=[Depends(local_only)])
async def summary(request: Request, limit: int = 200) -> dict[str, Any]:
    recent = await run_in_threadpool(requests_summary, min(max(limit, 1), 1000))
    guards = getattr(request.app.state, "guards", None)
    live = {"cache": await guards.cache.stats(), "gate_in_use": await guards.gate.in_use(),
            "gate_slots": guards.gate.slots, "gate_hold_s": await guards.gate.hold_s()} if guards else {}
    return {"stats": stats(recent), "live": live, "requests": recent[:50]}


@router.get("/dashboard/api/traces/{trace_id}", dependencies=[Depends(local_only)])
async def trace(trace_id: str) -> list[dict[str, Any]]:
    spans = await run_in_threadpool(trace_detail, trace_id)
    if not spans:
        raise HTTPException(404, "no such trace")
    return spans
