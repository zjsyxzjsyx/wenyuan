"""Trace: one tree of spans per request, ingest job or eval run, kept in MySQL.

Adapted from zhiqiu's src/zhiqiu/tracing.py (same author). Changes: SQLAlchemy
instead of sqlite3 so the API and several Celery workers can all write; zhiqiu's
learning-specific link columns dropped; trace ids can be carried across
processes via `inject()` / `attach()` (Celery task headers in W2).

A span is written when it starts (status "running") and updated when it ends,
so the dashboard can show what is running right now. Tracing never decides
anything and must never break a flow: every write is best-effort.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from sqlalchemy import BigInteger, Column, Integer, MetaData, String, Table, Text, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

metadata = MetaData()
spans_table = Table(
    "spans", metadata,
    Column("span_id", String(16), primary_key=True),
    Column("trace_id", String(32), index=True),
    Column("parent_span_id", String(16), nullable=True),
    Column("kind", String(32)),             # request | ingest | eval | stage | llm | retrieval
    Column("name", String(255)),
    Column("started_at_ns", BigInteger, index=True),
    Column("duration_ms", Integer, nullable=True),
    Column("status", String(16)),           # running | ok | error | interrupted
    Column("error_kind", String(128), nullable=True),
    Column("attrs_json", Text),
)

# A running span older than this whose process cannot be checked is treated as abandoned.
STALE_NS = 6 * 3600 * 10**9

_current: ContextVar["Span | None"] = ContextVar("docrag_span", default=None)
_engine: Engine | None = None


def configure(engine: Engine) -> None:
    """Set where spans go and create the table if needed. Without it, tracing is a no-op."""
    global _engine
    _engine = engine
    metadata.create_all(engine)


def _write(stmt) -> None:
    if _engine is None:
        return
    try:
        with _engine.begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError:
        pass


def _hex(size: int) -> str:
    return os.urandom(size).hex()


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    kind: str
    name: str
    started_ns: int
    attrs: dict[str, Any] = field(default_factory=dict)

    def set(self, **attrs: Any) -> None:
        """Attach facts learned while running: token counts, hit counts, cache hit, model name."""
        self.attrs.update(attrs)
        _write(update(spans_table).where(spans_table.c.span_id == self.span_id)
               .values(attrs_json=json.dumps(self.attrs, ensure_ascii=False, default=str)))


def current() -> Span | None:
    return _current.get()


def set_attrs(**attrs: Any) -> None:
    span = _current.get()
    if span is not None:
        span.set(**attrs)


def inject() -> dict[str, str]:
    """Carrier for another process: put it in Celery task headers."""
    span = _current.get()
    return {"trace_id": span.trace_id, "parent_span_id": span.span_id} if span else {}


_remote_parent: ContextVar[dict[str, str]] = ContextVar("docrag_remote_parent", default={})


@contextmanager
def attach(carrier: dict[str, str]) -> Iterator[None]:
    """Continue a trace started elsewhere: spans opened inside become children of the remote span."""
    token = _remote_parent.set(carrier or {})
    try:
        yield
    finally:
        _remote_parent.reset(token)


def _status_for(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted", "KeyboardInterrupt"
    return "error", type(exc).__name__


@contextmanager
def span(kind: str, name: str, **attrs: Any) -> Iterator[Span]:
    """Open a span under the current one (or a remote parent, or a new trace) and close it on exit."""
    parent = _current.get()
    remote = _remote_parent.get() if parent is None else {}
    trace_id = parent.trace_id if parent else remote.get("trace_id") or _hex(16)
    parent_id = parent.span_id if parent else remote.get("parent_span_id")
    item = Span(trace_id, _hex(8), parent_id, kind, name, time.time_ns(), {"pid": os.getpid(), **attrs})
    _write(insert(spans_table).values(
        span_id=item.span_id, trace_id=trace_id, parent_span_id=parent_id, kind=kind, name=name[:255],
        started_at_ns=item.started_ns, status="running",
        attrs_json=json.dumps(item.attrs, ensure_ascii=False, default=str)))
    token = _current.set(item)
    began = time.perf_counter_ns()
    status, error_kind = "ok", None
    try:
        yield item
    except BaseException as exc:
        status, error_kind = _status_for(exc)
        raise
    finally:
        _current.reset(token)
        _write(update(spans_table).where(spans_table.c.span_id == item.span_id).values(
            status=status, error_kind=error_kind, duration_ms=(time.perf_counter_ns() - began) // 10**6))


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) would send CTRL_C_EVENT on Windows, so ask the kernel instead.
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reconcile() -> int:
    """Mark spans left running by a process that died as interrupted; returns how many."""
    if _engine is None:
        return 0
    now = time.time_ns()
    try:
        with _engine.begin() as conn:
            rows = conn.execute(select(spans_table.c.span_id, spans_table.c.started_at_ns, spans_table.c.attrs_json)
                                .where(spans_table.c.status == "running")).all()
            stale = []
            for span_id, started, attrs_json in rows:
                pid = int(json.loads(attrs_json or "{}").get("pid", 0))
                if (now - started > STALE_NS and pid != os.getpid()) or not _alive(pid):
                    stale.append(span_id)
            if stale:
                conn.execute(update(spans_table).where(spans_table.c.span_id.in_(stale))
                             .values(status="interrupted", error_kind="process_exited"))
            return len(stale)
    except SQLAlchemyError:
        return 0


def tree(trace_id: str) -> list[dict[str, Any]]:
    """All spans of a trace, parents before children, for the dashboard and tests."""
    if _engine is None:
        return []
    with _engine.connect() as conn:
        rows = conn.execute(select(spans_table).where(spans_table.c.trace_id == trace_id)
                            .order_by(spans_table.c.started_at_ns)).mappings().all()
    return [{**row, "attrs": json.loads(row["attrs_json"] or "{}")} for row in rows]
