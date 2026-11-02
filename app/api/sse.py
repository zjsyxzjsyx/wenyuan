"""Run a blocking query chain in a worker thread and stream its stage events as server-sent events.

The chains are synchronous (httpx, pymilvus, SQLAlchemy), so each request runs in a
thread and hands events to the event loop through a queue. The thread copies the
request's context, so its spans join the request's trace.

What is streamed are stages (understanding, sources, answer), not tokens: the answer
is only sent after its citations are checked, so a learner never sees a sentence
that is later withdrawn.
"""
from __future__ import annotations

import asyncio
import json

import anyio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from app.cache.limits import GateBusy, Ticket
from app.query.ask import OnEvent


def format_event(name: str, data: Any) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n".encode()


async def stream(run: Callable[[OnEvent], dict], on_error: Callable[[Exception], dict],
                 ticket: Ticket | None = None,
                 on_done: Callable[[dict], Awaitable[dict]] | None = None) -> AsyncIterator[bytes]:
    """`run(emit)` does the work and returns the final payload, sent as event "done"
    (after `on_done`, e.g. storing it in the answer cache, which may also annotate it).
    With a gate ticket, the client first gets "queued" events while it waits for a GPU slot."""
    if ticket is not None:
        try:
            async for position in ticket.wait():
                yield format_event("queued", {"position": position})
        except GateBusy as exc:
            yield format_event("error", on_error(exc))
            return
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

    def emit(name: str, data: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (name, data))

    def work() -> None:
        try:
            emit("done", run(emit))
        except Exception as exc:  # reported to the client as an event; the stream itself already started
            emit("error", on_error(exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(work))   # to_thread copies the current context
    try:
        while (item := await queue.get()) is not None:
            name, data = item
            if name == "done" and on_done is not None:
                data = await on_done(data)
            yield format_event(name, data)
    finally:
        # A disconnected client cannot interrupt the thread (the calls are blocking); it finishes
        # and its result is dropped. The slot is held until then, so the gate's count stays true.
        # Shielded, because a disconnect cancels this generator and would cancel these awaits too.
        with anyio.CancelScope(shield=True):
            await task
            if ticket is not None:
                await ticket.release()
