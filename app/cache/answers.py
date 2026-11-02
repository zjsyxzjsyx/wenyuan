"""Answer cache with single-flight: identical questions are answered once.

Key = normalized question + libraries + conversation history + the cache version
of each library in scope. A user who has private notes gets keys of their own (their
notes' version is part of the key): an answer built from one user's notes must
never be served to another user asking the same question. Users without notes
still share answers. Re-ingesting a library bumps its version (`invalidate`),
so every cached answer that could cite it stops matching; nothing has to be found
and deleted, and stale entries simply expire.

Single-flight: when the same question arrives while it is being answered (a class
working through the same chapter), the first request takes a short lock and
computes; the others wait for its result instead of each taking a GPU slot. If the
leader fails, its lock is released (or expires) and a waiter computes itself.

Exact match only: a semantic cache ("similar enough" questions share an answer)
needs a similarity threshold validated on data, or it serves wrong answers.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Sequence

import redis
from redis.asyncio import Redis

from app.config import get_settings

ALL = "*"   # version bumped by every invalidation, used when no library is given


def invalidate_now(library: str, prefix: str = "ans") -> None:
    """Synchronous `AnswerCache.invalidate` for ingest workers: called once per indexed page,
    which is cheap (two INCRs) and harmless when a task is retried."""
    client = redis.Redis.from_url(get_settings().redis_url)
    client.incr(f"{prefix}:ver:{library}")
    client.incr(f"{prefix}:ver:{ALL}")


def private_scope(owner_id: int) -> str:
    """Cache scope of one user's notes; `invalidate(private_scope(uid))` on every upload or delete."""
    return f"private:{owner_id}"


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().casefold()


class AnswerCache:
    def __init__(self, redis: Redis, ttl_s: int = 86400, lock_s: float = 120, poll_s: float = 0.1,
                 prefix: str = "ans") -> None:
        self.redis, self.ttl_s, self.lock_ms, self.poll_s, self.prefix = redis, ttl_s, int(lock_s * 1000), poll_s, prefix

    async def key(self, route: str, question: str, libraries: Sequence[str] | None, history: Sequence[dict],
                  owner_id: int | None = None) -> str:
        scope = sorted(libraries) if libraries else [ALL]
        if owner_id is not None and int(await self.redis.get(f"{self.prefix}:ver:{private_scope(owner_id)}") or 0):
            scope.append(private_scope(owner_id))
        versions = await self.redis.mget([f"{self.prefix}:ver:{lib}" for lib in scope])
        material = json.dumps([route, normalize(question), scope, [int(v or 0) for v in versions],
                               [[t["role"], normalize(t["content"])] for t in history]], ensure_ascii=False)
        return f"{self.prefix}:{route}:{hashlib.sha256(material.encode()).hexdigest()}"

    async def get(self, key: str) -> dict | None:
        raw = await self.redis.get(key)
        return json.loads(raw) if raw else None

    async def put(self, key: str, value: dict) -> None:
        await self.redis.set(key, json.dumps(value, ensure_ascii=False), ex=self.ttl_s)

    async def claim(self, key: str) -> bool:
        """Become the one request computing this key."""
        return bool(await self.redis.set(f"{key}:lock", "1", nx=True, px=self.lock_ms))

    async def release(self, key: str) -> None:
        await self.redis.delete(f"{key}:lock")

    async def wait(self, key: str) -> dict | None:
        """Follower: the leader's result, or None once the leader is gone without one."""
        deadline = time.monotonic() + self.lock_ms / 1000
        while time.monotonic() < deadline:
            if (value := await self.get(key)) is not None:
                return value
            if not await self.redis.exists(f"{key}:lock"):
                return await self.get(key)
            await asyncio.sleep(self.poll_s)
        return None

    async def count(self, outcome: str) -> None:
        """hit (served from cache), shared (waited for a concurrent leader), miss (computed)."""
        await self.redis.hincrby(f"{self.prefix}:stats", outcome, 1)

    async def stats(self) -> dict[str, int]:
        return {k.decode(): int(v) for k, v in (await self.redis.hgetall(f"{self.prefix}:stats")).items()}

    async def invalidate(self, library: str) -> None:
        """Call after (re)ingesting a library, or with private_scope(uid) after a note changes."""
        await self.redis.incr(f"{self.prefix}:ver:{library}")
        if not library.startswith("private:"):
            await self.redis.incr(f"{self.prefix}:ver:{ALL}")
