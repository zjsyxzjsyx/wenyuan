"""Redis guards for scarce resources: a per-user token bucket and a fair GPU gate.

Both run as Lua scripts, so each check-and-update is atomic across API processes,
and both read time from Redis (`TIME`), so API hosts with skewed clocks agree.

Token bucket: each (user, route) has `burst` tokens refilled at `per_minute`; a
request spends one or gets 429 with the seconds until the next token.

Gate: at most `slots` requests use the GPU at once (= the LLM server's parallel
slots), at most `per_user` of them from one user. The rest wait in arrival order;
a waiter whose user is at their cap is skipped rather than blocking everyone
behind it. Without the gate, extra requests still queue inside the LLM server, but
invisibly and without bound: every request slows down until clients time out. With
it, a waiter knows its position, and one that waits longer than `max_wait_s` gets a
fast "busy" instead of a timeout.

Admission control: the gate keeps a moving average of how long a request holds a
slot. A waiter at position k can expect about k * hold / slots seconds; when that
already exceeds `max_wait_s`, it is told "busy" at once, with that estimate as the
retry hint, instead of after waiting `max_wait_s` for nothing (W3 load test: busy
answers came after 20 s before this, in ~0.1 s after).

Crash safety: a slot is a lease that expires after `lease_s` (longer than the LLM
timeout), and a waiter that vanished is dropped once it is older than `max_wait_s`.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import anyio
from redis.asyncio import Redis

_BUCKET = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1e6
local burst, rate = tonumber(ARGV[1]), tonumber(ARGV[2])
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = math.min(burst, (tonumber(b[1]) or burst) + (now - (tonumber(b[2]) or now)) * rate)
local allowed, retry = 0, 0
if tokens >= 1 then tokens = tokens - 1; allowed = 1 else retry = (1 - tokens) / rate end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], math.ceil(burst / rate) + 1)
return {allowed, tostring(retry)}
"""

# KEYS: holders (zset token -> lease expiry ms), owners (hash token -> user),
#       queue (zset token -> enqueue ms), waiters (hash token -> user)
# ARGV: token, user, slots, per_user, lease_ms, stale_ms
_ACQUIRE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local token, user = ARGV[1], ARGV[2]
local slots, per_user = tonumber(ARGV[3]), tonumber(ARGV[4])
for _, m in ipairs(redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now)) do redis.call('HDEL', KEYS[2], m) end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
local stale = now - tonumber(ARGV[6])
for _, m in ipairs(redis.call('ZRANGEBYSCORE', KEYS[3], '-inf', stale)) do redis.call('HDEL', KEYS[4], m) end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', stale)
if not redis.call('ZSCORE', KEYS[3], token) then
  redis.call('ZADD', KEYS[3], now, token)
  redis.call('HSET', KEYS[4], token, user)
end
local held, holders = {}, redis.call('ZRANGE', KEYS[1], 0, -1)
for _, m in ipairs(holders) do
  local u = redis.call('HGET', KEYS[2], m)
  held[u] = (held[u] or 0) + 1
end
local free, ahead = slots - #holders, 0
for i, m in ipairs(redis.call('ZRANGE', KEYS[3], 0, -1)) do
  local u = redis.call('HGET', KEYS[4], m)
  local eligible = (held[u] or 0) < per_user
  if m == token then
    if eligible and ahead < free then
      redis.call('ZREM', KEYS[3], token)
      redis.call('HDEL', KEYS[4], token)
      redis.call('ZADD', KEYS[1], now + tonumber(ARGV[5]), token)
      redis.call('HSET', KEYS[2], token, user)
      return 0
    end
    return i
  end
  if eligible then
    ahead = ahead + 1
    held[u] = (held[u] or 0) + 1
  end
end
return -1
"""

# Exponential moving average of slot hold times (seconds); the first sample seeds it.
_HOLD = """
local old = tonumber(redis.call('GET', KEYS[1]))
local new = old and (old * 0.8 + tonumber(ARGV[1]) * 0.2) or tonumber(ARGV[1])
redis.call('SET', KEYS[1], tostring(new))
return tostring(new)
"""

_LEAVE = """
redis.call('ZREM', KEYS[1], ARGV[1]); redis.call('HDEL', KEYS[2], ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[1]); redis.call('HDEL', KEYS[4], ARGV[1])
return 1
"""


@dataclass
class RateDecision:
    allowed: bool
    retry_after_s: float


class RateLimiter:
    def __init__(self, redis: Redis, burst: int, per_minute: float) -> None:
        self.redis, self.burst, self.rate = redis, burst, per_minute / 60
        self._script = redis.register_script(_BUCKET)

    async def hit(self, user: int | str, route: str) -> RateDecision:
        allowed, retry = await self._script(keys=[f"rl:{route}:{user}"], args=[self.burst, self.rate])
        return RateDecision(bool(int(allowed)), float(retry))


class GateBusy(Exception):
    def __init__(self, position: int, waited_s: float, expected_wait_s: float | None = None) -> None:
        detail = f"expected wait {expected_wait_s:.0f}s" if expected_wait_s is not None else f"waited {waited_s:.1f}s"
        super().__init__(f"GPU busy: position {position}, {detail}")
        self.position, self.waited_s, self.expected_wait_s = position, waited_s, expected_wait_s

    @property
    def retry_after_s(self) -> int:
        return max(1, round(self.expected_wait_s or 5))


class Gate:
    def __init__(self, redis: Redis, name: str = "gpu", slots: int = 4, per_user: int = 2,
                 lease_s: float = 360, max_wait_s: float = 30, poll_s: float = 0.1) -> None:
        self.redis, self.slots, self.per_user = redis, slots, per_user
        self.lease_ms, self.max_wait_s, self.poll_s = int(lease_s * 1000), max_wait_s, poll_s
        self.keys = [f"gate:{name}:{k}" for k in ("holders", "owners", "queue", "waiters")]
        self.hold_key = f"gate:{name}:hold_s"
        self._acquire = redis.register_script(_ACQUIRE)
        self._leave = redis.register_script(_LEAVE)
        self._hold = redis.register_script(_HOLD)

    def ticket(self, user: int | str) -> Ticket:
        return Ticket(self, str(user))

    async def in_use(self) -> int:
        return await self.redis.zcard(self.keys[0])

    async def hold_s(self) -> float | None:
        """Average time a request holds a slot, or None before the first one finished."""
        value = await self.redis.get(self.hold_key)
        return float(value) if value is not None else None

    def expected_wait_s(self, position: int, hold_s: float | None) -> float | None:
        return position * hold_s / self.slots if hold_s is not None else None


class Ticket:
    """One request's place at the gate: `async for position in ticket.wait()` yields queue
    positions until a slot is held; `release()` gives it back (and leaves the queue)."""

    def __init__(self, gate: Gate, user: str) -> None:
        self.gate, self.user, self.token = gate, user, uuid.uuid4().hex
        self.waited_s = 0.0
        self._acquired_at: float | None = None

    async def wait(self) -> AsyncIterator[int]:
        g, start, last = self.gate, time.monotonic(), None
        stale_ms = int((g.max_wait_s + 5) * 1000)
        try:
            while True:
                position = int(await g._acquire(keys=g.keys, args=[self.token, self.user, g.slots, g.per_user,
                                                                     g.lease_ms, stale_ms]))
                self.waited_s = time.monotonic() - start
                if position == 0:
                    self._acquired_at = time.monotonic()
                    return
                expected = g.expected_wait_s(position, await g.hold_s())
                if expected is not None and self.waited_s + expected > g.max_wait_s:
                    raise GateBusy(position, self.waited_s, expected)
                if self.waited_s > g.max_wait_s:
                    raise GateBusy(position, self.waited_s)
                if position != last:
                    last = position
                    yield position
                await asyncio.sleep(g.poll_s)
        except BaseException:   # busy, or the client went away while queued
            await self.release()
            raise

    async def release(self) -> None:
        """Shielded: it runs when a client disconnects, i.e. inside a cancelled scope where any
        plain await is cancelled again. Unshielded, a disconnect leaked the slot until its
        lease expired (W3 load test: two timed-out clients blocked the gate for minutes)."""
        with anyio.CancelScope(shield=True):
            await self.gate._leave(keys=self.gate.keys, args=[self.token])
            if self._acquired_at is not None:
                held, self._acquired_at = time.monotonic() - self._acquired_at, None
                await self.gate._hold(keys=[self.gate.hold_key], args=[held])
