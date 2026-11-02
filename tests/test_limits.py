import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from app.cache.limits import Gate, GateBusy, RateLimiter


def run(coro):
    return asyncio.run(coro)


async def acquire(gate: Gate, user) -> tuple:
    ticket, positions = gate.ticket(user), []
    async for p in ticket.wait():
        positions.append(p)
    return ticket, positions


def test_token_bucket_allows_burst_then_reports_retry_after():
    async def go():
        rl = RateLimiter(FakeAsyncRedis(), burst=3, per_minute=60)
        decisions = [await rl.hit(1, "ask") for _ in range(4)]
        other = await rl.hit(2, "ask")
        return decisions, other
    decisions, other = run(go())
    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert 0 < decisions[-1].retry_after_s <= 1.0     # one token per second
    assert other.allowed                               # buckets are per user


def test_gate_caps_concurrency_and_serves_waiters_in_order():
    async def go():
        gate = Gate(FakeAsyncRedis(), slots=2, per_user=2, poll_s=0.01)
        a, _ = await acquire(gate, "u1")
        b, _ = await acquire(gate, "u2")
        order = []

        async def waiter(user):
            t, positions = await acquire(gate, user)
            order.append(user)
            return t, positions
        w3 = asyncio.create_task(waiter("u3"))
        await asyncio.sleep(0.03)
        w4 = asyncio.create_task(waiter("u4"))
        await asyncio.sleep(0.03)
        assert await gate.in_use() == 2 and not w3.done()
        await a.release()
        t3, p3 = await w3
        await b.release()
        t4, p4 = await w4
        return order, p3, p4
    order, p3, p4 = run(go())
    assert order == ["u3", "u4"]
    assert p3 == [1] and p4[0] == 2                 # waiters see their queue position


def test_user_at_cap_does_not_block_others():
    async def go():
        gate = Gate(FakeAsyncRedis(), slots=3, per_user=1, poll_s=0.01)
        await acquire(gate, "heavy")
        blocked = asyncio.create_task(acquire(gate, "heavy"))   # queued first, but at its cap
        await asyncio.sleep(0.03)
        _, positions = await asyncio.wait_for(acquire(gate, "light"), 1)
        blocked.cancel()
        return positions
    assert run(go()) == []                          # "light" got a free slot straight away


def test_waiting_too_long_is_busy_and_leaves_the_queue():
    async def go():
        redis = FakeAsyncRedis()
        gate = Gate(redis, slots=1, per_user=1, max_wait_s=0.05, poll_s=0.01)
        await acquire(gate, "u1")
        with pytest.raises(GateBusy) as busy:
            await acquire(gate, "u2")
        return busy.value, await redis.zcard(gate.keys[2])
    busy, queued = run(go())
    assert busy.position == 1 and queued == 0


def test_expired_lease_frees_the_slot():
    async def go():
        gate = Gate(FakeAsyncRedis(), slots=1, per_user=1, lease_s=0.05, poll_s=0.01)
        await acquire(gate, "crashed")              # never released
        await asyncio.sleep(0.1)
        _, positions = await asyncio.wait_for(acquire(gate, "next"), 1)
        return positions
    assert run(go()) == []


def test_waiter_is_turned_away_at_once_when_the_expected_wait_is_too_long():
    async def go():
        redis = FakeAsyncRedis()
        gate = Gate(redis, slots=1, per_user=1, max_wait_s=20, poll_s=0.01)
        first, _ = await acquire(gate, "u1")
        await asyncio.sleep(0.05)
        await first.release()                       # seeds the average hold time (~0.05 s)
        assert 0.03 < await gate.hold_s() < 0.5
        await redis.set(gate.hold_key, 15)          # pretend answers take 15 s each
        await acquire(gate, "u1")                   # holds the only slot
        began = asyncio.get_running_loop().time()
        queued = asyncio.create_task(acquire(gate, "u3"))   # position 1: expects 15 s, still allowed
        await asyncio.sleep(0.05)
        with pytest.raises(GateBusy) as busy:
            await acquire(gate, "u4")               # position 2: expects 30 s > 20 s
        queued.cancel()
        return busy.value, asyncio.get_running_loop().time() - began
    busy, elapsed = run(go())
    assert busy.position == 2 and busy.expected_wait_s == 30 and busy.retry_after_s == 30
    assert elapsed < 1                              # answered at once, not after max_wait_s


def test_slot_is_released_even_when_the_holder_is_cancelled():
    """A client disconnect cancels the request; the slot must still come back."""
    import anyio

    async def go():
        gate = Gate(FakeAsyncRedis(), slots=1, per_user=1, poll_s=0.01)
        ticket, _ = await acquire(gate, "u1")
        leave = gate._leave

        async def networked_leave(**kw):                # real Redis yields on I/O; fakeredis does not
            await asyncio.sleep(0.001)
            return await leave(**kw)
        gate._leave = networked_leave

        async def request():
            try:
                await asyncio.sleep(10)                 # the answer is still being generated
            finally:
                await ticket.release()
        async with anyio.create_task_group() as tg:     # Starlette runs requests in task groups
            tg.start_soon(request)
            await asyncio.sleep(0.02)
            tg.cancel_scope.cancel()                    # the client went away
        return await gate.in_use()
    assert run(go()) == 0
