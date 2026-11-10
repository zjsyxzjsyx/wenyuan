import asyncio

from fakeredis import FakeAsyncRedis

from app.cache.answers import AnswerCache, normalize


def run(coro):
    return asyncio.run(coro)


async def single_flight(cache: AnswerCache, key: str, compute) -> tuple[dict, str]:
    """The same protocol the API uses (app.api.main._from_cache / _store)."""
    if (hit := await cache.get(key)) is not None:
        return hit, "hit"
    token = await cache.claim(key)
    if token is None and (shared := await cache.wait(key)) is not None:
        return shared, "shared"
    try:
        value = await compute()
        await cache.put(key, value)
        return value, "miss"
    finally:
        if token is not None:
            await cache.release(key, token)


def test_key_ignores_spacing_and_case_but_not_history_or_scope():
    async def go():
        c = AnswerCache(FakeAsyncRedis())
        base = await c.key("ask", "How  do I use EXPIRE?", ["redis"], [])
        return [base == await c.key("ask", " how do i use expire? ", ["redis"], []),
                base == await c.key("ask", "How do I use EXPIRE?", ["redis", "python"], []),
                base == await c.key("ask", "How do I use EXPIRE?", ["redis"], [{"role": "user", "content": "x"}]),
                base == await c.key("search", "How do I use EXPIRE?", ["redis"], [])]
    assert run(go()) == [True, False, False, False]
    assert normalize("  A\tB\n c ") == "a b c"


def test_concurrent_identical_questions_compute_once():
    async def go():
        cache, calls = AnswerCache(FakeAsyncRedis(), poll_s=0.01), []

        async def compute():
            calls.append(1)
            await asyncio.sleep(0.05)          # a slow LLM answer
            return {"answer": "FLUSHALL"}
        key = await cache.key("ask", "q", ["redis"], [])
        results = await asyncio.gather(*(single_flight(cache, key, compute) for _ in range(10)))
        later = await single_flight(cache, key, compute)
        return len(calls), sorted(o for _, o in results), later[1]
    calls, outcomes, later = run(go())
    assert calls == 1
    assert outcomes == ["miss"] + ["shared"] * 9
    assert later == "hit"


def test_failed_leader_lets_a_waiter_compute():
    async def go():
        cache = AnswerCache(FakeAsyncRedis(), poll_s=0.01)
        key = await cache.key("ask", "q", None, [])

        async def boom():
            await asyncio.sleep(0.03)
            raise RuntimeError("LLM down")

        async def ok():
            return {"answer": "ok"}
        leader = asyncio.create_task(single_flight(cache, key, boom))
        await asyncio.sleep(0.01)
        follower = await single_flight(cache, key, ok)
        try:
            await leader
        except RuntimeError:
            pass
        return follower
    assert run(go()) == ({"answer": "ok"}, "miss")


def test_slow_leader_does_not_release_the_lock_of_the_request_after_it():
    async def go():
        redis = FakeAsyncRedis()
        cache = AnswerCache(redis, lock_s=0.05)
        key = await cache.key("ask", "q", None, [])
        slow = await cache.claim(key)
        await asyncio.sleep(0.1)                     # queued for the GPU past lock_s: the lock expired
        second = await cache.claim(key)              # another request takes over
        await cache.release(key, slow)               # the slow leader finishes and releases
        after_stale_release = await redis.get(f"{key}:lock")
        third = await cache.claim(key)
        await cache.release(key, second)
        return slow, second, after_stale_release, third, await redis.exists(f"{key}:lock")
    slow, second, after_stale_release, third, left = run(go())
    assert slow and second and slow != second
    assert after_stale_release == second.encode()    # the second request's lock survives
    assert third is None                             # so a third request waits instead of computing again
    assert left == 0                                 # and the owner can still release it


def test_reingest_invalidates_answers_for_that_library_only():
    async def go():
        c = AnswerCache(FakeAsyncRedis())
        redis_key, py_key, any_key = (await c.key("ask", "q", ["redis"], []), await c.key("ask", "q", ["python"], []),
                                      await c.key("ask", "q", None, []))
        await c.invalidate("redis")
        return (redis_key == await c.key("ask", "q", ["redis"], []), py_key == await c.key("ask", "q", ["python"], []),
                any_key == await c.key("ask", "q", None, []))
    assert run(go()) == (False, True, False)


def test_users_with_notes_never_share_cached_answers():
    from app.cache.answers import private_scope

    async def go():
        c = AnswerCache(FakeAsyncRedis())
        before = [await c.key("ask", "q", ["redis"], [], uid) for uid in (1, 2)]
        await c.invalidate(private_scope(1))          # user 1 uploads a note
        after = [await c.key("ask", "q", ["redis"], [], uid) for uid in (1, 2)]
        public = await c.key("ask", "q", ["redis"], [], None)
        return before, after, public
    before, after, public = run(go())
    assert before[0] == before[1] == public          # no notes yet: everyone shares the public answer
    assert after[0] != after[1] and after[1] == public   # user 1's answers are now theirs alone
