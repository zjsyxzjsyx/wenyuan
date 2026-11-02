"""Load test: N learners asking at once against one GPU.

`python -m eval.loadtest --label gate --users 1 5 10 20 --per-user 3`
Each virtual user has its own API key and asks `per-user` different real Stack
Overflow questions back to back (no think time), streaming the answer. A client
gives up after `--timeout` seconds, like a learner who stops waiting. The answer
cache is invalidated before each round so every request reaches the GPU.

Reported per N: outcomes (ok / busy / rate_limited / timeout / error), latency of
successful answers, time to the first event (queue position or understanding), and
goodput: answers delivered within the timeout per minute. Goodput, not raw
throughput, is the point: a request the client already abandoned still burns GPU
time if nothing stops it before the model runs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from app.api.auth import create_user
from app.cache.answers import invalidate_now
from app.db.models import create_tables
from app.ingest.sites import SITES
from eval.metrics import percentile

DATASET = Path("eval/datasets/so_v1.jsonl")
KEYS = Path(".cache/loadtest_keys.json")
OUT_DIR = Path("experiments")


def user_keys(n: int) -> list[str]:
    """API keys for loadtest users 1..n, created once and kept in a gitignored file."""
    keys = json.loads(KEYS.read_text()) if KEYS.exists() else []
    if len(keys) < n:
        create_tables()
        stamp = int(time.time())
        keys += [create_user(f"loadtest-{stamp}-{i}") for i in range(len(keys), n)]
        KEYS.parent.mkdir(exist_ok=True)
        KEYS.write_text(json.dumps(keys))
    return keys[:n]


def questions(seed: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    rows = [{"question": r["question"], "libraries": [r["library"]]} for r in rows if r["library"] in SITES]
    random.Random(seed).shuffle(rows)
    return rows


async def one_request(client: httpx.AsyncClient, key: str, body: dict, timeout: float) -> dict:
    began, first, name, record = time.perf_counter(), None, None, {"status": "error"}
    try:
        async with asyncio.timeout(timeout):
            async with client.stream("POST", "/v1/ask", json=body, headers={"Authorization": f"Bearer {key}"}) as r:
                if r.status_code != 200:
                    await r.aread()
                    kind = {429: "rate_limited", 503: "busy"}.get(r.status_code, "error")
                    return {"status": kind, "first_event_s": None, "total_s": time.perf_counter() - began}
                async for line in r.aiter_lines():
                    if line.startswith("event: "):
                        name = line[7:]
                        first = first if first is not None else time.perf_counter() - began
                    elif line.startswith("data: ") and name in ("done", "error"):
                        data = json.loads(line[6:])
                        if name == "done":
                            record = {"status": "ok", "found": data["answer"]["found"], "cached": data.get("cached")}
                        else:
                            record = {"status": "busy" if data["type"] == "busy" else "error", "error": data["type"]}
    except TimeoutError:
        record = {"status": "timeout"}
    except httpx.HTTPError as exc:
        record = {"status": "error", "error": type(exc).__name__}
    return {**record, "first_event_s": first, "total_s": time.perf_counter() - began}


async def round_(base_url: str, n_users: int, per_user: int, timeout: float, pool: list[dict]) -> dict:
    for library in SITES:      # cold answer cache: every request must reach the model
        invalidate_now(library)
    keys = user_keys(n_users)
    work = iter(pool)
    limits = httpx.Limits(max_connections=n_users + 5)
    async with httpx.AsyncClient(base_url=base_url, timeout=None, limits=limits) as client:
        async def user(i: int) -> list[dict]:
            out = []
            for _ in range(per_user):
                out.append({"user": i, **await one_request(client, keys[i], {**next(work), "stream": True}, timeout)})
            return out
        began = time.perf_counter()
        results = [r for rs in await asyncio.gather(*(user(i) for i in range(n_users))) for r in rs]
        wall = time.perf_counter() - began
    ok = [r for r in results if r["status"] == "ok"]
    firsts = [r["first_event_s"] for r in results if r["first_event_s"] is not None]
    outcomes = {s: sum(r["status"] == s for r in results) for s in ("ok", "busy", "rate_limited", "timeout", "error")}
    return {
        "users": n_users, "requests": len(results), "outcomes": outcomes, "wall_s": round(wall, 1),
        "ok_p50_s": round(percentile([r["total_s"] for r in ok], 0.5), 2) if ok else None,
        "ok_p95_s": round(percentile([r["total_s"] for r in ok], 0.95), 2) if ok else None,
        "first_event_p95_s": round(percentile(firsts, 0.95), 2) if firsts else None,
        "goodput_per_min": round(len(ok) / wall * 60, 1),
        "requests_detail": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="e.g. gate / nogate (the API's settings decide which)")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--per-user", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    pool, rounds = questions(), []
    offset = 0
    for n in args.users:
        need = n * args.per_user
        r = asyncio.run(round_(args.base_url, n, args.per_user, args.timeout, pool[offset:offset + need]))
        offset += need
        rounds.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "requests_detail"}, ensure_ascii=False), flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"W3-load-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"label": args.label, "timeout_s": args.timeout, "per_user": args.per_user,
                                "rounds": rounds}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
