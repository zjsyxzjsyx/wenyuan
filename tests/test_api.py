import json

import httpx
import pytest
from fakeredis import FakeAsyncRedis, FakeRedis, FakeServer
from fastapi.testclient import TestClient

from app.api import main
from app.api.auth import Caller, current_user, hash_key
from app.cache.answers import AnswerCache
from app.cache.limits import Gate, RateLimiter
from app.generation.answer import Answer
from app.models.llm import LLMError
from app.query import ask as chain
from app.query.understand import Understanding
from app.retrieval.context import Source

U = Understanding("How do I delete everything in Redis?", "delete all keys redis", candidate_apis=["FLUSHALL"])
SRC = [Source("S1", "https://redis.io/flushall", "FLUSHALL", "FLUSHALL", "Deletes all keys.", 4)]


def fake_ask(question, history=None, libraries=None, on_event=chain._noop, **_):
    on_event("understanding", {"standalone_question": U.standalone_question})
    on_event("sources", {"sources": [chain.source_view(s) for s in SRC]})
    return chain.AskResult(Answer(True, "Use FLUSHALL [S1].", [{"sid": "S1", "url": SRC[0].url}]), U, "t1")


COURSES = [{"id": "redis", "name": "Redis course", "role": "student"},
           {"id": "cs50", "name": "CS50x", "role": "teacher"}]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "_courses", lambda user_id: COURSES)
    monkeypatch.setattr(main.insights, "file_question", lambda db, user_id, courses, text: None)
    monkeypatch.setattr(chain, "ask", fake_ask)
    monkeypatch.setattr(chain, "search", lambda q, h=None, l=None, on_event=chain._noop, **_:
                        chain.SearchResult(U, SRC, [], "t2"))
    main.app.dependency_overrides[current_user] = lambda: Caller(1, "alice")
    redis = FakeAsyncRedis()
    main.app.state.guards = main.Guards(RateLimiter(redis, burst=100, per_minute=600), Gate(redis, poll_s=0.01),
                                        AnswerCache(redis, poll_s=0.01))
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_ask_streams_stages_then_the_checked_answer(client):
    r = client.post("/v1/ask", json={"question": "怎么删光 Redis", "libraries": ["redis"]})
    assert r.headers["content-type"].startswith("text/event-stream")
    got = events(r.text)
    assert [name for name, _ in got] == ["understanding", "sources", "done"]
    assert got[1][1]["sources"][0]["url"] == "https://redis.io/flushall"
    assert got[2][1]["answer"]["text"] == "Use FLUSHALL [S1]." and got[2][1]["trace_id"] == "t1"


def test_ask_without_stream_returns_one_json(client):
    r = client.post("/v1/ask", json={"question": "q", "stream": False})
    assert r.status_code == 200 and r.json()["answer"]["found"] is True


def test_search_returns_source_texts(client):
    r = client.post("/v1/search", json={"question": "q"})
    assert r.json()["sources"][0]["text"] == "Deletes all keys."


def test_validation_rejects_unknown_library_and_bad_history(client):
    refused = client.post("/v1/ask", json={"question": "q", "libraries": ["mongodb"]})
    assert refused.status_code == 403 and refused.json()["detail"]["type"] == "not_enrolled"
    assert client.post("/v1/ask", json={"question": "q", "history": [{"role": "system", "content": "x"}]}
                       ).status_code == 422
    assert client.post("/v1/ask", json={"question": ""}).status_code == 422


def test_dependency_failures_are_named(client, monkeypatch):
    def down(*a, **k):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(chain, "ask", down)
    assert events(client.post("/v1/ask", json={"question": "q"}).text) == [
        ("error", {"type": "upstream_unavailable", "message": "ConnectError: connection refused"})]
    r = client.post("/v1/ask", json={"question": "q", "stream": False})
    assert r.status_code == 503 and r.json()["detail"]["type"] == "upstream_unavailable"
    monkeypatch.setattr(chain, "ask", lambda *a, **k: (_ for _ in ()).throw(LLMError("no JSON")))
    assert client.post("/v1/ask", json={"question": "q", "stream": False}).json()["detail"]["type"] == "llm_error"


def test_me_names_the_caller(client):
    assert client.get("/v1/me").json() == {"id": 1, "name": "alice"}


def test_requests_without_a_valid_key_are_rejected(monkeypatch):
    monkeypatch.setattr("app.api.auth.lookup", lambda key: Caller(7, "bob") if key == "wy_good" else None)
    monkeypatch.setattr(main, "_courses", lambda user_id: [])
    c = TestClient(main.app)
    assert c.get("/v1/libraries").status_code == 401
    assert c.get("/v1/libraries", headers={"Authorization": "Bearer wy_bad"}).status_code == 401
    assert c.get("/v1/libraries", headers={"Authorization": "Bearer wy_good"}).status_code == 200
    assert hash_key("a") != hash_key("b") and len(hash_key("a")) == 64


def test_rate_limit_returns_429_with_retry_after(client):
    redis = FakeAsyncRedis()
    main.app.state.guards = main.Guards(RateLimiter(redis, burst=2, per_minute=6), Gate(redis), AnswerCache(redis))
    codes = [client.post("/v1/ask", json={"question": "q", "stream": False}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/v1/ask", json={"question": "q", "stream": False})
    assert r.json()["detail"]["type"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1


def test_full_gate_streams_queue_position_then_busy(client):
    server = FakeServer()
    gate = Gate(FakeAsyncRedis(server=server), slots=1, per_user=1, max_wait_s=0.1, poll_s=0.01)
    main.app.state.guards = main.Guards(RateLimiter(FakeAsyncRedis(server=server), burst=100, per_minute=600),
                                        gate, AnswerCache(FakeAsyncRedis(server=server)))
    holder = FakeRedis(server=server)          # another user's request holds the only slot
    holder.zadd(gate.keys[0], {"other-request": 9e15})
    holder.hset(gate.keys[1], "other-request", "other")
    got = events(client.post("/v1/ask", json={"question": "q"}).text)
    assert got[0] == ("queued", {"position": 1})
    assert got[-1][0] == "error" and got[-1][1]["type"] == "busy"
    r = client.post("/v1/ask", json={"question": "q", "stream": False})
    assert r.status_code == 503 and r.json()["detail"]["type"] == "busy" and r.headers["Retry-After"] == "5"
    assert holder.zcard(gate.keys[2]) == 0     # the busy waiters left the queue


def test_second_identical_ask_is_served_from_cache(client, monkeypatch):
    calls = []
    monkeypatch.setattr(chain, "ask", lambda *a, **k: calls.append(1) or fake_ask(*a, **k))
    first = client.post("/v1/ask", json={"question": "Redis 删光", "libraries": ["redis"], "stream": False}).json()
    again = events(client.post("/v1/ask", json={"question": "redis  删光 ", "libraries": ["redis"]}).text)
    assert first["cached"] == "miss" and again == [("done", {**first, "cached": "hit"})]
    assert len(calls) == 1


def test_notes_endpoints_scope_to_the_caller(client, monkeypatch):
    from app.ingest import private as notes
    store = {"d1": 1}
    monkeypatch.setattr(notes, "add_note", lambda db, mc, models, tok, coll, owner, fn, content:
                        {"doc_id": "d1", "filename": fn, "title": "t", "chunks": 2})
    monkeypatch.setattr(notes, "delete_note", lambda db, mc, coll, owner, doc_id: store.get(doc_id) == owner)
    monkeypatch.setattr(main, "_db", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(main.milvus_store, "client", lambda: None)
    main.app.dependency_overrides[main.clients] = lambda: {"models": None}
    r = client.post("/v1/notes", json={"filename": "redis.md", "content": "# Redis\nEXPIRE sets a TTL."})
    assert r.status_code == 200 and r.json()["chunks"] == 2
    assert client.delete("/v1/notes/d1").status_code == 200          # caller (id 1) owns d1
    store["d2"] = 2
    assert client.delete("/v1/notes/d2").status_code == 404          # someone else's note looks missing
    monkeypatch.setattr(notes, "add_note", lambda *a: (_ for _ in ()).throw(notes.NoteError("bad name")))
    assert client.post("/v1/notes", json={"filename": "x.exe", "content": "x"}).json()["detail"]["type"] == "bad_note"


def test_dashboard_is_local_only_and_summarizes_stages(monkeypatch):
    from app.api import dashboard
    rows = [{"trace_id": "t1", "name": "POST /v1/ask", "started_ns": 1, "duration_ms": 3000, "status": "ok",
             "error": None, "user_id": 1, "queued_ms": 500, "stages": {"understand": 800, "generate": 1500}},
            {"trace_id": "t2", "name": "POST /v1/ask", "started_ns": 2, "duration_ms": None, "status": "running",
             "error": None, "user_id": 2, "queued_ms": 0, "stages": {}}]
    monkeypatch.setattr(dashboard, "requests_summary", lambda limit: rows)
    local = TestClient(main.app)                                   # TestClient's host is "testclient"
    body = local.get("/dashboard/api/summary").json()
    assert body["stats"]["total_ms"]["p50"] == 3000 and body["stats"]["stages_ms"]["generate"]["p50"] == 1500
    opened = local.get("/dashboard", follow_redirects=False)
    assert opened.status_code == 307 and opened.headers["location"] == "/ui#trace"
    remote = TestClient(main.app, client=("10.0.0.5", 5000))
    assert remote.get("/dashboard").status_code == 403
    assert remote.get("/dashboard/api/summary").status_code == 403


def test_outline_rejects_unknown_libraries(client):
    assert client.get("/v1/outline", params={"library": "mongodb"}).status_code == 403


def test_workbench_page_and_local_key_setup(monkeypatch):
    from app.api import workbench
    taken = {"koyo"}
    monkeypatch.setattr(workbench, "create_account", lambda name: None if name in taken else f"wy_{name}")
    local = TestClient(main.app)
    assert "<title>文渊</title>" in local.get("/ui").text
    assert local.post("/ui/api/key", json={"name": "小秋-web"}).json() == {"name": "小秋-web", "key": "wy_小秋-web"}
    assert local.post("/ui/api/key", json={"name": "koyo"}).status_code == 409
    assert local.post("/ui/api/key", json={"name": "a b"}).status_code == 422
    remote = TestClient(main.app, client=("10.0.0.5", 5000))
    assert remote.get("/ui").status_code == 200                    # the page itself holds nothing
    assert remote.post("/ui/api/key", json={"name": "x"}).status_code == 403


def test_traces_show_only_the_callers_own_requests(client, monkeypatch):
    from sqlalchemy import create_engine, insert
    from sqlalchemy.pool import StaticPool

    from app.api import dashboard
    from app.tracing.spans import spans_table
    db = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    spans_table.metadata.create_all(db)

    def span(trace, sid, parent, name, user=None, at=0):
        return {"span_id": sid, "trace_id": trace, "parent_span_id": parent, "kind": "http" if parent is None else "stage",
                "name": name, "started_at_ns": at, "duration_ms": 10, "status": "ok", "error_kind": None,
                "attrs_json": json.dumps({"user_id": user} if user else {"question": "secret"})}
    with db.begin() as conn:
        conn.execute(insert(spans_table), [span("mine", "a", None, "POST /v1/ask", user=1, at=1),
                                           span("mine", "b", "a", "answer"),
                                           span("theirs", "c", None, "POST /v1/ask", user=2, at=2),
                                           span("theirs", "d", "c", "answer")])
    monkeypatch.setattr(dashboard, "engine", lambda: db)
    body = client.get("/v1/traces").json()                          # the client is user 1
    assert [r["trace_id"] for r in body["requests"]] == ["mine"] and set(body["live"]) == {"gate_in_use", "gate_slots"}
    assert len(client.get("/v1/traces/mine").json()) == 2
    assert client.get("/v1/traces/theirs").status_code == 404        # someone else's trace looks missing
    assert client.get("/v1/traces/nothing").status_code == 404


def test_queries_search_only_the_callers_courses(client, monkeypatch):
    seen = []
    monkeypatch.setattr(chain, "ask", lambda q, h=None, libraries=None, **kw: seen.append(libraries) or fake_ask(q))
    client.post("/v1/ask", json={"question": "a", "stream": False})
    client.post("/v1/ask", json={"question": "b", "libraries": ["cs50", "private"], "stream": False})
    assert seen == [["redis", "cs50"], ["cs50"]]          # default: every course of mine; "private" adds nothing
    monkeypatch.setattr(main, "_courses", lambda user_id: [])
    client.post("/v1/ask", json={"question": "c", "stream": False})
    assert seen[-1] == []                                   # no course: notes only, never "everything"
    assert client.get("/v1/courses").json() == [] and client.get("/v1/libraries").json() == []


def test_course_management_is_for_its_teachers(client, monkeypatch):
    from app import courses as courses_
    from app.ingest import materials
    monkeypatch.setattr(main, "_db", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(main.milvus_store, "client", lambda: None)
    monkeypatch.setattr(courses_, "create_invite", lambda db, cid, uid, uses, days: {"code": "join-x", "course_id": cid})
    monkeypatch.setattr(materials, "add_material", lambda db, mc, models, tok, coll, cid, name, fn, text:
                        {"doc_id": "d1", "filename": fn, "course": name})
    invalidated = []
    monkeypatch.setattr(main.AnswerCache, "invalidate", lambda self, scope: invalidated.append(scope) or _done())
    monkeypatch.setattr(main.app.state, "clients", {"models": None}, raising=False)
    assert client.post("/v1/courses/cs50/invites", json={}).json()["code"] == "join-x"
    added = client.post("/v1/courses/cs50/materials", json={"filename": "week1.md", "content": "# W1"}).json()
    assert added["course"] == "CS50x" and invalidated == ["cs50"]
    for path, body in [("/v1/courses/redis/invites", {}), ("/v1/courses/redis/materials", {"filename": "a.md", "content": "x"}),
                       ("/v1/courses/algo/invites", {})]:
        assert client.post(path, json=body).status_code == 403         # a student, and not in the course at all
    assert client.get("/v1/courses/redis/materials").status_code == 403


async def _done():
    return None


def test_joining_with_a_bad_code_is_a_clear_error(client, monkeypatch):
    from app import courses as courses_

    def refuse(db, uid, code):
        raise courses_.CourseError("this invite code is not valid (unknown, expired or used up)")
    monkeypatch.setattr(main, "_db", lambda: __import__("contextlib").nullcontext())
    monkeypatch.setattr(courses_, "join", refuse)
    r = client.post("/v1/courses/join", json={"code": "join-nope"})
    assert r.status_code == 422 and r.json()["detail"]["type"] == "bad_invite"


def test_questions_are_filed_per_course_and_only_teachers_see_the_summary(client, monkeypatch):
    filed = []
    monkeypatch.setattr(main.insights, "file_question", lambda db, uid, courses, text: filed.append(([c["id"] for c in courses], text)))
    monkeypatch.setattr(main, "_db", lambda: __import__("contextlib").nullcontext())
    client.post("/v1/ask", json={"question": "malloc 失败怎么办", "libraries": ["redis"], "stream": False})
    assert filed == [(["redis"], "malloc 失败怎么办")]
    monkeypatch.setattr(main.insights, "frequent_questions", lambda db, models, cid, days: [{"question": "q", "asked": 3}])
    monkeypatch.setattr(main.app.state, "clients", {"models": None}, raising=False)
    assert client.get("/v1/courses/cs50/questions").json() == [{"question": "q", "asked": 3}]     # teacher of cs50
    assert client.get("/v1/courses/redis/questions").status_code == 403                          # a student there


def test_a_failed_question_log_never_fails_the_answer(client, monkeypatch):
    def broken(db, uid, courses, text):
        raise RuntimeError("database down")
    monkeypatch.setattr(main.insights, "file_question", broken)
    monkeypatch.setattr(main, "_db", lambda: __import__("contextlib").nullcontext())
    r = client.post("/v1/ask", json={"question": "q", "stream": False})
    assert r.status_code == 200 and r.json()["answer"]["found"]
