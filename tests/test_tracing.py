import json

import pytest
from sqlalchemy import create_engine, insert

from app.tracing import spans


@pytest.fixture(autouse=True)
def sqlite_engine():
    spans.configure(create_engine("sqlite://"))
    yield
    spans._engine = None


def test_nested_spans_form_a_tree():
    with spans.span("request", "ask") as root:
        with spans.span("retrieval", "dense_search") as child:
            child.set(hits=10)
        spans.set_attrs(cached=False)
    rows = {r["name"]: r for r in spans.tree(root.trace_id)}
    assert rows["dense_search"]["parent_span_id"] == rows["ask"]["span_id"]
    assert rows["dense_search"]["attrs"]["hits"] == 10
    assert rows["ask"]["attrs"]["cached"] is False
    assert all(r["status"] == "ok" and r["duration_ms"] is not None for r in rows.values())


def test_error_is_recorded_and_reraised():
    with pytest.raises(ValueError):
        with spans.span("request", "ask") as root:
            raise ValueError("boom")
    (row,) = spans.tree(root.trace_id)
    assert row["status"] == "error" and row["error_kind"] == "ValueError"


def test_remote_parent_via_carrier():
    with spans.span("request", "add_source") as root:
        carrier = spans.inject()
    with spans.attach(carrier):
        with spans.span("ingest", "crawl_page"):
            pass
    rows = {r["name"]: r for r in spans.tree(root.trace_id)}
    assert rows["crawl_page"]["parent_span_id"] == root.span_id


def test_reconcile_marks_dead_process_spans():
    with spans._engine.begin() as conn:
        conn.execute(insert(spans.spans_table).values(
            span_id="dead", trace_id="t", kind="ingest", name="x", started_at_ns=1, status="running",
            attrs_json=json.dumps({"pid": 999999})))
    assert spans.reconcile() == 1
    assert spans.tree("t")[0]["status"] == "interrupted"


def test_unconfigured_tracing_is_a_noop():
    spans._engine = None
    with spans.span("request", "ask") as s:
        s.set(x=1)
    assert spans.tree(s.trace_id) == []
