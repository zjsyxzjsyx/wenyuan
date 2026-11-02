import asyncio
import json

import httpx
import pytest

from app import mcp_server as m


def fake_api(monkeypatch, handler):
    monkeypatch.setattr(m, "_client", lambda: httpx.Client(base_url="http://wenyuan", transport=httpx.MockTransport(handler)))


def test_tools_are_registered():
    names = {t.name for t in asyncio.run(m.mcp.list_tools())}
    assert names == {"libraries", "search", "ask", "outline", "get_section", "add_note", "list_notes"}


def test_search_text_carries_every_source_url_within_budget(monkeypatch):
    sources = [{"sid": f"S{i}", "title": "Redis", "heading_path": f"EXPIRE {i}", "url": f"https://redis.io/{i}",
                "text": "x" * 5000, "private": i == 2} for i in range(1, 6)]

    def handler(request):
        body = json.loads(request.content)
        assert body["history"] == [{"role": "user", "content": "Redis 怎么删 key"}]   # system turns dropped
        return httpx.Response(200, json={"sources": sources})
    fake_api(monkeypatch, handler)
    text = m.search("那它呢", ["redis"], [{"role": "system", "content": "x"}, {"role": "user", "content": "Redis 怎么删 key"}])
    assert len(text) <= m.BUDGET + 20
    assert all(f"https://redis.io/{i}" in text for i in range(1, 6))
    assert "（我的笔记）" in text


def test_ask_lists_citations_or_says_there_is_no_basis(monkeypatch):
    found = {"answer": {"found": True, "text": "用 FLUSHALL [S1]。", "citations": [
        {"sid": "S1", "title": "FLUSHALL", "heading_path": "FLUSHALL", "url": "https://redis.io/flushall"}]}}
    fake_api(monkeypatch, lambda r: httpx.Response(200, json=found))
    assert "https://redis.io/flushall" in m.ask("怎么删光 Redis")
    refused = {"answer": {"found": False, "text": "资料里没有找到能回答这个问题的内容。", "citations": []}}
    fake_api(monkeypatch, lambda r: httpx.Response(200, json=refused))
    assert "不要凭记忆补充" in m.ask("Redis 的作者喜欢什么颜色")


def test_busy_and_not_found_become_readable_errors(monkeypatch):
    fake_api(monkeypatch, lambda r: httpx.Response(503, json={"detail": {"type": "busy"}}, headers={"Retry-After": "12"}))
    with pytest.raises(m.WenyuanError, match="12 秒后重试"):
        m.ask("q")
    fake_api(monkeypatch, lambda r: httpx.Response(404, json={"detail": {"type": "not_found"}}))
    with pytest.raises(m.WenyuanError, match="不是你的私有笔记"):
        m.get_section("abc:1")


def test_outline_is_indented_with_section_ids_and_summaries():
    text = m.format_outline([{"title": "Persistence", "url": "https://redis.io/p", "sections": [
        {"section_id": "d1:1", "level": 1, "heading": "Persistence", "summary": "RDB and AOF."},
        {"section_id": "d1:2", "level": 2, "heading": "RDB"}]}])
    assert "- Persistence  [d1:1]" in text and "  - RDB  [d1:2]" in text and "摘要：RDB and AOF." in text


def test_missing_key_is_explained(monkeypatch):
    monkeypatch.delenv("WENYUAN_API_KEY", raising=False)
    with pytest.raises(m.WenyuanError, match="WENYUAN_API_KEY"):
        m._client()
