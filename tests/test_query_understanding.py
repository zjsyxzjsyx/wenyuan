import json

import pytest

from app.models import llm as llm_module
from app.models.llm import LLM, LLMError, parse_json
from app.query.understand import understand


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def chat_json(self, messages, temperature=0.0):
        self.prompts.append(messages[0]["content"])
        return self.reply


def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    assert parse_json('<think>\nthe user wants {x}\n</think>\n{"a": 3}') == {"a": 3}


def test_no_think_only_for_qwen3_user_turn():
    from app.models.llm import _no_think
    msgs = [{"role": "user", "content": "hi"}]
    assert _no_think("qwen3:8b", msgs)[-1]["content"] == "hi\n/no_think"
    assert _no_think("deepseek-chat", msgs) == msgs
    assert msgs[0]["content"] == "hi"  # caller's list is not mutated
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_understand_normalizes_and_filters_libraries():
    fake = FakeLLM({
        "standalone_question": "如何删除 Redis 里的所有数据？",
        "search_query_en": "delete all keys in redis",
        "keywords": ["redis", "delete", "redis", ""],
        "candidate_apis": ["FLUSHALL", "FLUSHDB"],
        "libraries": ["redis", "mongodb"],
        "needs_retrieval": True,
    })
    u = understand(fake, "那怎么全部删掉？", history=[{"role": "user", "content": "Redis 怎么删一个 key？"}],
                   libraries=["redis", "python"])
    assert u.keywords == ["redis", "delete"]
    assert u.libraries == ["redis"]                      # mongodb is not indexed
    assert u.bm25_text().startswith("FLUSHALL FLUSHDB")
    assert "Redis 怎么删一个 key？" in fake.prompts[0]   # history reaches the prompt


def test_understand_falls_back_to_question_on_sparse_reply():
    u = understand(FakeLLM({}), "What does EXPIRE do?", libraries=["redis"])
    assert u.standalone_question == u.search_query_en == "What does EXPIRE do?"
    assert u.candidate_apis == [] and u.needs_retrieval is True


def test_llm_uses_cache_and_requires_key(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_module, "CACHE", tmp_path / "llm.jsonl")
    client = LLM(base_url="http://unused", api_key="", model="m")
    messages = [{"role": "user", "content": "hi"}]
    with pytest.raises(LLMError):
        client.chat_json(messages)
    (tmp_path / "llm.jsonl").write_text(json.dumps({"key": client._key(messages), "response": {"ok": 1}}) + "\n")
    assert LLM(base_url="http://unused", api_key="", model="m").chat_json(messages) == {"ok": 1}



def test_one_server_error_is_retried_but_client_errors_are_not(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(llm_module, "CACHE", tmp_path / "llm.jsonl")
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})
    client = LLM(base_url="http://llm", api_key="k", model="m", use_cache=False)
    client._http = httpx.Client(base_url="http://llm", transport=httpx.MockTransport(handler))
    assert client.chat_json([{"role": "user", "content": "x"}]) == {"ok": 1} and len(calls) == 2
    calls.clear()
    client._http = httpx.Client(base_url="http://llm", transport=httpx.MockTransport(
        lambda r: calls.append(1) or httpx.Response(400, text="bad request")))
    with pytest.raises(httpx.HTTPStatusError):
        client.chat_json([{"role": "user", "content": "y"}])
    assert len(calls) == 1



def test_a_repetition_loop_is_retried_with_sampling(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    temperatures = []

    def handler(request):
        temperatures.append(json.loads(request.content)["temperature"])
        if temperatures[-1] == 0.0:
            return httpx.Response(500, json={"error": {"message": "prediction aborted, token repeat limit reached"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})
    client = LLM(base_url="http://llm", api_key="k", model="m", use_cache=False)
    client._http = httpx.Client(base_url="http://llm", transport=httpx.MockTransport(handler))
    assert client.chat_json([{"role": "user", "content": "x"}]) == {"ok": 1}
    assert temperatures == [0.0, 0.0, 0.4]          # the plain retry, then one with sampling
    temperatures.clear()
    client._http = httpx.Client(base_url="http://llm", transport=httpx.MockTransport(
        lambda r: temperatures.append(1) or httpx.Response(500, text="token repeat limit reached")))
    with pytest.raises(LLMError, match="looping"):
        client.chat_json([{"role": "user", "content": "z"}])
    assert len(temperatures) == 4                    # plain, plain retry, then two sampled attempts
