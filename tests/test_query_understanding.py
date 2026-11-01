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
