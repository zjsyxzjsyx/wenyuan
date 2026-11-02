"""OpenAI-compatible chat client (local Ollama, DeepSeek, Qwen/DashScope, ...) with JSON output and a disk cache.

The cache key is (model, messages): evaluation reruns reuse earlier answers, so an
ablation run costs nothing the second time and its LLM-dependent parts are
reproducible. Token usage is attached to the current trace span. The client is
safe to share across threads (eval runs calls in parallel).

Qwen3 models "think" before answering by default, which multiplies latency for
the short structured outputs used here (318 hidden tokens for a 10-token JSON).
Ollama ignores the `/no_think` prompt switch and only honours `reasoning_effort:
"none"`; other Qwen3 servers (vLLM, DashScope) honour the prompt switch, so both
are sent. Any <think> block that still comes back is stripped.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.tracing import spans

CACHE = Path(".cache/llm_responses.jsonl")
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict[str, Any]:
    """Models sometimes add a <think> block, ```json fences or a sentence; take the outermost object."""
    text = _THINK.sub("", text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _is_qwen3(model: str) -> bool:
    return model.lower().startswith("qwen3")


def _no_think(model: str, messages: list[dict]) -> list[dict]:
    if not _is_qwen3(model) or not messages or messages[-1]["role"] != "user":
        return messages
    return messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + "\n/no_think"}]


class LLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 300, use_cache: bool = True) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.use_cache = use_cache
        self._http = httpx.Client(base_url=base_url or settings.llm_base_url, timeout=timeout)
        self._cache: dict[str, dict] | None = None
        self._lock = threading.Lock()

    def _key(self, messages: list[dict]) -> str:
        return hashlib.sha1(json.dumps([self.model, messages], ensure_ascii=False).encode()).hexdigest()

    def _load_cache(self) -> dict[str, dict]:
        with self._lock:
            if self._cache is None:
                self._cache = {}
                if CACHE.exists():
                    for line in CACHE.read_text(encoding="utf-8").splitlines():
                        row = json.loads(line)
                        self._cache[row["key"]] = row["response"]
            return self._cache

    def _store(self, key: str, result: dict) -> None:
        with self._lock:
            self._cache[key] = result
            CACHE.parent.mkdir(exist_ok=True)
            with CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": result}, ensure_ascii=False) + "\n")

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict[str, Any]:
        key = self._key(messages)
        cache = self._load_cache() if self.use_cache else {}
        if key in cache:
            spans.set_attrs(llm_cache="hit")
            return cache[key]
        if not self.api_key:
            raise LLMError("LLM_API_KEY is not set; put it in .env (any non-empty value for a local Ollama)")
        body = {"model": self.model, "messages": _no_think(self.model, messages), "temperature": temperature,
                "response_format": {"type": "json_object"}}
        if _is_qwen3(self.model):
            body["reasoning_effort"] = "none"
        response = self._http.post("/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"},
                                   json=body)
        response.raise_for_status()
        body = response.json()
        result = parse_json(body["choices"][0]["message"]["content"])
        usage = body.get("usage", {})
        spans.set_attrs(llm_model=self.model, llm_cache="miss", prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"))
        if self.use_cache:
            self._store(key, result)
        return result
