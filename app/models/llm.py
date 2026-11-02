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
import time
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
                 timeout: float = 300, use_cache: bool = True, think: bool = False) -> None:
        """`think=True` keeps a reasoning model's thinking on (slower; used for the eval judge)."""
        settings = get_settings()
        self.model = model or settings.llm_model
        self.think = think
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.use_cache = use_cache
        self._http = httpx.Client(base_url=base_url or settings.llm_base_url, timeout=timeout)
        self._cache: dict[str, dict] | None = None
        self._lock = threading.Lock()

    def _key(self, messages: list[dict]) -> str:
        parts = [self.model, messages] + (["think"] if self.think else [])
        return hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()

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

    def _post(self, body: dict, retries: int = 1, backoff_s: float = 2.0) -> httpx.Response:
        """One retry on a server error or a dropped connection: the local server occasionally answers
        500 mid-run (W4: one in ~300 calls), and one such call used to end a whole eval run.
        Client errors (4xx) are not retried; they will not change."""
        for attempt in range(retries + 1):
            try:
                response = self._http.post("/chat/completions", json=body,
                                           headers={"Authorization": f"Bearer {self.api_key}"})
                if response.status_code < 500 or attempt == retries:
                    response.raise_for_status()
                    return response
            except httpx.TransportError:
                if attempt == retries:
                    raise
            spans.set_attrs(llm_retries=attempt + 1)
            time.sleep(backoff_s * (attempt + 1))
        raise AssertionError("unreachable")

    def _unloop(self, body: dict) -> httpx.Response:
        """At temperature 0 a model that falls into a repetition loop falls into it every time, and
        Ollama aborts with "token repeat limit reached" (W4: one in ~100 answers). Sampling with a
        frequency penalty usually breaks it; a single sampled retry still looped once, hence two."""
        for attempt, (temperature, penalty) in enumerate(((0.4, 0.3), (0.8, 0.6)), start=1):
            spans.set_attrs(llm_unloop_attempts=attempt)
            try:
                return self._post({**body, "temperature": temperature, "frequency_penalty": penalty}, retries=0)
            except httpx.HTTPStatusError as exc:
                if "repeat" not in exc.response.text:
                    raise
        raise LLMError("model output kept looping (token repeat limit) at every temperature tried")

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict[str, Any]:
        key = self._key(messages)
        cache = self._load_cache() if self.use_cache else {}
        if key in cache:
            spans.set_attrs(llm_cache="hit")
            return cache[key]
        if not self.api_key:
            raise LLMError("LLM_API_KEY is not set; put it in .env (any non-empty value for a local Ollama)")
        body = {"model": self.model, "messages": messages if self.think else _no_think(self.model, messages),
                "temperature": temperature, "response_format": {"type": "json_object"}}
        if _is_qwen3(self.model) and not self.think:
            body["reasoning_effort"] = "none"
        try:
            response = self._post(body)
        except httpx.HTTPStatusError as exc:
            if "repeat" not in exc.response.text:
                raise
            response = self._unloop(body)
        body = response.json()
        result = parse_json(body["choices"][0]["message"]["content"])
        usage = body.get("usage", {})
        spans.set_attrs(llm_model=self.model, llm_cache="miss", prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"))
        if self.use_cache:
            self._store(key, result)
        return result
