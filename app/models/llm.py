"""OpenAI-compatible chat client (DeepSeek, Qwen/DashScope, ...) with JSON output and a disk cache.

The cache key is (model, messages): evaluation reruns reuse earlier answers, so an
ablation run costs nothing the second time and its LLM-dependent parts are
reproducible. Token usage is attached to the current trace span.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.tracing import spans

CACHE = Path(".cache/llm_responses.jsonl")


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> dict[str, Any]:
    """Models sometimes wrap JSON in ```json fences or add a sentence; take the outermost object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


class LLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 120, use_cache: bool = True) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.use_cache = use_cache
        self._http = httpx.Client(base_url=base_url or settings.llm_base_url, timeout=timeout)
        self._cache: dict[str, dict] | None = None

    def _key(self, messages: list[dict]) -> str:
        return hashlib.sha1(json.dumps([self.model, messages], ensure_ascii=False).encode()).hexdigest()

    def _load_cache(self) -> dict[str, dict]:
        if self._cache is None:
            self._cache = {}
            if CACHE.exists():
                for line in CACHE.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["response"]
        return self._cache

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict[str, Any]:
        key = self._key(messages)
        if self.use_cache and key in self._load_cache():
            spans.set_attrs(llm_cache="hit")
            return self._cache[key]
        if not self.api_key:
            raise LLMError("LLM_API_KEY is not set; put it in .env")
        response = self._http.post("/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"}, json={
            "model": self.model, "messages": messages, "temperature": temperature,
            "response_format": {"type": "json_object"},
        })
        response.raise_for_status()
        body = response.json()
        result = parse_json(body["choices"][0]["message"]["content"])
        usage = body.get("usage", {})
        spans.set_attrs(llm_model=self.model, llm_cache="miss", prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"))
        if self.use_cache:
            self._load_cache()[key] = result
            CACHE.parent.mkdir(exist_ok=True)
            with CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": result}, ensure_ascii=False) + "\n")
        return result
