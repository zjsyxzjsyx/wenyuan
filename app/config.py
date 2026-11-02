"""Settings read from the environment (and .env); defaults match docker-compose.yml."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    milvus_uri: str = "http://127.0.0.1:19530"
    mysql_url: str = "mysql+pymysql://docrag:change-me@127.0.0.1:3306/docrag?charset=utf8mb4"
    redis_url: str = "redis://127.0.0.1:6379/0"
    model_server_url: str = "http://127.0.0.1:8001"

    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3:8b"
    # The model that answers whenever a private note is in the context. Unset: the LLM_* model,
    # which is right while that is local; set these when LLM_* points at a remote API.
    local_llm_base_url: str | None = None
    local_llm_model: str | None = None

    # GPU gate: concurrent query requests, per-user cap, and how long a request may queue before
    # it gets a fast "busy" instead of a slow timeout. Set slots from measured capacity, not from
    # the LLM server's config: on a 16 GB card shared with the embedding model, 4 concurrent
    # generations ran out of GPU memory and 2 did not (W3 load test).
    gate_slots: int = 2
    gate_per_user: int = 2
    gate_max_wait_s: float = 20
    # Per-user token bucket for /v1/search and /v1/ask.
    rate_burst: int = 10
    rate_per_minute: float = 20

    # Token counting for chunk sizes and context budgets (not the embedding model's own tokenizer).
    tokenizer_encoding: str = "cl100k_base"


@lru_cache
def get_settings() -> Settings:
    return Settings()
