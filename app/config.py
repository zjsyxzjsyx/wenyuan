"""Settings read from the environment (and .env); defaults match docker-compose.yml."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    milvus_uri: str = "http://localhost:19530"
    mysql_url: str = "mysql+pymysql://docrag:change-me@localhost:3306/docrag?charset=utf8mb4"
    redis_url: str = "redis://localhost:6379/0"
    model_server_url: str = "http://localhost:8001"

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3:8b"

    # Token counting for chunk sizes and context budgets (not the embedding model's own tokenizer).
    tokenizer_encoding: str = "cl100k_base"


@lru_cache
def get_settings() -> Settings:
    return Settings()
