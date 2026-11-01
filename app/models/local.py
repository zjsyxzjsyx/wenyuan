"""Clients for the host GPU model service (model_server/server.py)."""
from __future__ import annotations

import httpx

from app.config import get_settings


class ModelServer:
    def __init__(self, base_url: str | None = None, timeout: float = 600) -> None:
        self._http = httpx.Client(base_url=base_url or get_settings().model_server_url, timeout=timeout)

    def embed(self, texts: list[str], model: str | None = None, batch: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            body = {"texts": texts[start:start + batch]}
            if model:
                body["model"] = model
            response = self._http.post("/embed", json=body)
            response.raise_for_status()
            vectors.extend(response.json()["vectors"])
        return vectors

    def rerank(self, query: str, documents: list[str], model: str | None = None) -> list[float]:
        body = {"query": query, "documents": documents}
        if model:
            body["model"] = model
        response = self._http.post("/rerank", json=body)
        response.raise_for_status()
        return response.json()["scores"]
