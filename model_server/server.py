"""Embedding and rerank over HTTP on the local GPU.

Run from this directory: `uv run uvicorn server:app --port 8001`.
Models load lazily on first use and stay resident; GPU calls are serialized by a
lock because one process owns the card and batches are already parallel inside.
Model names are Hugging Face ids, so the Qwen3 comparison (E7) only needs a
different `model` in the request; `prompt_name` selects a model's own prompt
(Qwen3-Embedding wants its "query" instruction on queries and none on documents).
"""
from __future__ import annotations

import os
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

DEFAULT_EMBED = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DEFAULT_RERANK = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Chunks are ~300 tokens; capping the sequence length keeps a model with a 32k default (Qwen3) from
# sizing its batches for texts we never send.
MAX_SEQ = int(os.getenv("MAX_SEQ_LENGTH", "2048"))

app = FastAPI(title="wenyuan-model-server")
_lock = threading.Lock()
_embedders: dict[str, SentenceTransformer] = {}
_rerankers: dict[str, CrossEncoder] = {}


def _embedder(name: str) -> SentenceTransformer:
    if name not in _embedders:
        model = SentenceTransformer(name, device=DEVICE)
        model.max_seq_length = min(model.max_seq_length or MAX_SEQ, MAX_SEQ)
        if DEVICE == "cuda":
            model.half()
        _embedders[name] = model
    return _embedders[name]


def _reranker(name: str) -> CrossEncoder:
    if name not in _rerankers:
        model = CrossEncoder(name, device=DEVICE, max_length=1024)
        if DEVICE == "cuda":
            model.model.half()
        _rerankers[name] = model
    return _rerankers[name]


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=512)
    model: str = DEFAULT_EMBED
    batch_size: int = 32
    prompt_name: str | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(min_length=1, max_length=512)
    model: str = DEFAULT_RERANK
    batch_size: int = 16


@app.get("/healthz")
def healthz() -> dict:
    return {
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "loaded": {"embed": list(_embedders), "rerank": list(_rerankers)},
    }


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    began = time.perf_counter()
    try:
        with _lock:
            model = _embedder(req.model)
            if req.prompt_name and req.prompt_name not in (model.prompts or {}):
                raise HTTPException(400, f"{req.model} has no prompt {req.prompt_name!r}")
            vectors = model.encode(req.texts, batch_size=req.batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, prompt_name=req.prompt_name)
    except OSError as exc:  # unknown model id or download failure
        raise HTTPException(400, str(exc)) from exc
    return {
        "model": req.model,
        "dim": int(vectors.shape[1]),
        "vectors": vectors.astype("float32").tolist(),
        "ms": round((time.perf_counter() - began) * 1000, 1),
    }


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict:
    began = time.perf_counter()
    try:
        with _lock:
            scores = _reranker(req.model).predict(
                [(req.query, doc) for doc in req.documents], batch_size=req.batch_size, convert_to_numpy=True
            )
    except OSError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "model": req.model,
        "scores": [float(s) for s in scores],
        "ms": round((time.perf_counter() - began) * 1000, 1),
    }
