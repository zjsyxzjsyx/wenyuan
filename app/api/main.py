"""FastAPI entry point: `uvicorn app.api.main:app --reload`."""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="wenyuan")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
