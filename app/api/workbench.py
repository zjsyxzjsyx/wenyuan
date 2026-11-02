"""Web workbench: GET /ui is a page for using wenyuan without a terminal.

Ask questions (stages stream in over SSE, citations link to the docs), keep
private notes, browse a library's outline. The page talks to the same /v1 API as
any client, with the caller's API key kept in the browser.

POST /ui/api/key creates a user and returns its key once, so a first-time local
user needs no command line. It is answered only for this machine, like the
dashboard: on a shared deployment, accounts are still made by an administrator
(`python -m app.api.auth create <name>`).
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.api.auth import create_user
from app.api.dashboard import local_only
from app.db.models import User, create_tables, engine

router = APIRouter()
PAGE = Path(__file__).with_name("workbench.html")
NAME = re.compile(r"^[\w\-.一-鿿]{1,32}$")


class KeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=32)


def create_account(name: str) -> str | None:
    """A new user's key, or None when the name is taken (its key is not recoverable)."""
    create_tables()
    with DBSession(engine()) as db:
        if db.scalar(select(User.id).where(User.name == name)) is not None:
            return None
    return create_user(name)


@router.get("/ui", response_class=HTMLResponse)
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


@router.post("/ui/api/key", dependencies=[Depends(local_only)])
async def new_key(req: KeyRequest) -> dict:
    name = req.name.strip()
    if not NAME.match(name):
        raise HTTPException(422, {"type": "bad_name", "message": "名字只能用字母、数字、中文、下划线、点和短横线"})

    key = await run_in_threadpool(create_account, name)
    if key is None:
        raise HTTPException(409, {"type": "name_taken", "message": "这个名字已经有账号了，请粘贴它的 API key"})
    return {"name": name, "key": key}
