"""API keys: `Authorization: Bearer <key>` resolves to a user.

`python -m app.api.auth create alice` prints a new key once. Keys are random
(32 bytes), so a plain SHA-256 is enough to store them: unlike passwords they
cannot be guessed from a dictionary, and a fast hash keeps the per-request lookup cheap.
"""
from __future__ import annotations

import argparse
import hashlib
import secrets
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import User, create_tables, engine

PREFIX = "wy_"
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Caller:
    id: int
    name: str


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_user(name: str) -> str:
    key = PREFIX + secrets.token_urlsafe(32)
    with DBSession(engine()) as db:
        db.add(User(name=name, key_hash=hash_key(key)))
        db.commit()
    return key


def lookup(key: str) -> Caller | None:
    with DBSession(engine()) as db:
        user = db.scalar(select(User).where(User.key_hash == hash_key(key)))
    return Caller(user.id, user.name) if user else None


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Caller:
    caller = lookup(credentials.credentials) if credentials else None
    if caller is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key",
                            headers={"WWW-Authenticate": "Bearer"})
    return caller


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create").add_argument("name")
    args = parser.parse_args()
    create_tables()
    print(f"API key for {args.name} (shown once): {create_user(args.name)}")


if __name__ == "__main__":
    main()
