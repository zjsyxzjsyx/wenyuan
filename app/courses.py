"""Courses: who may read a course's shared materials, and who manages them (design doc section 0.3).

A course's id is the library id of its materials, so "which courses is this caller in" is exactly
the list of libraries their searches are limited to. A teacher uploads materials and makes invite
codes; a student joins with a code. Invite codes are stored as hashes and shown once, like API keys.

`python -m app.courses create <id> <name> --teacher <user>` | `enroll <id> <user> [--role]` |
`invite <id> --by <teacher>` (prints a code) | `list <user>`
"""
from __future__ import annotations

import argparse
import hashlib
import re
import secrets
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Course, Enrollment, Invite, User, create_tables, engine
from app.ingest.sites import SITES
from app.retrieval.milvus_store import PRIVATE

ROLES = ("teacher", "student")
INVITE_USES = 200
INVITE_DAYS = 14
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")
# v0.7 public-doc libraries still have documents under these ids; a course reusing one would mix them in.
RESERVED = {PRIVATE} | {lib for lib, site in SITES.items() if site.source != "markdown"}


class CourseError(ValueError):
    pass


def _hash(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()


def create_course(db: DBSession, course_id: str, name: str, teacher_id: int) -> dict:
    if not _ID.match(course_id) or course_id in RESERVED:
        raise CourseError("course id: 2-41 lowercase letters, digits, - or _, and not a reserved library name")
    if db.get(Course, course_id) is not None:
        raise CourseError(f"course {course_id!r} already exists")
    db.add(Course(id=course_id, name=name.strip()[:255] or course_id))
    db.flush()
    db.add(Enrollment(user_id=teacher_id, course_id=course_id, role="teacher"))
    db.commit()
    return {"id": course_id, "name": name, "role": "teacher"}


def enroll(db: DBSession, user_id: int, course_id: str, role: str = "student") -> None:
    if role not in ROLES:
        raise CourseError(f"role must be one of {ROLES}")
    if db.get(Course, course_id) is None:
        raise CourseError(f"no course {course_id!r}")
    found = db.get(Enrollment, (user_id, course_id))
    if found is None:
        db.add(Enrollment(user_id=user_id, course_id=course_id, role=role))
    elif role == "teacher":
        found.role = role          # joining with a student code never demotes a teacher
    db.commit()


def courses_of(db: DBSession, user_id: int) -> list[dict]:
    rows = db.execute(select(Course.id, Course.name, Enrollment.role).join(Enrollment, Enrollment.course_id == Course.id)
                      .where(Enrollment.user_id == user_id).order_by(Course.id))
    return [{"id": cid, "name": name, "role": role} for cid, name, role in rows]


def role_in(db: DBSession, user_id: int, course_id: str) -> str | None:
    found = db.get(Enrollment, (user_id, course_id))
    return found.role if found else None


def create_invite(db: DBSession, course_id: str, created_by: int, max_uses: int = INVITE_USES,
                  days: int = INVITE_DAYS) -> dict:
    if role_in(db, created_by, course_id) != "teacher":
        raise CourseError("only a teacher of the course can invite")
    code = f"join-{secrets.token_urlsafe(9)}"
    expires = datetime.now() + timedelta(days=days)
    db.add(Invite(code_hash=_hash(code), course_id=course_id, created_by=created_by, max_uses=max_uses, uses=0,
                  expires_at=expires))
    db.commit()
    return {"code": code, "course_id": course_id, "max_uses": max_uses, "expires_at": expires.isoformat(timespec="minutes")}


def join(db: DBSession, user_id: int, code: str) -> dict:
    """Enroll the caller as a student. The invite row is locked, so two people redeeming the last use
    at once cannot both get in."""
    invite = db.scalar(select(Invite).where(Invite.code_hash == _hash(code)).with_for_update())
    already = db.get(Enrollment, (user_id, invite.course_id)) if invite else None
    if already is None:
        if invite is None or invite.expires_at < datetime.now() or invite.uses >= invite.max_uses:
            db.rollback()
            raise CourseError("this invite code is not valid (unknown, expired or used up)")
        invite.uses += 1
        db.add(Enrollment(user_id=user_id, course_id=invite.course_id, role="student"))
    db.commit()
    course = db.get(Course, invite.course_id)
    return {"id": course.id, "name": course.name, "role": already.role if already else "student"}


def _user_id(db: DBSession, name: str) -> int:
    found = db.scalar(select(User.id).where(User.name == name))
    if found is None:
        raise SystemExit(f"no account named {name!r}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.courses")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("id")
    c.add_argument("name")
    c.add_argument("--teacher", required=True)
    e = sub.add_parser("enroll")
    e.add_argument("id")
    e.add_argument("user")
    e.add_argument("--role", default="student", choices=ROLES)
    i = sub.add_parser("invite")
    i.add_argument("id")
    i.add_argument("--by", required=True, help="a teacher of the course")
    i.add_argument("--uses", type=int, default=INVITE_USES)
    i.add_argument("--days", type=int, default=INVITE_DAYS)
    ls = sub.add_parser("list")
    ls.add_argument("user")
    args = parser.parse_args()
    create_tables()
    with DBSession(engine()) as db:
        try:
            if args.cmd == "create":
                print(create_course(db, args.id, args.name, _user_id(db, args.teacher)))
            elif args.cmd == "enroll":
                enroll(db, _user_id(db, args.user), args.id, args.role)
                print(f"{args.user} -> {args.id} ({args.role})")
            elif args.cmd == "invite":
                invite = create_invite(db, args.id, _user_id(db, args.by), args.uses, args.days)
                print(f"invite code for {args.id} (shown once, {invite['max_uses']} uses, until "
                      f"{invite['expires_at']}): {invite['code']}")
            else:
                print(courses_of(db, _user_id(db, args.user)))
        except CourseError as exc:
            raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
