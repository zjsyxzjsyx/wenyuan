from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import courses as c
from app.db.models import Base, Invite, User


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([User(id=1, name="teacher", key_hash="a"), User(id=2, name="ann", key_hash="b"),
                         User(id=3, name="bob", key_hash="c")])
        session.commit()
        yield session


def test_the_creator_teaches_and_ids_are_checked(db):
    assert c.create_course(db, "cs50", "CS50x 2024", 1)["role"] == "teacher"
    assert c.courses_of(db, 1) == [{"id": "cs50", "name": "CS50x 2024", "role": "teacher"}]
    for bad in ("CS50", "a", "private", "redis", "has space"):
        with pytest.raises(c.CourseError):
            c.create_course(db, bad, "x", 1)
    with pytest.raises(c.CourseError):
        c.create_course(db, "cs50", "again", 1)


def test_students_join_with_a_code_and_only_teachers_make_codes(db):
    c.create_course(db, "cs50", "CS50x", 1)
    invite = c.create_invite(db, "cs50", 1, max_uses=1)
    assert db.get(Invite, c._hash(invite["code"])) is not None and "join-" in invite["code"]
    assert all(row.code_hash != invite["code"] for row in db.query(Invite))         # only the hash is stored
    assert c.join(db, 2, invite["code"]) == {"id": "cs50", "name": "CS50x", "role": "student"}
    assert c.join(db, 2, invite["code"])["role"] == "student"                         # joining twice costs nothing
    with pytest.raises(c.CourseError):
        c.join(db, 3, invite["code"])                                                   # the one use is spent
    with pytest.raises(c.CourseError):
        c.create_invite(db, "cs50", 2)                                                  # a student cannot invite
    assert c.role_in(db, 3, "cs50") is None and c.courses_of(db, 3) == []


def test_expired_or_unknown_codes_are_refused_and_teachers_are_never_demoted(db):
    c.create_course(db, "cs50", "CS50x", 1)
    invite = c.create_invite(db, "cs50", 1)
    db.get(Invite, c._hash(invite["code"])).expires_at = datetime.now() - timedelta(minutes=1)
    db.commit()
    for code in (invite["code"], "join-unknown"):
        with pytest.raises(c.CourseError):
            c.join(db, 2, code)
    fresh = c.create_invite(db, "cs50", 1)
    assert c.join(db, 1, fresh["code"])["role"] == "teacher"
    c.enroll(db, 2, "cs50", "teacher")
    c.enroll(db, 2, "cs50", "student")
    assert c.role_in(db, 2, "cs50") == "teacher"
