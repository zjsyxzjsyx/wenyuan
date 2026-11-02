"""MySQL tables: users, courses and who is in them, and documents with their section trees
(what `outline` and small-to-big read)."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from sqlalchemy import (JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, func,
                        inspect, text)
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings

LongText = Text().with_variant(MEDIUMTEXT(), "mysql")


class Base(DeclarativeBase):
    pass


class User(Base):
    """An API caller. Only a hash of the key is stored; the key itself is shown once at creation."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)   # sha256 hex of the API key
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Course(Base):
    """A class. Its id is also the library id of its materials in the vector store and documents."""
    __tablename__ = "courses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Enrollment(Base):
    """Who is in a course, and as what: a teacher manages its materials and invites, a student reads."""
    __tablename__ = "enrollments"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str] = mapped_column(String(16))                        # teacher | student
    joined_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Invite(Base):
    """A code that enrolls whoever redeems it as a student. Only its hash is stored, like an API key."""
    __tablename__ = "invites"

    code_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), index=True)
    created_by: Mapped[int] = mapped_column(Integer)
    max_uses: Mapped[int] = mapped_column(Integer)
    uses: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)          # sha1(url)[:16]
    library: Mapped[str] = mapped_column(String(64), index=True)
    url: Mapped[str] = mapped_column(String(768), unique=True)
    title: Mapped[str] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(64))
    owner_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)   # set for private notes
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (UniqueConstraint("doc_id", "idx"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)                    # order within the page
    parent_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    level: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str] = mapped_column(String(512))
    path: Mapped[list] = mapped_column(JSON)
    text: Mapped[str] = mapped_column(LongText)


@lru_cache
def engine() -> Engine:
    return create_engine(get_settings().mysql_url, pool_pre_ping=True)


def create_tables() -> None:
    Base.metadata.create_all(engine())
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Columns added after a table was first created (create_all never alters a table)."""
    columns = {c["name"] for c in inspect(engine()).get_columns("documents")}
    if "owner_id" not in columns:
        with engine().begin() as conn:
            conn.execute(text("ALTER TABLE documents ADD COLUMN owner_id INT NULL"))
            conn.execute(text("CREATE INDEX ix_documents_owner_id ON documents (owner_id)"))
