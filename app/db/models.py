"""MySQL tables for documents and their section trees (what `outline` and small-to-big read)."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, func
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings

LongText = Text().with_variant(MEDIUMTEXT(), "mysql")


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)          # sha1(url)[:16]
    library: Mapped[str] = mapped_column(String(64), index=True)
    url: Mapped[str] = mapped_column(String(768), unique=True)
    title: Mapped[str] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(64))
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
