"""A page ready to be split into sections and chunks.

v0.7 made these from crawled HTML (app.ingest.extract, removed with v0.7's public libraries);
v2's course loader, teachers' uploads and students' notes build them from Markdown directly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Page:
    url: str
    library: str
    title: str
    markdown: str
