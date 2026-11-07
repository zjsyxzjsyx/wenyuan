"""Libraries known to the code, with the name shown in context headers and citations.

v0.7 registered 18 public documentation sites here (crawled, or downloaded from DevDocs), with the
selectors used to cut their pages; v2 dropped them with their crawler and build scripts (see
`git tag v0.7.0`). What is left is the course loaded from Markdown by app.ingest.cs50. Courses that
teachers open in the workbench are rows in the `courses` table, not entries here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    library: str                      # library id used in MySQL, Milvus and MCP calls
    name: str                         # shown in context headers and citations


SITES: dict[str, Site] = {
    "cs50": Site(library="cs50", name="CS50x 2024"),
}
