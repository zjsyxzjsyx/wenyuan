"""Markdown -> section tree, split at ATX headings (ignoring `#` lines inside code fences)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Section:
    index: int                   # order within the page
    level: int                   # heading level; 0 = text before the first heading
    heading: str
    path: list[str]              # headings from the page title down to this section
    text: str                    # body without the heading line
    parent_index: int | None = None
    children: list[int] = field(default_factory=list)


def split_sections(markdown: str, title: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[Section] = []    # open ancestors, shallowest first
    body: list[str] = []
    current = Section(index=0, level=0, heading=title, path=[title], text="")
    in_fence = False

    def close() -> None:
        current.text = "\n".join(body).strip()
        sections.append(current)

    for line in markdown.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if not match:
            body.append(line)
            continue
        close()
        body = []
        level, heading = len(match.group(1)), match.group(2).strip()
        stack = [s for s in stack + [current] if 0 < s.level < level]
        parent = stack[-1] if stack else None
        path = (parent.path if parent else [title]) + ([] if level == 1 and heading == title else [heading])
        current = Section(index=len(sections), level=level, heading=heading, path=path, text="",
                          parent_index=parent.index if parent else None)
        if parent:
            parent.children.append(current.index)
    close()
    # Empty sections (e.g. a bare H1 repeating the title) stay so indices remain valid; chunking skips them.
    return sections
