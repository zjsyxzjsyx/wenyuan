import pytest

from app.ingest.cs50 import load

NOTES = """---
link: https://cs50.harvard.edu/x/2024/notes/3/#lecture-3
related_files:
  - week-3-transcription.md
title: Lecture 3
type: notes
week/lecture: "3"
---

# [Lecture 3](https://cs50.harvard.edu/x/2024/notes/3/#lecture-3)

- [Welcome!](#welcome)
- [Binary Search](#binary-search)

## [Binary Search](https://cs50.harvard.edu/x/2024/notes/3/#binary-search)

- Halve the problem each step.
"""

TRANSCRIPT = """---
link: https://youtu.be/jZzyERW7h1A
title: "Lecture 3: Week 3 - Algorithms"
type: transcription
week/lecture: "3"
---

# [Lecture 3: Week 3 - Algorithms](https://youtu.be/jZzyERW7h1A)

TABLE OF CONTENTS:

0:00 - Introduction
1:01 - Overview

## [Overview](https://youtu.be/jZzyERW7h1A?t=61)

1:01
DAVID MALAN: All right.
"""


def test_notes_keep_heading_text_and_lose_the_table_of_contents():
    page, kind = load(NOTES)
    assert (page.url, page.library, page.title, kind) == (
        "https://cs50.harvard.edu/x/2024/notes/3/#lecture-3", "cs50", "Week 3 · Lecture 3", "notes")
    assert page.markdown == "# Lecture 3\n\n## Binary Search\n\n- Halve the problem each step."


def test_transcripts_lose_timestamp_lines_and_the_timestamp_index():
    page, kind = load(TRANSCRIPT.replace("\n", "\r\n"))          # Windows line endings too
    assert kind == "transcript" and page.url == "https://youtu.be/jZzyERW7h1A"
    assert page.markdown == "# Lecture 3: Week 3 - Algorithms\n\n## Overview\n\nDAVID MALAN: All right."


def test_a_file_without_front_matter_is_rejected():
    with pytest.raises(ValueError):
        load("# no front matter\n")
