"""What a class keeps asking: the teacher's summary of students' questions (design doc section 0.3).

Every question a student asks in a course is filed under it (app.api.main, POST /v1/ask). The summary
embeds the questions of the last days with bge-m3 and groups those whose cosine similarity to a
group's centroid reaches `threshold`, greedily in order. A group is shown with the question closest to
its centroid, how often it was asked and by how many students, plus a few other wordings.

Privacy: the teacher sees what the class asks, never who asked. A group appears only when at least
`min_students` different students asked it, so a question only one student had (often about their own
notes) never reaches the teacher; teachers' own questions are not filed at all. The 0.80 threshold is
a starting point, not yet measured against labelled data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Question
from app.models.local import ModelServer

THRESHOLD = 0.80
MIN_STUDENTS = 2


@dataclass
class Group:
    members: list[tuple[int, str]] = field(default_factory=list)    # (student id, question)
    vectors: list[list[float]] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def group_questions(questions: list[tuple[int, str]], vectors: list[list[float]], threshold: float = THRESHOLD,
                    min_students: int = MIN_STUDENTS, limit: int = 20) -> list[dict]:
    """Groups of similar questions, most widely asked first; only groups at least `min_students` asked."""
    groups: list[Group] = []
    for member, vector in zip(questions, map(_unit, vectors)):
        best = max(groups, key=lambda g: _dot(g.centroid, vector), default=None)
        if best is None or _dot(best.centroid, vector) < threshold:
            best = Group()
            groups.append(best)
        best.members.append(member)
        best.vectors.append(vector)
        best.centroid = _unit([sum(col) for col in zip(*best.vectors)])
    out = []
    for g in groups:
        students = {user for user, _ in g.members}
        if len(students) < min_students:
            continue
        ranked = sorted(zip(g.members, g.vectors), key=lambda mv: -_dot(g.centroid, mv[1]))
        texts = list(dict.fromkeys(text for (_, text), _ in ranked))
        out.append({"question": texts[0], "asked": len(g.members), "students": len(students), "also": texts[1:4]})
    return sorted(out, key=lambda g: (-g["students"], -g["asked"]))[:limit]


def file_question(db: DBSession, user_id: int, courses: list[dict], text: str) -> None:
    """File a question under each course it was asked in where the asker is a student."""
    db.add_all(Question(course_id=c["id"], user_id=user_id, text=text[:2000]) for c in courses if c["role"] == "student")
    db.commit()


def frequent_questions(db: DBSession, models: ModelServer, course_id: str, days: int = 14) -> list[dict]:
    since = datetime.now() - timedelta(days=days)
    rows = db.execute(select(Question.user_id, Question.text).where(Question.course_id == course_id,
                                                                    Question.asked_at >= since)
                      .order_by(Question.asked_at)).all()
    if not rows:
        return []
    questions = [(user, text) for user, text in rows]
    return group_questions(questions, models.embed([text for _, text in questions]))
