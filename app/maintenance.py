"""Clear out what v0.7 and testing left behind (design doc section 0).

`python -m app.maintenance cleanup`        lists what would be removed, and removes nothing
`python -m app.maintenance cleanup --yes`  removes it

- Milvus collections other than the live one (`chunks_v2`): v0.7's 18-library corpus, the
  Qwen3-embedding comparison, section summaries.
- Shared documents (and their section trees) that belong to no course: v0.7's public libraries.
  Notes (documents with an owner) and course materials are never touched.
- Test courses and test accounts made while building the workbench and load test, with their
  enrollments (cascade), their materials, and their notes if any.
Deleting cannot be undone, so the default only prints the plan.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session as DBSession

from app.db.models import Course, Document, User, engine
from app.query.ask import COLLECTION
from app.retrieval import milvus_store
from app.retrieval.milvus_store import PRIVATE

TEST_USERS = ("ui-test", "ui-test2", "ui-teacher-demo", "ui-student-demo", "ui-verify")
TEST_USER_PREFIX = "loadtest-"
TEST_COURSES = ("ui-demo",)


@dataclass
class Plan:
    collections: list[str] = field(default_factory=list)
    libraries: dict[str, int] = field(default_factory=dict)      # orphan shared library -> documents
    courses: list[str] = field(default_factory=list)
    users: list[str] = field(default_factory=list)

    def empty(self) -> bool:
        return not (self.collections or self.libraries or self.courses or self.users)


def make_plan(collections: list[str], docs_per_library: dict[str, int], course_ids: list[str],
              user_names: list[str], keep: str = COLLECTION) -> Plan:
    """What to remove, from listings alone (no service needed, so it can be tested)."""
    test_courses = [c for c in course_ids if c in TEST_COURSES]
    live = set(course_ids) - set(test_courses)
    return Plan(collections=sorted(c for c in collections if c != keep),
                libraries={lib: n for lib, n in sorted(docs_per_library.items()) if lib != PRIVATE and lib not in live},
                courses=test_courses,
                users=sorted(u for u in user_names if u in TEST_USERS or u.startswith(TEST_USER_PREFIX)))


def current_plan(db: DBSession, mc) -> Plan:
    docs = dict(db.execute(select(Document.library, func.count()).where(Document.owner_id.is_(None))
                           .group_by(Document.library)).all())
    return make_plan(mc.list_collections(), docs, list(db.scalars(select(Course.id))), list(db.scalars(select(User.name))))


def apply(db: DBSession, mc, plan: Plan) -> None:
    for name in plan.collections:
        mc.drop_collection(name)
    for lib in list(plan.libraries) + plan.courses:          # sections go with their documents (cascade)
        db.execute(delete(Document).where(Document.library == lib, Document.owner_id.is_(None)))
        mc.delete(COLLECTION, filter=f'library == "{lib}"')
    for course_id in plan.courses:
        db.delete(db.get(Course, course_id))                  # enrollments and invites cascade
    for user in db.scalars(select(User).where(User.name.in_(plan.users))):
        mc.delete(COLLECTION, filter=f'library == "{PRIVATE}" and owner_id == {user.id}')
        db.execute(delete(Document).where(Document.owner_id == user.id))
        db.delete(user)                                       # enrollments cascade
    db.commit()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.maintenance")
    parser.add_argument("command", choices=["cleanup"])
    parser.add_argument("--yes", action="store_true", help="really delete (without it, only list)")
    args = parser.parse_args()
    mc = milvus_store.client()
    with DBSession(engine()) as db:
        plan = current_plan(db, mc)
        print(f"Milvus collections to drop ({len(plan.collections)}): {plan.collections}")
        print(f"shared libraries outside any course: {plan.libraries} (documents)")
        print(f"test courses: {plan.courses}")
        print(f"test accounts ({len(plan.users)}): {plan.users}")
        if plan.empty():
            print("nothing to remove")
        elif not args.yes:
            print("nothing removed; run again with --yes to delete the above (cannot be undone)")
        else:
            apply(db, mc, plan)
            print("removed")


if __name__ == "__main__":
    main()
