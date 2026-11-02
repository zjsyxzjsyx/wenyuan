import random
import uuid

import pytest

from app.ingest.private import NoteError, clean_filename, note_url, title_of
from app.retrieval import milvus_store
from app.retrieval.milvus_store import PRIVATE, scope_filter


def test_filenames_are_reduced_to_a_safe_basename():
    assert clean_filename("../../etc/redis笔记.md") == "redis笔记.md"
    assert clean_filename("C:\\notes\\week1.markdown") == "week1.markdown"
    for bad in ("", "a.exe", "notes", "../"):
        with pytest.raises(NoteError):
            clean_filename(bad)
    assert note_url(7, "a.md") == "private://7/a.md"


def test_title_is_first_h1_or_file_stem():
    assert title_of("intro\n# Redis 持久化\n## RDB", "x.md") == "Redis 持久化"
    assert title_of("no heading", "week-1.md") == "week-1"


def test_scope_filter_never_admits_other_users_notes():
    assert scope_filter(["redis"]) == "library in ['redis']"
    assert scope_filter(None) == 'library != "private"'
    # A client listing "private" as a library gets nothing extra without an owner.
    assert scope_filter(["private"]) == 'library != "private"'
    assert scope_filter(["redis", "private"], 7) == \
        '(library in [\'redis\']) or (library == "private" and owner_id == 7)'


@pytest.fixture
def milvus():
    try:
        mc = milvus_store.client()
        mc.list_collections()
    except Exception:
        pytest.skip("Milvus is not running")
    name = f"test_private_{uuid.uuid4().hex[:8]}"
    milvus_store.create_collection(mc, name, dim=8)
    yield mc, name
    mc.drop_collection(name)


def test_users_only_ever_retrieve_their_own_notes(milvus):
    mc, name = milvus
    rng = random.Random(0)
    vec = lambda: [rng.random() for _ in range(8)]

    def row(cid, library, owner, text):
        return {"chunk_id": cid, "library": library, "doc_id": cid, "doc_type": "guide", "url": f"u/{cid}",
                "section_index": 0, "parent_index": 0, "heading_path": "", "text": text, "dense": vec(),
                "owner_id": owner}
    milvus_store.insert_rows(mc, name, [
        row("pub", "redis", 0, "redis expire notes"),
        row("alice", PRIVATE, 1, "redis expire notes alice secret"),
        row("bob", PRIVATE, 2, "redis expire notes bob secret"),
    ])
    for owner, libs in [(1, ["redis"]), (1, None), (1, ["redis", PRIVATE]), (None, None), (None, [PRIVATE])]:
        for search in (
            lambda: milvus_store.dense_search(mc, name, vec(), top_k=10, libraries=libs, owner_id=owner),
            lambda: milvus_store.bm25_search(mc, name, "redis secret", top_k=10, libraries=libs, owner_id=owner),
            lambda: milvus_store.hybrid_search(mc, name, vec(), "redis secret", top_k=10, libraries=libs,
                                               owner_id=owner),
        ):
            seen = {h["chunk_id"] for h in search()}
            assert "bob" not in seen, (owner, libs)
            assert ("alice" in seen) == (owner == 1), (owner, libs, seen)
            assert "pub" in seen or libs == [PRIVATE]
