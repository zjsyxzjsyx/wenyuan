import pytest

from app.ingest import devdocs
from eval.build_so import DevDocsResolver


@pytest.fixture
def fake_db(monkeypatch):
    db = {
        "library/functions": "", "library/stdtypes": "", "tutorial/classes": "",
        "patterns/fileuploads/index": "", "reference/headers/cache-control": "", "sql-select": "",
    }
    monkeypatch.setitem(devdocs.DEVDOCS, "python", "python~3.14")
    monkeypatch.setattr(devdocs, "_db", lambda library: db)
    return {devdocs.page_url("python", p) for p in db}


@pytest.mark.parametrize("link,expected", [
    ("https://docs.python.org/3/library/functions.html#len", "library/functions"),
    ("https://docs.python.org/3.8/library/stdtypes.html", "library/stdtypes"),
    ("https://docs.python.org/2/tutorial/classes.html", "tutorial/classes"),
    ("https://docs.python.org/3/library/missing.html", None),
    ("https://docs.python.org/3/", None),
])
def test_python_links(fake_db, link, expected):
    resolver = DevDocsResolver("python", fake_db)
    want = devdocs.page_url("python", expected) if expected else None
    assert resolver.resolve(link) == want


def test_index_suffix_and_tail_match(fake_db, monkeypatch):
    resolver = DevDocsResolver("python", fake_db)
    resolver.prefix = r"^/(en/[^/]+/)?"
    assert resolver.resolve("https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/") \
        == devdocs.page_url("python", "patterns/fileuploads/index")
    resolver.prefix = r"^/[a-z]{2}(-[A-Z]{2})?/docs/Web/HTTP/"
    assert resolver.resolve("https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control") \
        == devdocs.page_url("python", "reference/headers/cache-control")


def test_unindexed_pages_are_not_gold(fake_db):
    resolver = DevDocsResolver("python", fake_db - {devdocs.page_url("python", "sql-select")})
    resolver.prefix = r"^/docs/[^/]+/"
    assert resolver.resolve("https://www.postgresql.org/docs/current/sql-select.html") is None
