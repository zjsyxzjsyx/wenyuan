"""DevDocs (devdocs.io) as the main public library: clean, pre-split documentation.

Each docset is two files: index.json (entry names, paths, categories) and db.json
(page path -> article HTML, already stripped of site navigation). We download them
once into data/raw/devdocs/<slug>/ and read pages from there; nothing is crawled.

`python -m app.ingest.devdocs`  downloads every docset in DEVDOCS that is missing.
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import httpx

RAW_DIR = Path("data/raw/devdocs")
BASE = "https://documents.devdocs.io"

# library id -> DevDocs slug. FastAPI and Redis stay on the crawled official sites:
# their eval labels already point there and Redis on DevDocs has commands only.
DEVDOCS: dict[str, str] = {
    "python": "python~3.14",
    "postgresql": "postgresql~18",
    "sqlite": "sqlite",
    "django": "django~5.2",
    "flask": "flask",
    "celery": "celery",
    "rabbitmq": "rabbitmq",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "git": "git",
    "nginx": "nginx",
    "http": "http",
    "bash": "bash",
    "numpy": "numpy~2.5",
    "pandas": "pandas~3",
    "pytorch": "pytorch~2.9",
}

# Page paths that are noise for a learner, found by profiling each docset's path prefixes and
# biggest pages: generated indexes, changelogs/release notes, C APIs, raw RFC texts.
EXCLUDE: dict[str, str] = {
    "python": r"^(genindex|contents$|download$|whatsnew/|c-api/|py-modindex)",
    "sqlite": r"^(c3ref/|capi3ref$|session/|syntaxdiagrams$|releaselog|changes$)",
    "numpy": r"^(reference/c-api/|release)",
    "http": r"^rfc\d+",
    "flask": r"^changes/",
    "nginx": r"(^|/)changes$",
    "bash": r"(-index$|^indexes$)",
    "celery": r"^internals/",
    "postgresql": r"^release",
}


def doc_dir(library: str) -> Path:
    return RAW_DIR / DEVDOCS[library].replace("~", "_")


def download(library: str, client: httpx.Client) -> dict[str, int]:
    out = doc_dir(library)
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name in ("index.json", "db.json"):
        path = out / name
        if not path.exists():
            response = client.get(f"{BASE}/{DEVDOCS[library]}/{name}")
            response.raise_for_status()
            path.write_bytes(response.content)
        sizes[name] = path.stat().st_size
    return sizes


@lru_cache(maxsize=4)
def _db(library: str) -> dict[str, str]:
    return json.loads((doc_dir(library) / "db.json").read_text(encoding="utf-8"))


def page_url(library: str, path: str) -> str:
    return f"https://devdocs.io/{DEVDOCS[library]}/{path}"


def wanted(library: str, path: str) -> bool:
    # The docset's top-level `index` page is a bare link list in every docset.
    return path != "index" and not re.search(EXCLUDE.get(library, r"^$"), path)


def pages(library: str) -> dict[str, str]:
    """url -> article HTML for every wanted page."""
    return {page_url(library, path): html for path, html in _db(library).items() if wanted(library, path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("libraries", nargs="*", default=sorted(DEVDOCS))
    args = parser.parse_args()
    with httpx.Client(timeout=httpx.Timeout(30, read=600), follow_redirects=True, headers={"User-Agent": "wenyuan/0.1 (+https://github.com/zjsyxzjsyx/wenyuan)"}) as client:
        for library in args.libraries:
            sizes = download(library, client)
            print(f"{library:12} {DEVDOCS[library]:15} db={sizes['db.json'] / 1e6:5.1f}MB pages={len(_db(library)) - 1}")


if __name__ == "__main__":
    main()
