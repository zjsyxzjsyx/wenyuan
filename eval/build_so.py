"""Build the retrieval eval set from real Stack Overflow questions.

A question qualifies when its *accepted* answer links to a page we crawled: the
asker is a real learner, and the answerer's link is an independent judgement of
where the answer lives in the official docs. Those linked pages (and their
#anchors, when given) are the gold labels. The questions never enter the index,
so there is no leakage from the eval set into retrieval.

`python -m eval.build_so --tags redis fastapi --pages 10`
Writes eval/datasets/so_v1.jsonl. Content is CC BY-SA; every row keeps its link.
Stack Exchange allows ~300 requests/day per IP without a key; this uses ~40.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import urldefrag, urlparse

import httpx
from sqlalchemy import select

from app.db.models import Document, Section, engine
from app.ingest import devdocs
from app.ingest.crawl import cached_pages
from app.ingest.sites import SITES

API = "https://api.stackexchange.com/2.3"
OUT = Path("eval/datasets/so_v1.jsonl")
# library -> (official docs hosts, regex for the version/language prefix to strip from the link path).
# The tag of a library's Stack Overflow questions is the library id.
DOC_HOSTS: dict[str, tuple[tuple[str, ...], str]] = {
    "redis": (("redis.io",), ""),
    "fastapi": (("fastapi.tiangolo.com",), ""),
    "python": (("docs.python.org",), r"^/(\d(\.\d+)?/)?"),
    "django": (("docs.djangoproject.com",), r"^/[a-z]{2}/[^/]+/"),
    "postgresql": (("postgresql.org",), r"^/docs/[^/]+/(static/)?"),
    "flask": (("flask.palletsprojects.com", "flask.pocoo.org"), r"^/(en/[^/]+/|docs/[^/]+/)?"),
    "numpy": (("numpy.org", "docs.scipy.org"), r"^/doc/(numpy(-[\d.]+)?/|stable/|[\d.]+/)"),
    "pandas": (("pandas.pydata.org",), r"^/(pandas-docs/(stable|version/[^/]+)/|docs/)"),
    "pytorch": (("pytorch.org",), r"^/docs/(stable|master|main|[\d.]+)/"),
    "git": (("git-scm.com",), r"^/docs/"),
    "nginx": (("nginx.org",), r"^/en/docs/"),
    "sqlite": (("sqlite.org",), r"^/"),
    "celery": (("docs.celeryq.dev", "docs.celeryproject.org", "celery.readthedocs.io"), r"^/(en/[^/]+/)?"),
    "http": (("developer.mozilla.org",), r"^/[a-z]{2}(-[A-Z]{2})?/docs/Web/HTTP/"),
    "bash": (("gnu.org",), r"^/software/bash/manual/(html_node/)?"),
    "rabbitmq": (("rabbitmq.com",), r"^/(docs/)?"),
    "docker": (("docs.docker.com",), r"^/"),
}


def se_get(client: httpx.Client, path: str, **params) -> dict:
    response = client.get(f"{API}{path}", params={"site": "stackoverflow", **params})
    response.raise_for_status()
    data = response.json()
    if data.get("backoff"):
        time.sleep(data["backoff"])
    return data


def html_to_text(body: str, limit: int = 2000) -> str:
    body = re.sub(r"<pre[^>]*>(.*?)</pre>", lambda m: "\n```\n" + m.group(1) + "\n```\n", body, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", "", body))
    return re.sub(r"\n{3,}", "\n\n", text).strip()[:limit]


def canonical(url: str) -> str:
    parts = urlparse(url)
    path = parts.path if parts.path.endswith("/") or "." in parts.path.rsplit("/", 1)[-1] else parts.path + "/"
    return f"https://{parts.netloc.lower()}{path}"


class LinkResolver:
    """Map a (possibly old) docs link to a crawled page URL by following redirects."""

    def __init__(self, crawled: set[str]) -> None:
        self.crawled = {canonical(u): u for u in crawled}
        self.cache: dict[str, str | None] = {}
        self.http = httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": "wenyuan-eval (+https://github.com/zjsyxzjsyx/wenyuan)"})

    def resolve(self, link: str) -> str | None:
        key = canonical(urldefrag(link)[0])
        if key in self.crawled:
            return self.crawled[key]
        if key not in self.cache:
            try:
                final = str(self.http.head(key).url)
                self.cache[key] = self.crawled.get(canonical(urldefrag(final)[0]))
            except httpx.HTTPError:
                self.cache[key] = None
            time.sleep(0.5)
        return self.cache[key]


class DevDocsResolver:
    """Map an official-docs link to the DevDocs page holding the same content, without network calls.

    DevDocs keeps each project's own page paths, lower-cased and sometimes with an
    `/index` suffix, so after stripping host, version prefix and `.html` we try:
    exact path, path + /index, then a unique match on the last two path segments
    and finally the last one (this catches reorganized sites such as MDN's HTTP docs).
    """

    def __init__(self, library: str, indexable: set[str]) -> None:
        self.library = library
        self.prefix = DOC_HOSTS[library][1]
        self.keys = {path.lower(): path for path in devdocs._db(library)
                     if devdocs.page_url(library, path) in indexable}

    def _tail_match(self, parts: list[str], n: int) -> str | None:
        tail = "/".join(parts[-n:])
        found = [k for k in self.keys if k == tail or k.endswith("/" + tail) or k.endswith("/" + tail + "/index")]
        return self.keys[found[0]] if len(found) == 1 else None

    def resolve(self, link: str) -> str | None:
        path = re.sub(self.prefix, "/", urlparse(urldefrag(link)[0]).path, count=1)
        path = re.sub(r"(\.html?|/index\.html?)$", "", path).strip("/").lower()
        if not path:
            return None
        for candidate in (path, path + "/index"):
            if candidate in self.keys:
                return devdocs.page_url(self.library, self.keys[candidate])
        parts = path.split("/")
        for n in (2, 1):
            if len(parts) >= n and (key := self._tail_match(parts, n)):
                return devdocs.page_url(self.library, key)
        return None


def indexable_urls() -> set[str]:
    """Pages with at least one non-empty section. A JS-rendered page (e.g. the Redis
    command list) is crawled but empty, so it can never be retrieved and must not be gold."""
    with engine().connect() as conn:
        return set(conn.execute(select(Document.url).join(Section, Section.doc_id == Document.id)
                                .where(Section.text != "").distinct()).scalars())


def build(tags: list[str], pages: int) -> list[dict]:
    rows: list[dict] = []
    indexable = indexable_urls()
    with httpx.Client(timeout=30) as client:
        for tag in tags:
            hosts = DOC_HOSTS[tag][0]
            if SITES[tag].source == "devdocs":
                resolver = DevDocsResolver(tag, indexable)
            else:
                resolver = LinkResolver(set(cached_pages(SITES[tag].library)) & indexable)
            questions = []
            for page in range(1, pages + 1):
                data = se_get(client, "/questions", tagged=tag, sort="votes", order="desc",
                              pagesize=100, page=page, filter="withbody")
                questions += [q for q in data["items"] if q.get("accepted_answer_id")]
                if not data.get("has_more"):
                    break
            answers = {}
            ids = [q["accepted_answer_id"] for q in questions]
            for start in range(0, len(ids), 100):
                data = se_get(client, "/answers/" + ";".join(map(str, ids[start:start + 100])),
                              pagesize=100, filter="withbody")
                answers.update({a["answer_id"]: a for a in data["items"]})
            for q in questions:
                answer = answers.get(q["accepted_answer_id"])
                if not answer:
                    continue
                links = re.findall(r'href="([^"]+)"', answer["body"])
                links = [html.unescape(l) for l in links if urlparse(l).netloc.lower().endswith(hosts)]
                gold, anchors = [], []
                for link in links:
                    page_url = resolver.resolve(link)
                    if page_url and page_url not in gold:
                        gold.append(page_url)
                        anchors.append(urldefrag(link)[1] or None)
                if gold:
                    rows.append({
                        "id": f"so-{q['question_id']}", "library": SITES[tag].library, "type": "real",
                        "question": html.unescape(q["title"]), "body": html_to_text(q["body"]),
                        "answer_text": html_to_text(answer["body"], limit=1500),
                        "gold_urls": gold, "gold_anchors": anchors,
                        "source": q["link"], "answer": f"https://stackoverflow.com/a/{answer['answer_id']}",
                        "score": q["score"],
                    })
            print(f"{tag}: {len(questions)} questions with accepted answers -> "
                  f"{sum(r['library'] == SITES[tag].library for r in rows)} linked to indexed docs "
                  f"(API quota left: {data.get('quota_remaining')})", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=["redis", "fastapi"], choices=sorted(DOC_HOSTS))
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--merge", action="store_true",
                        help="keep existing rows of other libraries instead of rewriting the whole file")
    args = parser.parse_args()
    rows = build(args.tags, args.pages)
    if args.merge and OUT.exists():
        kept = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines()]
        rows = [r for r in kept if r["library"] not in args.tags] + rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
