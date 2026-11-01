"""Polite sitemap crawler with an on-disk HTML cache.

`python -m app.ingest.crawl redis [--limit N]` downloads every wanted page of a
preset site into data/raw/<library>/, skipping pages already cached, obeying
robots.txt and waiting `Site.delay_s` between requests. W2 moves this into a
Celery task; the cache layout stays the same.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from app.ingest.sites import CRAWLED, SITES, Site

RAW_DIR = Path("data/raw")
USER_AGENT = "wenyuan/0.1 (+https://github.com/zjsyxzjsyx/wenyuan; polite crawler)"


def page_path(library: str, url: str) -> Path:
    return RAW_DIR / library / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.html"


def sitemap_urls(client: httpx.Client, url: str) -> list[str]:
    """All page URLs in a sitemap, following one level of sitemap index."""
    xml = client.get(url).text
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)
    if "<sitemapindex" in xml:
        return [page for child in locs for page in sitemap_urls(client, child)]
    return locs


def robots_for(client: httpx.Client, url: str) -> RobotFileParser:
    parts = urlparse(url)
    parser = RobotFileParser()
    response = client.get(f"{parts.scheme}://{parts.netloc}/robots.txt")
    parser.parse(response.text.splitlines() if response.status_code == 200 else [])
    return parser


def crawl(site: Site, limit: int | None = None) -> dict[str, int]:
    stats = {"wanted": 0, "cached": 0, "fetched": 0, "blocked": 0, "failed": 0}
    out_dir = RAW_DIR / site.library
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
        robots = robots_for(client, site.sitemap)
        urls = sorted({u for u in sitemap_urls(client, site.sitemap) if site.wants(u)})[:limit]
        stats["wanted"] = len(urls)
        indexed = set(cached_pages(site.library))
        with index_path.open("a", encoding="utf-8") as index:
            for url in urls:
                path = page_path(site.library, url)
                if path.exists():
                    stats["cached"] += 1
                    if url not in indexed:  # repair: page saved but index line lost
                        index.write(json.dumps({"url": url, "file": path.name, "fetched_at": path.stat().st_mtime}) + "\n")
                    continue
                if not robots.can_fetch(USER_AGENT, url):
                    stats["blocked"] += 1
                    continue
                try:
                    response = client.get(url)
                    response.raise_for_status()
                except httpx.HTTPError:
                    stats["failed"] += 1
                else:
                    path.write_text(response.text, encoding="utf-8")
                    index.write(json.dumps({"url": url, "file": path.name, "fetched_at": time.time()}) + "\n")
                    index.flush()  # a crash must not leave cached pages missing from the index
                    stats["fetched"] += 1
                time.sleep(site.delay_s)
    return stats


def cached_pages(library: str) -> dict[str, Path]:
    """url -> cached HTML file, for every page this library has on disk."""
    index_path = RAW_DIR / library / "index.jsonl"
    if not index_path.exists():
        return {}
    pages = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        path = RAW_DIR / library / row["file"]
        if path.exists():
            pages[row["url"]] = path
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", choices=CRAWLED)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(crawl(SITES[args.library], args.limit))


if __name__ == "__main__":
    main()
