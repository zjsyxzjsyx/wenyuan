"""Preset documentation sites: where to find pages and how to cut the noise out of them."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Site:
    library: str                      # library id used in MySQL, Milvus and MCP calls
    name: str                         # shown in context headers and citations
    sitemap: str
    content_selector: str             # main article element
    include: str = r".*"              # URL regex a page must match
    exclude: str = r"^$"              # URL regex that drops a page
    drop_selectors: tuple[str, ...] = ()
    keep_code_langs: frozenset[str] = field(default_factory=frozenset)  # empty = keep all
    delay_s: float = 1.0              # politeness delay between requests to this host
    source: str = "crawl"             # "crawl" (sitemap + crawler) or "devdocs" (downloaded docset)

    def wants(self, url: str) -> bool:
        return bool(re.search(self.include, url)) and not re.search(self.exclude, url)


COMMON_DROP = ("script", "style", "svg", "button", "nav", "img", "noscript")

SITES: dict[str, Site] = {
    "fastapi": Site(
        library="fastapi",
        name="FastAPI",
        sitemap="https://fastapi.tiangolo.com/sitemap.xml",
        content_selector="article.md-content__inner",
        # English pages only: translations live under a two-letter (or zh-hant) prefix.
        exclude=r"fastapi\.tiangolo\.com/(?:[a-z]{2}|zh-hant)/",
        drop_selectors=(".headerlink", "a.md-content__button"),
    ),
    "redis": Site(
        library="redis",
        name="Redis",
        sitemap="https://redis.io/docs/latest/sitemap.xml",
        content_selector="section.prose",
        # Open-source Redis only: commands, developer guides, OSS operations.
        include=r"/docs/latest/(commands|develop/(data-types|use-cases|get-started|reference|using-commands"
                r"|clients|tools|whats-new|programmability|interact)|operate/oss_and_stack)/",
        # Noise found in the E0/E1 bad cases: enterprise modules, release notes, per-version command
        # lists, archived installs, and the same use case or client guide repeated per language
        # (kept: redis-py, matching keep_code_langs).
        exclude=r"/docs/latest/(operate/oss_and_stack/(stack-with-enterprise|install/archive)/"
                r"|commands/redis-\d+-\d+-commands/|develop/whats-new/|develop/tools/insight/release-notes/"
                r"|develop/use-cases/[^/]+/(java-jedis|java-lettuce|php|rust|dotnet|go|nodejs|ruby)/"
                r"|develop/clients/(dotnet|go|hiredis|ioredis|jedis|lettuce|nodejs|php|ruby|rust|om-clients)/)",
        drop_selectors=(".command-group-foldout", ".codetabs-header", ".tab-nav", ".api-methods-tab-content"),
        # Each example ships in ~15 client languages; redis-cli plus Python is enough to learn from.
        keep_code_langs=frozenset({"redis-cli", "python"}),
    ),
}

# DevDocs pages are article fragments; the only chrome left is PostgreSQL's per-page table of
# contents and PyTorch's hidden schema.org breadcrumb.
DEVDOCS_NAMES = {
    "python": "Python 3.14", "postgresql": "PostgreSQL 18", "sqlite": "SQLite", "django": "Django 5.2",
    "flask": "Flask", "celery": "Celery", "rabbitmq": "RabbitMQ", "docker": "Docker", "kubernetes": "Kubernetes",
    "git": "Git", "nginx": "nginx", "http": "HTTP (MDN)", "bash": "Bash", "numpy": "NumPy", "pandas": "pandas",
    "pytorch": "PyTorch 2.9",
}
SITES.update({
    library: Site(library=library, name=name, sitemap="", content_selector="body", source="devdocs",
                  drop_selectors=(".toc", "div[style*='display:none']"))
    for library, name in DEVDOCS_NAMES.items()
})
CRAWLED = sorted(lib for lib, site in SITES.items() if site.source == "crawl")
