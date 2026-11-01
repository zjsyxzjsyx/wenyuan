"""HTML page -> clean Markdown: keep the article, drop site chrome and extra code languages."""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from markdownify import markdownify

from app.ingest.sites import COMMON_DROP, Site


@dataclass
class Page:
    url: str
    library: str
    title: str
    markdown: str


def _code_language(pre) -> str:
    # markdownify passes the <pre>. DevDocs marks it with data-language; MkDocs and Hugo
    # sites put a language-* class on the inner <code>.
    if pre.get("data-language") not in (None, "", "none"):
        return pre["data-language"]
    for el in (pre, pre.find("code")):
        for cls in (el.get("class") or []) if el is not None else []:
            if cls.startswith("language-"):
                return cls.removeprefix("language-")
    return ""


def extract(site: Site, url: str, html: str) -> Page | None:
    soup = BeautifulSoup(html, "lxml")
    article = soup.select_one(site.content_selector)
    if article is None:
        return None
    for el in article.select(",".join(COMMON_DROP + site.drop_selectors)):
        el.decompose()
    if site.keep_code_langs:
        # Removing a tab panel also removes tabs nested in it, so skip ones already gone.
        for el in list(article.find_all(attrs={"data-lang": True})):
            if not el.decomposed and el.get("data-lang", "").lower() not in site.keep_code_langs:
                el.decompose()

    markdown = markdownify(str(article), heading_style="ATX", code_language_callback=_code_language)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    h1 = article.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else url)
    return Page(url=url, library=site.library, title=title, markdown=markdown)
