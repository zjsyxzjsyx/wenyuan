from app.ingest.chunking.structural import chunk_sections, top_level
from app.ingest.extract import extract
from app.ingest.sections import split_sections
from app.ingest.sites import Site
from test_fixed_token import CharTokenizer

MD = """# Sorted sets

Intro text.

## Adding members

Use ZADD.

```python
# not a heading
r.zadd("k", {"a": 1})
```

### Options

NX and XX.

## Ranges

ZRANGE returns members.
"""


def test_split_sections_tree_and_code_fence():
    sections = split_sections(MD, "Sorted sets")
    by_heading = {s.heading: s for s in sections}
    assert "# not a heading" in by_heading["Adding members"].text
    assert by_heading["Options"].path == ["Sorted sets", "Adding members", "Options"]
    assert by_heading["Options"].parent_index == by_heading["Adding members"].index
    assert by_heading["Ranges"].parent_index == by_heading["Sorted sets"].index
    assert top_level(sections, by_heading["Options"].index) == by_heading["Sorted sets"].index


def test_chunk_sections_header_and_parent():
    sections = split_sections(MD, "Sorted sets")
    chunks = chunk_sections(CharTokenizer(), sections, library="redis", site_name="Redis",
                            url="u", doc_id="d", max_tokens=400)
    options = next(c for c in chunks if c.heading_path.endswith("Options"))
    assert options.text.startswith("Redis > Sorted sets > Adding members > Options\n\n")
    assert all(c.tokens <= 400 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_merge_sections_folds_short_page_into_one_chunk():
    sections = split_sections(MD, "Sorted sets")
    chunks = chunk_sections(CharTokenizer(), sections, library="redis", site_name="Redis",
                            url="u", doc_id="d", max_tokens=1000, merge_sections=True)
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Sorted sets"
    assert "## Adding members" in chunks[0].text and "### Options" in chunks[0].text and "## Ranges" in chunks[0].text


def test_merge_sections_respects_budget():
    sections = split_sections(MD, "Sorted sets")
    chunks = chunk_sections(CharTokenizer(), sections, library="redis", site_name="Redis",
                            url="u", doc_id="d", max_tokens=120, merge_sections=True)
    assert len(chunks) > 1 and all(c.tokens <= 120 for c in chunks)


def test_extract_drops_noise_and_extra_languages():
    site = Site(library="x", name="X", sitemap="", content_selector="section.prose",
                drop_selectors=(".tab-nav",), keep_code_langs=frozenset({"python"}))
    html = """<html><body><nav>menu</nav><section class="prose"><h1>ZADD</h1>
    <div class="tab-nav">Syntax | API</div><p>Adds members.</p>
    <div data-lang="python"><pre><code class="language-python">r.zadd()</code></pre></div>
    <div data-lang="java"><pre><code class="language-java">jedis.zadd()</code></pre>
      <div data-lang="go">nested</div></div>
    <script>x()</script></section></body></html>"""
    page = extract(site, "https://x/zadd", html)
    assert page.title == "ZADD"
    assert "r.zadd()" in page.markdown and "```python" in page.markdown
    assert "jedis" not in page.markdown and "Syntax | API" not in page.markdown and "x()" not in page.markdown
