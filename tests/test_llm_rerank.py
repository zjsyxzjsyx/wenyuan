from app.retrieval.llm_rerank import rerank_pages


class FakeLLM:
    def __init__(self, ranking):
        self.ranking, self.prompts = ranking, []

    def chat_json(self, messages, temperature=0.0):
        self.prompts.append(messages[0]["content"])
        return {"ranking": self.ranking}


def hits(*urls):
    return [{"url": u, "heading_path": f"Docs > {u}", "text": f"text of {u}"} for u in urls]


def order(result):
    return list(dict.fromkeys(h["url"] for h in result))


def test_ranked_pages_move_first_and_the_rest_keep_their_order():
    # page "b" has two chunks; they stay together and keep their order
    pool = hits("a", "b", "c", "b", "d", "e")
    llm = FakeLLM([3, 1])                                    # c, then a
    out = rerank_pages(llm, "q", pool, top_pages=4)
    assert order(out) == ["c", "a", "b", "d", "e"]          # e was beyond top_pages: untouched, last
    assert [h["text"] for h in out if h["url"] == "b"] == ["text of b", "text of b"]
    assert len(out) == len(pool)
    assert "[4] Docs > d" in llm.prompts[0] and "Docs > e" not in llm.prompts[0]


def test_bad_ids_from_the_model_are_ignored():
    out = rerank_pages(FakeLLM(["2", 2, 9, "x", 0, None]), "q", hits("a", "b", "c"))
    assert order(out) == ["b", "a", "c"]


def test_a_single_page_needs_no_call():
    llm = FakeLLM([1])
    assert order(rerank_pages(llm, "q", hits("a", "a"))) == ["a"] and not llm.prompts


def test_a_page_shows_its_other_matching_sections_and_fits_the_budget():
    pool = [{"url": "a", "heading_path": "Docs > A > Intro", "text": "x" * 500},
            {"url": "a", "heading_path": "Docs > A > Lifespan events", "text": "use lifespan " * 40},
            {"url": "a", "heading_path": "Docs > A > Shutdown", "text": "..."},
            {"url": "b", "heading_path": "Docs > B", "text": "y" * 500}]
    llm = FakeLLM([2, 1])
    rerank_pages(llm, "q", pool)
    prompt = llm.prompts[0]
    assert "... Lifespan events: use lifespan" in prompt and "Also matched: Shutdown" in prompt
    assert prompt.count("q") >= 2 and prompt.rstrip().endswith("most useful ids.")   # the question comes twice


def test_the_listing_is_counted_in_tokens_and_drops_the_last_candidates_to_fit():
    chinese = [{"url": u, "heading_path": f"My notes > {u}", "text": "滑动窗口适合求最长无重复子串用哈希表记录位置" * 20}
               for u in "abcdefghij"]
    small = FakeLLM([2])
    out = rerank_pages(small, "q", chinese, max_tokens=150)
    assert "[1] My notes > a" in small.prompts[0] and "[10]" not in small.prompts[0]
    assert order(out) == list("bacdefghij")         # candidates left out of the prompt keep their place after it


def test_sections_of_one_page_can_be_ranked_separately():
    pool = [{"url": "lecture", "heading_path": "Week 3 · Lecture 3 > Lecture 3 > Binary Search", "text": "halve"},
            {"url": "lecture", "heading_path": "Week 3 · Lecture 3 > Lecture 3 > Merge Sort", "text": "merge"},
            {"url": "lecture", "heading_path": "Week 3 · Lecture 3 > Lecture 3 > Merge Sort > Runtime", "text": "n log n"},
            {"url": "notes", "heading_path": "My notes > sorting", "text": "mine"}]
    llm = FakeLLM([2, 3, 1])
    out = rerank_pages(llm, "q", pool, unit="section")
    assert [h["text"] for h in out] == ["merge", "n log n", "mine", "halve"]   # a subsection stays with its section
    assert "[3] My notes > sorting" in llm.prompts[0]
