from types import SimpleNamespace

from eval.answers import cites_gold, page_of
from eval.refusal import language, outcome, summarize


def test_a_citation_to_a_section_or_a_video_second_still_names_its_page():
    assert page_of("https://cs50.harvard.edu/x/2024/notes/4/#malloc") == "https://cs50.harvard.edu/x/2024/notes/4"
    assert page_of("https://youtu.be/jZzyERW7h1A?t=61") == "https://youtu.be/jZzyERW7h1A"
    gold = ["https://cs50.harvard.edu/x/2024/notes/4/#lecture-4"]
    assert cites_gold([{"url": "https://cs50.harvard.edu/x/2024/notes/4/#malloc"}], gold)
    assert not cites_gold([{"url": "https://cs50.harvard.edu/x/2024/notes/5/"}, {"url": "private://7/a.md"}], gold)


def test_unsupported_claims_are_sorted_and_only_wrong_ones_count_as_harmful():
    from eval.answers import classify_unsupported, summarize

    class Judge:
        def chat_json(self, messages, temperature=0.0):
            return {"claims": [{"id": "C1", "kind": "outside_correct"}, {"id": "C2", "kind": "outside_wrong"},
                               {"id": "C3", "kind": "whatever"}]}
    kinds = classify_unsupported(Judge(), ["a", "b", "c"], [])
    assert kinds == {"a": "outside_correct", "b": "outside_wrong", "c": "unclassified"}
    claim = lambda judge, removed, kind=None: {"text": "t", "verifier": "supported", "judge": judge, "removed": removed,
                                               **({"kind": kind} if kind else {})}
    rows = [{"id": "q1", "library": "cs50", "found_before": True, "found_after": True, "verify_ms": 1,
             "correct_before": 2, "correct_after": 2,
             "claims": [claim("supported", False), claim("unsupported", False, "outside_correct")]},
            {"id": "q2", "library": "cs50", "found_before": True, "found_after": True, "verify_ms": 1,
             "correct_before": 1, "correct_after": 1,
             "claims": [claim("unsupported", True, "outside_wrong"), claim("unsupported", False, "contradicts"),
                        claim("supported", False)]}]
    s = summarize(rows)
    assert s["unsupported_kinds_after"] == {"outside_correct": 1, "contradicts": 1}
    assert s["harmful_claims_before"] == 0.4 and s["harmful_claims_after"] == 0.25
    assert s["answers_with_harmful_before"] == 0.5 and s["answers_with_harmful_after"] == 0.5


def test_the_committed_run_keeps_every_number_but_no_answer_text():
    from eval.answers import summarize, without_texts
    q = {"id": "q1", "library": "cs50", "found_before": True, "found_after": True, "verify_ms": 5,
         "correct_before": 2, "correct_after": 2, "answer_before": "Use malloc [S1].", "answer_after": "Use malloc [S1].",
         "claims": [{"text": "Use malloc [S1].", "verifier": "supported", "judge": "supported", "removed": False}]}
    run = {"exp": "x", "summary": summarize([q]), "questions": [q]}
    stripped = without_texts(run)
    assert "malloc" not in str(stripped) and "malloc" in str(run)          # the input is left alone
    assert summarize(stripped["questions"]) == run["summary"]


def test_refusals_are_rated_per_language_and_errors_left_out():
    refused, answered = SimpleNamespace(found=False, citations=[]), SimpleNamespace(found=True, citations=[{}])
    results = [outcome({"id": "a1", "question": "How do I deploy to AWS?"}, refused, "refuse"),
               outcome({"id": "a2", "question": "课程会讲 Spring Boot 吗？"}, answered, "refuse"),
               outcome({"id": "q1", "question": "What does malloc return?"}, answered, "answer"),
               {"id": "q2", "expected": "answer", "error": "LLMError"}]
    assert language("为什么 malloc 返回 NULL") == "zh" and language("why NULL") == "en"
    s = summarize(results)
    assert s["errors"] == 1
    assert s["refused_when_absent"] == {"all": [0.5, 2], "zh": [0.0, 1], "en": [1.0, 1]}
    assert s["refused_when_answerable"] == {"all": [0.0, 1], "en": [0.0, 1]}
