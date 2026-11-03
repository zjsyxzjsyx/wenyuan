from app.insights import group_questions


def test_similar_questions_group_and_only_shared_ones_are_shown():
    malloc, sort, notes = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    questions = [(1, "malloc 返回 NULL 是什么意思"), (2, "malloc 失败了会怎样"), (3, "why is malloc returning NULL"),
                 (1, "merge sort 为什么是 n log n"), (2, "归并排序的复杂度怎么算"),
                 (4, "我笔记里那道题的思路对吗")]           # one student only: never shown to the teacher
    vectors = [malloc, [0.98, 0.05, 0.0], [0.95, 0.1, 0.0], sort, [0.1, 0.97, 0.0], notes]
    groups = group_questions(questions, vectors, threshold=0.8)
    assert [(g["students"], g["asked"]) for g in groups] == [(3, 3), (2, 2)]
    assert groups[0]["question"].startswith("malloc") and len(groups[0]["also"]) == 2
    assert all("笔记" not in g["question"] and all("笔记" not in t for t in g["also"]) for g in groups)


def test_one_student_asking_again_and_again_is_still_one_student():
    groups = group_questions([(7, "a"), (7, "a again"), (7, "a once more")], [[1, 0], [1, 0.01], [1, 0.02]])
    assert groups == []
