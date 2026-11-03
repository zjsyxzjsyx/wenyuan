from app.maintenance import make_plan


def test_cleanup_keeps_courses_notes_and_real_accounts():
    plan = make_plan(collections=["chunks_v2", "chunks_struct_merge", "chunks_qwen3e", "summaries_struct_merge"],
                     docs_per_library={"cs50": 70, "redis": 768, "python": 418, "ui-demo": 1, "private": 3},
                     course_ids=["cs50", "ui-demo"],
                     user_names=["koyo", "1", "cs50-teacher", "dev", "ui-test", "ui-verify", "loadtest-0912-3"])
    assert plan.collections == ["chunks_qwen3e", "chunks_struct_merge", "summaries_struct_merge"]
    assert plan.libraries == {"python": 418, "redis": 768, "ui-demo": 1}     # the test course's files go too
    assert plan.courses == ["ui-demo"]
    assert plan.users == ["loadtest-0912-3", "ui-test", "ui-verify"]
    assert make_plan(["chunks_v2"], {"cs50": 70}, ["cs50"], ["koyo"]).empty()
