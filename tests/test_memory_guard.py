from eval.memory_guard import Memory, read_memory, wait_for_memory


def test_waits_while_ram_is_short_even_if_commit_looks_fine():
    readings = iter([Memory(free_gb=0.8, commit_used=0.60),      # what froze the laptop: RAM, not commit
                     Memory(free_gb=3.0, commit_used=0.90),
                     Memory(free_gb=3.0, commit_used=0.50)])
    assert wait_for_memory(poll_s=0, read=lambda: next(readings)) == 0
    assert next(readings, None) is None          # it polled until both readings were fine


def test_unreadable_memory_never_blocks():
    assert wait_for_memory(poll_s=0, read=lambda: None) == 0
    m = read_memory()
    assert m is None or (m.free_gb > 0 and 0 < m.commit_used < 1)
