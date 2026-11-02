import json

from eval.gate import gate


def run_file(tmp_path, name, scores):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"exp": name, "questions": [
        {"id": f"q{i}", "recall@5": s, "mrr": s} for i, s in enumerate(scores)]}))
    return path


def test_gate_fails_only_a_real_drop(tmp_path):
    base = run_file(tmp_path, "base", [1.0, 0.0] * 50)
    same = run_file(tmp_path, "same", [1.0, 0.0] * 50)
    worse = run_file(tmp_path, "worse", [0.0, 0.0] * 50)                 # every hit lost
    slightly = run_file(tmp_path, "slightly", [1.0, 0.0] * 49 + [0.0, 0.0])  # one question lost
    assert gate(base, same, ["recall@5"], 0.0, 500)["passed"]
    assert gate(base, worse, ["recall@5", "mrr"], 0.0, 500)["failed"] == ["recall@5", "mrr"]
    assert gate(base, slightly, ["recall@5"], 0.0, 500)["passed"]          # within noise
    assert gate(base, worse, ["recall@5"], 0.6, 500)["passed"]             # tolerated drop
