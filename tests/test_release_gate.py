import json

from eval import gate


def run(tmp_path, name, dataset, owner, scores):
    path = tmp_path / f"{name}.json"
    questions = [{"id": f"q{i}", **({"ev_hit@5": s} if s is not None else {}), "hit@5": 1.0} for i, s in enumerate(scores)]
    path.write_text(json.dumps({"exp": name, "dataset": dataset, "owner": owner, "questions": questions}), encoding="utf-8")
    return path


def test_the_baseline_is_chosen_by_dataset_and_owner_and_missing_scores_are_skipped(tmp_path, monkeypatch):
    course = run(tmp_path, "course", "data/eval/cs50_gen_v1.jsonl", None, [1, 1, 1, 1, 0, None])
    notes = run(tmp_path, "notes", "data/eval/notes_koyo_v1.jsonl", "koyo", [1] * 6)
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"course": {"run": str(course), "metrics": ["ev_hit@5"]},
                                     "notes": {"run": str(notes), "metrics": ["ev_hit@5", "hit@5"]},
                                     "old": "experiments/old.json"}), encoding="utf-8")
    monkeypatch.setattr(gate, "BASELINES", baselines)
    worse = run(tmp_path, "worse", "data\\eval\\cs50_gen_v1.jsonl", None, [0, 0, 0, 0, 0, None])
    found, metrics = gate.resolve(worse, None)
    assert found == course and metrics == ["ev_hit@5"]
    result = gate.gate(found, worse, metrics, tolerance=0.0, resamples=500)
    assert result["metrics"]["ev_hit@5"]["n"] == 5 and result["failed"] == ["ev_hit@5"]
    assert gate.known()["old"]["metrics"] == ["recall@5", "recall@10", "mrr"]          # v0.7 bare-path entries
    assert gate.resolve(run(tmp_path, "mine", "data/eval/notes_koyo_v1.jsonl", "koyo", [1] * 6), None)[0] == notes
