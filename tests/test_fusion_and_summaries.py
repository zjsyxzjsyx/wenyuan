from app.ingest.summarize import Block, rows_for, summarize
from app.retrieval.fusion import best_per_block, rrf
from app.tokenizer import TiktokenTokenizer


def hit(cid, doc="d", parent=0):
    return {"chunk_id": cid, "doc_id": doc, "parent_index": parent, "url": f"u/{doc}"}


def test_rrf_rewards_items_in_both_lists():
    fused = rrf([[hit("a"), hit("b"), hit("c")], [hit("c"), hit("x")]])
    assert [h["chunk_id"] for h in fused][:2] == ["c", "a"]       # c: 1/63 + 1/61 beats a: 1/61
    assert len(rrf([[hit("a"), hit("b")], [hit("c")]], limit=2)) == 2


def test_best_per_block_keeps_first_row_of_each_block():
    rows = [hit("d:0:q1", parent=0), hit("d:3:s", parent=3), hit("d:0:s", parent=0)]
    assert [h["chunk_id"] for h in best_per_block(rows)] == ["d:0:q1", "d:3:s"]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat_json(self, messages, temperature=0.0):
        return self.reply


def test_summary_rows_point_at_the_parent_block():
    block = Block("redis", "doc1", "https://redis.io/x", 2, "Redis > Keys", "FLUSHALL deletes all keys. " * 20)
    result = summarize(FakeLLM({"summary": "How to delete keys.", "questions": ["how do I wipe redis?", " ", "q2"]}),
                       TiktokenTokenizer(), block)
    assert result == {"summary": "How to delete keys.", "questions": ["how do I wipe redis?", "q2"]}
    rows = rows_for(block, result)
    assert [r["chunk_id"] for r in rows] == ["doc1:2:s", "doc1:2:q0", "doc1:2:q1"]
    assert {(r["doc_id"], r["parent_index"], r["doc_type"]) for r in rows} == {("doc1", 2, "guide")}
    assert summarize(FakeLLM({}), TiktokenTokenizer(), block) is None
