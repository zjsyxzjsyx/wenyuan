from app.retrieval.fusion import rrf


def hit(cid, doc="d", parent=0):
    return {"chunk_id": cid, "doc_id": doc, "parent_index": parent, "url": f"u/{doc}"}


def test_rrf_rewards_items_in_both_lists():
    fused = rrf([[hit("a"), hit("b"), hit("c")], [hit("c"), hit("x")]])
    assert [h["chunk_id"] for h in fused][:2] == ["c", "a"]       # c: 1/63 + 1/61 beats a: 1/61
    assert len(rrf([[hit("a"), hit("b")], [hit("c")]], limit=2)) == 2
