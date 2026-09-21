from polycite.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_boosts_documents_ranked_highly_in_multiple_lists():
    list_a = ["p1", "p2", "p3"]
    list_b = ["p2", "p1", "p4"]
    fused = reciprocal_rank_fusion([list_a, list_b])
    fused_ids = [pid for pid, _ in fused]
    assert fused_ids[0] in ("p1", "p2")  # both appear near top of both lists


def test_rrf_respects_top_k():
    lists = [[f"p{i}" for i in range(20)]]
    fused = reciprocal_rank_fusion(lists, top_k=5)
    assert len(fused) == 5
