from polycite.retrieval.metrics import aggregate, mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k_hit_and_miss():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, "b", k=5) == 1.0
    assert recall_at_k(ranked, "z", k=5) == 0.0
    assert recall_at_k(ranked, "d", k=2) == 0.0  # outside k


def test_mrr_at_k():
    ranked = ["a", "b", "c"]
    assert mrr_at_k(ranked, "a", k=3) == 1.0
    assert mrr_at_k(ranked, "b", k=3) == 0.5
    assert mrr_at_k(ranked, "z", k=3) == 0.0


def test_ndcg_at_k_monotonic_in_rank():
    ranked = ["a", "b", "c"]
    ndcg_first = ndcg_at_k(ranked, "a", k=3)
    ndcg_second = ndcg_at_k(ranked, "b", k=3)
    assert ndcg_first > ndcg_second > 0


def test_aggregate_empty():
    result = aggregate([])
    assert result["n"] == 0
