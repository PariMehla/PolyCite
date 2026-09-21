from polycite.cohere_client import CohereClient
from polycite.rerank.cohere_rerank import rerank


def test_rerank_maps_index_back_to_passage_id(tmp_path):
    def transport(endpoint, model, params):
        assert endpoint == "rerank"
        assert params["documents"] == ["doc about cats", "doc about dogs"]
        return {"results": [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}]}

    client = CohereClient(cache_dir=tmp_path, transport=transport)
    candidates = [("p_cats", "doc about cats"), ("p_dogs", "doc about dogs")]
    results = rerank(client, "query", candidates, top_n=2)
    assert results == [("p_dogs", 0.8), ("p_cats", 0.2)]
