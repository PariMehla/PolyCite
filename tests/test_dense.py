from polycite.cohere_client import CohereClient
from polycite.retrieval.dense import DenseIndex, _cosine


def test_cosine_identical_vectors():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_orthogonal_vectors():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_search_with_vector_ranks_closest_first():
    index = DenseIndex(["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    results = index.search_with_vector([1.0, 0.0], top_k=3)
    assert results[0][0] == "a"
    assert results[1][0] == "c"


def test_from_local_encoder_builds_index_from_injected_encoder():
    passages = {"p1": {"text": "hello"}, "p2": {"text": "world"}}
    index = DenseIndex.from_local_encoder(passages, encode_fn=lambda texts: [[float(len(t)), 0.0] for t in texts])
    assert index.passage_ids == ["p1", "p2"]
    assert index.embeddings[0] == [5.0, 0.0]


def test_from_cohere_batches_embed_calls(tmp_path):
    seen_batches = []

    def transport(endpoint, model, params):
        seen_batches.append(len(params["texts"]))
        return {"embeddings": {"float": [[0.1, 0.2] for _ in params["texts"]]}}

    client = CohereClient(cache_dir=tmp_path, transport=transport)
    passages = {f"p{i}": {"text": f"text {i}"} for i in range(150)}
    index = DenseIndex.from_cohere(client, passages)
    assert len(index.passage_ids) == 150
    assert seen_batches == [96, 54]  # two batches, capped at EMBED_MAX_TEXTS_PER_CALL
