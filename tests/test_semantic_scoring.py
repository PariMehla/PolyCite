from polycite.cohere_client import CohereClient
from polycite.generate.semantic_scoring import (
    _cosine,
    batch_semantic_similarities,
    is_semantically_correct,
    semantic_similarity,
)


def test_cosine_identical_vectors_is_one():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_orthogonal_vectors_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def _fixed_vectors_transport(vectors_by_text: dict[str, list[float]]):
    def transport(endpoint, model, params):
        assert endpoint == "embed"
        return {"embeddings": {"float": [vectors_by_text[t] for t in params["texts"]]}}

    return transport


def test_semantic_similarity_batches_both_texts_into_one_call(tmp_path):
    transport = _fixed_vectors_transport({"pred text": [1.0, 0.0], "gold text": [1.0, 0.0]})
    client = CohereClient(cache_dir=tmp_path, transport=transport)
    sim = semantic_similarity(client, "pred text", "gold text")
    assert sim == 1.0
    assert client.counter.count == 1  # one call, not two


def test_is_semantically_correct_respects_threshold(tmp_path):
    transport = _fixed_vectors_transport({"pred": [1.0, 0.0], "gold": [0.0, 1.0]})
    client = CohereClient(cache_dir=tmp_path, transport=transport)
    correct, sim = is_semantically_correct(client, "pred", "gold", threshold=0.5)
    assert correct is False
    assert sim == 0.0


def test_batch_semantic_similarities_is_order_preserving_and_batches_one_call(tmp_path):
    vectors = {
        "p1": [1.0, 0.0], "g1": [1.0, 0.0],  # sim 1.0
        "p2": [1.0, 0.0], "g2": [0.0, 1.0],  # sim 0.0
    }
    transport = _fixed_vectors_transport(vectors)
    client = CohereClient(cache_dir=tmp_path, transport=transport)
    sims = batch_semantic_similarities(client, [("p1", "g1"), ("p2", "g2")])
    assert sims == [1.0, 0.0]
    assert client.counter.count == 1  # both pairs batched into one call


def test_batch_semantic_similarities_splits_across_calls_when_over_the_limit(tmp_path):
    from polycite.cohere_client import EMBED_MAX_TEXTS_PER_CALL

    pairs = [(f"p{i}", f"g{i}") for i in range(EMBED_MAX_TEXTS_PER_CALL // 2 + 5)]  # 5 pairs over one call's worth
    vectors = {}
    for p, g in pairs:
        vectors[p] = [1.0, 0.0]
        vectors[g] = [1.0, 0.0]
    transport = _fixed_vectors_transport(vectors)
    client = CohereClient(cache_dir=tmp_path, transport=transport)
    sims = batch_semantic_similarities(client, pairs)
    assert len(sims) == len(pairs)
    assert all(s == 1.0 for s in sims)
    assert client.counter.count == 2  # split into two embed calls
