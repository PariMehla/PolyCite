"""Embedding-based semantic answer scoring: a supplement to, not a
replacement for, the deterministic gold-recall scorer in scoring.py.

Motivated by a real limitation found via manual judge review
(HYPOTHESES.md "RQ4, minimally"): gold-recall requires literal token
overlap, so it can't detect a correct answer phrased with synonyms (e.g.
Arabic "root of problems" vs. gold "cause of problems" -- same meaning,
zero shared tokens). Cosine similarity between Cohere embeddings of the
prediction and the gold answer can catch paraphrases that share no tokens
but do share meaning.

The similarity threshold is NOT chosen yet. Picking one without validation
would just swap an unvalidated guess (the current 0.6 recall cutoff) for
another. See scripts/test_semantic_scoring.py, which checks candidate
thresholds against a small hand-labeled set of known-correct-but-missed
and known-genuinely-wrong cases from the manual judge review before this
is trusted for anything beyond that spot check.
"""
from __future__ import annotations

import math

from polycite.cohere_client import EMBED_MAX_TEXTS_PER_CALL, CohereClient

# Validated against a hand-labeled set (scripts/test_semantic_scoring.py,
# 2026-09-22): RIGHT (known-correct-synonym) cases ranged 0.563-0.583,
# WRONG (known-genuinely-wrong) cases ranged 0.486-0.526, no overlap. This
# threshold sits in the gap. n=6 validates the *concept* (the two groups
# separate), not a precise production cutoff -- see scripts/rescore_semantic.py.
DEFAULT_SEMANTIC_THRESHOLD = 0.545


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_similarity(
    client: CohereClient, prediction: str, gold: str, model: str = "embed-multilingual-v3.0"
) -> float:
    """Cosine similarity between embeddings of prediction and gold answer.
    Batches both texts into one embed call."""
    resp = client.embed(model=model, input_type="classification", texts=[prediction, gold])
    v_pred, v_gold = resp["embeddings"]["float"]
    return _cosine(v_pred, v_gold)


def is_semantically_correct(
    client: CohereClient, prediction: str, gold: str, threshold: float, model: str = "embed-multilingual-v3.0"
) -> tuple[bool, float]:
    sim = semantic_similarity(client, prediction, gold, model=model)
    return sim >= threshold, sim


def batch_semantic_similarities(
    client: CohereClient, pairs: list[tuple[str, str]], model: str = "embed-multilingual-v3.0"
) -> list[float]:
    """Cosine similarity for many (prediction, gold) pairs, batched at
    Cohere Embed's max texts/call. Order-preserving: result[i] corresponds
    to pairs[i]. Used by scripts/rescore_semantic.py to rescore an entire
    results parquet without one embed call per row."""
    if not pairs:
        return []
    max_pairs_per_call = EMBED_MAX_TEXTS_PER_CALL // 2
    similarities: list[float] = []
    for i in range(0, len(pairs), max_pairs_per_call):
        chunk = pairs[i : i + max_pairs_per_call]
        texts = [text for pair in chunk for text in pair]
        resp = client.embed(model=model, input_type="classification", texts=texts)
        vectors = resp["embeddings"]["float"]
        for j in range(len(chunk)):
            similarities.append(_cosine(vectors[2 * j], vectors[2 * j + 1]))
    return similarities
