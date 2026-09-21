"""Thin wrapper turning Cohere Rerank into a (passage_id, score) reranker."""
from __future__ import annotations

from polycite.cohere_client import CohereClient


def rerank(
    client: CohereClient,
    query: str,
    candidates: list[tuple[str, str]],  # (passage_id, text), pre-retrieval order
    model: str = "rerank-v3.5",
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Returns up to top_n (passage_id, relevance_score), best first."""
    ids = [pid for pid, _ in candidates]
    texts = [text for _, text in candidates]
    resp = client.rerank(model=model, query=query, documents=texts, top_n=top_n)
    out = []
    for item in resp["results"]:
        idx = item["index"]
        out.append((ids[idx], item["relevance_score"]))
    return out
