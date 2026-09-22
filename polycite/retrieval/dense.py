"""Dense retrieval: Cohere Embed (via CohereClient) and a local baseline.

Local baseline (multilingual-e5-small via sentence-transformers) needs a
model download from huggingface.co, which this sandbox cannot reach. The
function is written and unit-testable with an injected encoder, but running
it for real requires internet the way `pip install sentence-transformers`
does not need.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

from polycite.cohere_client import EMBED_MAX_TEXTS_PER_CALL, CohereClient


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class DenseIndex:
    """Cosine-similarity search over pre-computed embeddings.

    Populate via `from_cohere` (batches embed calls at 96 texts/call, the
    trial-key max) or by passing embeddings computed some other way (e.g. a
    local sentence-transformers model in `encode_fn`).
    """

    def __init__(self, passage_ids: list[str], embeddings: list[list[float]]):
        self.passage_ids = passage_ids
        self.embeddings = embeddings

    @classmethod
    def from_cohere(
        cls, client: CohereClient, passages: dict[str, dict], model: str = "embed-multilingual-v3.0"
    ) -> "DenseIndex":
        ids = list(passages.keys())
        texts = [passages[pid]["text"] for pid in ids]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_MAX_TEXTS_PER_CALL):
            if i > 0:
                # Cohere's trial embed limit is token-volume-based (e.g. 100k
                # tokens/min), not just request-count -- CohereClient's rate
                # limiter only paces by request count, so a big multi-batch
                # document-embedding job like this one can burst well past
                # the token budget in a few seconds even while staying under
                # the request-count cap. This small proactive gap avoids
                # relying on 429 retries for something this predictable; it
                # costs a few seconds even on a fully cache-hit rerun, which
                # is an acceptable tradeoff for not bursting on a fresh one.
                time.sleep(0.5)
            batch = texts[i : i + EMBED_MAX_TEXTS_PER_CALL]
            resp = client.embed(model=model, input_type="search_document", texts=batch)
            vectors.extend(resp["embeddings"]["float"])
        return cls(ids, vectors)

    @classmethod
    def from_local_encoder(cls, passages: dict[str, dict], encode_fn: Callable[[list[str]], list[list[float]]]) -> "DenseIndex":
        ids = list(passages.keys())
        texts = [passages[pid]["text"] for pid in ids]
        return cls(ids, encode_fn(texts))

    def search_with_vector(self, query_vector: list[float], top_k: int = 50) -> list[tuple[str, float]]:
        scored = [(pid, _cosine(query_vector, vec)) for pid, vec in zip(self.passage_ids, self.embeddings)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_cohere(self, client: CohereClient, query: str, model: str = "embed-multilingual-v3.0", top_k: int = 50) -> list[tuple[str, float]]:
        resp = client.embed(model=model, input_type="search_query", texts=[query])
        query_vector = resp["embeddings"]["float"][0]
        return self.search_with_vector(query_vector, top_k)
