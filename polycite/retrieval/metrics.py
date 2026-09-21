"""Deterministic retrieval metrics: Recall@k, MRR@k, nDCG@k.

All take a ranked list of passage IDs (best first) and a single gold ID,
since Belebele questions have exactly one gold passage. Callers should
filter out unanswerable-split questions before aggregating these (there is
no gold passage in the index for them by construction).
"""
from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    for rank, pid in enumerate(ranked_ids[:k], start=1):
        if pid == gold_id:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    for rank, pid in enumerate(ranked_ids[:k], start=1):
        if pid == gold_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def aggregate(per_question_scores: list[float]) -> dict:
    n = len(per_question_scores)
    if n == 0:
        return {"mean": float("nan"), "n": 0}
    return {"mean": sum(per_question_scores) / n, "n": n}
