"""Reciprocal Rank Fusion of multiple ranked lists."""
from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60, top_k: int = 50
) -> list[tuple[str, float]]:
    """Standard RRF: score(d) = sum over lists of 1 / (k + rank_in_list).

    A document absent from a list contributes 0 for that list. k=60 is the
    conventional default from the original RRF paper.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]
