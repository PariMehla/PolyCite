"""Assign exactly one failure stage per failed question.

Decision tree, applied in this order (per PolyCite's design doc):
  1. retrieval_failure       gold passage not in top-50 retrieved
  2. reranking_failure       in top-50, reranked out of top-5
  3. reading_failure         gold passage in context, answer still wrong
  4. citation_failure        answer correct, but citation missing/wrong
  5. language_failure        correct + cited, but wrong response language
  6. hallucinated_confidence answered instead of abstaining on an
                             unanswerable-split question
A question with none of the above is a success (returns None).
"""
from __future__ import annotations

from typing import Optional

FAILURE_STAGES = [
    "retrieval_failure",
    "reranking_failure",
    "reading_failure",
    "citation_failure",
    "language_failure",
    "hallucinated_confidence",
]


def classify_failure(record: dict) -> Optional[str]:
    """record fields:
    is_unanswerable: bool
    abstained: bool                       (did the model actually abstain)
    retrieved_top50: list[str]
    reranked_top5: list[str]
    gold_passage_id: str
    answer_correct: Optional[bool]
    cited_passage_ids: set[str]
    response_language: Optional[str]      (None if undetectable)
    query_language: str
    """
    if record["is_unanswerable"]:
        return None if record["abstained"] else "hallucinated_confidence"

    gold = record["gold_passage_id"]
    if gold not in record["retrieved_top50"]:
        return "retrieval_failure"
    if gold not in record["reranked_top5"]:
        return "reranking_failure"
    if not record["answer_correct"]:
        return "reading_failure"
    if gold not in record.get("cited_passage_ids", set()):
        return "citation_failure"
    response_lang = record.get("response_language")
    if response_lang is not None and response_lang != record["query_language"]:
        return "language_failure"
    return None


def attribution_counts(records: list[dict]) -> dict[str, dict[str, int]]:
    """{language: {stage_or_'success': count}}"""
    counts: dict[str, dict[str, int]] = {}
    for r in records:
        lang = r["query_language"]
        stage = classify_failure(r) or "success"
        counts.setdefault(lang, {s: 0 for s in [*FAILURE_STAGES, "success"]})
        counts[lang][stage] += 1
    return counts
