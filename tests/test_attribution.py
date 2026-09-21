from polycite.analysis.attribution import attribution_counts, classify_failure

BASE = {
    "is_unanswerable": False,
    "abstained": False,
    "retrieved_top50": ["p1", "p2"],
    "reranked_top5": ["p1", "p2"],
    "gold_passage_id": "p1",
    "answer_correct": True,
    "cited_passage_ids": {"p1"},
    "response_language": "eng_Latn",
    "query_language": "eng_Latn",
}


def _record(**overrides):
    return {**BASE, **overrides}


def test_success_returns_none():
    assert classify_failure(_record()) is None


def test_retrieval_failure_when_gold_missing_from_top50():
    r = _record(retrieved_top50=["p2", "p3"], reranked_top5=["p2"])
    assert classify_failure(r) == "retrieval_failure"


def test_reranking_failure_when_gold_retrieved_but_not_reranked():
    r = _record(retrieved_top50=["p1", "p2"], reranked_top5=["p2"])
    assert classify_failure(r) == "reranking_failure"


def test_reading_failure_when_gold_in_context_but_wrong_answer():
    r = _record(answer_correct=False)
    assert classify_failure(r) == "reading_failure"


def test_citation_failure_when_answer_correct_but_wrong_citation():
    r = _record(cited_passage_ids={"p2"})
    assert classify_failure(r) == "citation_failure"


def test_language_failure_when_correct_and_cited_but_wrong_language():
    r = _record(response_language="fra_Latn")
    assert classify_failure(r) == "language_failure"


def test_language_failure_skipped_when_undetectable():
    r = _record(response_language=None)
    assert classify_failure(r) is None


def test_hallucinated_confidence_on_unanswerable_without_abstention():
    r = _record(is_unanswerable=True, abstained=False)
    assert classify_failure(r) == "hallucinated_confidence"


def test_correct_abstention_is_not_a_failure():
    r = _record(is_unanswerable=True, abstained=True)
    assert classify_failure(r) is None


def test_decision_tree_order_retrieval_beats_reading():
    # gold missing from top50 AND answer happens to be wrong -> still retrieval_failure, first in order
    r = _record(retrieved_top50=["p2"], reranked_top5=[], answer_correct=False)
    assert classify_failure(r) == "retrieval_failure"


def test_attribution_counts_aggregates_by_language():
    records = [
        _record(query_language="eng_Latn"),
        _record(query_language="eng_Latn", retrieved_top50=["p2"], reranked_top5=[]),
        _record(query_language="yor_Latn", retrieved_top50=["p2"], reranked_top5=[]),
    ]
    counts = attribution_counts(records)
    assert counts["eng_Latn"]["success"] == 1
    assert counts["eng_Latn"]["retrieval_failure"] == 1
    assert counts["yor_Latn"]["retrieval_failure"] == 1
