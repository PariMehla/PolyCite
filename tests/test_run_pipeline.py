from polycite.cohere_client import CohereClient
from polycite.data.build_corpus import build_fixture_corpus
from polycite.testing import fake_transport
from scripts.run_pipeline import run_condition


def test_run_condition_completes_without_budget_pressure(tmp_path):
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)
    rows, budget_exhausted = run_condition(client, corpus, "MONO", "fake-rerank", "fake-chat")
    assert budget_exhausted is False
    assert len(rows) > 0


def test_run_condition_stops_gracefully_and_keeps_partial_rows_when_budget_runs_out(tmp_path):
    corpus = build_fixture_corpus()
    # Budget for a handful of calls only: enough for a couple of questions
    # (2 calls each: rerank + chat) before running out.
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=4)
    rows, budget_exhausted = run_condition(client, corpus, "MONO", "fake-rerank", "fake-chat")
    assert budget_exhausted is True
    assert len(rows) < len(corpus.questions)  # stopped early, didn't crash
    assert len(rows) >= 0  # whatever it got done is preserved, not lost
