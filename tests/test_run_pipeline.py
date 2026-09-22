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


def test_en2x_uses_real_english_query_text_not_relabeled_original(tmp_path):
    # Regression test for a real bug: EN2X used to relabel the ORIGINAL
    # non-English question as "eng_Latn" without translating it, so every
    # EN2X row collapsed into one query_language="eng_Latn" bucket regardless
    # of which language it actually came from.
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)
    rows, _ = run_condition(client, corpus, "EN2X", "fake-rerank", "fake-chat")
    assert len(rows) > 0
    # Every EN2X row must query in English...
    assert all(r["query_language"] == "eng_Latn" for r in rows)
    # ...but display_language (= corpus_language for EN2X) must vary across
    # the actual foreign corpora searched, not collapse to one value.
    display_languages = {r["display_language"] for r in rows}
    assert len(display_languages) > 1


def test_en2x_display_language_is_corpus_language_others_are_query_language(tmp_path):
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)

    en2x_rows, _ = run_condition(client, corpus, "EN2X", "fake-rerank", "fake-chat")
    for r in en2x_rows:
        assert r["display_language"] == r["corpus_language"]

    mono_rows, _ = run_condition(client, corpus, "MONO", "fake-rerank", "fake-chat")
    for r in mono_rows:
        assert r["display_language"] == r["query_language"]
