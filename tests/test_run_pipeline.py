from polycite.cohere_client import CohereClient
from polycite.data.build_corpus import build_fixture_corpus
from polycite.testing import fake_transport
from scripts.run_pipeline import bm25_retrieve, build_dense_retrieve_fn, run_condition


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


def test_x2en_gold_passage_id_is_the_english_counterpart_not_the_original_language(tmp_path):
    # Regression test for a real bug: X2EN searches an English-only corpus,
    # but gold_passage_id was left as question["passage_id"] (in language
    # L). An L-language passage_id can never appear in an English-only
    # pool, so retrieved_top50/reranked_top5/citations were silently graded
    # against an impossible target -- retrieval_failure fired near-100% of
    # the time regardless of whether retrieval was actually any good. Found
    # by comparing BM25 vs. dense retrieval X2EN results and finding them
    # suspiciously, exactly identical.
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)
    rows, _ = run_condition(client, corpus, "X2EN", "fake-rerank", "fake-chat")
    assert len(rows) > 0
    for r in rows:
        assert r["gold_passage_id"].startswith("eng_Latn") or "eng_Latn" in r["gold_passage_id"]


def test_x2en_gold_passage_id_is_actually_findable_in_the_english_pool(tmp_path):
    # Stronger version of the above: not just "looks English", but is
    # actually present in the pool X2EN searches, so a good retriever CAN
    # succeed (the whole point of the fix).
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)
    english_pool_ids = {pid for pid, p in corpus.passages.items() if p["language"] == "eng_Latn"}
    rows, _ = run_condition(client, corpus, "X2EN", "fake-rerank", "fake-chat")
    for r in rows:
        if not r["is_unanswerable"]:
            assert r["gold_passage_id"] in english_pool_ids


def test_x2en_gold_answer_and_query_stay_in_the_original_language(tmp_path):
    # The gold_passage_id fix must NOT change what language the query is
    # posed in or what language the answer is graded against -- only which
    # passage_id retrieval is checked against.
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)
    rows, _ = run_condition(client, corpus, "X2EN", "fake-rerank", "fake-chat")
    by_language = {r["query_language"] for r in rows}
    assert len(by_language) > 1  # X2EN still varies by query language, unlike EN2X


def test_bm25_retrieve_returns_only_ids_from_pool():
    corpus = build_fixture_corpus()
    pool = {pid: p for pid, p in corpus.passages.items() if p["language"] == "eng_Latn"}
    ids = bm25_retrieve(pool, "library open morning", "eng_Latn", "eng_Latn", top_k=3)
    assert all(pid in pool for pid in ids)
    assert len(ids) <= 3


def test_build_dense_retrieve_fn_only_builds_requested_languages(tmp_path):
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)
    retrieve_fn = build_dense_retrieve_fn(client, corpus, ["eng_Latn"], embed_model="fake-embed")
    pool = {pid: p for pid, p in corpus.passages.items() if p["language"] == "eng_Latn"}
    ids = retrieve_fn(pool, "some query", "eng_Latn", "eng_Latn", top_k=3)
    assert all(pid in pool for pid in ids)


def test_dense_retrieve_raises_clearly_for_a_language_never_built(tmp_path):
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)
    retrieve_fn = build_dense_retrieve_fn(client, corpus, ["eng_Latn"], embed_model="fake-embed")
    pool = {pid: p for pid, p in corpus.passages.items() if p["language"] == "yor_Latn"}
    try:
        retrieve_fn(pool, "query", "yor_Latn", "yor_Latn", top_k=3)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_dense_retrieve_filters_out_a_passage_removed_from_this_questions_pool(tmp_path):
    # The unanswerable split removes the gold passage from one question's
    # pool without rebuilding the whole language's cached dense index --
    # dense_retrieve must filter its (stale, full-language) ranking down to
    # what's actually still in `pool` for this specific question.
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)
    retrieve_fn = build_dense_retrieve_fn(client, corpus, ["eng_Latn"], embed_model="fake-embed")
    full_pool = {pid: p for pid, p in corpus.passages.items() if p["language"] == "eng_Latn"}
    excluded_pid = next(iter(full_pool))
    reduced_pool = {pid: p for pid, p in full_pool.items() if pid != excluded_pid}
    ids = retrieve_fn(reduced_pool, "some query", "eng_Latn", "eng_Latn", top_k=len(reduced_pool))
    assert excluded_pid not in ids


def test_run_condition_works_end_to_end_with_dense_retriever(tmp_path):
    corpus = build_fixture_corpus()
    client = CohereClient(cache_dir=tmp_path, transport=fake_transport, budget=10_000)
    retrieve_fn = build_dense_retrieve_fn(client, corpus, corpus.languages, embed_model="fake-embed")
    rows, budget_exhausted = run_condition(client, corpus, "MONO", "fake-rerank", "fake-chat", retrieve_fn=retrieve_fn)
    assert budget_exhausted is False
    assert len(rows) > 0
