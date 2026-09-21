from polycite.data.build_corpus import build_fixture_corpus, corpus_excluding_gold


def test_fixture_corpus_has_all_languages():
    corpus = build_fixture_corpus()
    assert len(corpus.languages) == 8
    assert len(corpus.questions) == 24  # 3 articles x 8 languages


def test_unanswerable_fraction_applied():
    corpus = build_fixture_corpus(unanswerable_fraction=0.5, seed=1)
    unanswerable = [q for q in corpus.questions if q["is_unanswerable"]]
    assert len(unanswerable) == 12


def test_mono_index_only_includes_target_language():
    corpus = build_fixture_corpus()
    pool = corpus.index_for("MONO", "yor_Latn")
    assert all(p["language"] == "yor_Latn" for p in pool.values())


def test_x2en_index_is_always_english():
    corpus = build_fixture_corpus()
    pool = corpus.index_for("X2EN", "yor_Latn")
    assert all(p["language"] == "eng_Latn" for p in pool.values())


def test_mixed_index_pools_all_languages():
    corpus = build_fixture_corpus()
    pool = corpus.index_for("MIXED", "yor_Latn")
    assert pool is corpus.passages
    assert len({p["language"] for p in pool.values()}) == 8


def test_corpus_excluding_gold_removes_gold_passage_for_unanswerable_question():
    corpus = build_fixture_corpus(unanswerable_fraction=1.0, seed=1)
    question = corpus.questions[0]
    pool = corpus_excluding_gold(corpus, question, "MONO")
    assert question["passage_id"] not in pool


def test_corpus_excluding_gold_keeps_gold_passage_for_answerable_question():
    corpus = build_fixture_corpus(unanswerable_fraction=0.0, seed=1)
    question = corpus.questions[0]
    pool = corpus_excluding_gold(corpus, question, "MONO")
    assert question["passage_id"] in pool
