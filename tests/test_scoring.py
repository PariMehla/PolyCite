from polycite.generate.scoring import (
    citation_precision_recall,
    gold_recall,
    is_correct,
    normalize_answer,
    score_generation,
    token_f1,
)


def test_normalize_answer_strips_articles_and_punctuation():
    assert normalize_answer("The Cat, sat.", "eng_Latn") == "cat sat"


def test_token_f1_exact_match():
    assert token_f1("nine in the morning", "nine in the morning", "eng_Latn") == 1.0


def test_token_f1_partial_overlap():
    f1 = token_f1("around nine am", "nine in the morning", "eng_Latn")
    assert 0 < f1 < 1


def test_is_correct_threshold():
    assert is_correct("nine in the morning", "nine in the morning", "eng_Latn")
    assert not is_correct("completely unrelated text", "nine in the morning", "eng_Latn")


def test_gold_recall_full_containment():
    assert gold_recall("nine in the morning", "nine in the morning", "eng_Latn") == 1.0


def test_arabic_unicode_punctuation_is_stripped():
    # Real bug: string.punctuation is ASCII-only, so the Arabic comma "،"
    # (U+060C) survived normalization and broke an otherwise exact match.
    assert normalize_answer("بريطانيا،", "arb_Arab") == "بريطانيا"


def test_arabic_definite_article_is_stripped_like_english_the():
    # Real bug found on PolyCite's first live run: gold "أداء ممتاز" vs. a
    # correct prediction containing "الأداء الممتاز" scored 0 recall because
    # Arabic's "ال" is glued onto the word, not a separate token.
    assert is_correct(
        "الأداء الممتاز لا يمكن تحقيقه فقط من خلال الممارسات الغذائية",
        "أداء ممتاز",
        "arb_Arab",
    )


def test_arabic_article_stripping_does_not_eat_short_tokens():
    # Guard against stripping "ال" down to nothing or a 1-char token.
    from polycite.generate.scoring import _strip_arabic_article

    assert _strip_arabic_article("ال") == "ال"  # the word "al" itself: don't touch
    assert _strip_arabic_article("الكتاب") == "كتاب"  # "the book" -> "book"


def test_gold_recall_ignores_extra_words_in_prediction():
    # regression test: real bug found on PolyCite's first live run. Gold
    # "Energy" vs. a verbose-but-correct model answer scored 0 under F1
    # (precision tanked by the extra words) but should score 1.0 under recall.
    recall = gold_recall(
        "A useful fusion reactor would create energy in the same way as stars.",
        "Energy",
        "eng_Latn",
    )
    assert recall == 1.0


def test_gold_recall_rejects_off_topic_answer():
    recall = gold_recall(
        "Dr. Ur's research is still in its early days, and he is skeptical about a cure.",
        "Some previously diabetic mice are no longer diabetic",
        "eng_Latn",
    )
    assert recall < 0.6


def test_is_correct_accepts_verbose_but_correct_answer_via_recall():
    # This exact pattern (verbose, correct, different word order) was
    # marked incorrect by the old F1-based scorer.
    assert is_correct(
        "All fifty of Florida's delegates were awarded to Mitt Romney.",
        "All of the state's delegates",
        "eng_Latn",
    )


def test_citation_precision_recall_single_correct_citation():
    precision, recall = citation_precision_recall({"gold_id"}, "gold_id")
    assert precision == 1.0
    assert recall == 1.0


def test_citation_precision_recall_wrong_citation():
    precision, recall = citation_precision_recall({"wrong_id"}, "gold_id")
    assert precision == 0.0
    assert recall == 0.0


def test_citation_precision_penalizes_over_citing():
    precision, recall = citation_precision_recall({"gold_id", "wrong_id"}, "gold_id")
    assert precision == 0.5
    assert recall == 1.0


def test_score_generation_unanswerable_correct_abstention():
    score = score_generation(
        prediction_text="NO_ANSWER",
        abstained=True,
        cited_ids=set(),
        gold_answer="nine",
        gold_passage_id="p1",
        language="eng_Latn",
        is_unanswerable=True,
    )
    assert score["correct_abstention"] is True
    assert score["answer_correct"] is None


def test_score_generation_hallucinated_confidence():
    # is_unanswerable=True and the model answered anyway instead of abstaining
    score = score_generation(
        prediction_text="nine in the morning",
        abstained=False,
        cited_ids={"p1"},
        gold_answer="nine",
        gold_passage_id="p1",
        language="eng_Latn",
        is_unanswerable=True,
    )
    assert score["correct_abstention"] is False


def test_score_generation_answerable_correct():
    score = score_generation(
        prediction_text="nine in the morning",
        abstained=False,
        cited_ids={"p1"},
        gold_answer="nine in the morning",
        gold_passage_id="p1",
        language="eng_Latn",
        is_unanswerable=False,
    )
    assert score["answer_correct"] is True
    assert score["citation_recall"] == 1.0
