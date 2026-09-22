import pandas as pd

from scripts.compare_abstention import compare_false_abstention


def _row(question_id, condition, display_language, abstained, is_unanswerable=False):
    return {
        "question_id": question_id,
        "condition": condition,
        "display_language": display_language,
        "abstained": abstained,
        "is_unanswerable": is_unanswerable,
    }


def test_compare_false_abstention_filters_by_condition_and_language():
    baseline = pd.DataFrame(
        [
            _row("q1", "EN2X", "hin_Deva", True),
            _row("q2", "EN2X", "hin_Deva", False),
            _row("q3", "EN2X", "fra_Latn", True),  # not in the requested language subset
            _row("q4", "MONO", "hin_Deva", True),  # not in the requested condition
        ]
    )
    variant = pd.DataFrame(
        [
            _row("q1", "EN2X", "hin_Deva", False),
            _row("q2", "EN2X", "hin_Deva", False),
        ]
    )
    result = compare_false_abstention(baseline, variant, "EN2X", languages=["hin_Deva"])
    assert result["baseline"]["n"] == 2
    assert result["baseline"]["mean"] == 0.5
    assert result["variant"]["n"] == 2
    assert result["variant"]["mean"] == 0.0


def test_compare_false_abstention_excludes_unanswerable_questions():
    baseline = pd.DataFrame(
        [
            _row("q1", "EN2X", "hin_Deva", True, is_unanswerable=True),
            _row("q2", "EN2X", "hin_Deva", False, is_unanswerable=False),
        ]
    )
    variant = pd.DataFrame([_row("q2", "EN2X", "hin_Deva", False)])
    result = compare_false_abstention(baseline, variant, "EN2X", languages=["hin_Deva"])
    assert result["baseline"]["n"] == 1


def test_compare_false_abstention_pairs_when_question_ids_match_exactly():
    baseline = pd.DataFrame(
        [_row("q1", "EN2X", "hin_Deva", True), _row("q2", "EN2X", "hin_Deva", True)]
    )
    variant = pd.DataFrame(
        [_row("q1", "EN2X", "hin_Deva", False), _row("q2", "EN2X", "hin_Deva", False)]
    )
    result = compare_false_abstention(baseline, variant, "EN2X")
    assert result["paired"] is not None
    assert result["paired"]["diff"] == -1.0  # variant abstained 0%, baseline 100%


def test_compare_false_abstention_skips_pairing_when_question_sets_differ():
    baseline = pd.DataFrame([_row("q1", "EN2X", "hin_Deva", True)])
    variant = pd.DataFrame([_row("q2", "EN2X", "hin_Deva", False)])
    result = compare_false_abstention(baseline, variant, "EN2X")
    assert result["paired"] is None
