import pandas as pd

from scripts.rescore_semantic import rescore


def test_rescore_flips_only_the_checked_rows_above_threshold():
    df = pd.DataFrame(
        {
            "answer_correct": [False, False, True],
            "abstained": [False, False, False],
            "is_unanswerable": [False, False, False],
            "prediction_text": ["a", "b", "c"],
            "gold_answer_text": ["x", "y", "z"],
        }
    )
    # only rows 0 and 1 were sent to the API (row 2 already correct, never checked)
    similarities_by_index = {0: 0.6, 1: 0.3}
    out = rescore(df, similarities_by_index, threshold=0.545)
    assert out.loc[0, "semantically_correct"] == True  # noqa: E712
    assert out.loc[1, "semantically_correct"] == False  # noqa: E712
    assert out.loc[2, "semantically_correct"] == True  # untouched, was already correct


def test_rescore_leaves_unchecked_rows_at_original_value():
    df = pd.DataFrame(
        {
            "answer_correct": [True, False],
            "abstained": [False, True],
            "is_unanswerable": [False, False],
            "prediction_text": ["a", "b"],
            "gold_answer_text": ["x", "y"],
        }
    )
    out = rescore(df, {}, threshold=0.545)
    assert out["semantically_correct"].tolist() == [True, False]
    assert out["semantic_similarity"].isna().all()
