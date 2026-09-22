import pandas as pd

from scripts.list_reading_failures import main


def _write_fake_results(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_filters_to_reading_and_citation_failures_only(tmp_path, capsys):
    path = tmp_path / "results.parquet"
    _write_fake_results(
        path,
        [
            {
                "condition": "MONO", "display_language": "arb_Arab", "query_language": "arb_Arab",
                "question_id": "q1", "gold_answer_text": "gold1", "prediction_text": "pred1",
                "citation_recall": 1.0, "failure_stage": "reading_failure",
            },
            {
                "condition": "MONO", "display_language": "arb_Arab", "query_language": "arb_Arab",
                "question_id": "q2", "gold_answer_text": "gold2", "prediction_text": "pred2",
                "citation_recall": 1.0, "failure_stage": "citation_failure",
            },
            {
                "condition": "MONO", "display_language": "arb_Arab", "query_language": "arb_Arab",
                "question_id": "q3", "gold_answer_text": "gold3", "prediction_text": "pred3",
                "citation_recall": 0.0, "failure_stage": "retrieval_failure",
            },
        ],
    )
    import sys

    sys.argv = ["list_reading_failures.py", str(path)]
    main()
    out = capsys.readouterr().out
    assert "q1" in out
    assert "q2" in out
    assert "q3" not in out  # retrieval_failure excluded
    assert "2 reading/citation failures" in out


def test_language_filter(tmp_path, capsys):
    path = tmp_path / "results.parquet"
    _write_fake_results(
        path,
        [
            {
                "condition": "MONO", "display_language": "arb_Arab", "query_language": "arb_Arab",
                "question_id": "q_arb", "gold_answer_text": "g", "prediction_text": "p",
                "citation_recall": 1.0, "failure_stage": "reading_failure",
            },
            {
                "condition": "MONO", "display_language": "yor_Latn", "query_language": "yor_Latn",
                "question_id": "q_yor", "gold_answer_text": "g", "prediction_text": "p",
                "citation_recall": 1.0, "failure_stage": "reading_failure",
            },
        ],
    )
    import sys

    sys.argv = ["list_reading_failures.py", str(path), "--language", "arb_Arab"]
    main()
    out = capsys.readouterr().out
    assert "q_arb" in out
    assert "q_yor" not in out


def test_combines_multiple_parquet_files(tmp_path, capsys):
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    _write_fake_results(
        path_a,
        [{
            "condition": "MONO", "display_language": "arb_Arab", "query_language": "arb_Arab",
            "question_id": "q_a", "gold_answer_text": "g", "prediction_text": "p",
            "citation_recall": 1.0, "failure_stage": "reading_failure",
        }],
    )
    _write_fake_results(
        path_b,
        [{
            "condition": "X2EN", "display_language": "hin_Deva", "query_language": "hin_Deva",
            "question_id": "q_b", "gold_answer_text": "g", "prediction_text": "p",
            "citation_recall": 1.0, "failure_stage": "reading_failure",
        }],
    )
    import sys

    sys.argv = ["list_reading_failures.py", str(path_a), str(path_b)]
    main()
    out = capsys.readouterr().out
    assert "q_a" in out
    assert "q_b" in out
