import pandas as pd

from scripts.measure_fertility import pick_matched_passages


def _fake_aligned_df():
    rows = []
    for link, qnum in [("art1", 1), ("art2", 1)]:
        for lang in ["eng_Latn", "fra_Latn", "yor_Latn"]:
            rows.append({"link": link, "question_number": qnum, "language": lang, "flores_passage": f"{lang}:{link}"})
    # one group present in only 2 of 3 languages -- must be excluded
    rows.append({"link": "art3", "question_number": 1, "language": "eng_Latn", "flores_passage": "eng_Latn:art3"})
    rows.append({"link": "art3", "question_number": 1, "language": "fra_Latn", "flores_passage": "fra_Latn:art3"})
    return pd.DataFrame(rows)


def test_picks_only_groups_present_in_every_language():
    df = _fake_aligned_df()
    result = pick_matched_passages(df, ["eng_Latn", "fra_Latn", "yor_Latn"], n_passages=2)
    assert set(result.keys()) == {"eng_Latn", "fra_Latn", "yor_Latn"}
    assert len(result["eng_Latn"]) == 2
    assert "eng_Latn:art3" not in result["eng_Latn"]  # incomplete group excluded


def test_texts_are_real_matched_translations_not_relabeled():
    df = _fake_aligned_df()
    result = pick_matched_passages(df, ["eng_Latn", "fra_Latn", "yor_Latn"], n_passages=2)
    # each language's text differs (a real per-language string), same link
    assert result["eng_Latn"][0] != result["fra_Latn"][0]
    assert result["eng_Latn"][0].endswith("art1") or result["eng_Latn"][0].endswith("art2")


def test_raises_when_not_enough_full_groups_exist():
    df = _fake_aligned_df()
    try:
        pick_matched_passages(df, ["eng_Latn", "fra_Latn", "yor_Latn"], n_passages=5)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
