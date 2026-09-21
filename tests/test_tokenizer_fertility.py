from polycite.analysis.tokenizer_fertility import measure_fertility, word_count


def test_word_count_whitespace_languages():
    assert word_count("the quick fox", "eng_Latn") == 3


def test_word_count_cjk_counts_characters():
    assert word_count("这是中文", "zho_Hans") == 4


def test_measure_fertility_computes_language_tax(tmp_path_factory):
    from polycite.cohere_client import CohereClient

    def transport(endpoint, model, params):
        assert endpoint == "tokenize"
        # pretend non-English text costs 2x the tokens per word
        n = len(params["text"].split()) * (4 if "yor" not in params["text"] else 8)
        return {"tokens": list(range(n))}

    client = CohereClient(cache_dir=tmp_path_factory.mktemp("cache"), transport=transport)
    result = measure_fertility(
        client,
        {"eng_Latn": ["the cat sat"], "yor_Latn": ["yor text here"]},
        model="command-a",
    )
    assert result["yor_Latn"]["language_tax_vs_english"] == 2.0
