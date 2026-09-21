from polycite.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_normalizes_case_and_diacritics():
    assert tokenize("The Cat", "eng_Latn") == ["the", "cat"]


def test_tokenize_cjk_falls_back_to_char_ngrams():
    tokens = tokenize("图书馆", "zho_Hans")
    assert all(len(t) == 2 for t in tokens)
    assert len(tokens) == 2  # 3 chars -> 2 bigrams


def test_bm25_ranks_relevant_passage_first():
    passages = {
        "p1": {"language": "eng_Latn", "text": "The library opens at nine in the morning."},
        "p2": {"language": "eng_Latn", "text": "Farmers harvest rice in late summer."},
        "p3": {"language": "eng_Latn", "text": "The museum has a new painting exhibit."},
    }
    index = BM25Index(passages)
    results = index.search("what time does the library open", "eng_Latn", top_k=3)
    assert results[0][0] == "p1"


def test_bm25_search_respects_top_k():
    passages = {f"p{i}": {"language": "eng_Latn", "text": f"document number {i} about topic {i}"} for i in range(10)}
    index = BM25Index(passages)
    results = index.search("topic 3", "eng_Latn", top_k=3)
    assert len(results) == 3
