"""BM25 retrieval with language-aware tokenization.

Two real pitfalls, called out because they're easy to get silently wrong:
  - CJK (zho_Hans): whitespace tokenization is meaningless for Chinese; we
    fall back to per-character n-grams, which is crude but at least gives
    BM25 overlapping tokens to score on. A real run should swap in `jieba`.
  - Diacritics (yor_Latn, and Belebele Yoruba text in general): NFC-normalize
    before lowercasing so that visually-identical strings with different
    Unicode compositions don't get treated as different tokens.
"""
from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi

_CJK_RANGE = re.compile(r"[一-鿿]")


def tokenize(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    if language.startswith("zho"):
        # crude character n-gram fallback; no CJK segmenter available offline
        chars = [c for c in text if not c.isspace()]
        return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)] or chars
    text = text.lower()
    return re.findall(r"\w+", text, flags=re.UNICODE)


class BM25Index:
    def __init__(self, passages: dict[str, dict]):
        self.passage_ids = list(passages.keys())
        self._languages = {pid: p["language"] for pid, p in passages.items()}
        corpus_tokens = [tokenize(passages[pid]["text"], passages[pid]["language"]) for pid in self.passage_ids]
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, language: str, top_k: int = 50) -> list[tuple[str, float]]:
        query_tokens = tokenize(query, language)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.passage_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
