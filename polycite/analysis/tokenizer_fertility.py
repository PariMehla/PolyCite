"""Tokens-per-word ("language tax") via Cohere's real tokenize endpoint.

client.tokenize(..., offline=False) forces an API call instead of a local
tokenizer file download (blocked in this sandbox — see cohere_client.py).
It's cheap and deterministic, so it's worth spending a handful of budget
calls on: this number directly supports the enterprise cost story ("the
same content costs N times more per query in Yoruba than English").
"""
from __future__ import annotations

from polycite.cohere_client import CohereClient


def word_count(text: str, language: str) -> int:
    if language.startswith("zho"):
        return len([c for c in text if not c.isspace()])
    return len(text.split())


def measure_fertility(client: CohereClient, texts_by_language: dict[str, list[str]], model: str) -> dict[str, dict]:
    """texts_by_language: {language: [sample sentences]} -> per-language stats."""
    out: dict[str, dict] = {}
    for lang, texts in texts_by_language.items():
        total_tokens = 0
        total_words = 0
        for text in texts:
            resp = client.tokenize(text=text, model=model)
            total_tokens += len(resp["tokens"])
            total_words += max(word_count(text, lang), 1)
        out[lang] = {
            "tokens_per_word": total_tokens / total_words,
            "total_tokens": total_tokens,
            "total_words": total_words,
        }
    if "eng_Latn" in out:
        baseline = out["eng_Latn"]["tokens_per_word"]
        for lang, stats in out.items():
            stats["language_tax_vs_english"] = stats["tokens_per_word"] / baseline
    return out
