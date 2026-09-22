"""H5: tokens-per-word ("language tax") via Cohere's real tokenize endpoint.

Picks 2 real, professionally-translated Belebele passages -- the same
underlying article, present in all 8 languages -- rather than fixture
placeholder text, so fertility differences reflect real content, not
fixture artifacts. 2 passages x 8 languages = 16 tokenize calls.

Usage:
    python3 scripts/measure_fertility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from polycite.analysis.tokenizer_fertility import measure_fertility
from polycite.cohere_client import CohereClient, estimate_and_confirm
from polycite.data.belebele import build_aligned_table

LANGUAGES = ["eng_Latn", "fra_Latn", "zho_Hans", "arb_Arab", "hin_Deva", "ben_Beng", "swh_Latn", "yor_Latn"]
N_PASSAGES = 2


def pick_matched_passages(df, languages: list[str], n_passages: int) -> dict[str, list[str]]:
    """n_passages question-groups (link, question_number) present in every
    language in `languages` -> {language: [passage_text, ...]}, so every
    language's texts are real translations of the exact same source article."""
    counts = df.groupby(["link", "question_number"])["language"].nunique()
    full_groups = counts[counts == len(languages)].index[:n_passages]
    if len(full_groups) < n_passages:
        raise RuntimeError(
            f"Only found {len(full_groups)} question groups present in all {len(languages)} "
            f"languages; wanted {n_passages}. Check the aligned table."
        )

    texts_by_language: dict[str, list[str]] = {lang: [] for lang in languages}
    for link, qnum in full_groups:
        subset = df[(df["link"] == link) & (df["question_number"] == qnum)]
        for lang in languages:
            row = subset[subset["language"] == lang].iloc[0]
            texts_by_language[lang].append(row["flores_passage"])
    return texts_by_language


def main():
    config = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    generation_model = config["generation"]["model"]

    df = build_aligned_table(LANGUAGES)
    texts_by_language = pick_matched_passages(df, LANGUAGES, N_PASSAGES)

    n_calls = sum(len(v) for v in texts_by_language.values())
    estimate_and_confirm(n_calls, f"tokenizer fertility: {N_PASSAGES} passages x {len(LANGUAGES)} languages")

    client = CohereClient()
    results = measure_fertility(client, texts_by_language, model=generation_model)

    print(f"\n{'language':10s}  {'tokens/word':>12s}  {'tax vs English':>15s}")
    for lang in LANGUAGES:
        r = results[lang]
        tax = r.get("language_tax_vs_english")
        tax_str = f"{tax:.2f}x" if tax is not None else "--"
        print(f"{lang:10s}  {r['tokens_per_word']:12.2f}  {tax_str:>15s}")

    print(f"\n[polycite] Cohere calls made this run's client: {client.counter.count}")


if __name__ == "__main__":
    main()
