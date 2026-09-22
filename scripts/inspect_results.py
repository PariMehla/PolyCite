"""Print raw predictions vs gold answers, a full attribution breakdown, a
correctness-by-condition heatmap (as text), and a MIXED-condition language
bias check, from a results parquet. Exists to diagnose scoring/attribution
issues you can't see from the summary printout alone (e.g. "why is
correctness ~0"), and to read the cross-condition/language findings without
needing to view the PNG figures.

Usage:
    python3 scripts/inspect_results.py results/live_results.parquet
    python3 scripts/inspect_results.py results/live_results.parquet --condition MONO --language eng_Latn --n 10
"""
from __future__ import annotations

import argparse

import pandas as pd

KNOWN_LANGUAGES = ["eng_Latn", "fra_Latn", "zho_Hans", "arb_Arab", "hin_Deva", "ben_Beng", "swh_Latn", "yor_Latn"]


def passage_language(passage_id: str) -> str:
    """passage_id is "{language_code}_{url}"; language codes themselves
    contain underscores (eng_Latn), so a naive split-on-"_" doesn't work --
    match against the known set of codes instead."""
    for lang in KNOWN_LANGUAGES:
        if passage_id.startswith(lang + "_"):
            return lang
    return "UNKNOWN"


def print_correctness_heatmap(df: pd.DataFrame) -> None:
    print("\n=== Answer correctness by language x condition (text heatmap) ===")
    answerable = df[~df["is_unanswerable"]]
    pivot = answerable.groupby(["query_language", "condition"])["answer_correct"].mean().unstack()
    conditions = [c for c in ["MONO", "X2EN", "EN2X", "MIXED"] if c in pivot.columns]
    print(pivot[conditions].round(2).to_string())


def print_mixed_language_bias(df: pd.DataFrame) -> None:
    print("\n=== MIXED condition: language of the reranked_top5 passages (H4 check) ===")
    print("(if retrieval had no language bias, each language's share here should")
    print(" roughly match its share of the pooled 8-language corpus, ~12.5% each)\n")
    mixed = df[df["condition"] == "MIXED"]
    for query_lang in sorted(mixed["query_language"].unique()):
        rows = mixed[mixed["query_language"] == query_lang]
        counts: dict[str, int] = {}
        total = 0
        for reranked in rows["reranked_top5"]:
            for pid in reranked:
                lang = passage_language(pid)
                counts[lang] = counts.get(lang, 0) + 1
                total += 1
        if total == 0:
            continue
        share = {lang: round(100 * n / total, 1) for lang, n in sorted(counts.items(), key=lambda kv: -kv[1])}
        print(f"  query={query_lang:10s}  n_retrieved_passages={total:3d}  language_shares%={share}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)

    print(f"=== {len(df)} total rows, conditions={sorted(df['condition'].unique())}, "
          f"languages={sorted(df['query_language'].unique())} ===\n")

    print("=== Failure attribution, ALL conditions x languages ===")
    print(df.groupby(["condition", "query_language"])["failure_stage"]
            .apply(lambda s: s.fillna("success").value_counts().to_dict())
            .to_string())

    print_correctness_heatmap(df)
    print_mixed_language_bias(df)

    subset = df[~df["is_unanswerable"]].copy()
    if args.condition:
        subset = subset[subset["condition"] == args.condition]
    if args.language:
        subset = subset[subset["query_language"] == args.language]

    print(f"\n=== Sample predictions ({len(subset)} candidate rows, showing up to {args.n}) ===")
    for _, row in subset.head(args.n).iterrows():
        print(f"\n[{row['condition']} / {row['query_language']}] question_id={row['question_id']}")
        print(f"  gold_passage_id:  {row['gold_passage_id']}")
        print(f"  retrieved_top50 has gold: {row['gold_passage_id'] in row['retrieved_top50']}")
        print(f"  reranked_top5 has gold:   {row['gold_passage_id'] in row['reranked_top5']}")
        print(f"  gold_answer_text: {row.get('gold_answer_text')!r}")
        print(f"  prediction_text:  {row.get('prediction_text')!r}")
        print(f"  answer_correct:   {row['answer_correct']}")
        print(f"  citation_recall:  {row['citation_recall']}")
        print(f"  failure_stage:    {row['failure_stage']}")


if __name__ == "__main__":
    main()
