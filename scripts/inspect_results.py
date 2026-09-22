"""Print raw predictions vs gold answers, and a full attribution breakdown,
from a results parquet. Exists to diagnose scoring/attribution issues you
can't see from the summary printout alone (e.g. "why is correctness ~0").

Usage:
    python3 scripts/inspect_results.py results/live_results.parquet
    python3 scripts/inspect_results.py results/live_results.parquet --condition MONO --language eng_Latn --n 10
"""
from __future__ import annotations

import argparse

import pandas as pd


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
