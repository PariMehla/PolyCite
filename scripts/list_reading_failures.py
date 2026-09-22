"""Dump every reading_failure (and citation_failure) row -- gold retrieved
correctly, answer still marked wrong -- across one or more results parquets,
for manual/LLM-judge review. Entirely local: no Cohere calls, just reading
prediction_text/gold_answer_text already saved by a previous live run.

This is the raw material for two things at once:
  1. Investigating why reading_failure is high even once retrieval mostly
     works (the new bottleneck the Phase 6 fix surfaced).
  2. A lightweight, single-judge (not native-speaker) proxy for RQ4: does an
     LLM judge reading the same text agree with the deterministic
     gold-recall scorer, or does the scorer itself miss correct answers the
     way it did for Arabic before the punctuation/article fix?

Usage:
    python3 scripts/list_reading_failures.py results/live_results.parquet results/live_dense_results.parquet
    python3 scripts/list_reading_failures.py results/live_results.parquet --language arb_Arab --n 20
"""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_paths", nargs="+")
    parser.add_argument("--language", default=None, help="filter to one display_language / query_language")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    frames = []
    for path in args.parquet_paths:
        df = pd.read_parquet(path)
        df["source_file"] = path
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    failures = df[df["failure_stage"].isin(["reading_failure", "citation_failure"])].copy()
    lang_col = "display_language" if "display_language" in failures.columns else "query_language"
    if args.language:
        failures = failures[failures[lang_col] == args.language]
    if args.condition:
        failures = failures[failures["condition"] == args.condition]

    print(f"=== {len(failures)} reading/citation failures ({', '.join(args.parquet_paths)}) ===\n")

    for _, row in failures.head(args.n).iterrows():
        print(f"[{row['source_file']}] [{row['condition']} / {row.get(lang_col)}] {row['question_id']}")
        print(f"  gold_answer_text: {row.get('gold_answer_text')!r}")
        print(f"  prediction_text:  {row.get('prediction_text')!r}")
        print(f"  citation_recall:  {row.get('citation_recall')}")
        print(f"  failure_stage:    {row['failure_stage']}")
        print()


if __name__ == "__main__":
    main()
