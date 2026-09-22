"""Rescore a results parquet's wrong-but-not-abstained answers with
embedding similarity, to measure how much of the deterministic scorer's
"floor" (HYPOTHESES.md "RQ4, minimally": synonym-blind literal-recall
scoring undercounts true correctness) closes once synonyms are allowed.

Supplements, does not replace, the deterministic scorer: `answer_correct`
in the source parquet is untouched, and v1's metrics stay deterministic
(see README.md's "Engineering constraints"). This only adds a second,
explicitly-labeled column so both numbers can be reported side by side.

Only rescoring answer_correct == False rows is deliberate, not a
shortcut: the scorer's known failure mode (from the judge review) is
false negatives on synonymous answers, not false positives, so there is
no evidence spending calls on already-correct rows would change anything.

Usage:
    python3 scripts/rescore_semantic.py results/live_results.parquet
    python3 scripts/rescore_semantic.py results/live_results.parquet --threshold 0.545 --output results/live_results_semantic.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from polycite.cohere_client import CohereClient, estimate_and_confirm
from polycite.generate.semantic_scoring import DEFAULT_SEMANTIC_THRESHOLD, batch_semantic_similarities


def rescore(df: pd.DataFrame, similarities_by_index: dict[int, float], threshold: float) -> pd.DataFrame:
    """Adds `semantic_similarity` and `semantically_correct` columns.
    Rows not in similarities_by_index (already correct, abstained, or
    unanswerable -- never sent to the API) get NaN/original value."""
    df = df.copy()
    df["semantic_similarity"] = df.index.map(similarities_by_index)
    df["semantically_correct"] = df["answer_correct"]
    for idx, sim in similarities_by_index.items():
        df.loc[idx, "semantically_correct"] = sim >= threshold
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_parquet")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD)
    parser.add_argument("--output", default=None, help="write the rescored parquet here (default: don't write, just report)")
    args = parser.parse_args()

    df = pd.read_parquet(args.results_parquet)
    candidates = df[
        (df["answer_correct"] == False) & (~df["abstained"]) & (~df["is_unanswerable"])  # noqa: E712
    ]
    if len(candidates) == 0:
        print("No wrong-but-not-abstained rows to rescore.")
        return

    pairs = list(zip(candidates["prediction_text"], candidates["gold_answer_text"]))
    n_calls = (len(pairs) + 47) // 48  # EMBED_MAX_TEXTS_PER_CALL=96 -> 48 pairs/call
    estimate_and_confirm(n_calls, f"semantic rescoring of {len(pairs)} wrong answers")

    client = CohereClient()
    similarities = batch_semantic_similarities(client, pairs)
    similarities_by_index = dict(zip(candidates.index, similarities))

    rescored = rescore(df, similarities_by_index, args.threshold)

    flipped = rescored.loc[list(similarities_by_index.keys())]
    n_flipped = int((flipped["semantic_similarity"] >= args.threshold).sum())
    print(f"\n=== Semantic rescoring (threshold={args.threshold}) ===")
    print(f"  wrong-but-not-abstained rows checked: {len(pairs)}")
    print(f"  flipped to semantically_correct:      {n_flipped}")

    for label, col in [("literal recall", "answer_correct"), ("+ semantic", "semantically_correct")]:
        scored = rescored[~rescored["is_unanswerable"]]
        rate = scored[col].astype(float).mean()
        print(f"  overall correctness ({label}): {rate:.3f}")

    if args.output:
        rescored.to_parquet(args.output)
        print(f"\nWrote {args.output}")

    print(f"\n[polycite] Cohere calls made this run: {client.counter.count}")


if __name__ == "__main__":
    main()
