"""Compare false-abstention rate between two results parquets (e.g. a
baseline default-prompt run vs. an anti_abstain-prompt run) on the same
condition and language scope.

Exists because `summarize()`'s false-abstention report groups by condition
only, not by language -- so a prompt-variant experiment run on a language
subset (e.g. --languages hin_Deva,yor_Latn,swh_Latn,ben_Beng) can't be
compared against a full-8-language baseline number without re-filtering
the baseline to the same subset first. Also does a paired bootstrap diff
when both runs cover the exact same question set, which is the only way
to tell "helped" from "noise" at n=33.

Usage:
    python3 scripts/compare_abstention.py \\
        results/live_results.parquet results/live_anti_abstain_results.parquet \\
        --condition EN2X --languages hin_Deva,yor_Latn,swh_Latn,ben_Beng
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from polycite.analysis import stats


def _filter(df: pd.DataFrame, condition: str, languages: list[str] | None) -> pd.DataFrame:
    out = df[(df["condition"] == condition) & (~df["is_unanswerable"])]
    if languages is not None:
        out = out[out["display_language"].isin(languages)]
    return out


def compare_false_abstention(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    condition: str,
    languages: list[str] | None = None,
) -> dict:
    """Independent bootstrap CIs for both runs, plus a paired diff when the
    two runs cover the exact same question_id set -- a paired estimate is
    only valid there, so `paired` is None otherwise rather than a misleading
    number from mismatched samples."""
    base_rows = _filter(baseline, condition, languages)
    var_rows = _filter(variant, condition, languages)

    base_ci = stats.bootstrap_ci(base_rows["abstained"].astype(float).tolist(), n_resamples=2000)
    var_ci = stats.bootstrap_ci(var_rows["abstained"].astype(float).tolist(), n_resamples=2000)

    result = {"baseline": base_ci, "variant": var_ci, "paired": None}

    base_by_id = base_rows.set_index("question_id")["abstained"].astype(float)
    var_by_id = var_rows.set_index("question_id")["abstained"].astype(float)
    shared = base_by_id.index.intersection(var_by_id.index)
    if len(shared) > 0 and len(shared) == len(base_by_id) == len(var_by_id):
        result["paired"] = stats.paired_bootstrap_diff(
            var_by_id.loc[shared].tolist(), base_by_id.loc[shared].tolist(), n_resamples=2000
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_parquet")
    parser.add_argument("variant_parquet")
    parser.add_argument("--condition", default="EN2X")
    parser.add_argument(
        "--languages", default=None,
        help="comma-separated, e.g. hin_Deva,yor_Latn,swh_Latn,ben_Beng (default: no language filter)",
    )
    args = parser.parse_args()

    baseline = pd.read_parquet(args.baseline_parquet)
    variant = pd.read_parquet(args.variant_parquet)
    languages = args.languages.split(",") if args.languages else None

    result = compare_false_abstention(baseline, variant, args.condition, languages)

    b, v = result["baseline"], result["variant"]
    scope = f" ({', '.join(languages)})" if languages else ""
    print(f"\n=== False abstention: {args.condition}{scope} ===")
    print(f"  baseline (default prompt)      false_abstention={b['mean']:.2f}  95% CI [{b['low']:.2f}, {b['high']:.2f}]  n={b['n']}")
    print(f"  variant  (anti_abstain prompt) false_abstention={v['mean']:.2f}  95% CI [{v['low']:.2f}, {v['high']:.2f}]  n={v['n']}")

    if result["paired"]:
        d = result["paired"]
        print(f"\n  paired diff (variant - baseline): {d['diff']:+.2f}  95% CI [{d['low']:+.2f}, {d['high']:+.2f}]  n={d['n']}")
        if d["high"] < 0:
            print("  -> variant abstains LESS than baseline (CI excludes 0) -- the anti-abstain prompt helped")
        elif d["low"] > 0:
            print("  -> variant abstains MORE than baseline (CI excludes 0) -- the anti-abstain prompt hurt")
        else:
            print("  -> CI includes 0: no clear difference at this sample size")
    else:
        print(
            "\n  (question sets differ between the two runs -- not paired; "
            "compare the independent CIs above by eye; overlapping CIs mean "
            "no clear difference)"
        )


if __name__ == "__main__":
    main()
