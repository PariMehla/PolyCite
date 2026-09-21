"""Loader for facebook/belebele -> aligned per-language parquet.

NOT YET RUN AGAINST THE LIVE DATASET. This repo was scaffolded in a sandbox
with no egress to huggingface.co (org policy denial, not a code bug — see
CLAUDE.md). Run `inspect_schema()` first thing on a machine with internet,
eyeball 3 rows, and confirm the column names and language config codes below
still match before trusting anything downstream.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from polycite.data.fixtures import LANGUAGES

BELEBELE_DATASET = "facebook/belebele"

# Expected columns per the Belebele dataset card as of last documentation
# check. VERIFY against inspect_schema() output before trusting this.
EXPECTED_COLUMNS = [
    "link",
    "question_number",
    "flores_passage",
    "question",
    "mc_answer1",
    "mc_answer2",
    "mc_answer3",
    "mc_answer4",
    "correct_answer_num",
    "dialect",
    "ds",
]


def inspect_schema(language: str = "eng_Latn", n: int = 3) -> "pd.DataFrame":
    """Load one language config and print its schema + first n rows.

    Run this interactively before anything else: `python -m polycite.data.belebele`.
    """
    from datasets import load_dataset  # local import: heavy, HF-hub dependent

    ds = load_dataset(BELEBELE_DATASET, language, split="test")
    df = ds.to_pandas()
    print(f"columns: {list(df.columns)}")
    print(f"rows: {len(df)}")
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        print(f"WARNING: expected columns not found: {missing}. Update EXPECTED_COLUMNS.")
    print(df.head(n).to_string())
    return df.head(n)


def load_language(language: str) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(BELEBELE_DATASET, language, split="test")
    df = ds.to_pandas()
    df["language"] = language
    return df


def build_aligned_table(languages: list[str] = LANGUAGES) -> pd.DataFrame:
    """One row per (link, question_number) with all languages' text joined in.

    Belebele is parallel: the same (link, question_number) key identifies the
    same underlying question/passage across every language config. This is
    what makes paired statistics (McNemar, paired bootstrap) valid later.
    """
    frames = [load_language(lang) for lang in languages]
    combined = pd.concat(frames, ignore_index=True)
    combined["passage_id"] = combined["language"] + "_" + combined["link"].astype(str)
    combined["question_id"] = (
        combined["language"] + "_" + combined["link"].astype(str) + "_" + combined["question_number"].astype(str)
    )
    return combined


def save_aligned_table(out_path: Path | str = "data/belebele_aligned.parquet", languages: list[str] = LANGUAGES) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_aligned_table(languages)
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    inspect_schema()
