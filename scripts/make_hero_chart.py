"""Generate the README hero chart: retrieval_failure rate, BM25 vs. dense
retrieval, X2EN and EN2X, before/after the Phase 6 intervention.

The numbers here are hardcoded, not read from a live results parquet --
they're the already-published, git-committed results from HYPOTHESES.md's
"Phase 6 intervention" section (paired BM25/dense comparison, 3 languages,
10 sampled questions/language, 2026-09-22). This script exists so the
chart is regenerable/auditable from source instead of being a one-off
image with no traceable origin, not because it reads fresh data.

Usage:
    python3 scripts/make_hero_chart.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PATH = Path(__file__).parent.parent / "docs" / "assets" / "retrieval_fix_before_after.png"

# retrieval_failure rate (%), from HYPOTHESES.md "Phase 6 intervention"
# table -- BM25 baseline vs. Cohere Embed (embed-multilingual-v3.0) dense
# retrieval, same paired 10-questions/language sample per condition.
ROWS = [
    ("arb_Arab\nX2EN", 60, 0),
    ("hin_Deva\nX2EN", 60, 0),
    ("yor_Latn\nX2EN", 50, 10),
    ("arb_Arab\nEN2X", 60, 0),
    ("hin_Deva\nEN2X", 60, 0),
    ("yor_Latn\nEN2X", 44, 0),
]


def make_chart(rows: list[tuple[str, float, float]], out_path: Path) -> Path:
    labels = [r[0] for r in rows]
    bm25 = [r[1] for r in rows]
    dense = [r[2] for r in rows]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width / 2 for i in x], bm25, width, label="BM25 (lexical)", color="#C44E52")
    ax.bar([i + width / 2 for i in x], dense, width, label="Dense (Cohere Embed)", color="#4C72B0")

    for i, (b, d) in enumerate(zip(bm25, dense)):
        ax.text(i - width / 2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=9)
        ax.text(i + width / 2, d + 1.5, f"{d:.0f}%", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("retrieval_failure rate (%)")
    ax.set_ylim(0, 75)
    ax.set_title("BM25 can't cross a language boundary -- dense retrieval fixes it")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    path = make_chart(ROWS, OUT_PATH)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
