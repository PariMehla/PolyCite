"""The two headline figures: stage-attribution stacked bars, condition heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from polycite.analysis.attribution import FAILURE_STAGES

# Colorblind-safe, ordered from "closest to the model's fault" (reading, in
# warm tones) to "closest to the pipeline's fault" (retrieval, in cool
# tones) — retrieval_failure first/bottom since it's usually the largest bar
# and the headline claim.
STAGE_COLORS = {
    "retrieval_failure": "#4C72B0",
    "reranking_failure": "#55A868",
    "reading_failure": "#C44E52",
    "citation_failure": "#8172B2",
    "language_failure": "#CCB974",
    "hallucinated_confidence": "#937860",
    "success": "#DDDDDD",
}


def plot_attribution_stacked_bar(counts: dict[str, dict[str, int]], out_path: Path | str, order: list[str] | None = None) -> Path:
    """counts: {language: {stage_or_success: count}} from analysis.attribution.attribution_counts"""
    languages = order or list(counts.keys())
    fig, ax = plt.subplots(figsize=(max(6, len(languages) * 1.1), 5))
    bottoms = [0] * len(languages)
    for stage in [*FAILURE_STAGES, "success"]:
        heights = []
        for lang in languages:
            total = sum(counts[lang].values()) or 1
            heights.append(100 * counts[lang].get(stage, 0) / total)
        ax.bar(languages, heights, bottom=bottoms, label=stage.replace("_", " "), color=STAGE_COLORS[stage])
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_ylabel("Share of questions (%)")
    ax.set_title("Failure attribution by language")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_condition_heatmap(
    values: dict[str, dict[str, float]],  # {language: {condition: correctness}}
    out_path: Path | str,
    languages: list[str] | None = None,
    conditions: list[str] | None = None,
) -> Path:
    languages = languages or list(values.keys())
    conditions = conditions or list(next(iter(values.values())).keys())
    matrix = [[values[lang].get(cond, float("nan")) for cond in conditions] for lang in languages]

    fig, ax = plt.subplots(figsize=(max(5, len(conditions) * 1.3), max(4, len(languages) * 0.6)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(languages)
    for i in range(len(languages)):
        for j in range(len(conditions)):
            v = matrix[i][j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Answer correctness")
    ax.set_title("Answer correctness by language x condition")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
