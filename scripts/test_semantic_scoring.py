"""Validate embedding-based semantic scoring against a small hand-labeled
set of known-correct-but-scorer-missed and known-genuinely-wrong cases,
found via manual judge review (HYPOTHESES.md "RQ4, minimally"). Batches
all 12 texts (6 pairs) into ONE embed call.

This does NOT rewire scoring.py to use semantic similarity -- it only
checks whether cosine similarity cleanly separates the two labeled groups,
which is the evidence needed before trusting any threshold for that.

Usage:
    python3 scripts/test_semantic_scoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polycite.cohere_client import CohereClient, estimate_and_confirm
from polycite.generate.semantic_scoring import _cosine

# (gold, prediction, judge_correct, note) -- gold/prediction text and judge
# label are exactly the cases from the manual review session, 2026-09-22.
LABELED_CASES = [
    (
        "سبب المشكلات",
        "إيجاد جذور المشاكل البيئية وإبطالها يُعتبر حلاً عملياً وطويل الأمد للمشكلات البيئية.",
        True,
        "arb: root vs cause",
    ),
    (
        "أداء ممتاز",
        "لا يمكن تحقيق أداء رياضي متميز من خلال الممارسات الغذائية فقط، ولكنها تؤثر بشكل كبير على الصحة العامة للرياضيين الشباب.",
        True,
        "arb: distinguished vs excellent",
    ),
    (
        "مزيد من السرعة",
        "استخدام المنفاخ بضغط أو سرعة أكبر سيساعدك على زيادة الصوت عند العزف على الأكورديون.",
        True,
        "arb: greater vs more speed",
    ),
    (
        "Some previously diabetic mice are no longer diabetic",
        "Dr. Ehud Ur's research is still in its early days, and he is skeptical about whether diabetes can be cured.",
        False,
        "eng: genuinely off-topic",
    ),
    (
        "较强的对话能力",
        "根据文档，被动物养大的孩子不太可能符合野孩子的成长背景。",
        False,
        "zho: genuinely off-topic",
    ),
    (
        "বিজ্ঞান মিউজিয়ামে যাওয়া",
        "সাংস্কৃতিক পর্যটন প্রকৃতি ভিত্তিক পর্যটন কার্যকলাপের উদাহরণ নয়।",
        False,
        "ben: different MCQ non-example",
    ),
]


def main():
    model = "embed-multilingual-v3.0"
    texts = []
    for gold, pred, _, _ in LABELED_CASES:
        texts.extend([gold, pred])

    estimate_and_confirm(1, f"semantic scoring validation: {len(LABELED_CASES)} pairs, 1 batched embed call")
    client = CohereClient()
    resp = client.embed(model=model, input_type="classification", texts=texts)
    vectors = resp["embeddings"]["float"]

    rows = []
    for i, (gold, pred, judge_correct, note) in enumerate(LABELED_CASES):
        v_gold, v_pred = vectors[2 * i], vectors[2 * i + 1]
        sim = _cosine(v_gold, v_pred)
        rows.append((judge_correct, sim, note))

    print(f"\n{'judge':>6s}  {'cosine':>7s}  note")
    for judge_correct, sim, note in sorted(rows, key=lambda r: -r[1]):
        label = "RIGHT" if judge_correct else "WRONG"
        print(f"{label:>6s}  {sim:7.3f}  {note}")

    right_sims = [s for c, s, _ in rows if c]
    wrong_sims = [s for c, s, _ in rows if not c]
    print(f"\nRIGHT-labeled range: [{min(right_sims):.3f}, {max(right_sims):.3f}]")
    print(f"WRONG-labeled range: [{min(wrong_sims):.3f}, {max(wrong_sims):.3f}]")
    if min(right_sims) > max(wrong_sims):
        midpoint = (min(right_sims) + max(wrong_sims)) / 2
        print(f"CLEAN SEPARATION -- a threshold around {midpoint:.3f} separates the two groups.")
    else:
        print("NO CLEAN SEPARATION -- cosine similarity alone doesn't reliably distinguish these groups; "
              "a single global threshold would misclassify at least one labeled case either way.")

    print(f"\n[polycite] Cohere calls made this run's client: {client.counter.count}")


if __name__ == "__main__":
    main()
