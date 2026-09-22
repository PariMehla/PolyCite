"""Deterministic generation scoring: no LLM judge in v1.

Answer correctness is gold-recall: what fraction of the gold option text's
tokens appear in the model's free-form answer, thresholded at 0.6 — not
symmetric F1. This was an actual bug, not a design choice, found by hand-
inspecting PolyCite v1's first live run: the prompt asks for one sentence,
but the model routinely answers in a full explanatory sentence rather than
echoing the gold phrase, e.g. gold "Energy" vs. prediction "...would create
energy in the same way as stars...". Symmetric F1 penalizes every extra
word in a verbose-but-correct answer through precision, so genuinely correct
answers were scoring ~0. Recall alone (does the gold content appear
somewhere in the answer) fixed this and, spot-checked against 6 real
English MONO answers, still correctly rejects an off-topic answer that
never states the gold fact at all.

Still a known-imperfect proxy for languages without whitespace word
boundaries (zho_Hans falls back to character overlap, which is coarser) and
for morphologically rich languages where a correct paraphrase shares few
surface tokens. Flagged in the README as a v1 limitation; v2 adds LLM-judge
validation against native-speaker labels (see plan RQ4).
"""
from __future__ import annotations

import re
import string
import unicodedata


def normalize_answer(text: str, language: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    if language == "eng_Latn":
        text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str, language: str) -> list[str]:
    normalized = normalize_answer(text, language)
    if language.startswith("zho"):
        return [c for c in normalized if not c.isspace()]
    return normalized.split()


def token_f1(prediction: str, gold: str, language: str) -> float:
    pred_tokens = _tokens(prediction, language)
    gold_tokens = _tokens(gold, language)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common: dict[str, int] = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def gold_recall(prediction: str, gold: str, language: str) -> float:
    """What fraction of gold's tokens appear (with multiplicity) in prediction.
    Unlike token_f1, does not penalize a verbose-but-correct answer for
    containing extra words beyond the gold phrase."""
    pred_tokens = _tokens(prediction, language)
    gold_tokens = _tokens(gold, language)
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(pred_tokens))
    return num_same / len(gold_tokens)


def is_correct(prediction: str, gold: str, language: str, threshold: float = 0.6) -> bool:
    return gold_recall(prediction, gold, language) >= threshold


def citation_precision_recall(cited_ids: set[str], gold_passage_id: str) -> tuple[float, float]:
    if not cited_ids:
        return 0.0, 0.0
    hit = gold_passage_id in cited_ids
    precision = (1.0 / len(cited_ids)) if hit else 0.0
    recall = 1.0 if hit else 0.0
    return precision, recall


def score_generation(
    prediction_text: str,
    abstained: bool,
    cited_ids: set[str],
    gold_answer: str,
    gold_passage_id: str,
    language: str,
    is_unanswerable: bool,
) -> dict:
    if is_unanswerable:
        return {
            "correct_abstention": abstained,
            "false_abstention": None,  # not applicable: this question has no gold passage to falsely abstain from
            "answer_correct": None,
            "citation_precision": None,
            "citation_recall": None,
        }
    precision, recall = citation_precision_recall(cited_ids, gold_passage_id)
    return {
        "correct_abstention": None,
        "false_abstention": abstained,
        "answer_correct": (not abstained) and is_correct(prediction_text, gold_answer, language),
        "citation_precision": precision,
        "citation_recall": recall,
    }
