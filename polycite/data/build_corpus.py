"""Build per-condition retrieval corpora and the unanswerable split.

Works on either the real Belebele-aligned table (data/belebele.py) or the
offline fixtures (data/fixtures.py) — both are normalized to plain dicts
here so the retrieval/rerank/generation code never has to know which one
it's looking at.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from polycite.data.fixtures import Question as FixtureQuestion
from polycite.data.fixtures import Passage as FixturePassage


@dataclass
class Corpus:
    passages: dict[str, dict]  # passage_id -> {passage_id, language, text}
    questions: list[dict]  # question_id -> {..., is_unanswerable}
    languages: list[str]

    def index_for(self, condition: str, query_language: str) -> dict[str, dict]:
        """Passage pool a retriever should search for this (condition, language) pair.

        MONO:  L corpus for language L
        X2EN:  English corpus, regardless of query language
        EN2X:  L corpus, query language forced to English by the caller
        MIXED: every language pooled together
        """
        if condition == "MIXED":
            return self.passages
        target_lang = "eng_Latn" if condition == "X2EN" else query_language
        return {pid: p for pid, p in self.passages.items() if p["language"] == target_lang}


def passages_from_fixtures(passages: list[FixturePassage]) -> dict[str, dict]:
    return {p.passage_id: {"passage_id": p.passage_id, "language": p.language, "text": p.text} for p in passages}


def questions_from_fixtures(questions: list[FixtureQuestion]) -> list[dict]:
    return [
        {
            "question_id": q.question_id,
            "language": q.language,
            "passage_id": q.passage_id,
            "question": q.question,
            "options": q.options,
            "answer_index": q.answer_index,
        }
        for q in questions
    ]


def passages_from_belebele_df(df) -> dict[str, dict]:
    return {
        row.passage_id: {"passage_id": row.passage_id, "language": row.language, "text": row.flores_passage}
        for row in df.itertuples()
    }


def questions_from_belebele_df(df) -> list[dict]:
    out = []
    for row in df.itertuples():
        options = [row.mc_answer1, row.mc_answer2, row.mc_answer3, row.mc_answer4]
        out.append(
            {
                "question_id": row.question_id,
                "language": row.language,
                "passage_id": row.passage_id,
                "question": row.question,
                "options": options,
                "answer_index": int(row.correct_answer_num) - 1,
            }
        )
    return out


def apply_unanswerable_split(
    corpus_passages: dict[str, dict],
    questions: list[dict],
    fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, dict], list[dict]]:
    """Mark `fraction` of questions unanswerable by removing their gold passage.

    Returns a NEW passages dict per language-scoped removal is not applied
    here (removal is condition-specific, done at index_for time via the
    `is_unanswerable` flag) — this function only decides *which* questions
    are unanswerable and stores that as metadata, so every condition sees a
    consistent set of unanswerable question IDs.
    """
    rng = random.Random(seed)
    ids = sorted(q["question_id"] for q in questions)
    rng.shuffle(ids)
    n_unanswerable = int(len(ids) * fraction)
    unanswerable_ids = set(ids[:n_unanswerable])

    annotated = []
    for q in questions:
        q = dict(q)
        q["is_unanswerable"] = q["question_id"] in unanswerable_ids
        annotated.append(q)
    return corpus_passages, annotated


def corpus_excluding_gold(corpus: Corpus, question: dict, condition: str) -> dict[str, dict]:
    """The passage pool a retriever sees for one question, honoring the
    unanswerable split by dropping that question's gold passage."""
    pool = corpus.index_for(condition, question["language"])
    if question.get("is_unanswerable"):
        pool = {pid: p for pid, p in pool.items() if pid != question["passage_id"]}
    return pool


def build_fixture_corpus(unanswerable_fraction: float = 0.15, seed: int = 42) -> Corpus:
    from polycite.data.fixtures import LANGUAGES, build_fixture_dataset

    passages, questions = build_fixture_dataset()
    passages_dict = passages_from_fixtures(passages)
    questions_list = questions_from_fixtures(questions)
    passages_dict, questions_list = apply_unanswerable_split(passages_dict, questions_list, unanswerable_fraction, seed)
    return Corpus(passages=passages_dict, questions=questions_list, languages=LANGUAGES)


def build_belebele_corpus(
    languages: list[str], unanswerable_fraction: float = 0.15, seed: int = 42, sample_per_language: int | None = None
) -> Corpus:
    """Requires internet + `pip install datasets` to reach huggingface.co.
    Not runnable in the sandbox this repo was built in — see CLAUDE.md."""
    from polycite.data.belebele import build_aligned_table

    df = build_aligned_table(languages)
    passages_dict = passages_from_belebele_df(df)
    questions_list = questions_from_belebele_df(df)
    if sample_per_language:
        rng = random.Random(seed)
        sampled = []
        for lang in languages:
            lang_qs = [q for q in questions_list if q["language"] == lang]
            rng.shuffle(lang_qs)
            sampled.extend(lang_qs[:sample_per_language])
        questions_list = sampled
    passages_dict, questions_list = apply_unanswerable_split(passages_dict, questions_list, unanswerable_fraction, seed)
    return Corpus(passages=passages_dict, questions=questions_list, languages=languages)
