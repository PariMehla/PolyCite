"""PolyCite v1 end-to-end pipeline: retrieval -> rerank -> generation ->
scoring -> attribution -> figures.

Two modes:
  --mode dry-run (default): fixture data + a fake, deterministic Cohere
      transport (polycite.testing.fake_transport). Costs $0, needs no
      internet or API key, and exists to prove the pipeline's wiring is
      correct end to end. Output is clearly tagged synthetic; treat none of
      it as a research finding.
  --mode live: real Belebele data (needs `pip install datasets` + internet)
      and a real Cohere trial key (COHERE_API_KEY). This is the actual
      first real run of PolyCite v1 — nobody has run it yet from the
      sandbox this repo was built in, so expect and fix small surprises
      (see README "First real run checklist").

Usage:
    python scripts/run_pipeline.py --mode dry-run
    COHERE_API_KEY=... python scripts/run_pipeline.py --mode live
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from polycite.analysis import attribution, figures, stats
from polycite.analysis.language_id import detect_language_coarse
from polycite.cohere_client import BudgetExceededError, CohereClient, estimate_and_confirm
from polycite.data.build_corpus import Corpus, build_fixture_corpus, corpus_excluding_gold
from polycite.generate.cohere_chat import generate_answer
from polycite.generate.scoring import score_generation
from polycite.rerank.cohere_rerank import rerank
from polycite.retrieval.bm25 import BM25Index
from polycite.testing import fake_transport

CONDITIONS = ["MONO", "X2EN", "EN2X", "MIXED"]


def run_condition(
    client: CohereClient,
    corpus: Corpus,
    condition: str,
    rerank_model: str,
    generation_model: str,
    top_k_retrieve: int = 50,
    top_k_rerank: int = 5,
) -> tuple[list[dict], bool]:
    """Returns (rows, budget_exhausted). Stops early and returns whatever it
    has, rather than crashing, if the Cohere call budget runs out mid-loop —
    a live run that dies uncaught here would spend real trial-key calls and
    save nothing, since results are only written to disk after every
    condition finishes (see main())."""
    rows = []
    for question in corpus.questions:
        query_language = "eng_Latn" if condition == "EN2X" else question["language"]
        pool = corpus_excluding_gold(corpus, question, condition)
        if len(pool) < 2:
            continue  # fixture corpus is tiny; skip degenerate pools

        bm25 = BM25Index(pool)
        retrieved = bm25.search(question["question"], query_language, top_k=top_k_retrieve)
        retrieved_ids = [pid for pid, _ in retrieved]
        candidates = [(pid, pool[pid]["text"]) for pid in retrieved_ids]

        try:
            reranked = rerank(client, question["question"], candidates, model=rerank_model, top_n=top_k_rerank)
            reranked_ids = [pid for pid, _ in reranked]
            gen_candidates = [(pid, pool[pid]["text"]) for pid in reranked_ids]
            result = generate_answer(client, question["question"], query_language, gen_candidates, model=generation_model)
        except BudgetExceededError as e:
            print(f"\n[polycite] Cohere call budget exhausted mid-run: {e}", file=sys.stderr)
            print(f"[polycite] stopping here and saving the {len(rows)} rows already collected for condition={condition}.", file=sys.stderr)
            return rows, True

        gold_index = question["answer_index"]
        gold_answer = question["options"][gold_index]
        score = score_generation(
            prediction_text=result["text"],
            abstained=result["abstained"],
            cited_ids=result["cited_passage_ids"],
            gold_answer=gold_answer,
            gold_passage_id=question["passage_id"],
            language=query_language,
            is_unanswerable=question["is_unanswerable"],
        )

        record = {
            "question_id": question["question_id"],
            "query_language": query_language,
            "condition": condition,
            "is_unanswerable": question["is_unanswerable"],
            "gold_passage_id": question["passage_id"],
            "retrieved_top50": retrieved_ids,
            "reranked_top5": reranked_ids,
            "answer_correct": score["answer_correct"],
            "prediction_text": result["text"],
            "gold_answer_text": gold_answer,
            "cited_passage_ids": result["cited_passage_ids"],
            "response_language": detect_language_coarse(result["text"]),
            "abstained": result["abstained"],
            "false_abstention": score["false_abstention"],
            "citation_precision": score["citation_precision"],
            "citation_recall": score["citation_recall"],
        }
        record["failure_stage"] = attribution.classify_failure(record)
        rows.append(record)
    return rows, False


def summarize(rows: list[dict]) -> None:
    print("\n=== BM25 Recall@5 by language (MONO, pre-rerank -- raw retrieval quality) ===")
    for lang in sorted({r["query_language"] for r in rows if r["condition"] == "MONO"}):
        vals = [
            1.0 if r["gold_passage_id"] in r["retrieved_top50"][:5] else 0.0
            for r in rows
            if r["condition"] == "MONO" and r["query_language"] == lang and not r["is_unanswerable"]
        ]
        ci = stats.bootstrap_ci(vals, n_resamples=2000)
        print(f"  {lang:10s}  recall@5={ci['mean']:.2f}  95% CI [{ci['low']:.2f}, {ci['high']:.2f}]  n={ci['n']}")

    print("\n=== Post-rerank Recall@5 by language (MONO -- what the model actually saw) ===")
    for lang in sorted({r["query_language"] for r in rows if r["condition"] == "MONO"}):
        vals = [
            1.0 if r["gold_passage_id"] in r["reranked_top5"] else 0.0
            for r in rows
            if r["condition"] == "MONO" and r["query_language"] == lang and not r["is_unanswerable"]
        ]
        ci = stats.bootstrap_ci(vals, n_resamples=2000)
        print(f"  {lang:10s}  recall@5={ci['mean']:.2f}  95% CI [{ci['low']:.2f}, {ci['high']:.2f}]  n={ci['n']}")

    print("\n=== Answer correctness by language (MONO) ===")
    for lang in sorted({r["query_language"] for r in rows if r["condition"] == "MONO"}):
        vals = [
            1.0 if r["answer_correct"] else 0.0
            for r in rows
            if r["condition"] == "MONO" and r["query_language"] == lang and not r["is_unanswerable"]
        ]
        ci = stats.bootstrap_ci(vals, n_resamples=2000)
        print(f"  {lang:10s}  correct={ci['mean']:.2f}  95% CI [{ci['low']:.2f}, {ci['high']:.2f}]  n={ci['n']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    if args.mode == "dry-run":
        corpus = build_fixture_corpus()
        client = CohereClient(cache_dir="cache/dry-run", transport=fake_transport)
        rerank_model = "fake-rerank"
        generation_model = "fake-chat"
        print("[polycite] DRY RUN: fixture data + deterministic fake Cohere transport.")
        print("[polycite] These numbers prove the pipeline WORKS. They are not research findings.")
    else:
        from polycite.data.build_corpus import build_belebele_corpus

        languages = config.get("languages") or None
        if languages is None:
            langs_cfg = yaml.safe_load(Path("configs/languages.yaml").read_text())
            languages = [l["code"] for tier in langs_cfg["tiers"].values() for l in tier]
        n_per_lang = config["budget"]["generation_questions_per_language"]
        estimated_calls = len(languages) * n_per_lang * len(CONDITIONS) * 2  # 1 rerank + 1 chat per question per condition
        call_cap = config["budget"]["cohere_call_cap"]
        if estimated_calls > call_cap:
            print(
                f"[polycite] WARNING: this run is estimated at {estimated_calls} calls, over your "
                f"{call_cap}-call budget. It WILL stop partway through (results up to that point are "
                f"still saved) unless you lower budget.generation_questions_per_language in "
                f"{args.config}.",
                file=sys.stderr,
            )
        estimate_and_confirm(
            n_calls=estimated_calls,
            description=f"live run: {len(languages)} languages x {n_per_lang}/lang x {len(CONDITIONS)} conditions",
        )
        corpus = build_belebele_corpus(languages, sample_per_language=n_per_lang)
        client = CohereClient()
        rerank_model = config["rerank"]["model"]
        generation_model = config["generation"]["model"]

    all_rows = []
    for condition in CONDITIONS:
        rows, budget_exhausted = run_condition(client, corpus, condition, rerank_model, generation_model)
        all_rows.extend(rows)
        if budget_exhausted:
            print(f"[polycite] skipping remaining conditions after {condition}; saving partial results below.", file=sys.stderr)
            break

    if not all_rows:
        print("[polycite] No rows produced — check corpus size / conditions.", file=sys.stderr)
        return

    df = pd.DataFrame(all_rows)
    df["cited_passage_ids"] = df["cited_passage_ids"].apply(list)
    out_path = Path(config["results_dir"]) / f"{args.mode}_results.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[polycite] wrote {len(df)} rows to {out_path}")

    mono_rows = [r for r in all_rows if r["condition"] == "MONO"]
    counts = attribution.attribution_counts(mono_rows)
    fig_path = figures.plot_attribution_stacked_bar(counts, Path(config["figures_dir"]) / f"{args.mode}_attribution.png")
    print(f"[polycite] wrote {fig_path}")

    heatmap_values: dict[str, dict[str, float]] = {}
    for lang in corpus.languages:
        heatmap_values[lang] = {}
        for condition in CONDITIONS:
            vals = [
                1.0 if r["answer_correct"] else 0.0
                for r in all_rows
                if r["condition"] == condition and r["query_language"] == lang and not r["is_unanswerable"]
            ]
            heatmap_values[lang][condition] = sum(vals) / len(vals) if vals else float("nan")
    heat_path = figures.plot_condition_heatmap(heatmap_values, Path(config["figures_dir"]) / f"{args.mode}_heatmap.png", languages=corpus.languages, conditions=CONDITIONS)
    print(f"[polycite] wrote {heat_path}")

    summarize(all_rows)
    print(f"\n[polycite] Cohere calls made this run's client: {client.counter.count}")


if __name__ == "__main__":
    main()
