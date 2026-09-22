"""PolyCite v1 end-to-end pipeline: retrieval -> rerank -> generation ->
scoring -> attribution -> figures.

Two modes:
  --mode dry-run (default): fixture data + a fake, deterministic Cohere
      transport (polycite.testing.fake_transport). Costs $0, needs no
      internet or API key, and exists to prove the pipeline's wiring is
      correct end to end. Output is clearly tagged synthetic; treat none of
      it as a research finding.
  --mode live: real Belebele data (needs `pip install datasets` + internet)
      and a real Cohere trial key (COHERE_API_KEY).

Retriever: --retriever bm25 (default, validated) or --retriever dense
    (Cohere Embed, precomputed once per language and cached -- see
    build_dense_retrieve_fn). Dense results are written to separate files
    (*_dense_results.parquet etc.) so they never overwrite the validated
    bm25 run.

Scoping (live mode only, ignored with a warning-free no-op in dry-run
unless passed): --languages arb_Arab,hin_Deva,yor_Latn and/or --conditions
X2EN,EN2X restrict a run to specific languages/conditions -- used to test
the dense-retrieval fix for the X2EN/EN2X cross-lingual bottleneck (see
README "H3 and H4 are the same bug") without re-spending the full 8x4
budget. EN2X always needs eng_Latn loaded internally for its English-query
lookup regardless of --languages; that's handled automatically.

Usage:
    python scripts/run_pipeline.py --mode dry-run
    COHERE_API_KEY=... python scripts/run_pipeline.py --mode live
    COHERE_API_KEY=... python scripts/run_pipeline.py --mode live --retriever dense \\
        --languages arb_Arab,hin_Deva,yor_Latn --conditions X2EN,EN2X
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
from polycite.retrieval.dense import DenseIndex
from polycite.testing import fake_transport

CONDITIONS = ["MONO", "X2EN", "EN2X", "MIXED"]

# A retrieve_fn has signature (pool, query_text, query_language, corpus_language, top_k) -> ranked passage_ids


def bm25_retrieve(pool: dict, query_text: str, query_language: str, corpus_language: str, top_k: int) -> list[str]:
    return [pid for pid, _ in BM25Index(pool).search(query_text, query_language, top_k=top_k)]


def build_dense_retrieve_fn(client: CohereClient, corpus: Corpus, languages: list[str], embed_model: str):
    """Precomputes one DenseIndex per language's FULL passage pool -- built
    ONCE per language and reused across every question that searches that
    language's corpus, rather than re-embedding documents per question,
    which would blow the call budget for no reason (the same ~488-passage
    corpus is identical across all 10 questions sampled from it). Only
    builds indices for the languages actually passed in, so callers control
    embedding cost by scoping `languages` to what they're testing."""
    dense_indices: dict[str, DenseIndex] = {}
    for lang in languages:
        pool = {pid: p for pid, p in corpus.passages.items() if p["language"] == lang}
        if not pool:
            continue
        print(f"[polycite] embedding {len(pool)} passages for {lang} ({embed_model})...", file=sys.stderr)
        dense_indices[lang] = DenseIndex.from_cohere(client, pool, model=embed_model)

    def dense_retrieve(pool: dict, query_text: str, query_language: str, corpus_language: str, top_k: int) -> list[str]:
        index = dense_indices.get(corpus_language)
        if index is None:
            raise RuntimeError(
                f"No dense index built for corpus_language={corpus_language!r}. "
                f"Pass it in `languages` to build_dense_retrieve_fn (built for: {sorted(dense_indices)})."
            )
        resp = client.embed(model=embed_model, input_type="search_query", texts=[query_text])
        query_vector = resp["embeddings"]["float"][0]
        # Search past top_k then filter to what's actually in `pool`: the
        # unanswerable split may have removed one passage from the cached
        # full-language index for this specific question.
        ranked = index.search_with_vector(query_vector, top_k=top_k + 10)
        filtered = [pid for pid, _ in ranked if pid in pool]
        return filtered[:top_k]

    return dense_retrieve


def run_condition(
    client: CohereClient,
    corpus: Corpus,
    condition: str,
    rerank_model: str,
    generation_model: str,
    retrieve_fn=None,
    top_k_retrieve: int = 50,
    top_k_rerank: int = 5,
) -> tuple[list[dict], bool]:
    """Returns (rows, budget_exhausted). Stops early and returns whatever it
    has, rather than crashing, if the Cohere call budget runs out mid-loop —
    a live run that dies uncaught here would spend real trial-key calls and
    save nothing, since results are only written to disk after every
    condition finishes (see main())."""
    if retrieve_fn is None:
        retrieve_fn = bm25_retrieve
    rows = []
    for question in corpus.questions:
        if condition == "EN2X":
            # EN2X means "an English query over a foreign-language corpus" --
            # query_text must actually BE English, not the original-language
            # question relabeled as English (a real bug this fixes; see
            # README.md "Status"). Belebele is parallel, so the same
            # question exists in English under the same (link,
            # question_number) key.
            english_q = corpus.english_query_for(question)
            query_text = english_q["question"]
            query_language = "eng_Latn"
            gold_answer = english_q["options"][english_q["answer_index"]]
        else:
            query_text = question["question"]
            query_language = question["language"]
            gold_answer = question["options"][question["answer_index"]]

        corpus_language = {
            "MONO": question["language"],
            "X2EN": "eng_Latn",
            "EN2X": question["language"],  # the foreign corpus actually being searched
            "MIXED": "ALL",
        }[condition]

        pool = corpus_excluding_gold(corpus, question, condition)
        if len(pool) < 2:
            continue  # fixture corpus is tiny; skip degenerate pools

        retrieved_ids = retrieve_fn(pool, query_text, query_language, corpus_language, top_k_retrieve)
        candidates = [(pid, pool[pid]["text"]) for pid in retrieved_ids]

        try:
            reranked = rerank(client, query_text, candidates, model=rerank_model, top_n=top_k_rerank)
            reranked_ids = [pid for pid, _ in reranked]
            gen_candidates = [(pid, pool[pid]["text"]) for pid in reranked_ids]
            result = generate_answer(client, query_text, query_language, gen_candidates, model=generation_model)
        except BudgetExceededError as e:
            print(f"\n[polycite] Cohere call budget exhausted mid-run: {e}", file=sys.stderr)
            print(f"[polycite] stopping here and saving the {len(rows)} rows already collected for condition={condition}.", file=sys.stderr)
            return rows, True

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
            "corpus_language": corpus_language,
            # The one language axis to group by for a per-language comparison
            # under this condition: X2EN and MONO/MIXED vary by query
            # language (corpus_language is constant "eng_Latn"/"ALL" there),
            # EN2X varies by corpus_language (query is always English).
            "display_language": corpus_language if condition == "EN2X" else query_language,
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
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument(
        "--languages", default=None,
        help="comma-separated language codes to scope a live run to, e.g. arb_Arab,hin_Deva,yor_Latn (default: all 8)",
    )
    parser.add_argument(
        "--conditions", default=None,
        help="comma-separated conditions to scope a live run to, e.g. X2EN,EN2X (default: all 4)",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    run_conditions = args.conditions.split(",") if args.conditions else CONDITIONS
    bad = [c for c in run_conditions if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"--conditions has unknown condition(s) {bad}; must be a subset of {CONDITIONS}")

    if args.mode == "dry-run":
        corpus = build_fixture_corpus()
        dense_build_languages = corpus.languages  # full fixture set; embedding it is free (fake transport)
        if args.languages:
            target_languages = args.languages.split(",")
            corpus.questions = [q for q in corpus.questions if q["language"] in target_languages]
            corpus.languages = target_languages
        client = CohereClient(cache_dir="cache/dry-run", transport=fake_transport)
        rerank_model = "fake-rerank"
        generation_model = "fake-chat"
        retrieve_fn = bm25_retrieve if args.retriever == "bm25" else build_dense_retrieve_fn(client, corpus, dense_build_languages, "fake-embed")
        print("[polycite] DRY RUN: fixture data + deterministic fake Cohere transport.")
        print("[polycite] These numbers prove the pipeline WORKS. They are not research findings.")
    else:
        from polycite.data.build_corpus import build_belebele_corpus

        all_languages = config.get("languages") or None
        if all_languages is None:
            langs_cfg = yaml.safe_load(Path("configs/languages.yaml").read_text())
            all_languages = [l["code"] for tier in langs_cfg["tiers"].values() for l in tier]

        target_languages = args.languages.split(",") if args.languages else all_languages

        n_per_lang = config["budget"]["generation_questions_per_language"]
        n_test_questions = len(target_languages) * n_per_lang
        call_cap = config["budget"]["cohere_call_cap"]

        if args.retriever == "bm25":
            estimated_calls = n_test_questions * len(run_conditions) * 2  # 1 rerank + 1 chat per question
            estimate_description = f"live run (bm25): {len(target_languages)} languages x {n_per_lang}/lang x {len(run_conditions)} conditions"
        else:
            # Dense needs the corpora actually touched by run_conditions embedded once each,
            # plus 1 query-embed + 1 rerank + 1 chat per question. Batch size ~488 passages/
            # language / 96 per call =~ 6 embed calls/language; a rough estimate, not exact --
            # the real per-call budget check (BudgetExceededError) is what actually protects you.
            needed_dense_languages = set()
            if "X2EN" in run_conditions:
                needed_dense_languages.add("eng_Latn")
            if "EN2X" in run_conditions or "MONO" in run_conditions:
                needed_dense_languages.update(target_languages)
            if "MIXED" in run_conditions:
                needed_dense_languages.update(all_languages)
            doc_embed_estimate = len(needed_dense_languages) * 6
            estimated_calls = n_test_questions * len(run_conditions) * 3 + doc_embed_estimate  # rerank + chat + query-embed per question
            estimate_description = (
                f"live run (dense/{config['retrieval']['dense_cohere']['model']}): "
                f"{len(target_languages)} languages x {n_per_lang}/lang x {len(run_conditions)} conditions "
                f"+ ~{doc_embed_estimate} document-embed calls for {sorted(needed_dense_languages)}"
            )

        if estimated_calls > call_cap:
            print(
                f"[polycite] WARNING: this run is estimated at ~{estimated_calls} calls, over your "
                f"{call_cap}-call budget. It WILL stop partway through (results up to that point are "
                f"still saved) unless you lower budget.generation_questions_per_language in "
                f"{args.config}, or narrow --languages/--conditions.",
                file=sys.stderr,
            )
        estimate_and_confirm(n_calls=estimated_calls, description=estimate_description)

        # Build with the FULL language list regardless of --languages: EN2X needs
        # every language's real English counterpart (Corpus.english_query_for),
        # which requires eng_Latn's questions to be loaded even when eng_Latn
        # itself isn't in target_languages. Filter questions down afterward.
        corpus = build_belebele_corpus(all_languages, sample_per_language=n_per_lang)
        corpus.questions = [q for q in corpus.questions if q["language"] in target_languages]
        corpus.languages = target_languages

        client = CohereClient()
        rerank_model = config["rerank"]["model"]
        generation_model = config["generation"]["model"]
        if args.retriever == "bm25":
            retrieve_fn = bm25_retrieve
        else:
            embed_model = config["retrieval"]["dense_cohere"]["model"]
            # needed_dense_languages was computed above in the matching
            # (args.retriever == "dense") branch of the estimate step.
            retrieve_fn = build_dense_retrieve_fn(client, corpus, sorted(needed_dense_languages), embed_model)

    all_rows = []
    for condition in run_conditions:
        rows, budget_exhausted = run_condition(client, corpus, condition, rerank_model, generation_model, retrieve_fn=retrieve_fn)
        all_rows.extend(rows)
        if budget_exhausted:
            print(f"[polycite] skipping remaining conditions after {condition}; saving partial results below.", file=sys.stderr)
            break

    if not all_rows:
        print("[polycite] No rows produced — check corpus size / conditions.", file=sys.stderr)
        return

    tag = args.mode if args.retriever == "bm25" else f"{args.mode}_{args.retriever}"

    df = pd.DataFrame(all_rows)
    df["cited_passage_ids"] = df["cited_passage_ids"].apply(list)
    out_path = Path(config["results_dir"]) / f"{tag}_results.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[polycite] wrote {len(df)} rows to {out_path}")

    mono_rows = [r for r in all_rows if r["condition"] == "MONO"]
    if mono_rows:
        counts = attribution.attribution_counts(mono_rows)
        fig_path = figures.plot_attribution_stacked_bar(counts, Path(config["figures_dir"]) / f"{tag}_attribution.png")
        print(f"[polycite] wrote {fig_path}")

    heatmap_values: dict[str, dict[str, float]] = {}
    for lang in corpus.languages:
        heatmap_values[lang] = {}
        for condition in run_conditions:
            vals = [
                1.0 if r["answer_correct"] else 0.0
                for r in all_rows
                if r["condition"] == condition and r["display_language"] == lang and not r["is_unanswerable"]
            ]
            heatmap_values[lang][condition] = sum(vals) / len(vals) if vals else float("nan")
    heat_path = figures.plot_condition_heatmap(heatmap_values, Path(config["figures_dir"]) / f"{tag}_heatmap.png", languages=corpus.languages, conditions=run_conditions)
    print(f"[polycite] wrote {heat_path}")

    summarize(all_rows)
    print(f"\n[polycite] Cohere calls made this run's client: {client.counter.count}")


if __name__ == "__main__":
    main()
