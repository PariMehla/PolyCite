# PolyCite v1

Where does multilingual RAG break? PolyCite measures grounded-answer quality
across 8 languages on a parallel benchmark (Belebele), and attributes every
failure to a specific pipeline stage — retrieval, reranking, reading, or
citation — instead of reporting one opaque accuracy number per language.

Pre-registered hypotheses: [`HYPOTHESES.md`](HYPOTHESES.md) (committed before
any real run, so results can't be quietly reframed to fit them).

## Status: engineering complete, first real run still pending

This repo was built in a single day inside a sandboxed Claude Code session
with **no network access to `huggingface.co` or `api.cohere.com`**
(confirmed org egress policy denials, not a bug). That means:

- Every module below is real, working code, exercised by 53 passing unit
  tests and one full dry run of the whole pipeline (retrieval -> rerank ->
  generation -> scoring -> attribution -> figures) against a small synthetic
  fixture dataset and a deterministic fake Cohere transport.
- Nobody has run it against the **real Belebele dataset** or a **real
  Cohere key** yet. `make reproduce` (dry-run) proves the wiring; the first
  person with internet and a `COHERE_API_KEY` should run `make
  reproduce-live` as PolyCite's actual first real run, expect small
  surprises, and fix them (see the checklist below).

This is the honest state of the project, not a hedge: the code you'd want
someone to have written *before* spending trial-key budget is exactly what's
here.

## Quickstart

```bash
make install          # pip install -r requirements.txt
make test              # 53 unit tests, no network or API key needed
make reproduce          # full pipeline on fixture data + a fake Cohere transport, $0, no key needed
```

To run for real:

```bash
pip install datasets                       # needed for the live Belebele loader
export COHERE_API_KEY=...                  # free trial key, dashboard.cohere.com/api-keys, no card needed
python -m polycite.data.belebele            # inspect_schema(): eyeball 3 rows before trusting anything
make reproduce-live                        # asks for confirmation before spending any calls
```

### First real run checklist
1. `python -m polycite.data.belebele` and confirm the column names in
   `EXPECTED_COLUMNS` (`polycite/data/belebele.py`) and the 8 language config
   codes in `configs/languages.yaml` still match the live dataset — both were
   transcribed from memory/documentation, not re-verified against the HF
   dataset card, because this sandbox couldn't reach it.
2. Pin exact model version strings in `configs/experiment.yaml` from your
   Cohere dashboard (`command-a-03-2025` etc. are placeholders).
3. Watch `cache/call_count.json` — it persists your spend against the
   1,000-call/month trial budget across runs and crashes.
4. Expect the free-form-answer scoring (`generate/scoring.py`, SQuAD-style
   token F1) to need threshold tuning once you see real model outputs; see
   Limitations below.

## What this measures

**Conditions** (configs/languages.yaml): MONO (local query, local corpus),
X2EN (local query, English corpus — the realistic enterprise case), EN2X
(English query, local corpus), MIXED (local query, all 8 languages pooled —
reveals retrieval language bias).

**Languages**: eng_Latn, fra_Latn, zho_Hans, arb_Arab (high/optimized tier),
hin_Deva, ben_Beng (mid tier), swh_Latn, yor_Latn (low tier).

**Pipeline**: BM25 retrieval -> Cohere Rerank -> Cohere Chat with native
`documents=` citations -> deterministic scoring -> failure attribution.

**Metrics**: Recall@k/MRR (retrieval), answer correctness + citation
precision/recall + abstention rate (generation), all with 95% bootstrap CIs
(`polycite/analysis/stats.py`, 10k resamples, paired where the comparison is
paired — Belebele is parallel across languages, so McNemar/paired-bootstrap
apply, not independent two-sample tests).

**Signature output**: a stacked bar chart of failure stage by language
(`polycite/analysis/figures.py`), attributing each failure to exactly one of:
retrieval, reranking, reading, citation, language-fidelity, or hallucinated
confidence (answering instead of abstaining on the unanswerable split).

## Repo structure

```
polycite/
  cohere_client.py      budget-capped, cached, rate-limited Cohere wrapper — THE piece to trust
  data/                 belebele.py (live loader), build_corpus.py, fixtures.py (offline)
  retrieval/             bm25.py, dense.py, hybrid.py (RRF), metrics.py
  rerank/                cohere_rerank.py
  generate/               cohere_chat.py (native citations), scoring.py (deterministic, no judge yet)
  analysis/              attribution.py, stats.py, figures.py, tokenizer_fertility.py, language_id.py
  testing.py             deterministic fake Cohere transport, shared by tests and --mode dry-run
scripts/run_pipeline.py  orchestrator: make reproduce / make reproduce-live
configs/                 languages.yaml, experiment.yaml
tests/                   53 tests, all offline/mocked
cache/ results/ figures/ gitignored contents, regenerated by the pipeline
```

## Known v1 limitations (by design, not oversight)

- **No LLM-judge grading yet.** Answer correctness is SQuAD-style token F1
  against the gold option text, thresholded at 0.5 — a real but imperfect
  proxy that degrades for CJK (character-overlap fallback) and for
  morphologically rich languages where a correct paraphrase has low token
  overlap. Judge validation against native-speaker labels (RQ4 in the full
  plan) is v2 scope.
- **Language-fidelity detection is Unicode-script-based**
  (`analysis/language_id.py`), not a real langid model (GlotLID/fastText need
  an HF download this sandbox couldn't reach). It correctly separates the 4
  non-Latin languages but cannot distinguish eng/fra/swh/yor from each other.
- **Local dense-embedding baseline and the live Belebele loader are
  untested** against real data/models — both need `huggingface.co`, unlike
  the Cohere API path, which the trial key can reach directly.
- **No MIRACL distractor corpus.** Retrieval uses Belebele's own passages per
  language (~488, same-domain hard negatives) plus the MIXED condition's
  natural pooling across 8 languages (~3,900) as the hard retrieval test,
  instead of a 20K-passage-per-language corpus — that would burn the whole
  trial-key embed budget for one experiment.
- **Generation sampled to 50 questions/language** (configs/experiment.yaml
  `budget.generation_questions_per_language`) to fit a 1,000-call/month trial
  key across 8 languages x 4 conditions; the full design calls for 300.

## Roadmap to v2 (once real numbers exist and/or a Cohere Labs Catalyst Grant lands)
- LLM-judge validation against ~480 native-speaker labels, per-language kappa.
- MIRACL-backed 20K-passage corpora per language.
- Real langid (GlotLID) for language-fidelity scoring.
- The intervention phase: dual-query retrieval / translate-pivot baseline,
  reporting "closes X% of the gap at +Y ms, +$Z/1k queries."
- Fresh, post-training-cutoff question split per language as a contamination check.

## License note
Belebele and MIRACL each carry their own license — check and credit both
before publishing any derived dataset built from real (non-fixture) data.
