# PolyCite v1

Where does multilingual RAG break? PolyCite measures grounded-answer quality
across 8 languages on a parallel benchmark (Belebele), and attributes every
failure to a specific pipeline stage — retrieval, reranking, reading, or
citation — instead of reporting one opaque accuracy number per language.

Pre-registered hypotheses: [`HYPOTHESES.md`](HYPOTHESES.md) (committed before
any real run, so results can't be quietly reframed to fit them).

## Status: first real run complete (MONO condition validated)

This repo was scaffolded in a single day inside a sandboxed Claude Code
session with **no network access to `huggingface.co` or `api.cohere.com`**
(confirmed org egress policy denials, not a bug), then run for real on a
laptop with internet and a Cohere trial key. `make reproduce` (dry-run,
fixture data + fake transport) proves the pipeline's wiring; `make
reproduce-live` has now actually run against real Belebele data and a real
Cohere key (507 calls, well under the 1,000/month trial budget).

The first live run surfaced four real bugs, all found by hand-inspecting
actual model output against gold answers (`scripts/inspect_results.py`) and
all now fixed and covered by regression tests:
- Cohere rejects document ids over 100 chars; our URL-based passage ids
  routinely exceeded that (`generate/cohere_chat.py`).
- Answer scoring used symmetric F1, which penalizes a verbose-but-correct
  answer for every extra word — switched to gold-recall
  (`generate/scoring.py`).
- Arabic-specific: ASCII-only punctuation stripping missed Arabic
  punctuation, and Arabic's attached definite article "ال" broke exact-word
  matches in both directions — both fixed with Unicode-aware normalization
  and a light per-token article strip.
- **EN2X never actually queried in English.** It relabeled the original
  non-English question as `query_language="eng_Latn"` without translating
  the text, so every EN2X row collapsed into one bucket regardless of which
  language it came from, and the "English query" was never English. Fixed
  by looking up the real English-language version of each question via
  Belebele's shared `(link, question_number)` parallel key
  (`Corpus.english_query_for`), and by adding a `corpus_language` /
  `display_language` field since EN2X's meaningful per-language axis is
  which foreign corpus was searched, not the (now-always-English) query
  language. **This one changes what's actually sent to Cohere, so re-running
  it after the fix costs new real API calls for the EN2X condition — it
  does not replay from cache like the others did.**

**MONO-condition answer correctness after all three fixes** (8 languages x
~9 answerable questions each, 95% bootstrap CIs are wide at this sample
size — see the "First real run checklist" before treating this as final):

| Tier | Language | Correct | Post-rerank Recall@5 |
|---|---|---|---|
| High | eng_Latn | 0.78 | 1.00 |
| High | zho_Hans | 0.67 | 0.89 |
| High | arb_Arab | 0.67 | 1.00 |
| High | fra_Latn | 0.50 | 1.00 |
| Mid | hin_Deva | 0.12 | 0.75 |
| Mid | ben_Beng | 0.11 | 0.78 |
| Low | swh_Latn | 0.29 | 0.86 |
| Low | yor_Latn | 0.00 | 0.56 |

Directionally consistent with H1/H2: correctness degrades roughly by tier,
and Yoruba (outside Command's core supported languages) fails completely
even on the ~half of questions where the gold passage was retrieved —
suggesting a real generation-quality gap, not just a retrieval gap, is
worth investigating for that language specifically.

**H4 (MIXED-condition language bias) looks refuted, but for an interesting
reason.** Checked with `scripts/inspect_results.py`'s language-bias section:
in the pooled 8-language MIXED corpus, retrieval essentially never crosses
languages (Arabic queries retrieve ~100% Arabic passages, Yoruba ~92%
Yoruba, etc.) — not the hypothesized English over-representation. This
makes sense once you remember retrieval here is BM25, pure lexical
matching: a query in Arabic script has ~zero token overlap with English
documents, so lexical retrieval is language-siloed rather than
English-biased. Worth re-testing with Cohere Embed (dense/semantic
retrieval) once that's wired into the pipeline, since embedding-based
retrieval might show the originally-hypothesized bias where BM25 can't.

X2EN and the full attribution breakdown have been run once; EN2X needs
re-running after the bug fix above before its numbers mean anything (see
"First real run checklist"). `results/live_results.parquet` is gitignored
and local-only, not part of this repo's history.

**Not yet checked**, so hold these loosely: French's 0.50 hasn't been
hand-inspected the way Arabic and English were, and French has the same
general risk class as Arabic (elided articles like `l'eau` glue onto the
next word); sample size per language (~9 answerable questions) gives wide
CIs — this is a v1 pilot, not the full n=300/language design.

## Quickstart

```bash
make install          # python3 -m pip install -r requirements.txt
make test              # 82 unit tests, no network or API key needed
make reproduce          # full pipeline on fixture data + a fake Cohere transport, $0, no key needed
```

To run for real:

```bash
python3 -m pip install datasets            # needed for the live Belebele loader
export COHERE_API_KEY=...                  # free trial key, dashboard.cohere.com/api-keys, no card needed
python3 -m polycite.data.belebele            # inspect_schema(): eyeball 3 rows before trusting anything
make reproduce-live                        # asks for confirmation before spending any calls
```

### First real run checklist (done, kept here as a record)
1. ✅ `python3 -m polycite.data.belebele` schema matched `EXPECTED_COLUMNS`
   exactly on the first try — no loader fix needed.
2. ✅ `command-a-03-2025` confirmed present via `client.models.list()`;
   `embed-v4.0` was NOT on the trial account's model list and was swapped
   for `embed-multilingual-v3.0`. `rerank-v3.5` worked in practice (a real
   rerank call would have crashed loudly if it didn't).
3. ✅ `cache/call_count.json` tracked spend correctly across the 3 re-runs
   it took to land on working scoring; each re-run after the first replayed
   entirely from cache (0 new API calls) since only local scoring logic
   changed, not the underlying prompts/model calls.
4. ✅ Free-form scoring needed real fixing, not just threshold tuning — see
   "Status" above and Limitations below for what was found and fixed.

If you're the next person running this fresh: rerun `make reproduce-live`
and compare against the correctness table in "Status" above before assuming
anything here still holds — Cohere's model catalog and Belebele's schema can
both change, and `results/*.parquet` is gitignored (regenerated locally,
never committed), so this README is the only durable record of what a past
run found.

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
tests/                   82 tests, all offline/mocked
cache/ results/ figures/ gitignored contents, regenerated by the pipeline
```

## Known v1 limitations (by design, not oversight)

- **No LLM-judge grading yet.** Answer correctness is gold-recall (does the
  gold option text's content appear in the free-form answer), thresholded at
  0.6 — a real but imperfect proxy that degrades for CJK (character-overlap
  fallback) and for morphologically rich languages where a correct
  paraphrase shares few surface tokens. Arabic's specific normalization gaps
  (punctuation, attached definite article) are fixed; other languages'
  equivalent gaps (e.g. French elision) have not been hand-checked. Judge
  validation against native-speaker labels (RQ4 in the full plan) is v2 scope.
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
- **Generation sampled to 10 questions/language** (configs/experiment.yaml
  `budget.generation_questions_per_language`) so 8 languages x 4 conditions
  fits comfortably inside a 1,000-call/month trial key (~640 calls); the
  full design calls for 300/language. Small n means wide bootstrap CIs —
  don't over-read single-digit-percentage differences yet.

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
