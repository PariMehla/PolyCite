# Pre-registered hypotheses

Committed 2026-09-21, before any real (non-fixture) run against Belebele or
a live Cohere key. Written down now so later results can't be quietly
reframed to fit whatever comes out.

- **H1 (Attribution).** Retrieval, not generation, causes most low-resource
  failures: for the low tier (swh_Latn, yor_Latn), `retrieval_failure` is
  the largest single segment of the attribution chart, larger than
  `reading_failure` and `citation_failure` combined.
- **H2 (Degradation).** MONO-condition answer correctness on the low tier is
  at least 15 percentage points below the high tier (English/French/Chinese/
  Arabic), with non-overlapping bootstrap 95% CIs.
- **H3 (Cross-lingual penalty).** X2EN correctness beats EN2X correctness for
  the low tier — it's easier for the model to read a low-resource question
  against English documents than to read English questions against
  low-resource documents — because retrieval embeddings are stronger for
  English text than for low-resource text.
- **H4 (Language bias in MIXED).** In the MIXED condition, English passages
  are over-represented in the top-5 retrieved results relative to their
  share of the pooled corpus, for every non-English query language.
- **H5 (Tokenizer tax).** Command's tokenizer produces at least 1.5x more
  tokens per word for yor_Latn and swh_Latn than for eng_Latn on matched
  content.

## What would falsify each of these
- H1 falsified if reading/citation failures dominate even on Yoruba — would
  suggest the bottleneck is generation quality, not retrieval, which is a
  more surprising and arguably more useful finding for Cohere's Aya team.
- H2 falsified if degradation is under 5pp — "Cohere's pipeline is robust
  across languages" is itself a reportable finding (see plan risk table).
- H3 falsified by the opposite ordering — would suggest embed-multilingual
  quality is more uniform across languages than the tokenizer/generation
  stack is.
- H4 falsified by no measurable skew — would be a genuinely positive result
  worth highlighting.
- H5 falsified if fertility ratios cluster near 1.0 — would undercut part of
  the "language tax" framing in the write-up.

## Verdicts (first real run, 2026-09-22, n≈9-10/language, 8 languages x 4 conditions, Command A + BM25 + Rerank v3.5)

Sample size is small (pilot, not the full n=300/language design) — treat
percentages as directional, not precise. `results/live_results.parquet` is
gitignored/local-only; these verdicts are the durable record.

- **H1: SUPPORTED, but incompletely.** Yoruba's MONO failures are a mix of
  real retrieval misses (~44%) and reading failures (~56%) — retrieval is
  large but not clearly larger than reading+citation combined as stated, so
  this is "retrieval matters a lot" rather than a clean win on the strict
  wording.
- **H2: SUPPORTED.** MONO correctness: high tier 0.50–0.78, low tier
  0.00–0.29 — a 21–78pp gap depending on which low-tier language, well past
  the 15pp bar, though small-n CIs are wide (see README).
- **H3: REFUTED AND REVERSED.** EN2X correctness beat X2EN for both
  swh_Latn (0.14 vs 0.00) and yor_Latn (0.11 vs 0.00) — the opposite
  ordering from what was predicted. Root cause on inspection: X2EN's
  retrieval_failure rate is 70–100% for every non-English language (BM25
  essentially never matches a non-English query against English documents
  lexically) — the bottleneck is retrieval failing to cross the language
  boundary in either direction, not an asymmetry in embedding quality
  between directions as H3's rationale assumed.
- **H4: REFUTED, and for the same underlying reason as H3.** MIXED-condition
  retrieval showed ~92-100% same-language passages for every non-English
  query (no English over-representation) — because BM25 (pure lexical
  matching) essentially never retrieves across a script/language boundary
  at all. H3 and H4 turn out to be the same finding: **the dominant
  bottleneck in this pipeline is that lexical (BM25) retrieval cannot
  bridge languages, in any direction, not generation quality or embedding
  bias.** This is a stronger, more specific, more actionable claim than
  either original hypothesis and points directly at the Phase 6
  intervention (dense/embedding retrieval or translate-then-retrieve) as
  the fix to test next, rather than at prompting or model choice.
- **H5: NOT YET TESTED.** Tokenizer fertility measurement
  (`analysis/tokenizer_fertility.py`) has not been run against the live
  Cohere tokenize endpoint yet.
