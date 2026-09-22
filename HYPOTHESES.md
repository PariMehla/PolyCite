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
- **H5: SUPPORTED, strongly.** Real tokenizer fertility (2 matched
  professionally-translated passages/language via `scripts/measure_fertility.py`,
  Command A's live tokenizer): yor_Latn 1.95x English's tokens/word,
  swh_Latn 2.12x — both clear the 1.5x bar. Every non-English language
  costs more: fra_Latn 1.35x, arb_Arab 1.74x, hin_Deva 2.74x, ben_Beng
  **6.23x** (the most extreme "language tax" found). zho_Hans's 0.69x is a
  measurement artifact, not real efficiency — Chinese "words" are counted
  as raw characters (no whitespace), not comparable to the other rows.

## Correction: the H3/H4 numbers above were partly a measurement bug

While testing the Phase 6 intervention (below), comparing BM25 vs. dense
retrieval for X2EN produced suspiciously *identical* retrieval_failure
counts across two completely different retrieval mechanisms. Root cause:
X2EN's `gold_passage_id` was `question["passage_id"]` (in language L), but
X2EN's pool is English-only -- an L-language passage_id can never appear
there, so retrieval was being graded against a structurally impossible
target regardless of actual quality. Fixed in `scripts/run_pipeline.py` /
`polycite/data/build_corpus.py` (see git history); this only changed local
scoring, not what was sent to Cohere, so every number below is from a
zero-new-API-calls rescore of the same underlying data.

The corrected BM25 X2EN retrieval_failure rates are still high (60% arb,
60% hin, 50% yor for the 3 languages retested) -- H3's refutation and the
H3/H4-are-the-same-finding conclusion both survive the correction, just
with more trustworthy numbers behind them. What changed is confidence, not
conclusion.

## Phase 6 intervention: does dense retrieval fix it? (2026-09-22)

Tested Cohere Embed (embed-multilingual-v3.0) as a drop-in replacement for
BM25, scoped to X2EN/EN2X on 3 languages (arb_Arab, hin_Deva, yor_Latn;
same 10 sampled questions/language as the BM25 baseline, for a fair
paired comparison) -- `scripts/run_pipeline.py --retriever dense`.

**retrieval_failure rate, BM25 -> dense (both directions, corrected):**

| Language | X2EN (BM25→dense) | EN2X (BM25→dense) |
|---|---|---|
| arb_Arab | 60% → **0%** | 60% → **0%** |
| hin_Deva | 60% → **0%** | 60% → **0%** |
| yor_Latn | 50% → **10%** | 44% → **0%** |

**Verdict: the fix works, symmetrically, and dramatically.** Dense/semantic
retrieval essentially eliminates the cross-lingual retrieval wall BM25
couldn't cross in either direction -- this demonstrates the diagnosis
above (H3/H4's real root cause) rather than just asserting it.

**But it's not the whole fix.** Answer correctness rose much less than
retrieval did (e.g. arb_Arab X2EN: 0% retrieval_failure now, but
correctness only 0.44) -- once retrieval stops being the bottleneck,
`reading_failure` becomes the dominant one: the model now has the right
document most of the time and still often answers it wrong. That's a
distinct, real finding worth its own investigation, not solved by this
intervention and out of scope for today.

## RQ4, minimally: single-LLM-judge read of the reading_failure cases (2026-09-22)

`scripts/list_reading_failures.py` dumped every `reading_failure`/
`citation_failure` row (133 total, ~25-30 distinct underlying questions
once de-duplicated across conditions) from both the BM25 baseline and the
Phase 6 dense run. This session's AI coding assistant (not a separate API
call, not a native speaker, not RQ4's full design) read each one against
its gold answer and judged correctness independently of the deterministic
scorer.
This is NOT the real RQ4 experiment -- no native-speaker labels, one judge,
small unblinded sample -- but it's a free, honest, directional data point
in the meantime.

**Three failure categories emerged, and they are not the same problem:**

1. **False abstention (NO_ANSWER on an answerable question), ~9 distinct
   cases, judge agrees these are real failures.** Notably concentrated in
   EN2X specifically (Visa, Blended Learning, Communication Theory,
   Fukushima, Machu Picchu, Cold War, South Asian cuisine all abstained
   only in EN2X, not in MONO/MIXED for the same question) -- suggestive
   that reading a foreign-language document and answering in English makes
   the model more conservative, not just less accurate. Worth testing
   directly in v2 (compare abstention rate by condition, controlling for
   difficulty).
2. **Genuinely wrong answers, the majority, judge agrees with the scorer.**
   Including several "which option is NOT an example" MCQ items where the
   model picked a different, also-plausible wrong answer than Belebele's
   labeled one (Communication Theory, Nature Tourism) -- a real reading
   failure, not a labeling ambiguity worth chasing further.
3. **Semantically correct answers the deterministic scorer still misses,
   3 clear + 2 borderline out of ~25-30 questions (roughly 10-17%
   disagreement).** All 3 clear cases are Arabic synonym pairs the
   gold-recall metric can't detect because it requires literal token
   overlap: "جذور" (root) vs. gold "سبب" (cause); "متميز" (distinguished)
   vs. gold "ممتاز" (excellent); "أكبر" (greater) vs. gold "مزيد"
   (more). Unlike the earlier Arabic punctuation/article bugs (regex-
   fixable normalization bugs), this is a **synonym-blindness limitation
   intrinsic to any literal-token-overlap metric** -- not fixable with
   another normalization patch. It would need either a real LLM judge (the
   full RQ4 design) or embedding-similarity scoring (Cohere Embed is
   already wired into this pipeline for retrieval; reusing it to score
   prediction-vs-gold semantic similarity is the natural, low-effort v2
   extension) as a supplement to, not full replacement for, literal recall.

**Implication for every correctness number reported above:** treat them as
a floor, not a precise estimate. The deterministic scorer very likely
undercounts true correctness by something in the 10-17% range found here,
concentrated in languages/answers where a correct paraphrase doesn't share
the gold answer's exact root words -- exactly the risk flagged as unchecked
for French elsewhere in this file, and probably not unique to Arabic.

## Testing the two RQ4 findings as fixes, not just diagnoses (2026-09-22)

Both findings above are open questions about model/scorer behavior, not
code defects with one correct answer -- "fixing" them means running a
small, falsifiable experiment, not editing a formula. Both experiments are
committed and runnable inside the remaining trial-key budget.

### Fix attempt 1: embedding-similarity scoring for synonym-blindness (finding #3)

`scripts/test_semantic_scoring.py` validates Cohere Embed cosine similarity
against a hand-labeled set from the judge review above: 3 known-correct-
synonym cases the deterministic scorer missed (RIGHT) vs. 3 known-
genuinely-wrong cases (WRONG), all 12 texts batched into one embed call.

**Result: clean separation.**

| Group | Cosine similarity range |
|---|---|
| RIGHT (synonym, scorer-missed) | 0.563 - 0.583 |
| WRONG (genuinely off-topic) | 0.486 - 0.526 |

No overlap; a threshold around **0.545** sits cleanly in the gap. This is a
real, demonstrated result on real data (not a synthetic sanity check), and
it validates `polycite/generate/semantic_scoring.py`'s approach: embedding
similarity is a viable supplement to literal-recall scoring for exactly the
failure mode found in the judge review.

**Not done, deliberately:** wiring this into `scoring.py`'s default
`is_correct()` path -- v1's metrics are judge-free by design (README.md's
"Engineering constraints"), and n=6 validates the *concept* (the two
groups separate), not a precise
production threshold; that needs a larger labeled set, ideally pulled from
the same judge-review process at greater scale.

**Done:** `scripts/rescore_semantic.py` (new, unit-tested, zero Cohere
calls in tests) applies `DEFAULT_SEMANTIC_THRESHOLD = 0.545` to an entire
results parquet as an explicit, separate supplement -- it rescores only
`answer_correct == False, not abstained` rows (the scorer's known failure
mode is false negatives on synonyms, not false positives, so there's no
evidence rescoring already-correct rows would change anything), batches
Cohere Embed calls at 48 pairs/call, goes through `estimate_and_confirm()`
like every other batch job, and reports literal-recall vs. semantic-
adjusted correctness side by side without mutating the original
`answer_correct` column. This is the fix, runnable end to end -- what's
still open is running it against the real live-run parquet and recording
how much of the correctness floor it closes, which needs a machine with
both the result file and remaining trial-key budget:
```
python3 scripts/rescore_semantic.py results/live_results.parquet
```

### Fix attempt 2: anti-abstention prompt for EN2X false abstention (finding #1)

`polycite/generate/prompts/answer_with_citations_anti_abstain.txt` adds one
explicit rule: a document needing translation is never by itself a reason
for `NO_ANSWER`. Run via `--prompt-variant anti_abstain`, scoped to the 4
languages where EN2X false abstention was worst in the judge review
(hin_Deva, yor_Latn, swh_Latn, ben_Beng), same sampled questions as the
original run (same seed, same `--languages`/`--conditions` scoping, so
retrieval/rerank calls hit cache -- only chat calls were new).

**Raw result:** `false_abstention=0.64`, 95% CI [0.48, 0.79], n=33.

That number alone didn't say whether the prompt helped -- it needed to be
compared against the *original* EN2X false-abstention rate for these same
4 languages, not the all-8-language EN2X baseline `summarize()` reports
(different, wider scope -- would be apples to oranges), and
`summarize()`'s false-abstention report only ever broke down by condition,
not condition+language. `scripts/compare_abstention.py` fixed that: it
filters both runs to the same condition+language scope and reports a
paired bootstrap diff when both runs cover the exact same question set.

```
python3 scripts/compare_abstention.py \
    results/live_results.parquet results/live_anti_abstain_results.parquet \
    --condition EN2X --languages hin_Deva,yor_Latn,swh_Latn,ben_Beng
```

**Result (run 2026-09-22, same 33 matched questions both runs):**

| | false_abstention | 95% CI |
|---|---|---|
| baseline (default prompt) | 0.79 | [0.64, 0.91] |
| variant (anti_abstain prompt) | 0.64 | [0.48, 0.79] |

Paired diff (variant − baseline): **−0.15**, 95% CI **[−0.27, −0.03]** —
excludes 0.

**Verdict: the anti-abstain prompt works.** One added sentence telling the
model that a document needing translation isn't a reason to abstain cut
EN2X false abstention by ~15 percentage points on these 4 languages, and
the paired bootstrap CI rules out noise as the explanation. It doesn't
close the gap (0.64 is still high -- the model still over-abstains on
EN2X relative to MONO/MIXED for the same questions), so this is a real,
partial fix, not a solved problem. Worth carrying into v2 as the default
EN2X prompt, and worth testing whether it generalizes to the other 4
languages and to X2EN, which weren't in scope for this run.
