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
