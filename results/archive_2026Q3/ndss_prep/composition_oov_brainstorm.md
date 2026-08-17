# Composition / OOV brainstorm (literature-driven, 2026-08-05)

Deep-research survey of composition and OOV-name mechanisms from NMT/ASR, compositional
generalization, code intelligence, entity linking, and retrieval-augmented generation,
mapped against our measured failure modes (0/1,873 novel names; oracle capture 15-38%;
composability != predictability on lmdb/jansson; prefix-forcing 9.7x existence proof;
relevance-gating law). Ranked by bet-worthiness under the 14-day deadline.

## Executive summary
Bet #1: evidence-anchored lexicon biasing (mine candidate sub-tokens/names from the target
binary's own strings/.dynsym/PLT, bias decoding with a trie + shallow fusion, GENRE/ASR-style)
— decoding-time only, generalizes the one intervention with a measured 9.7x effect, and
replaces the failed binary-level gate with per-token soft biasing that only wins where the
decoder is at floor. Bet #2: delexicalized prefix slots (train-time transform replacing
package prefixes with <PFX>, filled from binary evidence at inference) — converts prefix
forcing into a trained mechanism. Both gated on a ONE-DAY DIAGNOSTIC to run first: measure
the fraction of GT names/sub-tokens on mbedtls/lmdb/jansson/gettext appearing verbatim in
the target binary's strings + dynamic symbols (C assert/log macros embed function names via
__func__; the "answer is in the binary" yield may be high and upper-bounds ideas 1/3/4/7).
Sub-token kNN-LM: predicted null as-is (datastore = the same memorized trajectories; cf.
kNN-LM long-tail crisis, arXiv:2503.22426). Sketch decoding / data recombination: paper
roadmap, not the 2-week play.

## Ranked ideas
1. **Evidence-anchored lexicon biasing** (BET #1) — per-binary candidate lexicon from
   strings/.dynsym/PLT + composed prefixed stems; trie/WFST soft per-token bonus
   (score = log p_dec + lambda*bonus), optionally GENRE-style constrained beam head.
   Lit: GENRE (De Cao et al., ICLR'21); trie deep biasing + shallow fusion (Le et al.,
   Interspeech'21, arXiv:2104.02194); CLAS (Pundak et al., SLT'18); Post & Vilar NAACL'18;
   Hokamp & Liu ACL'17. Feasible in 2 weeks, inference-only (~200 lines in our beam search).
   Decisive: day-1 lexical-yield diagnostic; then lambda sweep on ftdomains + 7 benchmark
   pkgs; success = lift off the 0.02-0.03 floor with no benchmark regression.
   Failure mode: mining recall on string-poor binaries; lambda is a gate in disguise —
   mitigate by scaling lambda with decoder entropy.
2. **Delexicalized prefix slots** (BET #2) — training rewrites `mbedtls_ssl_read` ->
   `<PFX> ssl read` with prob p; model learns prefix-invariant stems + an explicit
   emit-a-prefix action; inference fills <PFX> from evidence. Lit: Jia & Liang ACL'16;
   GECA (Andreas, ACL'20); lexicon learning (Akyurek & Andreas, ACL'21). One data-loader
   transform + one training run. Decisive: <PFX> slot recall on FT + mbedtls/lmdb F1 vs
   benchmark regression. Failure: prefix labeling heuristics noisy on GNU names; stems
   still in-vocab only (gettext-`python` needs idea 1/3).
3. **Copy-from-strings pointer with FORCED-copy supervision** — pointer-generator over
   referenced string literals (not ext-calls), with copy supervision where GT sub-tokens
   appear in strings, and OOV-simulation (mask those sub-tokens from softmax during
   training so copy is the only path — the missing pressure that likely nulled the 2026
   ext-call copy). Lit: See et al. ACL'17; CopyNet ACL'16; Gulcehre ACL'16; Allamanis
   ICML'16. Borderline 2 weeks; first retraining experiment after 1-2. Kill-switch:
   if <10% of train functions have GT sub-tokens in referenced strings, drop.
4. **Call-graph anchor propagation** — seed test-time lexical anchors from dynsym/string
   evidence, propagate lexicon along CFG-call edges (2-pass infra exists); makes gating
   structural (fires on the connected component containing evidence). Lit: GenNm NDSS'25;
   NATURALIZE FSE'14. Thin layer over idea 1. Decisive: mbedtls with dynsym hidden as
   input, used only as anchors; F1 on non-exported fns vs graph distance to anchor.
5. **Sub-token-level kNN-LM** — per-step interpolation p = l*p_kNN + (1-l)*p_dec.
   PREDICTED NULL as-is (datastore from training = sharpened recognition; kNN-LM
   long-tail crisis). Only interesting with a test-time evidence datastore, at which
   point idea 1 is more direct. Lit: Khandelwal ICLR'20/'21; adaptive kNN-MT ACL'21.
6. **Coarse-to-fine sketch decoding** (paper roadmap) — grammar-pattern sketch (PFX VERB
   NOUN) then slot filling. Lit: Dong & Lapata ACL'18; Newman et al. JSS'20 (identifier
   grammar patterns); LANE NeurIPS'20. CHEAP PRECURSOR worth running now: oracle-sketch
   forcing on FT (force GT pattern, no lexical content) — if F1 doesn't move, structure
   is not the binding constraint and the idea dies for free.
7. **Open-set candidate pool + cross-encoder rerank** (BLINK-style) — pool = beams +
   k-NN + evidence-mined + prefix-x-stem recombinations; tiny cross-encoder scores.
   Lit: BLINK EMNLP'20; Logeswaran ACL'19; Retrieve-and-Edit NeurIPS'18. First measure
   pool recall on FT (one script; bounds everything). Coordinate with the hard-negative
   selection work to avoid duplication.
8. **Name-side subword regularization** (BPE-dropout analog on Votes tokens) — target-side
   segmentation sampling (distinct from the failed INPUT token dropout). Lit: Provilkov
   ACL'20; Kudo ACL'18. Days of work; expect wash; only as cheap pairing with idea 3.

## Cross-cutting
- Day-1 diagnostic gates ideas 1/3/4/7: lexical yield of target-binary strings+dynsym vs
  GT names, per package — one script over existing artifacts (string_lexicon, labels.json).
- Differentiation vs SymGen/GenNm (both NDSS'25, LLM-based): "evidence-anchored decoding
  recovers OOV names WITHOUT a 7B prior" is the paper sentence ideas 1/2/4 support.
- Why past nulls rhyme: copy/string-encoder/cross-attention all lacked FORCING pressure —
  the softmax path could always win. Any retraining idea must include OOV simulation or
  delexicalized slots, or it repeats the nulls.
