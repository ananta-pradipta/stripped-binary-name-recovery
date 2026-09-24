# Unified Retrieval–Composition Experiment — Complete Results Report

**Date:** 2026-08-11 → 2026-08-13 · **Branch:** `unified` (commits `20b54965` … `e4e804d6`)
**Hardware:** Wulver A100 40GB (jobs 1172837, 1172878, 1173477, 1174048) + local RTX 4060
**Verdict up front:** **STOP composition — permanently.** Two independent validation rounds — a
parity-fair full-data run (Part 1, §3–§8) and a fully protocol-corrected rerun that answered
every methodological objection (Part 2, §10) — show that no composition head (linear,
lexical-residual, or fused) improves production retrieval on clean-7. In the final, maximally
conservative test (reconciled corpus, one embedding pipeline, rebuilt-and-anchored U0,
package-level out-of-fold selection through the production retrieval pipeline, and add-only
fusion that can never delete a retrieval token), fusion still **lost 0.034 macro-F1 and broke
66% of retrieval's exact-match predictions while creating zero new ones.**

---

## 1. Research question

> Can a simple multi-label name predictor, trained on exactly the same full corpus available to
> retrieval, contribute complementary correct name tokens that can be converted into a measurable
> improvement over strong identifier retrieval?

Both heads predict the same target: the ground-truth function name as an unordered canonical
token set. Retrieval (U0) predicts by copying a complete identifier from the training index;
composition (U1/U2) predicts the token set directly from the function embedding. A dual-head
architecture is justified only if a fused system (U3) beats U0 under honest selection.

Pre-registered decision criteria (user spec, 2026-08-12):
- **U2 gate:** U2 ≥ U1 + 0.02 on NOVEL_NAME_COMPOSABLE with rare/very-rare recall gains.
- **U3 gate (§12):** proceed to fusion only if composition-only correct tokens are non-trivial
  AND an oracle shows a meaningful gap (operationalized: ≥5% of functions gain ≥1 token,
  oracle gap ≥0.03).
- **Dual-head GO (§18):** U3 ≥ U0 + 0.01 ALL macro-F1 (prefer +0.02), clear NOVEL gain, no major
  SEEN loss.
- **STOP (§19):** U3 ≈ U0 with marginal NOVEL gain → close composition, no further heads.

---

## 2. Experimental setup

### 2.1 Frozen retrieval baseline (U0)
The shipped production system, completely frozen: contrastive 32M encoder → z_R (1024-d) →
cosine kNN over the full 243,289-function training index + BinFilter (Jaccard τ=0.5 over binary
ext-call fingerprints). Predictions are taken **verbatim** from the archived evaluation dump
(`results/ndss_dual_head_eval.json`, produced 2026-08-09 with two anchor asserts: decoder 0.5330,
retrieval 0.5913). No component was retrained, retuned, or re-run.

### 2.2 Evaluation set
Clean-7 cross-project packages (nginx118, angie, tengine, recutils, dash, gettext, psmisc):
**n = 13,581 functions**, none of whose packages appear anywhere in training.
Known data caveat: 2,987 rows share a (binary, name) key with another row (same-named functions
within a binary — e.g. static functions in different TUs); such rows inherit the same frozen-U0
prediction. This is a property of the archived dump's keying and affects all heads equally.

### 2.3 Canonical tokenizer (single implementation everywhere)
`semantic_modules.tokenizer.atoms`, version `v1-canonical-2026-08-10`, SHA-256(16)
`9c85684bb9d21d13`. Used for GT names, retrieved names, U1 labels, strata construction, fusion,
and every metric. Identifier-level matching uses `norm_name` (clone-suffix stripping).

### 2.4 Honest strata (rebuilt from the FULL training corpus)
Stratum of each clean-7 function, computed against all 243,289 training-index names:

| Stratum | Definition | n |
|---|---|---|
| SEEN_NAME | complete canonical identifier occurs in training | 10,151 |
| NOVEL_NAME_COMPOSABLE | identifier unseen; **all** GT tokens seen | 1,509 |
| LEXICAL_OOV | identifier unseen; ≥1 GT token unseen | 1,762 |
| FULL_LEXICAL_OOV | all GT tokens unseen (subset of LEXICAL_OOV) | 159 |
| **Total** | | **13,581** ✓ |
| RETR_FAIL | frozen-U0 token-F1 == 0 exactly, current tokenizer | 4,217 (overlay) |

RETR_FAIL was **redefined** for this study: mean U0 F1 on the subset is 0.0000 by construction
(asserted per-row). An earlier report showed 0.004 because it reused the dump's `f1R` field,
computed with an older tokenizer version. LEXICAL_OOV rows in the results tables below include
the FULL_LEXICAL_OOV subset (n=1,921 total).

An earlier in-script strata variant (drawn from a 75K composer-train subsample) inflated
NOVEL_NAME_COMPOSABLE to n=4,336 with U0 ≈ 0.667 — those names were "novel" only w.r.t. the
subsample while remaining verbatim-retrievable from the full index. **Only the full-corpus strata
above should be cited.**

### 2.5 Training-data parity (the audit that motivated the re-run)

| Quantity | U0 (retrieval index) | U1 as previously run | U1 in this study |
|---|---|---|---|
| Functions | 243,289 | 75,136 | **228,369** |
| Unique canonical names | 40,808 | — | 37,663 |
| Unique name tokens | 11,479 | 8,994 | 10,695 |

The previous U1 was trained on a ~75K random subsample — an unfair comparison. In this study U1
trains on the full corpus; the **only** exclusion is the 6-package dev split (14,920 functions:
strace 6,961, libsodium 2,263, zlib 2,162, texinfo 1,568, cflow 1,118, sed 848), required for
package-disjoint selection. Exactness of the corpus identification was proven by reproducing the
dump's own `random.Random(1234).sample(train, 80000)` fit subsample from recovered metadata —
`fit_idx`/`val_idx` match the archived cache element-for-element.

### 2.6 Selection protocol (no clean-7 leakage)
All model selection — checkpoint epoch, threshold τ, class weighting (pos_weight ∈ {1, 3, 5}) and
the U3 fusion parameters (λ, τ) — was performed **only** on the package-disjoint dev split.
Normalization statistics (z-score μ/σ) come from U1-train rows only. Clean-7 was evaluated
exactly once per frozen configuration. Vocabulary (|V| = 10,695) comes from U1-train names only.

### 2.7 Embedding provenance and the space check
The full-corpus z_R was freshly embedded with the frozen production checkpoint
(`best_model_cont_control.pt`) on the A100 and persisted
(`strlex_ws/results/ztr_full_control.npz`: embs + names + binaries, 862 MB).
Fresh embeddings drift from the Aug-9 cached probe sample (mean cos 0.9706; ~75% of functions
below 0.99 in nearly every package) because the preprocessing pipeline changed between Aug 9 and
Aug 12. The **functional** space check is authoritative: plain cosine top-1 retrieval from the
fresh index reproduces the frozen dump's final predictions on **86.5%** of clean-7 (a
deliberately-wrong-space control index scores 4.3%; the residual is BinFilter + tie-breaks).
Any remaining train/query pipeline skew biases **against** U1; at the observed margins (§4) this
cannot change any conclusion.

Hardware determinism control: the corrected U0/U1/U2 gate was executed independently on a local
RTX 4060 and on the A100 (job 1172837) — **identical results to 4 decimal places** on every
metric.

---

## 3. Phase 1 — Unified gate: U0 vs U1 vs U2 (seed 42, subsampled corpus)

Purpose: first execution of the unified composer comparing retrieval, a plain linear multi-label
head, and the unified lexical-residual composer (U2: frozen Phase-1 char-encoder embedding e(t) +
learnable per-token residual δ with frequency-weighted L2, q = MLP(z_R), cosine scoring).

### 3.1 Two implementation bugs found on first execution (both fixed, commit `20b54965`)
1. **U2 logit scaling.** With L2-normalized q and v, q·v ∈ [−1, 1]; the score divided by √256
   then ×10, compressing logits to ±0.625 around the per-token bias. The frequency-fitted bias
   dominated: every function received ~515 tokens, F1 ≈ 0.001. Fix: drop the √D division
   (it belongs to unnormalized attention). Post-fix U2 predicts 9.9 tokens/function and trains.
2. **Disabled oracle.** The complementarity ceiling contained
   `union & gt if False else union` — reporting plain-union F1 (meaningless with a noisy
   partner) instead of the documented precision-1 GT-intersection. Enabled.

### 3.2 Corrected results (clean-7, honest strata, macro-F1)

| Model | seen (10,151) | novel_comp (1,509) | oov (1,921) | ALL | RETR_FAIL |
|---|---|---|---|---|---|
| U0 retrieval | **0.757** | 0.085 | 0.084 | **0.587** | 0.004* |
| U1 linear (75K, 8,994 vocab) | 0.377 | 0.091 | 0.105 | 0.307 | 0.069 |
| U2 lexical-residual (fixed) | 0.176 | 0.069 | 0.062 | 0.148 | 0.040 |

\* dump-tokenizer f1R basis; exactly 0 under the redefinition of §2.4.

Token-recall by training-frequency band (frequent ≥22 / medium 8–21 / rare 5–7 / very-rare 1–4):
U1 = .458/.071/.054/.111 · U2 = .412/.033/.038/.019.

**U2 verdict: NO-GO, decisively.** U2 needed ≥ U1+0.02 on NOVEL; it is −0.14, and it loses the
rare and very-rare bands — the exact bands the frequency-regularized shared-embedding residual
was designed to win. The shared char-embedding space is not degenerate (pairwise cos: mean 0.133,
p99 0.485), so this is a genuine negative for the architecture, not an artifact. Confirmed
bit-identically on two GPUs.

---

## 4. Phase 2 — Final validation: full-data U1 (3 seeds)

U1 = Linear(1024 → 10,695) + sigmoid, BCE, AdamW lr 1e-3, wd 1e-4, batch 2048, ≤25 epochs.
Seeds 42/123/7. All three seeds independently select pos_weight 5.0, τ = 0.40, epoch 24
(dev macro-F1 0.1655 / 0.1657 / 0.1661). Note: pos_weight is selected at the grid edge and dev
selection is monotone in it — a larger value might marginally shift dev numbers; given §5–§6 the
margins make this immaterial.

### 4.1 Main table (clean-7 macro-F1; micro-F1 and EM in parentheses)

| Model | ALL | SEEN_NAME | NOVEL_COMP | LEXICAL_OOV | FULL_OOV | RETR_FAIL |
|---|---|---|---|---|---|---|
| U0 | 0.587 (.663 / .490) | 0.757 (.815 / .655) | 0.085 (.108 / 0) | 0.084 (.104 / 0) | 0 | 0 |
| U1 s42 | 0.320 (.367 / .019) | 0.395 (.437 / .025) | 0.093 (.112 / 0) | 0.102 (.118 / 0) | 0 | 0.054 (.065 / .005) |
| U1 s123 | 0.316 (.363 / .016) | 0.390 (.430 / .022) | 0.092 (.113 / 0) | 0.102 (.119 / 0) | 0 | 0.052 (.066 / .000) |
| U1 s7 | 0.313 (.358 / .018) | 0.387 (.425 / .024) | 0.093 (.110 / 0) | 0.099 (.119 / 0) | 0 | 0.048 (.064 / .000) |
| **U1 mean±std** | **0.3165 ± 0.0027** | 0.3906 ± 0.0035 | **0.0927 ± 0.0003** | 0.1011 ± 0.0013 | 0 ± 0 | 0.0513 ± 0.0025 |

### 4.2 Token recall by frequency band

| Model | frequent | medium | rare | very_rare |
|---|---|---|---|---|
| U0 | .7404 | .6005 | .6465 | .3914 |
| U1 s42 | .4680 | .0564 | .1238 | .0974 |
| U1 s123 | .4627 | .0601 | .1238 | .1054 |
| U1 s7 | .4635 | .0516 | .1131 | .0945 |
| **U1 mean±std** | .4647 ± .0023 | .0560 ± .0035 | .1202 ± .0050 | .0991 ± .0046 |

### 4.3 The parity question, answered
Full data (228,369 fns, 3.04× the subsample, +1,701 vocabulary tokens) moved:
**ALL +0.010, NOVEL_NAME_COMPOSABLE +0.002** (0.091 → 0.0927). Seed variance on NOVEL is ±0.0003.
The spec's §20 contingency is confirmed: **the 75K subsample was never the bottleneck. Output-side
multi-label reformulation does not learn to compose novel names, at any tested data scale.**
This eliminates the last data-side explanation for the composition failure.

---

## 5. Complementarity and oracle analyses (D)

Per-function sets: GT, R = U0 tokens, C = U1 tokens (seed 42 shown; seeds agree to 3 decimals).

### 5.1 Unique-contribution analysis

| Subset | % fns with ≥1 comp-only correct token | comp-only correct /fn | retr-only correct /fn |
|---|---|---|---|
| ALL (13,581) | 7.89% | 0.092 | 1.168 |
| NOVEL_COMP (1,509) | 15.37% | 0.168 | 0.131 |
| RETR_FAIL (4,217) | 14.25% | 0.173 | 0 (by def.) |

Retrieval contributes **12.7× more** unique correct tokens than composition overall. Composition's
unique contribution concentrates where retrieval fails, but at ~0.17 tokens/function.

### 5.2 Correct-Token Union Oracle (precision = 1 by construction — an upper bound, NOT a model)

| Subset | U0 correct-only F1 | U1 correct-only F1 | U0+U1 union F1 | gap over U0 |
|---|---|---|---|---|
| ALL | 0.6152 | 0.4301 | 0.6498 | **+0.035** |
| NOVEL_COMP | 0.1227 | 0.1471 | 0.1920 | +0.069 |
| RETR_FAIL | 0 | 0.0865 | 0.0865 | +0.087 |

### 5.3 Best-Head Oracle (per-function max of real F1s — selection ceiling)

| Subset | U0 real F1 | Best-head oracle | gap |
|---|---|---|---|
| ALL | 0.587 | 0.6133 | **+0.026** |
| NOVEL_COMP | 0.085 | 0.1267 | +0.042 |
| RETR_FAIL | 0 | 0.0538 | +0.054 |

Both ceilings are small in absolute terms, and they are **ceilings**: the union oracle assumes an
impossible perfect-precision filter; the best-head oracle assumes a perfect per-function router —
a construct already shown unlearnable in this setting (learned-gate study, 2026-08-07: AUC 0.671
yet negative at every operating threshold under package shift).

---

## 6. U3 — simple global fusion (the pre-registered final test)

Gate check (§12): 7.89% ≥ 5% ✓ and union gap 0.0346 ≥ 0.03 ✓ → fusion ran per protocol.

Design (per spec, deliberately minimal): S(t) = λ·r(t) + (1−λ)·c(t); r(t) = 1 iff t ∈ tokens of
the top-1 retrieved name (frozen dump for clean-7; plain package-disjoint cosine top-1 for dev);
c(t) = U1 seed-42 sigmoid probabilities. λ ∈ {0.0 … 1.0 step 0.1} × τ ∈ {0.05 … 0.90 step 0.05}
swept **once** on dev; the single best dev setting evaluated **once** on clean-7.
No router, no per-stratum rules, no neural fusion.

**Dev selection:** λ = 0.2, τ = 0.35 (dev macro-F1 0.1901). **Clean-7 result:**

| Model | ALL | SEEN_NAME | NOVEL_COMP | LEXICAL_OOV | RETR_FAIL |
|---|---|---|---|---|---|
| U0 | **0.587** | **0.757** | 0.085 | 0.084 | 0 |
| U1 (s42) | 0.320 | 0.395 | 0.093 | 0.102 | 0.054 |
| U3 | 0.4398 | 0.5531 | 0.1012 | 0.1067 | 0.0454 |

(U3 micro-F1 ALL 0.5121, EM 0.0511 — vs U0 0.6633 / 0.4897.)

**U3 is 0.147 below U0.** The failure mechanism is fully quantified by the saved dev sweep
(`fusion_dev_sweep.tsv` — best dev F1 per λ):

| λ | 0.0 | 0.1 | **0.2** | 0.3 | 0.4–0.9 | 1.0 |
|---|---|---|---|---|---|---|
| best dev F1 | .1655 | .1766 | **.1901** | .1859 | .1838 | .1535 |

Dev ranks **pure retrieval (λ=1.0) dead last** — package-disjoint dev retrieval is genuinely weak
(these are held-out packages queried cross-package) — while production clean-7 retrieval scores
0.587. Dev selection is therefore *anti-correlated* with test behavior: any (λ, τ) chosen on a
package-disjoint dev discounts retrieval exactly where the deployed system is strongest. This is
the **third independent instance of the same prior-shift failure** (learned routing gate,
2026-08-07; retrieval-augmented decoder, 2026-08-07; global λ-fusion, this study), now
demonstrated with the simplest possible fusion family and a fully pre-registered protocol.

---

## 7. Decision against the pre-registered criteria

| Criterion | Threshold | Observed | Outcome |
|---|---|---|---|
| U2 gate | ≥ U1+0.02 NOVEL, rare recall up | −0.14, rare recall down | **NO-GO** |
| U3 gate (§12) | ≥5% adds ∧ oracle gap ≥0.03 | 7.89% / +0.035 | passed → fusion ran |
| Dual-head GO (§18) | U3 ≥ U0+0.01, NOVEL gain, SEEN held | U3 = U0 − 0.147; NOVEL +0.016; SEEN −0.204 | **FAIL** |
| STOP (§19) | U3 ≈ U0, marginal NOVEL | exceeded (U3 ≪ U0) | **STOP** |

**Answer to the research question: No.** Complementary correct tokens exist (7.9% of all
functions; 15.4% of novel-name functions) but cannot be converted into an aggregate improvement:
the honest ceilings are +0.026 (selection) to +0.035 (precision-1 union), and every realizable
selection mechanism tested across this research programme — learned gate, retrieve-and-edit
decoder, and now dev-tuned global fusion — lands **below** pure retrieval under cross-project
shift. Composition is closed. Per §25, the paper's contribution should center on retrieval
coverage/generalization (the coverage-boundary framing), not added architecture.

Corroborating context from the wider programme: the model is a pure recognizer (0/1,873 unseen
names ever produced by the decoder); function-specific observable evidence covers only ~1–4% of
functions (Item-A census correction); dynamic-memory/set-decoder variants improved tail recall
but not aggregate F1.

---

## 8. Threats to validity (disclosed)

1. **Pipeline drift between index and queries.** Fresh train z vs Aug-9 cached query z differ
   for ~75% of functions at mean cos 0.97 (preprocessing moved). Functional agreement is 86.5%;
   the skew biases against U1. It cannot explain a 0.27 ALL gap or a 0.147 fusion deficit; a
   query re-embed would be required only if any result had been within ~0.02 of a gate.
2. **pos_weight at grid edge** (5.0), dev-monotone. Could shift U1 dev numbers slightly; NOVEL
   variance across seeds (±0.0003) and the size of the gaps make this immaterial.
3. **Duplicate (binary, name) keys** in clean-7 (2,987 rows) inherit shared frozen-U0 rows —
   affects all heads identically; inherited from the archived production dump.
4. **Dev retrieval for U3 tuning is plain top-1 without BinFilter**, while clean-7 r(t) comes
   from the BinFiltered dump — the mismatch *flatters* dev retrieval quality if anything, and the
   observed failure direction is the opposite.
5. **U3 uses seed-42 U1 only** (per protocol simplicity); seed spread elsewhere is ≤0.003.
6. **Single dev composition** (6 packages, strace-heavy at 47%). A different dev could pick a
   different λ — but that is precisely the finding: no package-disjoint dev predicts clean-7
   fusion behavior, and clean-7 itself may not be touched for selection.

---

## 9. Reproducibility & artifacts

| Artifact | Location (repo `results/dualhead_final/` unless noted) |
|---|---|
| Parity report | `training_parity.json` |
| Tokenizer audit | `tokenizer_audit.json` |
| Strata + per-function U0 F1 | `strata_full_training.tsv`, `strata_audit.json` |
| Frozen U0 dump (tokenized) | `u0_frozen.tsv` |
| U1 per-seed predictions | `u1_seed{42,123,7}.tsv` |
| U1 full metrics (micro/macro/EM/bands/selection) | `u1_summary.json`, `u1_summary.tsv` |
| Complementarity + both oracles | `complementarity.json`, `complementarity.tsv` |
| U3 gate decision | `u3_gate.json` |
| Full dev fusion sweep | `fusion_dev_sweep.tsv` |
| U3 predictions + metrics | `u3_final.tsv`, `u3_final.json`, `final_comparison.tsv` |
| Phase-1 gate (corrected U0/U1/U2) | `results/unified_gate.json`, `results/unified_predictions.tsv`, `results/unified_posthoc.json`, `results/unified_dh_strata.json` |
| Full-corpus production-space embeddings | Wulver `strlex_ws/results/ztr_full_control.npz` (862 MB) |
| Job logs | Wulver `strlex_ws/results/{unified.1172837,dualhead.1173477}.out` |
| Code | `experiments_semantic/unified_composer.py`, `unified_composer_local.py`, `unified_posthoc.py`, `dualhead_final_full.py` |

Determinism: seeds fixed (42/123/7; corpus sampling rng 1234); identical results verified on
RTX 4060 vs A100 to 4 decimals for the Phase-1 gate; per-run EFFECT asserts (split reproduction,
functional space check ≥0.80, RETR_FAIL ≡ 0, strata sum = 13,581) abort rather than report.

---

# Part 2 — Protocol-Corrected Final Validation (2026-08-13, job 1174048)

Part 1's STOP was challenged on five protocol grounds: an unresolved corpus discrepancy
(241,174 vs 243,289), residual embedding-pipeline drift between index and queries, ambiguous
function keying (2,987 duplicate `(binary, name)` rows), dev selection that used a *weaker*
retrieval pipeline than test, and a convex fusion that could delete retrieval tokens. Part 2
fixed all five and re-ran the entire validation. **Every conclusion survived and sharpened.**

## 10.1 Corpus reconciliation (Priority Zero) — exact

```
paper protocol of record   = 241,174     frozen-dump index = 243,289
intersection               = 241,174     difference        = 2,115
2,115 = grep (1,267) + sed (848), exactly (leakage-suspect, excluded by the
paper-clean train definition; the Aug-9 dump retained them — recorded deviation)
```
`D_train := 241,174` (paper-clean). All of: index, U1 training, vocabulary, strata, frequency
statistics derive from exactly this set (`production_train_function_ids.txt`, one row per member
with package/binary/address). A stray "drops curl" explanation in `probe_encoder.md` is refuted.

## 10.2 One pipeline, address-level keying, rebuilt U0 (anchored)

Clean-7 re-embedded through the shipped predict path against the same preprocessing state as the
index embeddings (pipeline hash `070fcc39527a8a3a`). Keying audit: 13,581 rows, **13,581 unique
`binary:address` function_ids, 0 duplicates**. U0 rebuilt from scratch (exact cosine kNN +
BinFilter Jaccard τ=0.5 implementation):

| | ALL | SEEN | NOVEL_COMP | PARTIAL_OOV | FULL_OOV |
|---|---|---|---|---|---|
| U0 rebuilt (macro-F1) | **0.5855** | 0.755 | 0.0844 | 0.0911 | 0 |

Anchor gate **passed**: 0.5855 vs 0.5913 (−0.0058, tolerance ±0.02); micro-F1 0.661, EM 48.6%;
BinFilter active on 8/77 query binaries (historical: 8/78); 92.8% exact-prediction agreement
with the archived dump. Strata counts unchanged (10,151 / 1,509 / 1,762 / 159 — grep+sed names
are covered by other packages). RETR_FAIL (rebuilt-U0 F1 ≡ 0): n = 4,238.

## 10.3 Out-of-fold development through the production pipeline

Six package-disjoint folds; in every fold the held-out packages are scored by U0 run through the
**same kNN+BinFilter pipeline** against the fold's index, and by a fold-trained U1. Pooled over
all 241,174 training functions:

```
OOF U0 (production pipeline)  = 0.2348
OOF U1 (pw=10, tau_C=0.5)     = 0.2615     <- composition BEATS retrieval out-of-fold
clean-7 U0                    = 0.5855     <- retrieval dominates in deployment
```
This is the prior-shift mechanism measured directly, with no pipeline asymmetry left to blame:
package-disjoint development inverts the head ranking relative to deployment. (pos_weight again
selected at the extended grid's edge — immaterial at these margins.)

## 10.4 Final U1 — refit on 100% of D_train (no dev exclusion), 3 seeds

| | ALL | SEEN | NOVEL_COMP | PARTIAL_OOV | FULL_OOV | RETR_FAIL |
|---|---|---|---|---|---|---|
| U1 mean±std (macro-F1) | 0.3191±0.0012 | 0.3945±0.0012 | **0.0889±0.0016** | 0.1110±0.0010 | 0 | 0.0491±0.0009 |

Fifth protocol variant (75K dev-excluded / 228K dev-excluded / 241K full / three seeds each) —
NOVEL_NAME_COMPOSABLE has now been 0.089–0.093 under every configuration ever tested.

## 10.5 Complementarity with precision accounting (the missing diagnostic)

| Subset | % fns U1 adds ≥1 correct token | comp-only correct /fn | **comp-only wrong /fn** | retr-only correct /fn |
|---|---|---|---|---|
| ALL | 8.31% | 0.099 | **2.859** | 1.090 |
| NOVEL_COMP | 15.51% | 0.169 | **3.183** | 0.119 |
| RETR_FAIL | 14.13% | 0.170 | **2.516** | 0 |

For every correct token composition uniquely contributes, it contributes ~29 wrong ones (ALL).
This is the number that dooms any additive use of the signal.

## 10.6 Add-only fusion — the most conservative possible test

U3 = R ∪ (top-k new U1 tokens with p ≥ τ_add), k ∈ {1,2}; retrieval tokens can never be removed.
τ_add = 0.6 selected on OOF, where fusion beats OOF U0 (0.2642/0.2669 vs 0.2348 → gate passed).
Single clean-7 evaluation:

| Model | ALL | SEEN | NOVEL_COMP | EM (ALL) |
|---|---|---|---|---|
| U0 rebuilt | **0.5855** | **0.755** | 0.0844 | **48.6%** |
| U1 (s42) | 0.3185 | 0.394 | 0.0894 | 1.9% |
| U3-top1 | 0.5519 | 0.706 | 0.0963 | 16.3% |
| U3-top2 | 0.5353 | 0.684 | 0.0973 | — |

Exact-match accounting (top1): of U0's 6,593 exact predictions, **4,376 broken (66%), 2,217
preserved, 0 newly exact**. Gate results: ALL gain −0.0336 (needed ≥ +0.01), SEEN drop 0.049
(allowed ≤ 0.005), NOVEL gain +0.0119. **GO = false.**

## 10.7 Final verdict and the supported conclusion

**STOP COMPOSITION PERMANENTLY** (spec §28). The supported conclusion, stated at the strength
the evidence allows (spec §29):

> Multi-label decomposition exposes some complementary correct name tokens, particularly where
> retrieval fails, but the signal is too sparse (≈0.1–0.17 unique correct tokens per function)
> and insufficiently precise (≈29 wrong per correct added token; OOF-selected thresholds do not
> transfer) to improve end-to-end function-name prediction over strong identifier retrieval.
> **Output-side compositional prediction does not improve this retrieval system under the tested
> cross-project protocol.** No claim is made that composition is impossible in general or that
> binary semantics cannot be learned.

The paper's contribution moves to retrieval coverage/generalization (coverage-boundary framing).

## 10.8 Part-2 artifacts

`results/final_composition/`: `production_index_reconciliation.md`,
`production_train_function_ids.txt`, `embedding_pipeline_config.json` + hash, `keying_audit.json`,
`u0_recomputed.tsv` + metrics, `strata_final.tsv`, `oof_u1_selection.tsv`,
`oof_addition_precision.tsv`, `oof_fusion_selection.json`, `u1_final_seed{42,123,7}.tsv`,
`u1_final_summary.json`, `complementarity_final.json`, `u3_top{1,2}.tsv`, `u3_final_metrics.json`,
`final_comparison.tsv`, job log `finalcomp.1174048.out`. Code:
`experiments_semantic/final_composition.py`.
