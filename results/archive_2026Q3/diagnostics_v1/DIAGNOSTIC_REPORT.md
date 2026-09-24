# Diagnostic Report — Retrieval + Composition/Generation (diagnostics_v1)

**Date:** 2026-08-16 · **Protocol:** D_train = 241,174 paper-clean; clean-7 = 13,581 fns keyed `binary:address`; measurement only — no training, no clean-7 tuning, no protocol changes. All tables in `results/diagnostics_v1/`; per-function packets in `error_cases/` (951 files, seeded, no cherry-picking).

## A. Reproducibility and artifact integrity

- Repo commit at collection: `bbee84e2` (dirty tree: diagnostic outputs only). Full inventory with sizes/sha256 in `artifact_manifest.tsv`. Environment: Python 3.10, torch 2.x + CUDA (RTX-class 8GB local GPU); P1 checkpoints pulled from HPC `strlex_ws/results/p1v2/`.
- **Missing artifacts (reported, not recreated):** BinFilter per-candidate pass flags (exist only inside the production hybrid eval on HPC; `binfilter_pass` column is null); U1 raw token scores (the U1 dump stores final sets only); train-index function addresses (the index dump carries name+binary only, so `retrieved_train_function_id` is `binary:name`-keyed).
- Retrieval provenance: top-20 from plain cosine over the paper-clean control index; top-1 agrees with the anchored `U0_name` on **92.2%** (gap = anchor's BinFilter layer + tie-breaks). §16/§5 numbers use this dump; U0 metrics themselves come from the frozen `u0_recomputed.tsv`.
- P1 systems evaluated on their cache coverage (13,084/13,581; 497 fns lack enriched graphs — angie_O3 et al., counted in §33 denominators).

## B. Current system performance (macro token-F1)

| system | n | ALL | SEEN | NOVEL | PARTIAL-OOV | RETR_FAIL |
|---|---|---|---|---|---|---|
| U0 | 13,581 | **0.5855** | **0.7550** | 0.0844 | 0.0911 | 0.0000 |
| U1 | 13,581 | 0.3204 | 0.3955 | 0.0941 | 0.1107 | 0.0545 |
| U1+lex | 13,581 | 0.3235 | 0.3960 | **0.1008** | **0.1260** | **0.0591** |
| P1-control | 13,084 | 0.1184 | 0.1524 | 0.0297 | 0.0153 | 0.0383 |
| P1-embfix | 13,084 | 0.1128 | 0.1443 | 0.0290 | 0.0190 | 0.0336 |
| P1-lex | 13,084 | 0.1195 | 0.1513 | 0.0318 | 0.0281 | 0.0367 |
| P1-copy | 13,084 | 0.1247 | 0.1562 | 0.0385 | 0.0346 | 0.0436 |

(Note: U0's NOVEL 0.0844 here uses the frozen per-fn `U0_f1`; strata as defined in the protocol dump. P1 numbers match their gate reports.)

## C. Retrieval quality beyond top-1

- **Union GT-token recall** (ALL / NOVEL): top-3 0.591/0.080 · top-5 0.609/0.092 · top-10 0.634/0.115 · top-20 0.661/**0.150**.
- **Edit distance, NOVEL:** within ≤1 edit of a top-10 name: 3.3% · ≤2: **28.4%** · ≤3: **70.0%**. Mean min-edit @10 = 3.10.
- **Neighborhood:** mean top-10 name entropy 1.58 (NOVEL 1.62), ~6 unique names per top-10, top-1/top-2 margin 0.064 (NOVEL **0.021** — retrieval is far less decisive exactly where it fails). Top-10 contains the GT name: 57.0% ALL, **0.0% NOVEL** (by construction of the stratum).
- Full distributions: `retrieval_topk.parquet`, `retrieval_summary.parquet`, `retrieval_token_freq.parquet`.

## D. Composition score behavior

- **GT-token ranks (NOVEL, V=5,400):** mean GT rank ~830–880 for all P1 heads; fraction of GT tokens in top-10: control 5.9% → P1-copy 7.6%; in top-100: ~29–30% for all. The heads never bring the bulk of GT tokens near the top.
- **Calibration is uniformly terrible:** ECE 0.56–0.60, Brier 0.38–0.42, AUPRC ≤ 0.052 (ALL) and ≤ 0.014 (NOVEL) for every P1 head — sigmoid scores are unusable as probabilities; any future selector needs recalibration or margin-based rules. Full 10-bin tables in `calibration.tsv`.
- **P1-copy component split** (in `token_diagnostics_P1-copy.parquet`): learned copy weight **5.484** — comparable to the whole score spread, i.e. the trained model chose to trust the wire maximally.
- Frequency effects (`frequency_recall.tsv`): U1 recall collapses outside FREQUENT (0.43 → 0.02 MEDIUM → 0.00 RARE); U0 degrades gracefully (0.72 → 0.58 → 0.38/0.42); UNSEEN tokens are 0 everywhere (closed vocab).

## E. Evidence availability (NOVEL stratum; full table `evidence_coverage_by_source.tsv`)

| source | % with evidence | exact-copy GT recall | ≥1 hit | all GT |
|---|---|---|---|---|
| SELF_STRING | 18.7 | 0.040 | 7.6% | 1.9% |
| SELF_EXTCALL | 42.6 | 0.008 | 2.3% | 0.0% |
| SELF_GREF | 21.9 | **0.046** | 8.5% | 1.9% |
| ALL_SELF | 48.4 | 0.053 | 10.4% | 1.9% |
| CALLEE / CALLER | 41.3 / 42.5 | 0.037 / 0.039 | 8.2 / 9.2% | ~1% |
| **ALL_RAW** | **74.6** | **0.106** | 21.1% | 3.1% |

Context inheritance doubles exact-copy recall over self-only; GREF is the most precise self source; ext-call names alone carry almost no direct name material (0.008) — their value is associative (translation memory), not copyable.

## F. Evidence generalization (NOVEL; `generalization_slices.parquet`)

- Clean-7 evidence is overwhelmingly **seen** at the atom level: mostly-seen (≥80%) n=702, partly-unseen n=28, mostly-unseen n=0. The unseen-atom regime is nearly empty — atom-level memorization cannot be separated from transfer on this eval, and claims either way are unsupported.
- Pair-level (evidence,GT-token) association dependence is measurable: functions with **all GT token-pairs seen** in train: U1+lex 0.124 vs **some pair unseen** 0.098; P1-copy 0.050 vs 0.037. Gains shrink ~20–25% when exact associations are novel but do not vanish.
- With **no evidence at all** (n=779 NOVEL): U1 0.108 / U1+lex 0.103 / P1-copy 0.028 — the learned head degrades far more than retrieval-composition when evidence is absent.

## G. Candidate ceiling (NOVEL oracle F1; full table `candidate_master.tsv`)

R@1 0.077 · R@10 0.160 · R@20 0.208 · E_SELF 0.107 · E_SELF+CALLEE+CALLER 0.171 · R10+E_SELF 0.202 · **R10+ALL_RAW 0.269** (+TM 0.326, +sparse 0.330 from Stage A). Mean candidate size ≤ ~26 (R10+ALL_RAW).

## H. Failure decomposition (§16; ALL, n=13,581; full `failure_decomposition.tsv`)

| category | n | % |
|---|---|---|
| A_RETRIEVAL_EXACT | 6,637 | 48.9 |
| B_RETRIEVAL_NEAR (≤2 edits @10) | 3,067 | 22.6 |
| F_OOV_TOKEN | 1,919 | 14.1 |
| D_GT_TOKEN_MISSING_FROM_ALL_CANDIDATES | 1,820 | 13.4 |
| C_CANDIDATE_AVAILABLE_SELECTION_FAIL | 51 | 0.4 |
| G_OTHER | 87 | 0.6 |

The remarkable number: **C is only 0.4%** — complete candidate coverage almost never coexists with a wrong prediction, because complete coverage is itself rare (all-GT-covered: 5% NOVEL at R10+ALL_RAW). The battle is B (near-misses needing small edits) + D/F (coverage/vocabulary).

## I. Rescue versus destructive edit analysis

- **U1+lex rescues** (F1 improved over U0): 1,836 functions; token-set exact created: see `rescue_U1lex.tsv`. Recovered tokens come predominantly from string evidence (df-filtered union only fires when U0∩U1=∅ — 4,236 gated functions).
- **"Breakage" caveat:** 6,409 functions have U0 exact-correct while U1-family *token sets* differ — but U1/U1+lex are token-set systems, not the production output; in a routed system these functions would be served by U0 (the U1+lex gate already fires only on U0/U1 disagreement, touching only 31% of functions). True production breakage of the banked rule = **0 SEEN regression** (Section B: SEEN 0.3950→0.3960).
- The §27 counterfactuals quantify why naive unions fail: adding all raw atoms to U1 costs −0.028 precision for +0.094 recall (NOVEL F1 −0.001) — **selection, not union, is where context evidence must enter.**

## J. Caller/callee diagnostics

- Context atoms exist for ~55% of NOVEL fns and add the largest raw-coverage increment (E: 0.053→0.106 with context included).
- Counterfactual unions (NOVEL): +self +0.0042 F1 · +callee +0.0027 · +caller −0.0011 · +allraw −0.0014 — raw inheritance carries recall (+3.3–3.5pp each) but needs precision filtering; caller inheritance is noisiest (precision −0.014).
- Wrapper linkage: mean w_wrap 0.566; wrapper-feature vs callee-hit curves in `context_master.parquet` + `wrapper_scores.tsv` (packets show wrapper-like functions inheriting callee strings correctly).

## K. Retrieval-conditioned semantic differences

`retrieval_delta_dataset.parquet` (67,905 query–neighbor pairs, top-5, evidence differences binary-observable, GT differences labels-only). Descriptive association mining over train-side deltas (§29) was **deferred** per §35 (P3, expensive: requires train-side retrieval pairs); the dataset is exported so it can be computed without re-collection.

## L. Representation information-loss diagnostics

- **Evidence present but token badly ranked (§24, `info_loss_probe.tsv`):** of GT tokens that literally appear among the function's own self-evidence atoms, **48.1% rank below 100** in P1-lex (trunk-fed) vs **30.6%** in P1-copy (direct wire) — the quantified version of "evidence dies in the trunk," and the wire recovers ~a third of the loss, not all of it.
- Stage stats (2,000-fn sample, `representation_diagnostics.parquet`): mean pairwise block cosine — raw-embedding mean-pool 0.356 → frozen trunk 0.339 → adapter **0.108**. The trained adapter *re-differentiates* blocks; the catastrophic homogenization seen earlier came from the random-init GAT (removed), not the frozen trunk.
- NN purity (trunk-mean space, within-sample): nearest neighbor shares a GT token 74.7% of the time, same package 52.6% — the space is semantically organized but strongly package-flavored.

## M. Data integrity and leakage

| check | answer | evidence |
|---|---|---|
| clean-7 GT names in training labels/index | NO | index built from D_train only; `assert not any(pkg in CLEAN7)` in dualhead_final_full.py stage 0 |
| caller/callee GT names used as model input | NO | ctx extraction uses `sub_` ids + raw atoms only (`raec_context.py`); GT names exported only in the marked AUDIT column of `context_master.parquet` |
| source/debug symbols in evidence | NO | lex/gref extractors read stripped ELFs (`extract_strings.py`, `gref_chase.py`); .bir lifted from stripped binaries |
| package identifiers exposed to any composer | NO (P1 heads see tokens only); package adversary was removed (P1_NOADV) | `p1v2_train.py` |
| clean-7 labels used for vocab/thresholds/associations | NO | vocab from train split (V=5,400 rebuild); TM thresholds a priori; U1+lex gate a priori; documented in Stage A report |
| symbol tables retained in graphs | NO | graphs_v2 carries `sub_` names only |
| CFG integrity (stored graphs, §31 sample n=1,000) | v2 invalid edges: 0 fns; legacy v3 graphs: 3 fns with out-of-range edges | `cfg_integrity_audit.tsv`; the known production issue is in the *batch collate*, not stored graphs — §25 experiment scripted (`cfg_integrity_u0.py`), run cancelled on request |

## N. 20 representative examples

Deterministic picks (seed 42, first two per category) from `error_cases_index.tsv`; full packets in `error_cases/`. Categories: breakage_u1lex, rescue_u1lex, rescue_p1copy, novel_near_retrieval, novel_far_retrieval, selection_fail (all 51 exported), missing_candidates, unseen_evidence, seen_ev_unseen_pair, partial_oov — 951 packets total, each with GT, all system predictions, U0 top-10 with edit distances, provenance-tagged evidence, P1-copy GT-token ranks, and generalization flags.

## O. Conclusions from data only

1. **Is retrieval usually semantically close when it fails?** Structurally close, lexically not: 70% of NOVEL functions are within ≤3 token edits of a top-10 name, but top-20 union token recall is only 0.150, and the neighborhood is ambiguous (6 distinct names/10, margin 0.021).
2. **Are the missing GT concepts observable anywhere?** Partially: 74.6% of NOVEL fns carry raw evidence, but exact-copy recall is 0.106; 13.4% of ALL functions have a GT token missing from R20+ALL_RAW and 14.1% have an OOV token — a hard floor no selector can cross.
3. **Which sources add nonredundant coverage?** Context inheritance (doubles exact-copy recall), GREF (most precise self source), translation memory (+0.057 oracle, Stage A). Ext-call names copy almost nothing directly (0.008) — associative value only.
4. **Does evidence help when the atom was unseen in train?** Unanswerable on clean-7: the unseen-atom slice is nearly empty (n=28 partly, 0 mostly-unseen). Needs a different eval design.
5. **Does help survive unseen evidence–token pairs?** Yes, attenuated: gains shrink ~20–25% (U1+lex 0.124→0.098) but persist.
6. **Coverage or selection?** **Coverage first** (D+F = 27.5% of ALL; all-GT-covered only 5% NOVEL), then *precision of selection from ~26–40 candidates* (oracle 0.27–0.33 vs achieved 0.10); pure "candidates complete but selection failed" is only 0.4%.
7. **Are caller/callee signals genuinely useful?** As candidates yes (largest coverage increment); as naive unions no (F1 ≈ 0 net; caller noisiest). They need scored selection.
8. **Are learned heads losing available evidence?** Yes, quantified: 48% of evidence-present GT tokens rank >100 trunk-fed; the copy wire cuts it to 31%.
9. **Does the representation discard semantic distinctions?** The frozen trunk keeps moderate block diversity (cos 0.34) and the adapter sharpens it (0.11); the space is token-organized (75% NN token overlap) but package-flavored (53% same-package NN). The historical homogenization was the GAT, not the trunk.
10. **Facts that should constrain the next architecture:** (a) direct evidence paths are mandatory — every trunk-routed variant lost evidence; (b) scores are uncalibrated (ECE ≥ 0.56) — selection must be rank/margin-based or recalibrated; (c) U0 must be protected by routing (it wins SEEN by 0.36 F1 over everything); (d) the exploitable regime is B+C+D margins: small-edit composition over a ~30-candidate inventory, with coverage expansion (GREF chasing, TM, context) as the enabling investment; (e) OOV (14.1%) needs vocabulary work, not modeling.

---
**Denominators (§33):** U0/U1/U1+lex: 13,581 scored, 0 dropped. P1 heads: 13,084 scored, 497 missing enriched graphs (reason: angie_O3 corruption + 8-binary re-lift gaps; subset-matched comparisons used throughout, consistent with prior gate reports). Metric definitions identical to `dualhead_final_full.py` (`prf`/`block_metrics`).
