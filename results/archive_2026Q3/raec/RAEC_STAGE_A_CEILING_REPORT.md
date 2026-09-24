# RAEC Stage A — Candidate-Ceiling Report (§36)

**Date:** 2026-08-16 · **Protocol:** clean-7 = 13,581 fns (`binary:address`), GT/strata from `u0_recomputed.tsv`; candidates built zero-training from D_train = 241,174 only; no tuning on clean-7 (all thresholds a priori). NOVEL stratum n = 1,509.

**Provenance notes (integrity):**
- Source R = plain cosine top-K over the paper-clean control index (241,174 rows after grep/sed exclusion from `ztr_full_control.npz`). Top-1 agrees with the anchored `U0_name` on **92.2%** of clean-7; the difference is the anchor's BinFilter layer. Ceiling numbers score candidates against GT directly, so this does not contaminate the audit; the 8% disagreement is noted for Stage B scaffold work.
- 68/837 binaries lack a stripped ELF (no string/GREF evidence there — context and R still apply); angie_O3 corrupt (known).
- Translation memory admissibility fixed a priori (≥3 pkgs, ≥5 names, ≥10 fns; atom df ≥ 10). Sparse memory: IDF overlap, postings capped df ≤ 2000.

---

## A. Evidence coverage (NOVEL stratum)

| source | % NOVEL with evidence | GT-token recall (among-with) | precision |
|---|---|---|---|
| E0 (self strings + ext calls) | 47.4% | 0.098 | 0.067 |
| GREF (pointer-chased strings) | 21.9% | **0.209** | **0.087** |
| CALLEE (one-hop raw) | 41.3% | 0.089 | 0.039 |
| CALLER (one-hop raw) | 42.5% | 0.091 | 0.026 |
| **ALL RAW** | **74.6%** | 0.142 | 0.040 |

The coverage constraint that closed the P1v2 phase (47% of novel fns with evidence) is directly relieved: **74.6%** of NOVEL functions now carry at least one raw evidence atom. GREF is the highest-precision single source — pointer-table chasing reaches identifier-like strings (module tables, command tables) that direct constants miss.

Extraction census (all clean-7): E0 6,869 fns / GREF 3,986 / CALLEE 4,608 / CALLER 3,969. Train side: 181,088/241,174 fns (75%) with evidence; GREF adds 28,028 pointer-derived strings (9,387 deref + 18,641 table) over 216,692 direct.

## B. Translation-memory gain (NOVEL candidate recall)

| inventory | recall | oracle F1 |
|---|---|---|
| R + ALL RAW | 0.208 | 0.2688 |
| R + ALL RAW + translation memory | 0.254 | 0.3258 |
| **delta** | **+0.046** | **+0.0570** |

Memory: 37,133 admitted (e,t) pairs, 25,348 with A>0 (A = LOR × leave-one-package-out stability). Non-identity associations are real — e.g. `mandatory → help` (Conf 0.94, 8 packages: GNU usage boilerplate predicting help/usage functions).

## C. Sparse-memory gain (NOVEL)

| inventory | recall | oracle F1 |
|---|---|---|
| R + ALL RAW + TM | 0.254 | 0.3258 |
| + sparse anchors | 0.257 | 0.3300 |
| **delta** | **+0.003** | **+0.0042** |

Sparse anchor memory is marginal on top of TM (index: 106,875 train fns, 8,240 anchors; neighbors for 4,522 clean-7 fns). Keep as a low-priority feature.

## D. Candidate ceiling (NOVEL_COMPOSABLE)

| source combination | NOVEL oracle F1 | all-GT-covered | mean candidates |
|---|---|---|---|
| R@1 | 0.0773 | 0% | 3 |
| R@10 | 0.1596 | 1% | 16 |
| R@20 | 0.2075 | 1% | 28 |
| R@10 + E0 | 0.2023 | 3% | 19 |
| R@10 + E0 + GREF | 0.2088 | 3% | 20 |
| R@10 + E0 + GREF + CALLEE | 0.2404 | 4% | 22 |
| R@10 + E0 + GREF + CALLER | 0.2447 | 4% | 24 |
| R@10 + ALL RAW | 0.2688 | 5% | 26 |
| R@10 + ALL RAW + TM | 0.3258 | 6% | 40 |
| **R@10 + ALL RAW + TM + S** | **0.3300** | 6% | 42 |
| evidence only (no R) | 0.2124 | 4% | 27 |

ALL-stratum ceiling for the full inventory: oracle F1 0.7199, candidate size 37 — the inventory stays in the tens (§9 target met; p90/p95 in `candidate_ceiling.tsv`).

Reference points: U1 actual NOVEL = 0.0931; U1+lex rule = 0.1008; best learned head = 0.0385. The full candidate inventory holds **3.3× U1's achieved NOVEL** as oracle headroom.

## E. Scaffold viability (NOVEL, canonical token edit distance)

| top-K | ≤1 edit | ≤2 edits | ≤3 edits | mean min-edits |
|---|---|---|---|---|
| 1 | 1.4% | 15.4% | — | 3.68 |
| 5 | 2.7% | 25.0% | — | 3.20 |
| 10 | 3.3% | 28.4% | 55.6% | 3.10 |
| 20 | 4.4% | 31.4% | — | 3.01 |

(Full histograms in `retrieval_edit_distance.tsv`.) The §13 viability target (≥40% of NOVEL within ≤2 edits of top-10) is **NOT met** (28.4%).

## F. Decision

**GO RAEC — with the §13-mandated mode choice.**

- Ceiling gate (§11): NOVEL oracle F1 **0.3300** ≥ 0.16 required / 0.18 preferred → **passes at ~2× the preferred bar**. Candidate recall (0.257) materially exceeds the lexical-only ceiling measured at phase close (~0.10 of GT tokens on 47% of functions).
- Scaffold gate (§13): 28.4% < 40% → **primary composer mode = SET COMPOSITION (Mode A)**, not scaffold editing. Mode B is evaluated as an ablation only, on the ~28% near-scaffold slice where it applies.
- Largest single levers, in order: translation memory (+0.057 oracle F1), CALLEE+CALLER inheritance (+0.060 combined), E0 (+0.043), GREF (+0.007 on top of E0 at K=10 but highest per-source precision), sparse (+0.004).

**Honest caveat:** oracle F1 assumes perfect selection (precision 1.0 over ~40 candidates containing ~26% of GT tokens). The achieved-vs-oracle gap in prior systems is large (U1 achieves 0.093 against its own oracle of ~0.16 at K=10). Stage B's linear scorer plays for the 0.10 → 0.33 corridor; the head-selection rule (§20–22) must protect U0 exact matches (<10% breakage target).

**Stage B does not start until this report is reviewed** (§36). The §25 CFG-integrity experiment (BUGGED-U0 vs FIXED-CFG-U0) is queued as the parallel independent check.

## Artifacts (all in `results/raec/`)
`candidate_retrieval.tsv` (271,620 rows) · `candidate_e0.tsv` · `candidate_gref.tsv` · `candidate_callee.tsv` · `candidate_caller.tsv` · `evidence_atom_census.tsv` · `gref_audit.tsv` · `context_coverage.tsv` · `wrapper_scores.tsv` (mean w_wrap 0.566; 3,383 fns > 0.8) · `translation_memory.tsv` / `_support.tsv` / `_xpkg.tsv` · `sparse_anchor_index_stats.tsv` / `sparse_anchor_neighbors.tsv` · `candidate_ceiling.tsv` / `_by_stratum.tsv` · `retrieval_edit_distance.tsv` / `scaffold_viability.tsv`

Code: `src/preprocessing/gref_chase.py`, `experiments_semantic/{raec_context,raec_train_evidence,raec_translation_memory,raec_sparse_anchor,raec_ceiling}.py`; caches `data/{gref_v1,ctx_v1,raec_train_ev.jsonl}`.
