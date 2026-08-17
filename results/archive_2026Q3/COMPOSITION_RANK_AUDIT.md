# E0 — True-Subtoken Rank Audit

**Date:** 2026-08-09 · **Branch:** `dualspace` · clean-7 n=13581 · vocab 4000 · bands from training-frequency quartiles **[4.0, 7.0, 21.0]**

**Causal question.** Are medium/rare true sub-tokens encoded in `z` but not *emitted*, or are they not represented at all? Thresholded recall cannot distinguish these; rank can.

**Forced deviation.** Job D persisted metrics but not logits or head weights, so the linear probe is re-derived here with the identical seed, sample, vocabulary and training procedure. Sanity check 5 below verifies the reproduction. `not_in_vocab` atoms are excluded from all rank statistics (the head has no class for them) and reported separately.

## Arm: `linear` (threshold 0.30, tuned on VAL only)

| band | true atoms | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR | median rank | mean %ile | thresholded recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frequent | 7304 | 0.1094 | 0.2007 | 0.2389 | 0.2881 | 0.3749 | 0.4480 | 0.1582 | 155 | 30.7% | 0.1536 |
| medium | 810 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0099 | 0.0296 | 0.0020 | 1320 | 39.3% | 0.0000 |
| rare | 202 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0011 | 1478 | 39.4% | 0.0000 |
| very_rare | 374 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0027 | 0.0010 | 1724 | 45.9% | 0.0000 |
| *random baseline* | — | 0.0003 | 0.0013 | 0.0025 | 0.0050 | 0.0125 | 0.0250 | 0.0021 | 2000 | 50.0% | — |

**Per-function best correct sub-token rank** (n=3171 functions with ≥1 in-vocab true atom):

| ≥1 correct atom in top… | share |
|---|---|
| 5 | 36.6% |
| 10 | 42.0% |
| 50 | 57.4% |
| 100 | 66.0% |
| *median best rank* | 27 |

## Arm: `mlp` (threshold 0.20, tuned on VAL only)

| band | true atoms | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR | median rank | mean %ile | thresholded recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frequent | 7304 | 0.0898 | 0.1843 | 0.2128 | 0.2629 | 0.3412 | 0.4069 | 0.1366 | 218 | 18.6% | 0.1258 |
| medium | 810 | 0.0000 | 0.0000 | 0.0099 | 0.0259 | 0.0395 | 0.0753 | 0.0047 | 1053 | 30.5% | 0.0000 |
| rare | 202 | 0.0000 | 0.0000 | 0.0000 | 0.0149 | 0.0198 | 0.0347 | 0.0023 | 1852 | 48.0% | 0.0000 |
| very_rare | 374 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0053 | 0.0160 | 0.0013 | 1358 | 38.6% | 0.0000 |
| *random baseline* | — | 0.0003 | 0.0013 | 0.0025 | 0.0050 | 0.0125 | 0.0250 | 0.0021 | 2000 | 50.0% | — |

**Per-function best correct sub-token rank** (n=3171 functions with ≥1 in-vocab true atom):

| ≥1 correct atom in top… | share |
|---|---|
| 5 | 31.6% |
| 10 | 36.4% |
| 50 | 51.4% |
| 100 | 59.7% |
| *median best rank* | 46 |

## Sanity checks (plan §2)

1. Same vocabulary mapping (top-4000 by training frequency): **OK**
2. Same clean-7 population via the shipped predict path: **n=13581, OK**
3. Same novel_comp/oov labelling rule: **OK**
4. Same frequency bands (quartiles [4.0, 7.0, 21.0], not redefined): **OK**
5. Frequent-band thresholded recall **0.1536** vs prior runs 0.1538 / 0.1873 — **OK**
6. `not_in_vocab` atoms excluded from rank stats, counted separately: **2452 atoms, OK**

## Branch verdict (plan §3)

- medium: R@50 **0.0099** (random 0.0125), median rank **1320** of 4000
- rare: R@50 **0.0000** (random 0.0125), median rank **1478** of 4000

**BRANCH B — medium/rare ranking is effectively random.** Per the plan, STOP downstream composer-head development. `z` exposes limited compositional information concentrated in frequent name components; the next bottleneck is upstream of the composition head. Do not run more MLP widths, depths, ordering, cardinality, or threshold searches.
