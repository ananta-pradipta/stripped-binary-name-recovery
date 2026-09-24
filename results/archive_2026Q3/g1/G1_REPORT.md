# G1 — Binary Semantic Prototype Memory: Report (spec §45/§48)

**Date:** 2026-08-13 · **Job:** 1174597 (A100) · **Commit:** `1c5a1c7f` (+fixes)
**Verdict: GATE FAILED — `go_to_g2 = false`. Do not build the G2 decoder.**
G1 NOVEL_NAME_COMPOSABLE = 0.0218 vs U1 = 0.0894 (Δ = **−0.0676**; required ≥ +0.03).

## Setup (protocol of record, job 1174048 artifacts)
D_train = 241,174 paper-clean; clean-7 n = 13,581 keyed `binary:address`; U0 frozen (0.5855);
U1-final baseline (0.3185 ALL / 0.0894 NOVEL). GeneratorAdapter (1024→512→2×res-MLP→256) on the
frozen pooled production embedding h; per-token L2-mean prototypes rebuilt each epoch from the
full corpus, stop-gradient; **exact same-complete-name exclusion** in training (126,444
(token,name) exclusion pairs) and in the main evaluation; hard negatives from frozen z_R kNN
(24 hard / 12 name-overlap / 12 random); multi-positive contrastive loss, T = 0.07.
PSEUDO_NOVEL_DEV: 2,000 complete names / 7,859 functions held out (complete-name disjoint,
every token retained in ≥2 other training names); generator train = 233,315 fns. All three
leakage assertions passed. Selection (epoch, rule) on PN-dev only; clean-7 evaluated once.
Documented deviations: pooled h (block-level states not cached); transformer blocks degenerate
to residual MLPs on a single vector.

## Token-support census (a result in itself)
Of 11,426 canonical token types in generator train: **5,400 (47%) occur in ≥2 distinct complete
names** (eligible for prototypes), 3,732 in ≥3, 2,604 in ≥5. The majority of the token
vocabulary is single-name identity material that no prototype method can predict by construction.

## §48 tables (clean-7, macro-F1; G1 = same-name-excluded, the mandated main result)

| model | ALL | SEEN | NOVEL_COMP | PARTIAL_OOV | RETR_FAIL |
|---|---|---|---|---|---|
| U0 | 0.5855 | 0.755 | 0.0844 | 0.0911 | 0 |
| U1 | 0.3185 | 0.3937 | 0.0894 | 0.1108 | 0.0499 |
| **G1** | **0.0795** | 0.0988 | **0.0218** | 0.0246 | 0.0308* |
| G1 (standard prototypes) | 0.0984 | 0.1242 | 0.0218 | 0.0246 | — |

\* from g1_results_by_stratum.json RETR_FAIL row.

Token recall (precision) by D_train frequency band:

| model | frequent | medium | rare | very_rare |
|---|---|---|---|---|
| U1 | .4921 (.3293) | .0657 (.3183) | .1321 (.4411) | .1172 (.4506) |
| G1 | .0982 (.2639) | .0685 (**.0224**) | .1330 (**.0155**) | .0404 (**.0077**) |

PSEUDO_NOVEL_DEV: **U1-control F1 = 0.3205 vs G1 = 0.1116** (G1 peaked at epoch 2 of 15 and
declined monotonically after — the contrastive objective overfits the corpus while PN-dev
generalization degrades).

Prototype retrieval (rank of the GT token's prototype among 5,400 concepts):

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| clean-7, same-name-excluded (46,536 GT-token instances) | 0.0267 | 0.1209 | 0.2270 |
| PN-dev, standard | 0.0488 | 0.1313 | 0.1872 |
| **PN-dev, cross-package-only** | **0.0135** | 0.0438 | 0.0637 |

Cross-package-only PN-dev F1 = 0.0395 (vs 0.1116 with same-package support): **roughly two
thirds of what the prototypes capture is package-local, not transferable concept semantics.**

## Gate (spec §22/§24)
| criterion | required | observed | pass |
|---|---|---|---|
| NOVEL_COMP vs U1 | ≥ +0.03 (≥0.12 abs) | −0.0676 (0.0218 abs) | **NO** |
| medium recall | improved | +0.0028 | marginal |
| rare recall | improved | +0.0009 | marginal |
| frequent precision | no catastrophe | 0.264 vs 0.329 | ok |
**STOP. G2 (autoregressive decoder) is not built.**

## Qualitative (from g1_predictions.tsv)
- **A (G1 adds a correct token U1 missed): 15 / 1,509 novel functions.** The additions are
  package-flavored primitives (`chash`, `robin`, `slab`) — cross-name transfer inside a domain,
  not general concept grounding.
- **D (failure despite abundant support): 1,431 / 1,509 novel functions** have a GT token with
  ≥20 distinct support names ranked outside the top-10 — e.g. `create` (421 support names),
  `conn` (85), `api` (58). G1's predictions for these functions are semantically scattered
  (`ancient browser conf memcached`). The method fails hardest on exactly the most reusable
  concepts it was designed to ground.

## Diagnosis (why, at the strength the evidence supports)
1. A mean-of-supports cosine prototype is a far weaker per-token classifier than U1's learned
   hyperplane over the same frozen representation — for frequent concepts, averaging thousands
   of diverse z_G vectors collapses discriminative structure (frequent-band recall 0.098 vs
   U1's 0.492).
2. The adapter cannot add information: z_G is a function of the retrieval-oriented pooled h.
   The anti-memorization and hard-negative machinery worked mechanically (loss fell smoothly;
   leakage asserts held) but what remained learnable in this space is largely package identity
   — shown directly by the cross-package collapse (R@1 0.0135; F1 0.0395).
3. Combined with the U1/U2/fusion results, three different output mechanisms over this frozen
   representation now agree: **the pooled production embedding does not carry
   package-transferable name-primitive structure that any tested readout can exploit.** Under
   the spec's own hypothesis (§4), this is evidence that changing the readout is insufficient;
   the representation itself (or block-level states / encoder training) would have to change —
   which is outside G1's frozen-encoder mandate.

## Artifacts
`results/g1/`: g1_gate.json, g1_results_by_stratum.{json,tsv}, g1_results_by_frequency.{json,tsv},
g1_prototype_recall.{json,tsv}, g1_predictions.tsv (per-function §46 dump),
token_support_stats.tsv, prototype_support.tsv, prototype_vectors.npz (HPC),
pseudo_novel_split.tsv + pseudo_novel_audit.json, g1_train_pairs.tsv, generator_proto_config.yaml,
g1_adapter.pt, hard_neg_pool.npz (HPC), job log g1.1174597.out.
Code: `experiments_semantic/g1_prototype_memory.py`, `embed_clean7.py`;
clean-7 fresh embeddings persisted at `results/zq_clean7_fresh.npz` (HPC).
