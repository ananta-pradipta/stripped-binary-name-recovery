# E1 — Fine-Grained Semantic Evidence Model: Report (spec §45/§47)

**Date:** 2026-08-13 · **Job:** 1174620 (A100, ~4h) · **Commit:** `744dc417`
**Verdict: GATE FAILED — `go_to_g2 = false`.** E1 NOVEL 0.0693 vs U1 0.0894 (Δ = −0.020;
required ≥ +0.03 and ≥ 0.12 absolute), and cross-package retention 20% (required ≥ 70%).
Per §48: do not build G2 from this representation.

**Answer to the §49 research question:** *No.* Fine-grained binary code — as represented by the
production encoder's pre-pooling block states — does not contain localized, package-transferable
evidence for reusable name concepts that token-conditioned extraction can exploit. The pooling
operation is not what is discarding name semantics; the semantics the probes keep finding are
function-identity and package artifacts at every granularity tested.

## Representation audit (§5)
241,174 train functions → 3,286,478 block states (512-d, pre-pooling GAT output; mean 13.5,
median 10, p90 30, 92.9% multi-block, 20.3% truncated at the 30-block cap); clean-7 13,581
functions → 214,512 states (mean 15.8), extracted via the exact production predict path.
Sequence-length validity satisfied. States cached (`finegrain_states_{train,clean7}.npz`).

## §47-A Tomography (linear token readout per representation; clean-7 macro-F1)

| probe | ALL | SEEN | NOVEL | PARTIAL_OOV | RETR_FAIL | PN-dev |
|---|---|---|---|---|---|---|
| T0 pooled z_R | **0.3109** | 0.3836 | **0.0862** | 0.1127 | 0.0450 | 0.3195 |
| T1 block mean | 0.2293 | 0.2831 | 0.0716 | 0.0755 | 0.0682 | 0.1177 |
| T2 block max | 0.2382 | 0.2975 | 0.0781 | 0.0550 | **0.0877** | 0.1203 |
| T3 learned attn pool | 0.2132 | 0.2656 | 0.0687 | 0.0543 | 0.0594 | 0.1042 |

**Interpretation (§7):** T1/T2/T3 ≤ T0 everywhere that matters — pre-pooling states carry *less*
globally-readable name signal than z_R (which additionally fuses ext-call/callee/caller context
the block states never see). Small exception: block-max beats T0 on RETR_FAIL (0.088 vs 0.045).
This put E1 in the "token-conditioned extraction required" branch; E1 then failed to find it.

## §47-B Main E1 (clean-7 macro-F1)

| model | ALL | SEEN | NOVEL | PARTIAL_OOV | RETR_FAIL |
|---|---|---|---|---|---|
| U1 (protocol baseline) | 0.3185 | 0.3937 | **0.0894** | 0.1108 | 0.0499 |
| G1 | 0.0795 | 0.0988 | 0.0218 | 0.0246 | 0.0308 |
| E1-A (top-3 MIL) | 0.2004 | 0.2473 | 0.0677 | 0.0618 | 0.0193 |
| E1-B (token attention) | 0.1955 | 0.2405 | 0.0693 | 0.0619 | 0.0184 |

PN-dev grid: E1-A λx=0/0.1 → 0.0720/0.0633; E1-B λx=0/0.1 → 0.0946/**0.0968** (winner).
§18 sampler ablation: uniform-by-function 0.1085 > uniform-by-name 0.0968 — the anti-duplicate
sampler *cost* performance here (duplicate realizations act as augmentation for this model;
the SymGen duplication concern does not transfer to this architecture).

## §47-C Frequency (clean-7, P / R / F1)

| model | frequent | medium | rare | very_rare |
|---|---|---|---|---|
| U1 | .33/.49/.40 | .32/.07/.11 | .44/.13/.20 | .45/.12/.19 |
| E1-A | .19/.33/.25 | 0/0/0 | 0/0/0 | 0/0/0 |
| E1-B | .20/.32/.25 | .007/.004/.006 | 0/0/0 | 0/0/0 |

E1 predicts essentially only frequent tokens. AUROC/AUPRC (E1-B): token macro-AUPRC 0.026;
per-band AUROC 0.64–0.69 — barely above chance discrimination for every band.

## §47-D Cross-package

- **E1 package-holdout fold** (retrained without fold-0 packages, evaluated on pseudo-novel
  names inside them): F1 0.0194 vs PN-dev 0.0968 → **retention 0.20** (gate ≥ 0.70).
- **§21 G1 matched-support control** (support counts equalized, M ≤ 20, 9,821 pairs):
  same-package cosine 0.426 vs cross-package 0.338 — a real but *moderate* ~21% drop.
  The earlier G1 cross-package F1 collapse (0.112 → 0.040) was partly a support-count artifact,
  but E1's fold test now shows the transfer failure directly at the model level.

## §47-E Evidence quality — the decisive diagnostic

- **Mean attention entropy 2.539** against a ≈2.7 uniform ceiling for typical block counts:
  the token queries attend **near-uniformly** — no localization occurs.
- Mean pairwise top-3 block overlap between different GT tokens: 0.443.
- The §46 evidence maps show it plainly: for `ngx_http_limit_req_create_main_conf`, every token
  (`ngx`, `create`, `limit`, `conf`, …) puts 0.19–0.22 on every block — indistinguishable maps.
- §20 instance coverage rules out eligibility as the bottleneck: **97.3% of NOVEL GT token
  instances occur in ≥2 training names** (88.4% in ≥5); the tokens were predictable in principle.

## Diagnosis
The GAT block states are already heavily homogenized: three rounds of message passing over small
CFGs (median 10 nodes) mix block information globally, so per-block states approximate
"function identity + position noise" rather than local semantics. Token-conditioned queries
therefore find nothing to localize (uniform attention), degenerate into a weaker global readout
over an information-poorer input than z_R (no fusion channels), and — as the fold test shows —
what they do learn is 80% package-local. Combined with U1/U2/fusion/G1, **five readout families
across two granularities of the frozen production encoder now agree: the representation does not
expose package-transferable name-primitive structure.** Per §42, the remaining generator-side
option is a genuinely richer semantic input (decompiled/higher-level IR) or new supervision —
a user-level scope decision, explicitly not implemented without a go-ahead.

## Artifacts
`results/e1/`: e1_summary.json (all tables), tomography_results.{json,tsv},
finegrain_representation_audit.json, reusable_token_instance_coverage.tsv,
g1_xpkg_matched_control.tsv, e1_crosspackage.tsv, e1_predictions.tsv, e1_evidence_maps.tsv,
e1_config.yaml, e1_models.pt (HPC), finegrain_states_{train,clean7}.npz (HPC; clean7 copy
local), job log e1.1174620.out. Code: `experiments_semantic/e1_evidence.py`.
