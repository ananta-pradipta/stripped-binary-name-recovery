# E2 — Layer-Selective Multi-Source Evidence: Phase-1 Report

**Date:** 2026-08-14 · **Jobs:** 1175470/71/72/967 (a100_20g MIG chain) · **Verdict: GATE FAILED
— Outcome C (spec §40): the production encoder does not expose transferable naming semantics at
any tested depth. `go=false`; no G2 from this representation; Phase-2 context grid not run (§45:
E2-C ≤ U1).**

## Gate
| criterion | required | observed |
|---|---|---|
| E2-C NOVEL (clean-7) | ≥0.12 and ≥U1+0.03 | **0.0518** (U1 0.0894, E1-B 0.0693; Δ −0.038) |
| token macro-AUPRC | ≥0.06 | 0.072 ✓ (ranking signal exists; not convertible to sets) |
| package-holdout retention | ≥0.60 | **0.206** (fold-0 OOF 0.018 / PN 0.0874) |

## §44-A/B Census + homogenization (cap 128; uncapped mean 27.3 / p99 229 / max 29,303;
E1's cap-30 truncated 19.5%, cap-128 truncates 2.67%)
| layer | pairwise cos | centroid dist | eff. rank | variance |
|---|---|---|---|---|
| H0 | 0.348 | 2.14 | 9.8 | 0.012 |
| H1 | 0.345 | 2.65 | 13.1 | 0.017 |
| H2 | 0.451 | 2.55 | 13.1 | 0.018 |
| H3 | 0.473 | 2.49 | 10.8 | 0.020 |
Mild progressive homogenization; near-full rank at all depths (median 10 blocks). "GAT destroys
diversity" is only weakly supported — the diversity survives but carries no name signal.

## §44-C Tomography (clean-7 macro-F1; PN-OOF = 3-fold package-disjoint pseudo-novel)
| rep | ALL | NOVEL | RETR_FAIL | PN-OOF |
|---|---|---|---|---|
| z_R (ref) | 0.311 | 0.086 | 0.045 | — |
| H0-mean/max | 0.194/0.249 | 0.065/0.075 | 0.059/**0.095** | 0.025/0.029 |
| H1-mean/max | 0.251/0.250 | 0.069/0.080 | 0.086/**0.095** | 0.028/**0.030** ← l* |
| H2-mean/max | 0.257/0.247 | 0.076/0.079 | 0.088/0.095 | 0.026/0.029 |
| H3-mean/max | 0.250/0.238 | 0.068/0.074 | 0.082/0.086 | 0.025/0.027 |
Layer gradient is weak (NOVEL 0.065–0.080 everywhere). Robust residue: early max-pooling
**doubles** z_R's RETR_FAIL signal. T-attn probes deferred (documented economy).

## §44-D Context census
ext-call evidence 49.5% of functions (p95=8); callee 71.4% (p95=8). Untested in combination
(gate failed before Phase 2).

## §44-E E2-C (H1, MLP adapter, token-conditioned attention, by-function sampler, λx=0)
Clean-7: ALL 0.199 / SEEN 0.254 / NOVEL 0.0518 / RETR_FAIL 0.059. PN-dev peak 0.0874 at epoch 5,
monotone decline after (same overfit shape as G1/E1). Attention entropy 2.577 vs 2.607 uniform
ceiling — **no localization, even on correct-edge H1 states**.

## DISCOVERY: production edge misalignment
`collate_fn` offsets each function's CFG edges by real block count while the model flattens
nodes with padded stride (B×30): in every multi-function batch, all but the first function ran
the GAT over scrambled edges. Affected: training-era batching (presumed), cache_z, the full z_R
index, E1's states (hence H3-vs-E1 cos 0.625 vs E2's correct-edge extraction). The system's
demonstrated insensitivity to this scrambling (anchor 0.5855 reproduces with correct-edge
queries against a misaligned index) implies the GAT **learned to be edge-insensitive — a set
encoder in practice** — which is the mechanistic root of "diversity without concept binding".

## Root cause (full chain)
1. Identity objectives (name CE + same-function contrastive) never demanded sub-function,
   package-aligned concept structure; ~19 probe/model configurations over this encoder converge
   on NOVEL 0.05–0.09 with ~20% package-holdout retention.
2. The GAT degenerated to edge-insensitive mixing (discovery above): block states are content
   summaries without concept binding → token queries attend uniformly at every depth.
3. Name-bearing lexical channels (ext-call/callee/caller, strings) fuse **after** pooling —
   the block stream itself carries only ~1.5K coarse instruction types (z_R > all block probes
   on ALL/SEEN).

## Implication
Frozen-encoder mining is closed. The generation branch requires either a **trained**
representation (P1: episodic package-disjoint + package-adversarial training, pre-CE init,
lexical channels injected INTO the block stream) or a **richer input** (P2: decompiled IR).
Run P1 first on the same BAP-IR input: if concept-supervised training also fails, the input
modality is binding → P2. Residue worth carrying: H0/H1-max RETR_FAIL signal; AUPRC-level
ranking signal (0.072); context-stream coverage stats.

Artifacts: results/e2/ (census, homogenization, tomography, folds, context census, summary,
e2c_model.pt); states e2_H{0-3}_{train,clean7}.npz on Wulver (~25GB, permanent);
job logs e2.1175470/71/72/967.out. Code: experiments_semantic/e2_phase1.py.
