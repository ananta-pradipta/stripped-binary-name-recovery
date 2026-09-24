# Holding the Dual Retrieval + Generation Architecture: Evidence-Based Diagnosis and Proposals

**Date:** 2026-08-13 · Programme evidence through jobs 1174048 (protocol), 1174597 (G1),
1174620 (E1); E2 Phase 1 queued (1175300). All numbers under the corrected protocol
(D_train = 241,174; clean-7 = 13,581; U0 anchor 0.5855).

---

## 1. Evidence inventory (what we have actually measured)

| # | Fact | Number | Source |
|---|---|---|---|
| E1 | Retrieval is coverage-bound, not model-bound | NCT packages 0.84–0.89 F1 vs FT packages 0.02–0.31; ALL 0.5855 | u0_recomputed |
| E2 | Composition over frozen z_R is protocol-invariant | U1 NOVEL 0.089 across 5 configurations (75K/228K/241K, 3 seeds each) | 1174048 |
| E3 | More data does not fix composition | 3.2× data → NOVEL +0.002 | 1174048 |
| E4 | Complementary tokens exist but are dirty | 8% of fns gain ≥1 comp-only correct token; **~29 wrong per correct added** | complementarity_final |
| E5 | Best-head oracle is small and scattered | +0.026 ALL (function-level); **+0.002 package-level** | 1174048 + §4 below |
| E6 | Any token addition destroys EM | 66% of U0-exact broken, 0 newly exact (add-only, τ=0.6, top-1) | u3_final |
| E7 | Selection does not transfer (prior shift) | OOF ranks pure retrieval LAST (0.2348 vs fused 0.267); clean-7 reverses; 4th confirmation | 1174048, retmem, gate |
| E8 | Composition beats retrieval beyond coverage — in OOF only | OOF U1 0.2615 > OOF U0 0.2348; clean-7 FT packages: U1 ≈ or ≪ U0 (dash 0.010 vs 0.227) | 1174048 + §4 |
| E9 | Mean prototypes fail; signal is package-tinted | G1 NOVEL 0.022; matched-support cos 0.426→0.338 cross-package | 1174597 |
| E10 | Post-GAT block states are homogenized and identity-flavored | E1 NOVEL 0.069; attention entropy 2.54/≈2.7 uniform; package-holdout retention 0.20 | 1174620 |
| E11 | Pre-pooling states carry LESS readable name signal than z_R | tomography T1–T3 ≤ T0 (fusion channels are post-pooling) | 1174620 |
| E12 | Token coverage is NOT the bottleneck | 97.3% of NOVEL GT token instances have ≥2-name train support | 1174620 |
| E13 | Anti-memorization constraints did not help | uniform-by-name sampler −0.012 vs by-function; G1 exclusions didn't rescue transfer | 1174620/1174597 |
| E14 | The decoder is a recognizer | 0/1,873 unseen names ever emitted | historical, confirmed |
| E15 | CE fine-tuning erases pretraining geometry | pre-vs-CE probe (dualspace work) | diag_pre_vs_ce |
| E16 | Ext-call context transfers only with disambiguation | ext alone −4.3pp demo; +callee/caller +12.8pp | Exp 3→4 |
| E17 | Richer input changes naming generalization elsewhere | SymGen: decompiled ≫ assembly input | literature |
| E18 | Small positive residue | block-max helps RETR_FAIL (0.088 vs 0.045); U1 RETR_FAIL 0.05 vs U0 0 | 1174620/1174048 |

**New measurement for this memo (§4):** simulated *package-level* routing on clean-7 with the
current U1: naive Jaccard gate (J<0.5→U1) = 0.5652 (−0.020, dash-driven); *oracle* per-package
routing = 0.5875 (+0.002). Even a perfect coverage router gains nothing with the current
composition head.

## 2. Root-cause diagnosis (three layers)

**(a) The representation was trained for identity, and identity is what it contains.**
The shared encoder's objectives (name CE + same-function contrastive) demand invariances that
*collapse* within-function structure and *separate* functions — exactly what retrieval needs and
exactly what concept transfer does not. Five readout families at two granularities (U1 linear,
U2 residual, add-only fusion, G1 prototypes, E1 MIL/attention) all extract the same ~0.07–0.09
NOVEL, all collapse cross-package (E9, E10). This is not readout underfitting: E12 rules out
token coverage, E3 rules out data volume, E10's uniform attention shows there is nothing local
to find in post-GAT states, and E11 shows pre-pooling states are poorer still. **Conclusion:
no further readout on the frozen encoder can pass the gates. The generator branch needs its own
representation with its own objective.** (E2 will test the last frozen-encoder depth; its
outcome C is anticipated but must be measured.)

**(b) Selection on any in-corpus development regime anti-transfers.**
E7/E8: coverage-poor development inverts head ranking relative to coverage-rich deployment.
Confidence-threshold routers are therefore structurally miscalibrated — measured four times.
The only signals that transferred are *index-side coverage* signals (BinFilter fingerprints,
Jaccard), because they measure the corpus, not the model.

**(c) Token-set outputs are the wrong interface for combining heads.**
E4/E6: token-level fusion has a precision floor (29:1 wrong:correct) and EM is destroyed by any
addition. Whole-name outputs with *exclusive* routing (each function answered by exactly one
head) is the only combination interface that preserves retrieval's EM structurally.

## 3. What is now excluded (do not revisit)

- Any further readout/head over frozen z_R or frozen block states (a–e above).
- Global convex or add-only token fusion; any token-level mixing of heads (E4, E6).
- Confidence-tuned routing calibrated on in-corpus dev (E7).
- Mean concept prototypes; same-name-exclusion and anti-duplicate sampling as *fixes* (E9, E13).
- Scaling composition data (E3). More epochs/capacity on U1-family (E2: seed σ = 0.001).

## 4. The quantified bar a paper-worthy dual head must clear

Clean-7 decomposes into a coverage-rich region (NCT: nginx118/angie/tengine, n=7,917, U0=0.855)
and a coverage-poor region (FT: recutils/dash/gettext/psmisc, n=5,664, U0=0.223). A coverage
gate at the package/binary level (J<0.5) reproduces this split exactly and is available at
inference. Therefore:

> **The generator must beat U0 on the FT region by a margin.** Every +0.024 of FT-region macro-F1
> converts to +0.01 ALL through the gate, with zero SEEN/EM risk (NCT never routed).
> Target: generator FT-region macro-F1 ≥ 0.30 (U0: 0.223) → ALL ≥ 0.617 (+0.032), a defensible
> headline. Minimum: ≥ 0.28 (+0.024 ALL). Current U1 FT-region: ~0.19 (fails; dash 0.010).

This reframes the dual-head story honestly: **asymmetric coverage-partitioned naming** —
identity retrieval where the corpus covers, semantic generation where it cannot — with the gate
carried by corpus-side evidence that provably transfers (E7c).

## 5. Proposals (ranked by evidence-fit; each with gate and cost)

### P1 — Concept-supervised generator encoder (the central fix)
Train a *separate lightweight encoder* for the generation branch — same BAP-IR inputs, ~2–5M
params — whose objective is transfer itself:
1. **Init from the pre-CE pretrained checkpoint** (`pretrained_encoder.pt`), not the control
   model: E15 says CE destroyed the geometry we want to keep.
2. **Episodic package-disjoint training**: every episode holds out whole packages and scores
   token prediction on pseudo-novel names in them — the PN-OOF metric *is* the loss regime
   (fixes E7 at the source: train/dev/deploy regimes match).
3. **Package-adversarial head** (gradient reversal on package ID) — directly attacks the
   measured 0.20-retention pathology (E10).
4. **Token supervision at the encoder** (multi-label BCE + our z_R-hard negatives), so concept
   structure is a first-class training target rather than a post-hoc probe (fixes 2a).
5. Keep E18's residue: concatenate block-max summary + ext/callee/caller channels with type
   embeddings (E16 says context transfers when disambiguated).
- **Novelty:** dual-head with *decoupled objectives* — an identity head and a transfer-trained
  concept head over the same lifted IR, routed by corpus coverage. No published binary-naming
  system trains its generation representation adversarially against project identity with
  episodic pseudo-novel objectives (BLens/SymLM/SymGen all train monolithically).
- **Gate:** PN-OOF NOVEL ≥ 0.12 with retention ≥ 0.6 (E2 criteria), then FT-region ≥ 0.28.
- **Cost:** ~2–3 A100 jobs (encoder training is small; dataset already cached). **Risk:** BAP-IR
  token granularity may cap concept signal regardless of objective — this is exactly what
  distinguishes P1 from P2.

### P2 — Decompiled-IR generation branch (representation source upgrade)
If P1 improves transfer but plateaus below the bar, the input modality is the binding
constraint (E17). Build the generator over decompiler pseudocode/SSA for *beyond-coverage
functions only*: Ghidra headless decompilation of the ~5.7K FT-region + RETR_FAIL functions
(cheap at inference: only ~40% of functions ever reach the generator — the asymmetric-compute
story), normalized pseudocode → small transformer encoder trained with P1's episodic recipe →
ordered-token decoder.
- **Novelty (per the E2 spec's boundary):** not decompilation per se (SymGen) — the
  *asymmetric dual-head inference*: cheap raw-binary identity retrieval for the covered mass +
  a decompilation-based compositional generator invoked only beyond the coverage boundary,
  with the boundary estimated from index-side evidence.
- **Gate:** same FT-region bar. **Cost:** new tooling (Ghidra pipeline; note Ghidra was dropped
  in March for dataset reasons — this use is eval/generation-side only, far smaller surface).

### P3 — Coverage-gated deployment harness (the protective router; not a fix by itself)
Group-robust binary-level gate: fingerprint Jaccard to nearest index binaries + top-k name
agreement, threshold chosen so *every* development package group keeps retrieval precision
(§40-style). Today's measurement is the honest caveat: with the current U1 this router gains
+0.002 at best — the router only pays once P1/P2 produces a generator worth routing to.
It is still the right harness because it structurally protects SEEN/EM (E6) and uses the only
transfer-proven signal family (E7c). **Cost: one day; build once, reuse.**

### P4 — Whole-name generation with neighbor-template priors, beyond coverage only
Where the generator runs, condition its decoder on retrieval's top-k *token bags* as a soft
vocabulary prior (r(t) as decoder bias, α tuned episodically). Rationale: E8 shows retrieval
still carries partial signal beyond coverage (0.223 ≠ 0); retrieve-and-edit failed *globally*
(retmem) but was never tested gated-beyond-coverage, where its failure mode (breaking correct
names) cannot occur. Strictly secondary to P1/P2 — an increment on whichever passes.

### P5 — RETR_FAIL specialist (cheap, immediate, bounded win)
31% of clean-7 has U0 = 0 exactly; anything > 0 there is pure gain if routing identifies the
region. Low-J + low top-1-cosine identifies most of it (measurable from existing dumps). Even
the current U1 adds ~0.05 there → +0.015 ALL ceiling with a conservative gate. Worth carrying
as the floor while P1/P2 mature; also the natural first deployment surface for them.

## 6. Recommended sequence

1. **Now:** E2 Phase 1 verdict (queued) — completes the frozen-encoder story; its homogenization
   table is paper evidence regardless of outcome.
2. **P3 harness + P5 floor** (days): build the coverage gate, quantify the RETR_FAIL-specialist
   floor with existing U1 — worst case +0.01–0.015 ALL, zero risk; establishes the deployment
   frame and the evaluation harness every later generator plugs into.
3. **P1** (1–2 weeks): concept encoder with episodic + adversarial training. Gate at PN-OOF,
   then FT-region. If ≥0.28 → dual-head paper: identity/concept decoupling + coverage
   partitioning, with the entire negative-result chain as motivation (reviewers get the
   why, not just the what).
4. **P2** only if P1 shows transfer-learning works but signal is input-capped (P1 retention
   rises but absolute F1 plateaus < 0.25): swap the generator input to decompiled IR, keep
   P1's training recipe. This is also the strongest-novelty variant if it clears the bar.
5. If both fail their gates: the coverage-boundary paper stands on its own — retrieval +
   boundary estimation + the five-family negative result is a complete, honest contribution;
   the dual-head becomes the explicitly-motivated future-work section.

## 7. Expected paper claims if P1 (or P2) passes

- Dual-head, coverage-partitioned binary naming: +0.024–0.032 ALL over a strong retrieval
  system on clean-7, all of it earned beyond the coverage boundary, SEEN/EM untouched
  structurally (not by tuning).
- The diagnosis chain as a contribution: identity-objective representations do not transfer
  concepts (5 readouts × 2 granularities); selection anti-transfers under coverage shift
  (4 instances); token-level head fusion has a measured 29:1 precision floor.
- Honesty constraints (from the spec's reviewer restrictions): claim decoupled-objective
  training + asymmetric coverage-gated inference, not "first fine-grained alignment" or
  "decompilation novelty".
