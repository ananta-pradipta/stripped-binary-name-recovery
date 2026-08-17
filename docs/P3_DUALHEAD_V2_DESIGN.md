# P3 — Dual-head v2 design (draft 2026-08-17, to be revised after P2 numbers)

Constraints (user, 2026-08-17): dual-head narrative must hold; BAP-IR only, no decompiler, no LLM;
"simple yet novel" — no bolted-on mechanisms; every claim measured on dataset v2 with the
package-disjoint dev tier for all tuning.

## What the dual head is (unchanged CCS skeleton)
* Encoder: block Transformer → GAT over the (now correct) CFG → cascaded gated fusion (ext calls →
  callee ctx → caller ctx). z ∈ R^1024.
* Head R (retrieval): cosine k-NN over the train-tier index of z; prediction = top-1 name
  (optionally top-5 vote).
* Head G (generation): GRU decoder with beam 5 over Votes sub-tokens.
* Router: decides per query which head answers, and whether to answer at all.

## The narrative that survives the Aug-2026 evidence
Composition of truly novel names is ≈0.07–0.10 F1 for every head ever tried, so the dual head is
NOT "R for known, G for novel". It is:
  **R for in-coverage recognition; G for graceful degradation; a router that knows the regime.**
The measurable claims:
  C1 (coverage-conditioned): F1 by stratum (seen-name / novel-composable / OOV) and by regime
      (NCT / FT / dev) with the leakage table published; symbol-visible functions excluded from
      targets and reported separately.
  C2 (routing): router-selected F1 ≥ best single head + δ on xproject (δ ≥ 0.01, both seeds),
      selected on val_xproj only.
  C3 (selective prediction): risk–coverage curves / AURC per head and for the routed system;
      F1@{50,70,90}% coverage; abstention makes the system deployable (FLIRT-like "say nothing
      when unsure"). Nobody in the sub-area reports this.
  C4 (calibration): ECE of the router's confidence (old heads had ECE ≥ 0.56).

## Router features (all computable on a stripped binary at inference)
  r1 top-1 cosine sim; r2 margin top1−top2; r3 top-5 name agreement share;
  r4 binary-level recognition rate = fraction of the binary's functions with r1 ≥ τ (the CCS
     per-binary gate, generalised); r5 decoder mean log-prob; r6 **cross-head agreement**
     (sub-token F1 between R's and G's outputs — consensus as confidence, cheap and strong);
  r7 string-evidence support (predicted sub-tokens present in the function's .rodata strings);
  r8 function size (n_tokens) and has-ext-call flags.
Router model: logistic regression (or 2-layer MLP ≤ 5K params) predicting P(R correct),
P(G correct); route to argmax, abstain if max < θ (θ from val_xproj for a target coverage).
Trained on val_xproj records (never on xproject).  Simplicity is deliberate.

## Enriched inputs (ablation, not the main claim)
  * B7 side channels: literal buckets/magic constants (in-stream, +~30 vocab entries), function
    strings (existing string encoder path, gated fusion), arg-register count.
  * Only kept if the gate says +F1 on val_xproj AND xproject with both seeds.

## Gates
  P3-G1: router beats best single head on val_xproj by ≥ 0.01 F1 (both seeds) → run on xproject.
  P3-G2: xproject routed F1 ≥ best single head + 0.01 AND AURC better → claim C2/C3.
  P3-G3: enriched-input variant beats plain on val_xproj by ≥ 0.005 with both seeds → keep.
  Otherwise the paper reports the coverage-boundary result honestly (C1, C3) and the dual head
  as a measured, not oversold, deployment mechanism.
