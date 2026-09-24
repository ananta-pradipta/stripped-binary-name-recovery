# RARC OOF Decision (Stage 3)

pseudo-NOVEL n = 5535 (5 package-disjoint folds, seed 42)

| quantity | value |
|---|---|
| OOF U0 top-1 F1 | 0.0934 |
| baseline U1+lex F1 (fold-trained, dev-selected) | 0.2154 |
| RARC routed F1 | 0.0000 |
| **delta vs baseline** | **+0.0000** |
| 95% bootstrap CI | [+0.0000, +0.0000] |
| packages positive | 0% |
| U0 exact-match breakage | 0.00% |
| new exact matches | 0 |
| false-edit rate (routed) | 0.0% |
| realized proposal-space oracle | 0.2232 |
| headroom efficiency | 0.000 |
| thresholds | tau_R=None, tau_E=None |
| route rate | 0.0% |

## Decision: **STOP: no safe threshold pair**

---

## Failure diagnostics (§62/§63 — computed, no tuning performed)

| quantity | value |
|---|---|
| U0 top-1 (all 6,000 OOF queries) | 0.1082 |
| proposal-space oracle (incl. NO_EDIT) | 0.2280 |
| must-edit oracle | 0.2094 |
| scorer-selected edit (always-edit) | **0.0892** |
| oracle routing over scorer picks (diagnostic) | 0.1797 |
| baseline U1+lex | 0.2154 |
| scorer rank of true-best edit | top-1 12.6% · top-5 43.0% · top-20 68.7% · median 8 |
| best unsafe threshold pair | F1 0.1007 @ 14% route rate, breakage 0.0% |

**§62 classification: F4 dominates** — the correct proposal is generated
(oracle 0.228 ≈ Stage-1 restricted oracle 0.232) but the linear scorer ranks
it wrong (median rank 8; always-edit selection performs below U0). Per §63:
"selection remains fixable" — candidate/edit construction is NOT the
bottleneck on OOF.

**Structural finding (deeper than the scorer):** on pseudo-NOVEL, the
fold-U1+lex baseline (0.2154) is ~2.1× its true clean-7 NOVEL strength
(0.1008), while the edit-space oracle stays ~0.228. The restricted edit
formulation's headroom over the baseline is therefore only **+0.0126 on
OOF — below the +0.025 gate even for a PERFECT scorer.** The Stage-1
feasibility gate (oracle vs clean-7 U1+lex, +0.13 headroom) did not
anticipate that pseudo-NOVEL inflates a token-set baseline far more than it
inflates U0-anchored editing: U1 exploits within-ecosystem token priors
that survive package holdout, whereas U0 scaffold quality does not.

Safety detail: routed breakage was 0.0% at every operating point (the
conservative gate works); the safety failure was entirely the per-package
delta-vs-baseline bound, driven by the baseline's OOF strength.

## Verdict: **STOP (Stage 3). Clean-7 NOT evaluated. No parameters tuned after gate failure.**
