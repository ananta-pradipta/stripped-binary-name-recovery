# RARC Stage 1 — Restricted Oracle Report

Scaffolds: production top-10 (Stage-0 export, 100% anchor agreement).
Edit budget <= 2. Eligibility per §12; safe-delete per §18; §27 rank>=2
scaffolds require >=1 edit. NOVEL n = 1509.

## A. Candidate inventory (all clean-7)
candidate tokens/fn: mean 40.9 · median 29 · p90 87 · p95 107 · p99 154
eligible-for-edit tokens/fn: mean 9.1 · median 5 · p90 21

## B. O1 (INSERT + SUBSTITUTE, no pure delete)
NOVEL oracle F1 **0.1686** · token-set EM 0.015 · improves-U0 29.4% · improves-U1+lex 29.6%
edits: 0/82% 1/13% 2/6% (mean 0.24)

## C. O2 (+ safe DELETE)
NOVEL oracle F1 **0.2324** · EM 0.016 · mean edits 0.72
O2 − O1 = +0.0638 → ADOPT safe DELETE

## D. Source contribution (oracle under restricted eligibility)
| combo | NOVEL oracle F1 |
|---|---|
| R | 0.0844 |
| R+SELF | 0.1288 |
| R+SELF+CALLEE | 0.1291 |
| R+SELF+CALLER | 0.1299 |
| R+SELF+CALLEE+CALLER | 0.1346 |
| R+all+TM | 0.1686 |

## E. Decision
best restricted oracle (O2) = **0.2324** · gate >= 0.220 · delta vs U1+lex +0.1316 (>= +0.100 required)

**GO RARC**
