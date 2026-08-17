# RCEM Stage 1 — Contrastive Memory Report

Eval pairs: 20000 (seeded 4,000/fold subsample of the residual
pair corpus; package-disjoint fold memories with leave-name-out).

## B. INSERT (the decisive comparison)
| method | Hit@1 | Hit@3 | Hit@5 | MRR | mean rank |
|---|---|---|---|---|---|
| Absolute-TM control | 0.007 | 0.026 | 0.052 | 0.0289 | 16.6 |
| **RCEM contrastive** | 0.011 | 0.027 | 0.039 | **0.0240** | 11.2 |

fold-by-fold INSERT MRR delta (RCEM − ABS): F0:+0.0074, F1:-0.0009, F2:-0.0108, F3:-0.0089, F4:-0.0155 → positive in 1/5

## C. DELETE (RCEM)
Hit@1 0.016 · Hit@3 0.035 · Hit@5 0.048 · MRR 0.0305 → FAILS 0.20 gate — pure DELETE disabled

## D. KEEP (RCEM)
Hit@1 0.066 · Hit@3 0.140 · Hit@5 0.179 · MRR 0.1196

## E. Source contribution (INSERT MRR)
- SELF: MRR 0.0237 Hit@5 0.035
- SELF+CALLEE: MRR 0.0232 Hit@5 0.036
- SELF+CALLER: MRR 0.0250 Hit@5 0.039
- ALL: MRR 0.0240 Hit@5 0.039

## F. Decision
gates: MRR +0.020 FAIL · Hit@5 +0.030 FAIL · folds 1/5 FAIL · DELETE FAIL (disable pure delete)

**STOP: contrastive INSERT memory does not beat absolute TM**
