# FEC Stage 1 — Ranking Report

## B. OOF_NOVEL_COMPOSABLE census
total eval 6000 over 5 folds (per-fold table in census tsv); exact GT
token sets removed from fold training (strict §32 protocol).

## C. Exact-TM vs FEC latent semantics (pooled GT-token ranks)
| method | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR | median rank |
|---|---|---|---|---|---|---|
| Absolute-TM (same matrix, unfactorized) | 0.032 | 0.085 | 0.117 | 0.158 | 0.0616 | 30.0 |
| **FEC SVD-64** | 0.013 | 0.037 | 0.051 | 0.075 | **0.0268** | 40.0 |

## D. Fold deltas (FEC − TM, MRR)
F0:-0.0308, F1:-0.0271, F2:-0.0439, F3:-0.0347, F4:-0.0402 → positive 0/5

## E. Fusion (GT-token MRR / Hit@10)
- RET: MRR 0.0067 · Hit@10 0.019 · Hit@20 0.034
- SEM: MRR 0.0259 · Hit@10 0.051 · Hit@20 0.075
- SEM+RET: MRR 0.0203 · Hit@10 0.046 · Hit@20 0.070
- SEM+RET+DIR: MRR 0.0284 · Hit@10 0.065 · Hit@20 0.100

## F. Candidate oracle (SEM+RET+DIR union)
GT recall 0.1914 · oracle F1 0.2402 · all-GT covered 5.7% · mean size 77

## G. Decision
gates: MRR×1.25 FAIL (0.0268 vs 0.0770 req) · Hit@10+0.030 FAIL · folds 0/5 FAIL · Hit@20>=0.20 FAIL · oracle>=0.250 FAIL

**STOP A: semantic ranking fails**
