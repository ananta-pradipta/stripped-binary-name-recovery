# SECC Stages A+B Report

Strict OOF_NOVEL_COMPOSABLE, 5746 queries, 5 folds (FEC protocol).

## Stage A — exact-matrix candidate ceiling
| combo | oracle F1 |
|---|---|
| R only | 0.0639 |
| Exact-TM only | 0.2754 |
| Exact-TM + R | 0.3143 |
| **full (+direct)** | **0.3143** |

GT recall 0.2560 (gate 0.22) · precision 0.0086 · all-GT 8.7% · size mean 87 / p95 192
Gate (oracle >= 0.280 AND recall >= 0.22): **PASS**

## Stage B — composer comparison (macro F1)
| system | F1 |
|---|---|
| U0 top-1 | 0.0048 |
| Exact-TM independent top-m | 0.0608 |
| Reciprocal-rank fusion | 0.0477 |
| **SECC coverage composer** | **0.0428** |

efficiency 0.136 (gate 0.40) · positive folds 5/5 · mean m 3.16 vs GT len 2.87

## Decision: **STOP B: composer fails**
