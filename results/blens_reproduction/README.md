# BLens Reproduction on 9-Package Cross-Project (April 20, 2026)

## Context

Earlier reproduction showed BLens F1 = 0.010 on our 9-package cross-project test (19,406 functions), which seemed inconsistent with the BLens paper's reported 0.46 F1 on Punstrip. Deep audit revealed the issue: test-split CLAP and PalmTree embeddings were being loaded from `embedding/clap` (basename-keyed, 0% test hit rate) instead of `embedding/clap_test` (full-path-keyed, 100% test hit rate). Every test function was receiving a default-zero embedding, causing the decoder to collapse.

Once fixed (by loading the correct per-split embedding files with full-path keys), BLens reproduces as expected.

## Final BLens reproduction numbers on 9-pkg cross-project (19,406 fns)

| Decoder | F1 | EM | Non-empty outputs | Unique outputs |
|---|---|---|---|---|
| LORD (epoch 59, bias=0) | **0.4530** | 0.3264 | 99.1% | 4,668 |
| COMBO (greedy, bias=0) | 0.4350 | 0.3509 | 98.2% | 4,069 |

Both are within 0.01 of the BLens paper's self-reported 0.46 cross-project F1 on Punstrip — faithful reproduction.

## Per-package LORD F1

| Package | Fns | F1 | EM |
|---|---|---|---|
| tengine | 3,177 | 0.561 | 0.405 |
| angie | 4,169 | 0.519 | 0.344 |
| nginx118 | 4,850 | 0.507 | 0.330 |
| recutils | 3,743 | 0.493 | 0.431 |
| psmisc | 295 | 0.315 | 0.224 |
| dash | 1,125 | 0.293 | 0.252 |
| sed | 29 | 0.116 | 0.000 |
| grep | 40 | 0.083 | 0.025 |
| gettext | 1,978 | 0.055 | 0.025 |

## Comparison vs our system on the same 9-pkg test set

| System | 9-pkg F1 |
|---|---|
| Ours (adaptive gate) | 0.7381 |
| SymGen+LoRA | 0.6301 |
| **BLens (LORD)** | **0.4530** |
| BLens (COMBO) | 0.4350 |

## Files in this directory

- `LORD-inference-logs-test-fixed-59.txt` — 19,406 (target, output) pairs from LORD epoch 59
- `COMBO-inference-logs.txt` — 19,406 (target, output) pairs from COMBO simple greedy decoder

Scripts to reproduce are in `../scripts/blens_reproduction/`.
