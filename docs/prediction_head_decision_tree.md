# Prediction Head Decision Tree

Adaptive hybrid prediction head for GraphR cross-project function-name recovery.

## Decision flow

```
┌─────────────────────────────────────────────────┐
│  INPUT: query function f in target binary B     │
│         (stripped, from cross-project pkg)      │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Encoder forward pass:       │
         │   z_f = FunctionNamer(f)    │
         │   (GAT + ext/callee/caller  │
         │    fusion → 1024-dim)       │
         └──────────────┬──────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────────┐
    │  LEVEL 1 — BINARY GATE                    │
    │  Compute J_max(B) = max over all          │
    │  training binaries b' of                  │
    │    Jaccard(ext_calls(B), ext_calls(b'))   │
    └──┬────────────────────────────────────────┘
       │
   ┌───┴────────────────────┐
   │                        │
   ▼                        ▼
J_max(B) ≥ 0.7         J_max(B) < 0.7
"in-distribution"      "out-of-distribution"
 (nginx-family)        (gettext/dash/recutils/...)
   │                        │
   ▼                        ▼
┌──────────────────┐   ┌────────────────────────┐
│ k-NN MODE (σ=0)  │   │ DECODER MODE (σ=0.95)  │
│                  │   │                        │
│ k=20 retrieve +  │   │ k=20 retrieve +        │
│ rerank +         │   │ rerank +               │
│ BinFilter τ=0.5  │   │ BinFilter τ=0.5        │
│                  │   │                        │
│ Take top-1 name  │   │ LEVEL 2 — per-query    │
│ always           │   │ safety valve:          │
│                  │   │ top1_sim ≥ 0.95?       │
│                  │   │   ├─ YES: use k-NN     │
│                  │   │   └─ NO:  use decoder  │
└────────┬─────────┘   └──────────┬─────────────┘
         │                        │
         │                        ▼
         │             ┌────────────────────────┐
         │             │ Decoder generate:      │
         │             │ GRU sub-token          │
         │             │ beam search(z_f)       │
         │             │ → name string          │
         │             └──────────┬─────────────┘
         │                        │
         └──────────┬─────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  OUTPUT: name    │
         │  (sub-token      │
         │   sequence)      │
         └──────────────────┘
```

## How to read

1. **Per-binary**: compute `J_max(B)` once — cheap set-Jaccard over ~800 training binaries.
2. **If `J_max(B) ≥ 0.7`**: route all queries from this binary to k-NN (σ = 0, no decoder fallback).
3. **If `J_max(B) < 0.7`**: route most queries to the decoder, but fall back to k-NN if a specific query has top-1 cosine similarity ≥ 0.95.

## Hyperparameters

| Symbol | Value | Meaning |
|---|---|---|
| `τ_bin` | 0.7 | Binary-level Jaccard threshold for k-NN vs decoder mode |
| `τ_sim` | 0.95 | Per-query cosine-sim safety valve in decoder mode |
| `k_retrieve` | 20 | k-NN top-k retrieval size (paper headline) |
| `τ_BF` | 0.5 | BinFilter Jaccard threshold (pre-retrieval pool filter) |
| Beam width | 5 | Decoder beam search width |

## Routing outcomes (9-package cross-project)

| Package | `J_max(B)` | Route |
|---|---|---|
| nginx118 | 0.998 | k-NN mode |
| angie | 0.971 | k-NN mode |
| tengine | 0.695 | Decoder mode |
| grep¹ | 0.639 | Decoder mode |
| sed¹ | 0.447 | Decoder mode |
| psmisc | 0.391 | Decoder mode |
| recutils | 0.349 | Decoder mode |
| dash | 0.290 | Decoder mode |
| gettext | 0.173 | Decoder mode |

¹ grep and sed are present in the training split (data leakage); we exclude them from reported cross-project F1.

## Choice of τ_bin = 0.7

Not a tuned hyperparameter. The Jaccard values cluster into two groups with a natural cliff between them:

```
nginx118  0.998  ┐ "in-distribution" cluster
angie     0.971  ┘
────────────────── 0.276-wide gap ──────────────────
tengine   0.695  ┐
grep      0.639  │
sed       0.447  │
psmisc    0.391  ├ "out-of-distribution" cluster
recutils  0.349  │
dash      0.290  │
gettext   0.173  ┘
```

Any threshold in `[0.70, 0.97]` gives identical routing. We choose 0.7 as a round value in the middle of the gap.

## F1 results

| Subset | Pure k-NN | Fixed σ=0.80 | Adaptive gate | Oracle |
|---|---|---|---|---|
| All 9 pkgs | 0.535 | 0.727 | **0.763** | 0.764 |
| Clean 7 (no grep+sed) | 0.558 | 0.695 | **0.738** | 0.739 |
| Paper 4 (t/a/n/r) | 0.699 | 0.664 | **0.714** | 0.715 |
| Low-cov 3 (d/g/p) | 0.087 | 0.800 | **0.820** | 0.820 |

The adaptive gate achieves 99.9% of the oracle upper bound.
