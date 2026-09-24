# Retrieval-vs-Generation Complementarity — 7-pkg xproj set

- Predictions: $WORKSPACE/results/hybrid_xfl_REPRO.json (headline pdec_xfl checkpoint); 13581 functions.
- Heads: decoder-only, k-NN-only, hybrid-as-routed (knn if top_sim >= 0.80 else decoder; rule agreement 100.0%).
- Plus hybrid_adaptive_paper: package-level routing (k-NN for nginx118/angie/tengine, decoder for dash/gettext/psmisc/recutils) — reproduces the paper 0.738 headline (0.7338 here; residual gap = P2-reranked k-NN not recorded per-function).
- Overall 7-pkg F1: decoder 0.6826, k-NN 0.5576, hybrid(sigma) 0.6948, hybrid(adaptive) 0.7338.
- F1 = project sub-token F1 (src/evaluation/metrics.py); EM on normalized names.
- NOTE: the same (leaky-checkpoint) predictions are scored under both definitions; A/B changes only the categorization. Under B, "unseen" dash/gettext/psmisc names were still seen by this decoder.

## Definition A (headline pdec_xfl train — incl. dash/gettext/psmisc)

| Category | n | % of xproj | Decoder F1 (EM) | k-NN F1 (EM) | Hybrid-sigma F1 (EM) | Hybrid-adaptive F1 (EM) | % routed to k-NN (sigma) |
|---|---|---|---|---|---|---|---|
| whole_name_seen | 11947 | 88.0 | 0.758 (62.0%) | 0.615 (50.9%) | 0.772 (64.5%) | 0.813 (69.3%) | 56.3 |
| novel_composition | 471 | 3.5 | 0.153 (0.0%) | 0.209 (0.0%) | 0.152 (0.0%) | 0.216 (0.0%) | 34.8 |
| oov | 1163 | 8.6 | 0.118 (0.0%) | 0.104 (0.0%) | 0.118 (0.0%) | 0.129 (0.0%) | 10.6 |

mc2-vocab (min_count>=2) category sizes for A: whole_name_seen=11947, novel_composition=467, oov=1167 (full metrics in JSON).

## Definition B (paper-clean train — all 7 xproj pkgs excluded)

| Category | n | % of xproj | Decoder F1 (EM) | k-NN F1 (EM) | Hybrid-sigma F1 (EM) | Hybrid-adaptive F1 (EM) | % routed to k-NN (sigma) |
|---|---|---|---|---|---|---|---|
| whole_name_seen | 10122 | 74.5 | 0.749 (60.7%) | 0.722 (60.1%) | 0.771 (64.1%) | 0.813 (69.2%) | 65.2 |
| novel_composition | 1260 | 9.3 | 0.561 (45.1%) | 0.091 (0.0%) | 0.547 (44.0%) | 0.585 (45.1%) | 17.8 |
| oov | 2199 | 16.2 | 0.447 (31.7%) | 0.067 (0.0%) | 0.431 (30.2%) | 0.453 (31.7%) | 8.5 |

mc2-vocab (min_count>=2) category sizes for B: whole_name_seen=10122, novel_composition=1247, oov=2212 (full metrics in JSON).

