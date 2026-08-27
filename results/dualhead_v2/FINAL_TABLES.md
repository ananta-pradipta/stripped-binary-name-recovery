# Final tables — dual-head function naming (numbers only; 2026-08-27)
All F1 = metric v2 (camelCase-aware sub-token F1, C++ demangled). Test = 268,178 functions / 50 packages, package-disjoint, body-dedup vs train, exported symbols excluded. Micro / macro(per package). Router and abstainer fit on val_xproj only.

## T1. System
| system | micro | macro | FT | NCT | novel-name | sel. F1 @5/10/20/30% |
|---|---|---|---|---|---|---|
| BAP decoder (A1a) | 0.100 | 0.339 | 0.027 | 0.465 | 0.024 | — |
| BAP retrieval + string rerank (A1a) *(prev. best)* | 0.118 | 0.381 | — | — | — | 0.954 / 0.754 / 0.448 / — |
| BAP retrieval, C1 name-aware contrastive (λ1.0) | 0.127 | 0.385 | 0.040 | 0.564 | — | — |
| generation head A4 (CodeT5+ 220M, our data) | 0.185 | 0.368 | 0.121 | 0.504 | 0.123 | — |
| regime router (NCT→retrieval, FT→A4) | 0.193 | 0.424 | 0.121 | 0.553 | — | — |
| conf router (kNN sim threshold) | 0.198 | 0.430 | 0.116 | 0.611 | — | 0.705 / 0.657 / 0.559 / 0.471 |
| **learned GBT router, C1 λ1.0 ∪ A4** | **0.205** | **0.439** | 0.120 | 0.633 | — | **0.943 / 0.880 / 0.690 / 0.534** |
| learned GBT router, A1a ∪ A4 | 0.204 | 0.439 | 0.120 | 0.627 | — | 0.958 / 0.881 / 0.686 / 0.529 |
| oracle head choice (C1 λ1.0 ∪ A4) | 0.224 | 0.471 | 0.134 | 0.675 | — | — |
| oracle 3 heads (+ decoder) | 0.228–0.231 | 0.476–0.480 | — | — | — | — |

## T2. Router
| router | routing acc. on disagreements (31.6% of fns) | regret F1 | →retrieval | micro / macro |
|---|---|---|---|---|
| conf (sim1) | 76.8% | 0.024 | — | 0.198 / 0.430 |
| logistic (11 feats) | 80.2% | 0.020 | 15.8% | 0.202 / 0.434 |
| GBT (11 feats) | 82.1% | 0.018 | 16.3% | 0.204 / 0.439 |
Feature ablation (GBT, A1a∪A4): −kNN-sim → 0.201/0.433, acc 80.2%; −A4-conf → 0.203/0.433 but selective@20% 0.686→0.544; only sim1 → 0.196/0.423; only a4_conf → 0.183/0.376.

## T3. Baselines on matched keys (same functions for every system)
| system | FT sample (7,471 fns, 24 pkgs) micro / macro / EM | NCT sample (6,986 fns, 23 pkgs) micro / macro / EM |
|---|---|---|
| SymGen (CodeLlama-34B + LoRA) | 0.120 / 0.137 / 2.8% | 0.236 / 0.234 / 7.3% |
| BLens (retrained) | 0.026 / 0.030 / — | — |
| ours: BAP retrieval (A1a) | 0.039 / 0.047 / 1.4% | 0.744 / 0.747 / 68.8% |
| ours: BAP decoder (A1a) | 0.034 / 0.044 / 1.2% | 0.660 / 0.659 / 57.8% |
| ours: A4 gen head (run 1 / run 2) | 0.115 / 0.124 / 3.3%  ·  0.119 / 0.122 / 3.8% | 0.633 / 0.633 / 44.0%  ·  0.591 / 0.590 / 38.3% |
| ours: GBT union (run 1 / run 2) | 0.112 / 0.120 / 3.5%  ·  0.114 / 0.118 / 3.7% | 0.780 / 0.781 / 67.9%  ·  0.765 / 0.767 / 65.6% |
| oracle union | 0.130 / 0.139 / 3.7% | 0.827 / 0.828 / 73.8% |
Caveat: SymGen's LoRA was trained on its own corpus, which overlaps 9 of our test packages; BLens per-row preds unavailable beyond the FT sample.

## T4. External benchmark — SymGen 5-package holdout (gmp, libpng, libmicrohttpd, poke, libredwg; 9,991 scorable fns, exported symbols dropped)
| head | micro | macro | EM |
|---|---|---|---|
| A4 run 1 (our data) | 0.223 | 0.189 | 16.1% |
| A4 run 2 (+SymGen corpus) | 0.233 | 0.199 | 17.1% |

## T5. Data / objective variants (val vs test)
| variant | val_xproj | test (matched rows) | verdict |
|---|---|---|---|
| A3+ literal/ABI/rodata tokens (decoder) | 0.132 (= A1a) | 0.091 / 0.294 (A1a 0.100 / 0.339) | negative |
| A4 run 2 (+SymGen rows) | 0.237 (run 1: 0.200) | 0.178 / 0.348 (run 1: 0.185 / 0.368) | val-selected; test −0.008 |
| BAP v3 retrain (+SymGen corpus): decoder | 0.164 (A1a 0.133) | 0.098 / 0.329 (A1a 0.100 / 0.340) | wash |
| BAP v3 retrain: retrieval | 0.193 (A1a 0.154) | 0.115 / 0.377 (A1a 0.117 / 0.380) | wash |
| BAP v3 retrain: union | 0.264 (0.232) | 0.204 / 0.439 (= A1a) | wash |
| C1 exact-name contrastive (control) | 0.160 | retrieval 0.117 / 0.386 | +0.002 |
| C1 soft λ0.3 / λ1.0 | 0.160 / 0.159 | retrieval 0.125 / 0.387  ·  0.127 / 0.385 | +0.009 / +0.011 |
Finding: val_xproj (GNU-heavy FT packages) cannot arbitrate GNU-heavy data additions — three cases.

Sources: results/dualhead_v2/RESULTS_LEDGER.md (job ids per row), results/*.json.
