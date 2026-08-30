# Final tables — dual-head function naming (numbers only; 2026-08-27; audited against source JSONs)
All F1 = metric v2 (camelCase-aware sub-token F1, C++ demangled). Test = 268,178 functions / 50 packages, package-disjoint, body-dedup vs train, exported symbols excluded. Micro / macro(per package). Router and abstainer fit on val_xproj only.

## T1. System
| system | micro | macro | FT | NCT | novel-name | sel. F1 @5/10/20/30% |
|---|---|---|---|---|---|---|
| BAP decoder (A1a) | 0.100 | 0.339 | 0.027 | 0.465 | 0.024 | — |
| BAP retrieval + string rerank (A1a) *(prev. best)* | 0.118 | 0.381 | — | — | — | 0.954 / 0.754 / 0.448 / — |
| BAP retrieval, C1 name-aware contrastive (λ1.0) | 0.127 | 0.385 | 0.040 | 0.564 | — | — |
| generation head A4 (CodeT5+ 220M, our data) | 0.185 | 0.368 | 0.121 | 0.504 | 0.123 | — |
| A4 seed 43 (same recipe) | 0.185 | 0.370 | 0.121 | 0.506 | 0.123 | — |
| A4 mean ± half-range over seeds {42, 43} | 0.185 ± 0.000 | 0.369 ± 0.001 | 0.121 ± 0.000 | 0.505 ± 0.001 | 0.123 ± 0.000 | — |
| regime router (NCT→retrieval, FT→A4), C1 λ1.0 ∪ A4 | 0.195 | 0.424 | 0.121 | 0.564 | — | — |
| conf router (kNN sim threshold), C1 λ1.0 ∪ A4 | 0.196 | 0.429 | 0.112 | 0.617 | — | 0.656 / 0.645 / 0.560 / 0.468 |
| conf router, A1a ∪ A4 | 0.198 | 0.430 | 0.116 | 0.609 | — | 0.705 / 0.657 / 0.559 / 0.471 |
| **learned GBT router, C1 λ1.0 ∪ A4** | **0.205** | **0.439** | 0.120 | 0.633 | — | **0.943 / 0.880 / 0.690 / 0.534** |
| learned GBT router, C1 λ1.0 ∪ A4 seed 43 | 0.204 | 0.436 | 0.119 | 0.630 | — | 0.946 / 0.877 / 0.688 / 0.527 |
| routed union, mean ± half-range over A4 seeds | 0.205 ± 0.001 | 0.438 ± 0.001 | 0.119 ± 0.001 | 0.632 ± 0.002 | — | 0.944 / 0.879 / 0.689 / 0.531 (±≤0.004) |
| learned GBT router, A1a ∪ A4 | 0.204 | 0.439 | 0.120 | 0.627 | — | 0.958 / 0.881 / 0.686 / 0.529 |
| oracle head choice (C1 λ1.0 ∪ A4) | 0.224 | 0.471 | 0.134 | 0.675 | — | — |
| oracle head choice, A4 seed 43 | 0.224 | 0.470 | 0.134 | 0.673 | — | — |
| learned GBT 3-head router (R ∪ D ∪ A4), C1 λ1.0 | 0.203 | 0.430 | 0.120 | 0.617 | — | — |
| oracle 3 heads (+ decoder), C1 λ1.0 / A1a | 0.230 / 0.228 | 0.478 / 0.476 | — | — | — | — |

## T2. Router
| router | routing acc. on disagreements (31.6% of fns) | regret F1 | →retrieval | micro / macro |
|---|---|---|---|---|
| conf (sim1) | 76.8% | 0.024 | — | 0.198 / 0.430 |
| logistic (11 feats) | 80.2% | 0.020 | 15.8% | 0.202 / 0.434 |
| GBT (11 feats) | 82.1% | 0.018 | 16.3% | 0.204 / 0.439 |
Feature ablation (GBT, A1a∪A4): −kNN-sim → 0.201/0.433, acc 80.2%; −A4-conf → 0.203/0.433 but selective@20% 0.686→0.544; only sim1 → 0.196/0.423; only a4_conf → 0.183/0.376.

## T3. Baselines on matched keys — every system trained on OUR train tier (v2 protocol, 190,151 fns)
Same functions for every system; metric v2 (c++filt, template/arg stripping). Baseline retrains 2026-08-27→30: SymGen = CodeLlama-34B + LoRA on our train tier (1 ep, 1,485 steps, 4×A100-40G, job 1201270 → infer 1201284/1201285); BLens = CLAP+PalmTree "ablation-c+p" recipe, COMBO 80 ep + LORD 80 ep, inferBest = epoch 67 (jobs 1200059 → 1202627). Scorers: `results/matched_baselines_sgours_a4v1.json` (1204611), `results/blens_ours_v2_matched.json` (1204612).

**T3a. SymGen samples (FT 7,471 fns / 24 pkgs; NCT 6,986 fns / 23 pkgs)** — micro / macro / EM
| system | FT sample | NCT sample |
|---|---|---|
| SymGen, LoRA retrained on our tier | 0.124 / 0.138 / 2.7% | 0.260 / 0.258 / 7.7% |
| SymGen, original LoRA (own corpus; overlaps 9 test pkgs) — legacy | 0.120 / 0.137 / 2.8% | 0.236 / 0.234 / 7.3% |
| ours: BAP retrieval (A1a) | 0.039 / 0.047 / 1.4% | 0.744 / 0.747 / 68.8% |
| ours: BAP decoder (A1a) | 0.034 / 0.044 / 1.2% | 0.660 / 0.659 / 57.8% |
| ours: A4 gen head (run 1 / run 2) | 0.115 / 0.124 / 3.3%  ·  0.119 / 0.122 / 3.8% | 0.633 / 0.633 / 44.0%  ·  0.591 / 0.590 / 38.3% |
| ours: GBT union (run 1 / run 2) | 0.112 / 0.120 / 3.5%  ·  0.114 / 0.118 / 3.7% | 0.780 / 0.781 / 67.9%  ·  0.765 / 0.767 / 65.6% |
| oracle union | 0.130 / 0.139 / 3.7% | 0.827 / 0.828 / 73.8% |
Fair retraining moves SymGen by +0.004 (FT) / +0.025 (NCT): the 34B model with our train tier still reads FT at 0.124 vs A4-220m 0.115 (A4 has the higher EM, 3.3% vs 2.7%), and NCT at 0.260 vs our retrieval 0.744.

**T3b. BLens on ALL matched test keys (267,668 fns, 50 pkgs; C1 λ1.0 retrieval encoder, A4 run 1)** — micro / macro / EM
| system | all | FT (223,158; 27 pkgs) | NCT (44,510; 23 pkgs) | seen-name (32,996) | novel-name (234,672) |
|---|---|---|---|---|---|
| BLens (retrained on our tier) | 0.059 / 0.171 / 1.3% | 0.013 / 0.021 / 0.2% | 0.287 / 0.346 / 7.1% | 0.382 / 0.223 / 10.7% | 0.013 / 0.044 / 0.0% |
| ours: retrieval (R) | 0.126 / 0.378 / 8.8% | 0.039 / 0.061 / 0.7% | 0.562 / 0.750 / 49.4% | 0.769 / 0.530 / 70.8% | 0.035 / 0.098 / 0.0% |
| ours: BAP decoder (D) | 0.076 / 0.236 / 3.0% | 0.029 / 0.045 / 0.3% | 0.311 / 0.461 / 16.5% | 0.426 / 0.333 / 24.5% | 0.027 / 0.057 / 0.0% |
| ours: A4 gen head | 0.184 / 0.359 / 6.5% | 0.121 / 0.132 / 2.4% | 0.502 / 0.626 / 27.3% | 0.618 / 0.493 / 39.2% | 0.123 / 0.181 / 1.9% |
| ours: GBT union (R ∪ A4) | 0.204 / 0.431 / 10.2% | 0.119 / 0.128 / 2.5% | 0.632 / 0.786 / 49.1% | 0.796 / 0.576 / 69.2% | 0.121 / 0.178 / 1.9% |
| oracle union | 0.223 / 0.463 / 11.1% | 0.133 / 0.151 / 2.6% | 0.673 / 0.829 / 54.1% | 0.849 / 0.635 / 76.3% | 0.135 / 0.218 / 2.0% |
BLens caveats (report them): (i) it abstains on 46.0% of rows (LORD confidence threshold; empty output = F1 0, EM 0); (ii) it emits its own expanded name vocabulary (init→initialise, dir→directory, mbedtls→mb_ed_tls, 20-token cap) while every system is scored against the raw truth — scored against BLens's *own* canonical targets it reaches 0.090 micro over all rows / 0.167 on the 54% it answers, still below our retrieval head alone (0.126); (iii) its COMBO validation loss rose monotonically on our package-disjoint val (11.3→13.8) while train loss fell (10.3→3.6) — the recipe was run as published. The April-model row (BLens 0.026 on the FT sample) is retired.

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
| C1 soft λ0.3 / λ1.0 (final ckpts) | 0.158 / 0.159 | retrieval 0.124 / 0.384  ·  0.126 / 0.385; union 0.2045 / 0.4367  ·  0.2052 / 0.4387 | +0.009 / +0.011 retrieval; union wash → λ1.0 kept |
Finding: val_xproj (GNU-heavy FT packages) cannot arbitrate GNU-heavy data additions — three cases.

## T6. Per-package (2-head GBT union, C1 λ1.0 ∪ A4; largest packages)
| package | regime | n | retrieval | A4 | union | oracle |
|---|---|---|---|---|---|---|
| bdb | FT | 96,210 | 0.038 | 0.155 | 0.153 | 0.162 |
| icu | FT | 48,369 | 0.022 | 0.042 | 0.042 | 0.058 |
| mbedtls | FT | 30,322 | 0.030 | 0.104 | 0.099 | 0.114 |
| fossil | NCT | 16,196 | 0.282 | 0.249 | 0.366 | 0.409 |
| libsodium | FT | 9,363 | 0.025 | 0.079 | 0.083 | 0.096 |
| gettext | FT | 6,713 | 0.124 | 0.191 | 0.196 | 0.223 |
| openresty | NCT | 5,028 | 0.477 | 0.564 | 0.636 | 0.668 |
| mutt | FT | 4,587 | 0.031 | 0.105 | 0.100 | 0.117 |
| cvs | FT | 4,096 | 0.150 | 0.330 | 0.340 | 0.371 |
| recutils | FT | 3,577 | 0.182 | 0.266 | 0.253 | 0.300 |
| angie | NCT | 3,467 | 0.685 | 0.677 | 0.788 | 0.831 |
| nginx118 | NCT | 3,438 | 0.763 | 0.715 | 0.853 | 0.891 |
Full table: results/router3_c1l10_a4v1.json → per_package_2head.

Sources: results/dualhead_v2/RESULTS_LEDGER.md (job ids per row), results/*.json.
