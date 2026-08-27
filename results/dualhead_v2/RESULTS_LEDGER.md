# CURRENT HEADLINE (2026-08-26 17:40 UTC) — metric v2, split policy v3, test = 268,178 fns / 50 pkgs (package-disjoint, body-dedup, exported symbols dropped)
| system | test micro | test macro | test FT | test NCT | novel | selective F1 @5/10/20/30% |
|---|---|---|---|---|---|---|
| previous best (A1a retrieval + string rerank) | 0.118 | 0.381 | 0.029 | 0.553 | — | 0.95 / 0.75 / 0.45 / — |
| **routed union: A1a retrieval ∪ A4 gen head (run 1), learned GBT router** | **0.204** | **0.439** | 0.120 | 0.627 | — | **0.96 / 0.88 / 0.69 / 0.53** |
| same with A4 run 2 (val-selected head) | 0.200 | 0.431 | 0.117 | 0.612 | — | 0.93 / 0.85 / 0.65 / 0.50 |
| oracle union (upper bound) | 0.222 | 0.469 | 0.133 | 0.671 | — | — |
| A4 gen head alone (run 1 / run 2) | 0.185 / 0.178 | 0.368 / 0.348 | 0.121 / 0.119 | 0.504 / 0.473 | 0.123 / 0.120 | — |
Retrieval head alone: A1a 0.115/0.380 → **C1 soft λ1.0 (ep7) 0.126/0.385 (+rerank 0.128/0.386)**; in the routed union the gain is +0.001 (0.205/0.439). External: SymGen 5-pkg holdout (9,991 fns) A4 run 1 0.223 / run 2 0.233 micro. Matched-key vs SymGen-34B (FT sample): A4-220m 0.115–0.119 vs 0.120; union NCT sample 0.765–0.780 vs 0.236. Router accuracy on contested rows 82%. Details in the sections below.

# Dual-Head v2 — results ledger (numbers only; prose comes after experiments per workflow rule)
All numbers: dataset v2 split policy v3 (package-disjoint, body-dedup eval), metric v2
(camelCase-aware sub-token F1; C++ demangled in rescore contexts), greedy decode unless noted.
Micro = function-weighted, macro = package-weighted. Regimes: NCT = near-clone transfer,
FT = far transfer. Strata: seen-name / novel-name = GT name present/absent in train tier.

## Main table — full honest TEST (268,178 fns, 50 pkgs)
| System | micro F1 | macro F1 | EM | NCT | FT | seen-name | novel-name |
|---|---|---|---|---|---|---|---|
| P2-Baseline decoder (CCS arch) | 0.086 | 0.308 | 5.2% | 0.383* | 0.019* | 0.526* | 0.016* |
| P2 retrieval (k-NN, frozen enc) | 0.104 | 0.355 | 8.0%* | 0.506* | 0.021* | 0.699* | 0.016* |
| P2 retrieval + string rerank | 0.108 | 0.369 | – | – | – | – | – |
| A1a decoder (+string channel) | 0.100 | 0.339 | 6.1% | 0.465 | 0.027 | 0.635 | 0.024 |
| A1a retrieval | 0.115 | 0.380 | – | – | – | – | – |
| **A1a retrieval + rerank (best)** | **0.118** | **0.381** | – | – | – | – | – |
| A3+ (lit/ABI/rodata) | pending | | | | | | |
*P2 rows partially metric-v1 (pre-camel-fix); shifts ≤ +0.006.

## Selective prediction (router-confidence ranked, test)
| Encoder | F1@5% | F1@10% | F1@20% | F1@50% | AURC↓ | ECE |
|---|---|---|---|---|---|---|
| P2 | 0.960 | 0.700 | 0.407 | 0.188 | 0.708 | 0.0059 |
| A1a | 0.968 | 0.754 | 0.448 | 0.210 | 0.684 | 0.0054 |
Head-choice router: degenerate (always-retrieval) on both encoders — reported as negative finding.

## External baselines — interim, matched clean subsets (metric v2)
| Regime slice | SymGen 34B | our retrieval | our decoder | BLens 0.1B |
|---|---|---|---|---|
| FT clean-24 pkgs (7,532 fns) | **0.120 / 0.135** | 0.037 / 0.050 | 0.032 / 0.047 | 0.026 / 0.030 |
| NCT 23 pkgs (7,063 fns) | 0.234 / 0.232 (EM 7.1%) | **0.707 / 0.709 (EM 64.3%)** | 0.609 / – | not run |
| NCT seen-name | 0.229 | **0.792** | 0.682 | – |
| novel-name (NCT slice) | **0.267** | 0.116 | 0.109 | – |
(our heads = P2 encoder era on these slices; A1a slices TBD with final table re-run)
Final baseline retrains on the frozen final corpus: deferred to end of redesign (user-approved).

## Key supporting measurements
- Leakage (old protocol): test 89.5% body-identical to train; only reserve pool clean (14.7%).
- A1 strings census (test): FT 29.1% fns have ≥1 string; GT-subtoken recall 0.387 within; full name present 4.3% of ALL FT fns.
- A2 magic-immediates census: negative (325/183K fns, 0% precision). Rodata-table variant in A3+.
- Vocab-oracle (sampled): test 0.484; retrieval reaches ~21-24% of it (selection gap).
- Efficiency: ours 34M params, ms/function, BAP-only; SymGen 34B + Ghidra decomp, ~2.6 s/function.

## A3+ (p3b_a3_v2_seed42, lit/ABI/rodata channels on top of A1a) — eval_v2 greedy, metric v2, 2026-08-26
| ckpt | val micro | val macro | test micro | test EM | test macro | test FT | test NCT | seen | novel |
|---|---|---|---|---|---|---|---|---|---|
| P2  | 0.1078 | 0.2487 | 0.0797 | 0.0523 | 0.3057 | 0.0192 | 0.3827 | 0.5263 | 0.0160 |
| A1a | 0.1327 | 0.2864 | 0.1000 | 0.0613 | 0.3394 | 0.0270 | 0.4650 | 0.6353 | 0.0236 |
| A3+ | 0.1324 | 0.3002 | 0.0912 | 0.0551 | 0.2937 | 0.0258 | 0.4178 | 0.5764 | 0.0219 |
Per-package test (A1a→A3+): units2 0.741→0.416, gzip2 0.785→0.519, tar2 0.791→0.614, patch2 0.782→0.623, sed2 0.763→0.620, diffutils2 0.822→0.682, grep2 0.819→0.690; better: bsdtar 0.703→0.742, coreutils4 0.719→0.725. 12 better / 37 worse.
Retrieval (emb dump, kNN top-1, metric v2) and router — A1a vs A3+:
| head | val micro/macro | test micro/macro | test +string rerank (α=0.8) | router selective F1 @5/10/20% (test) |
|---|---|---|---|---|
| A1a | 0.1556 / 0.3298 (rerank) | 0.1147 / 0.3797 | 0.1176 / 0.3814 | 0.954 / 0.754 / 0.448 |
| A3+ | 0.1436 / 0.3150 (rerank) | 0.1039 / 0.3588 | 0.1067 / 0.3591 | 0.954 / 0.695 / 0.407 |
Note: the retrieval-head VAL already showed the drop (0.156→0.144) while the decoder val did not — gate BAP-side changes on retrieval val + per-regime decoder val. A3+ CLOSED (negative).

## A4 generation head (CodeT5+ 220m, masked stripped-Ghidra decomp → name), run 1 = our 190K rows, 3 ep — 2026-08-26
| head | val_xproj micro | val FT | val NCT |
|---|---|---|---|
| BAP decoder A1a (greedy) | 0.1327 | 0.0693 | 0.6894 |
| BAP retrieval A1a + rerank | 0.1556 | — | — |
| **A4 run 1 (greedy)** | **0.2004** | **0.1470** | 0.6691 |
Test + SymGen-holdout pending (job 1196560).
**Correction (audit 2026-08-26):** the A3+ retrieval/router rows above were computed on UN-ENRICHED inputs (dump/router scripts lacked `enrich_a3`; fixed 98a935a1) and are superseded by the enriched rerun (jobs 1196472/1196473/1196474, enrichment verified in logs):
| head | test retrieval micro/macro | + string rerank | router selective F1 @5/10/20% |
|---|---|---|---|
| A1a | 0.1147 / 0.3797 | 0.1176 / 0.3814 | 0.954 / 0.754 / 0.448 |
| A3+ (enriched, valid) | 0.1068 / 0.3572 | 0.1091 / 0.3578 (+emit 0.1101 / 0.3581) | 0.956 / 0.706 / 0.415 |
A3+ negative holds on decoder, retrieval and router. CLOSED.

## A4 run 1 — TEST (job 1196560, greedy, metric v2, split policy v3; SymGen holdout = 5 pkgs, in_dynsym dropped)
| head | test micro | test macro | EM | FT micro (macro) | NCT micro (macro) | seen | novel | SymGen holdout micro / macro |
|---|---|---|---|---|---|---|---|---|
| BAP decoder A1a | 0.1000 | 0.3394 | 0.061 | 0.0270 (0.0606) | 0.4650 (0.6667) | 0.6353 | 0.0236 | — |
| BAP retrieval A1a + rerank | 0.1176 | 0.3814 | — | — | — | — | — | — |
| **A4 gen head (220m, ours only)** | **0.1852** | 0.3680 | 0.055 | **0.1214 (0.1437)** | 0.5042 (0.6314) | 0.6205 | **0.1231** | **0.2231 / 0.1887** |
Val_xproj: 0.2003 (FT 0.147 / NCT 0.668; macro 0.361). Pred-uniqueness test 0.155; no_decomp 40/268K. Files: results/a4_codet5p220m_v1/val_test_symgen_holdout_{eval.json,preds.tsv}.

## HEAD UNION — A1a retrieval (R) + A1a decoder (D) + A4 gen head (job 1196617, `scripts/union_eval.py`, metric v2, test 268,178 / val 10,617; τ tuned on val)
| system | test micro | test macro | test FT | test NCT | val micro | val macro |
|---|---|---|---|---|---|---|
| R (retrieval top-1, A1a) | 0.1163 | 0.3800 | 0.0289 | 0.5529 | 0.1543 | 0.3325 |
| D (decoder, A1a) | 0.1001 | 0.3394 | 0.0271 | 0.4650 | 0.1327 | 0.2864 |
| A4 (gen head, run 1) | 0.1852 | 0.3680 | 0.1214 | 0.5042 | 0.2003 | 0.3606 |
| regime router (NCT→R, FT→A4) | 0.1933 | 0.4237 | 0.1214 | 0.5529 | 0.2179 | 0.3985 |
| **conf router sim1 ≥ 0.635 → R else A4** | **0.1983** | **0.4300** | 0.1162 | 0.6090 | 0.2127 | 0.3950 |
| conf router margin ≥ 0.07 | 0.1989 | 0.4189 | 0.1187 | 0.5998 | 0.2157 | 0.3984 |
| oracle R∪A4 | 0.2223 | 0.4692 | 0.1327 | 0.6707 | 0.2435 | 0.4367 |
| oracle R∪D∪A4 | 0.2283 | 0.4762 | 0.1386 | 0.6772 | 0.2526 | 0.4459 |
Standing best before A4: R + string rerank 0.1176 / 0.3814. Join: 278,204 rows by (binary, entry addr), 591 by name, 0 missing.

## LEARNED 2-HEAD ROUTER + SELECTIVE PREDICTION (job 1196651, `scripts/router2_eval.py`; router/abstainer trained on VAL only, features = kNN sim1, margin, ext_jacc, str_jacc, d_conf, d_len, n_ext, n_str, n_blocks, A4 conf (teacher-forced geometric-mean token prob), sim1−A4conf)
| system (test) | micro | macro | FT | NCT | %→retrieval |
|---|---|---|---|---|---|
| conf router sim1 ≥ 0.615 | 0.1981 | 0.4303 | 0.1155 | 0.6110 | — |
| learned logreg | 0.2021 | 0.4342 | 0.1196 | 0.6144 | 15.8% |
| **learned GBT** | **0.2044** | **0.4386** | 0.1199 | 0.6269 | 16.3% |
| oracle R∪A4 | 0.2223 | 0.4692 | 0.1327 | 0.6707 | — |
Selective F1 @ coverage (test): | 5% | 10% | 20% | 30% | 50% | 100% |
| retrieval-only, ranked by sim1 (old system) | 0.524 | 0.467 | 0.389 | 0.298 | 0.200 | 0.116 |
| conf router, ranked by max(sim1, A4conf) | 0.705 | 0.657 | 0.559 | 0.471 | 0.333 | 0.198 |
| **GBT router, ranked by GBT-regressed expected F1** | **0.958** | **0.881** | **0.686** | **0.529** | **0.361** | 0.204 |
A4 confidence quintiles on test (F1): 0.048 / 0.068 / 0.084 / 0.143 / 0.583. Previous best selective (retrieval-only router_v2, A1a): 0.954 @5% / 0.754 @10% / 0.448 @20%.

## MATCHED-KEY BASELINE COMPARISON (job 1197222, `scripts/matched_baselines.py`; P4 sample keys; metric v2 for every system; GBT router refit on val)
| system | FT sample (7,471 joined / 24 pkgs) micro / macro / EM | NCT sample (6,986 / 23 pkgs) micro / macro / EM |
|---|---|---|
| SymGen (CodeLlama-34B + LoRA) | **0.1200 / 0.1374** / 0.028 | 0.2358 / 0.2339 / 0.073 |
| BLens (retrained; from P4 interim, same FT keys) | 0.026 / 0.030 / — | — |
| ours A1a retrieval R | 0.0386 / 0.0467 / 0.014 | 0.7440 / 0.7466 / 0.688 |
| ours A1a decoder D | 0.0344 / 0.0442 / 0.012 | 0.6598 / 0.6592 / 0.578 |
| ours A4 gen head (220m, run 1) | 0.1152 / 0.1236 / **0.033** | 0.6329 / 0.6331 / 0.440 |
| ours conf router | 0.1060 / 0.1152 / 0.034 | 0.7672 / 0.7701 / 0.675 |
| **ours learned GBT router (R ∪ A4)** | 0.1123 / 0.1201 / 0.035 | **0.7797 / 0.7811** / 0.679 |
| oracle R ∪ A4 | 0.1297 / 0.1387 / 0.037 | 0.8273 / 0.8281 / 0.738 |
61 FT / 77 NCT sample keys had no row in our router features (dropped by split policy v3 body-dedup) and are excluded for all systems.

## A4 RUN 2 (ours 190K + SymGen 254K capped, 2 ep; job 1196561 / predict 1197243 / union+router 1197244) — vs run 1
| | val micro (FT/NCT) | test micro / macro | test FT | test NCT | novel | SymGen holdout micro/macro | GBT union test | oracle test | GBT selective 5/10/20% |
|---|---|---|---|---|---|---|---|---|---|
| A4 run 1 | 0.2003 (0.147/0.668) | 0.1852 / 0.3680 | 0.1214 | 0.5042 | 0.1231 | 0.2231 / 0.1887 | **0.2044 / 0.4386** | 0.2223 / 0.4692 | 0.958 / 0.881 / 0.686 |
| A4 run 2 | **0.2368 (0.190/0.645)** | 0.1776 / 0.3477 | 0.1185 | 0.4729 | 0.1195 | **0.2331 / 0.1993** | 0.1998 / 0.4309 | 0.2188 / 0.4668 | 0.929 / 0.853 / 0.653 |
Val and the external holdout prefer run 2; our test tier prefers run 1 (per-package: NCT GNU version-pairs lose up to −0.13, FT packages gain ≤ +0.03). Protocol selection (val) → run 2; both reported.
Routing stats (test; heads disagree on 31.6% of functions, ties 68.4%): oracle prefers A4 22.8% / R 8.7% (run 1); GBT routing accuracy on contested rows **82.1%** (run 1) / 82.0% (run 2), overall 94.3%; mean regret 0.018 / 0.019 F1; conf_sim1 76.8% / 75.9%; logreg 80.2% / 79.6%.
Matched-key (run 2): FT sample A4 0.1185/0.1221 EM 3.75% (SymGen 0.120/0.137, 2.8%); NCT sample GBT 0.765/0.767 (R 0.744, SymGen 0.236).

## ROUTER FEATURE ABLATION (job 1197390, `scripts/ablation_router.py`, A4 run 1 ∪ A1a retrieval, GBT refit on val per row; test)
| features | micro | macro | NCT | routing acc (contested) | regret | selective @10% / @20% |
|---|---|---|---|---|---|---|
| all 11 | **0.2044** | **0.4386** | 0.627 | **0.820** | 0.018 | **0.881 / 0.686** |
| − retrieval-sim (sim1, margin) | 0.2014 | 0.4328 | 0.614 | 0.802 | 0.021 | 0.863 / 0.672 |
| − overlap (ext_jacc, str_jacc) | 0.2037 | 0.4375 | 0.624 | 0.813 | 0.019 | 0.873 / 0.682 |
| − decoder (d_conf, d_len) | 0.2040 | 0.4368 | 0.625 | 0.818 | 0.018 | 0.872 / 0.675 |
| − size (n_ext, n_str, n_blocks) | 0.2038 | 0.4375 | 0.625 | 0.817 | 0.019 | 0.878 / 0.674 |
| − A4 conf | 0.2032 | 0.4333 | 0.621 | 0.818 | 0.019 | 0.784 / 0.544 |
| only retrieval-sim | 0.1990 | 0.4227 | 0.603 | 0.789 | 0.023 | 0.565 / 0.414 |
| only A4 (conf, sim1−conf) | 0.1973 | 0.4252 | 0.604 | 0.761 | 0.025 | 0.785 / 0.598 |
| only sim1 | 0.1961 | 0.4234 | 0.596 | 0.763 | 0.026 | 0.429 / 0.378 |
| only a4_conf | 0.1833 | 0.3761 | 0.518 | 0.693 | 0.039 | 0.770 / 0.571 |
Reading: routing is driven by retrieval similarity (largest drop when removed); abstention is driven by A4 confidence (selective@20% 0.686 → 0.544 without it). The two signals are complementary: neither alone gets both.

## PENDING — C1 name-aware contrastive sweep (v2 corpus; jobs 1197461 soft λ0.3 / 1197462 soft λ1.0 / 1197463 exact-name control) and BAP v3 retrain (job 1197330)
Gates (val_xproj, eval_v2): retrieval top-1 > 0.1543 and +rerank > 0.1556 (A1a); decoder per-regime ≥ A1a (FT 0.0693 / NCT 0.6894); novel stratum ≥ 0.06 (C1 target); router selective F1 @10% ≥ 0.754. Test reference: retrieval 0.1147/0.3797 (+rerank 0.1176/0.3814), decoder 0.1000/0.3394, router 0.754@10%/0.448@20%. BAP v3 retrain FINAL train-time val 0.1650 @ep44 (A1a 0.1333 @ep41; val scored 10,373 vs 10,617). Eval chain jobs 1198809–1198813 → rows below when done. Rows to be filled from results/c1<tag>_eval_greedy.json, emb_c1<tag>, router_c1<tag>.json, and results/p4sg_* for the v3 retrain.

## C1 INTERIM — soft-label InfoNCE λ=1.0, EPOCH-7 checkpoint (job 1198233/1198234; v2 corpus; retrieval = kNN top-1 on the encoder embedding, + string rerank α=0.8)
| encoder | val top-1 | val +rerank | test top-1 micro / macro | test +rerank | test +rerank+emit |
|---|---|---|---|---|---|
| A1a (final, ep41) | 0.1543 | 0.1556 | 0.1147 / 0.3797 | 0.1176 / 0.3814 | 0.1176 / 0.3814 |
| **C1 soft λ1.0 @ ep7** | **0.1593** | **0.1617** | **0.1255 / 0.3850** | **0.1278 / 0.3856** | 0.1301 / 0.3864 (thr 0.4) |
Gates G1/G2 passed at epoch 7 (decoder val of this ckpt: 0.1053 — the contrastive term trades decoder CE for embedding quality, as designed). Runs continue; final read on the finished checkpoints.

## C1 RETRIEVAL READS (v2 corpus; kNN top-1 on encoder embedding; + string rerank; metric v2). A1a = standing BAP encoder.
| encoder | training state | val top-1 | val +rerank | test top-1 micro / macro | test +rerank | router selective @10% / @20% (test) |
|---|---|---|---|---|---|---|
| A1a | final (ep41) | 0.1543 | 0.1556 | 0.1147 / 0.3797 | 0.1176 / 0.3814 | 0.754 / 0.448 |
| C1 exact-name control λ0.3 (legacy sampler) | final (ep41) | 0.1595 | 0.1604 | 0.1168 / 0.3861 | 0.1186 / 0.3856 | 0.770 / 0.456 |
| C1 soft λ0.3 (name-aware sampler + hard negs) | **interim ep8** | 0.1595 | 0.1609 | 0.1234 / 0.3859 | 0.1253 / 0.3875 | — |
| **C1 soft λ1.0** (same sampler) | **interim ep7** | 0.1593 | 0.1617 | **0.1255 / 0.3850** | **0.1278 / 0.3856** | — |
Val is saturated (~0.1595 for all three) and cannot separate the variants; test does: graded positives + hard negatives give +0.009–0.011 micro over A1a vs +0.002 for exact-name. Jobs 1198229/1198249/1198234/1198260. Final soft checkpoints pending.
Decoder read of the C1 exact control (eval_v2 greedy, job 1198257): val 0.1350/0.2900 (FT 0.0698 / NCT 0.7077); test 0.1024/0.3465, FT 0.0279 (macro 0.0628), NCT 0.4745 (macro 0.6795), seen 0.6514, novel 0.0240 — vs A1a 0.1000/0.3394, FT 0.0270, NCT 0.4650, seen 0.6353, novel 0.0236. Uniform small plus; the decoder is not where C1's effect lives.

## INTERIM SYSTEM with C1 λ1.0 ep7 encoder ∪ A4 run 1 (jobs 1198275/76/77; router/abstainer fit on val)
| system (test) | micro | macro | FT | NCT | routing acc (contested) | selective @5/10/20/30% |
|---|---|---|---|---|---|---|
| retrieval (C1 ep7, router_v2 r_pred) | 0.1269 | 0.3853 | 0.0395 | 0.5638 | — | — |
| A4 run 1 | 0.1852 | 0.3680 | 0.1214 | 0.5042 | — | — |
| conf router | 0.1972 | 0.4274 | 0.1137 | 0.6145 | 0.747 | — |
| learned logreg | 0.2026 | 0.4333 | 0.1199 | 0.6162 | 0.796 | — |
| **learned GBT** | **0.2052** | **0.4387** | 0.1196 | 0.6332 | 0.805 | 0.943 / 0.880 / 0.690 / 0.534 |
| oracle R∪A4 | 0.2241 | 0.4713 | 0.1340 | 0.6746 | — | — |
| (reference: same with A1a encoder) | 0.2044 | 0.4386 | 0.1199 | 0.6269 | 0.821 | 0.958 / 0.881 / 0.686 / 0.529 |
Reading: C1's +0.011 on the retrieval head becomes +0.001 in the routed union — the router already prefers A4 on most contested rows and C1's gains largely overlap A4's. Oracle +0.002. Final C1 checkpoints may move this slightly.
