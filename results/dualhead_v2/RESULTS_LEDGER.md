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
