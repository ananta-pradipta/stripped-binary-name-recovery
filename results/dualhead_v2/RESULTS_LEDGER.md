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
