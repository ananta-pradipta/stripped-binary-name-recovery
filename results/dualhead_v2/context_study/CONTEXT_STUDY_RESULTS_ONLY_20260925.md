# Context-source study: results only (organised by the action plan v2 sections)

Common setup for every trained head: CodeT5+ 220M generation head, LineageBench training tier (190,133 rows), 3 epochs,
bs 4 × accum 8, lr 5e-5, max-src 1280, bf16, greedy decoding, canonical (demangled) targets, first-occurrence masking,
40-token context digest of string-literal + named-library-call identifier tokens ranked by number of distinct source
functions containing them; only the SET OF SOURCE FUNCTIONS differs. Scorer: sub-token F1 (metric v2, canonical names),
exact match case-sensitive on canonical names. Test tier: 268,136 scored functions, 50 packages (27 FT, 23 RHT).
Raw files: this folder (eval_*.json, sens_*.json, stats_*.json, tu_locality_*.json, layout_delta.json, bootstrap.json,
coverage.json).

## Plan §5 / §7 / §8 / §9 — Controlled context matrix on LineageBench (trained heads)

| Variant | Context source | val F1 | test F1 (fn) | test F1 (pkg) | EM | FT F1 | RHT F1 | seen F1 | novel F1 | pred. uniqueness |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | none (decompiled text only) | 0.200 | 0.184 | 0.360 | – | – | – | – | – | – |
| C1 | 20 random same-binary functions (seed 20260925) | 0.2065 | 0.1927 | 0.3891 | 5.52% | 0.1243 | 0.5346 | 0.6395 | 0.1289 | 0.201 |
| C2 | callers + callees only (cap 40) | 0.2158 | 0.2138 | 0.4030 | 6.13% | 0.1451 | 0.5578 | 0.6795 | 0.1474 | 0.160 |
| C3 | ±10 address neighbours (adopted) | 0.2176 | 0.2125 | 0.4000 | 5.91% | 0.1444 | 0.5530 | 0.6659 | 0.1478 | 0.169 |
| C4 | ±10 address ∪ callers + callees, one ranking, 40 tokens | 0.2192 | 0.2165 | 0.4052 | 5.98% | 0.1479 | 0.5600 | 0.6765 | 0.1509 | – |
| C5 | address + call-graph + binary-pool (3-tier, Aug 2026; call tier resolved on non-PIE binaries only) | 0.214 | 0.217 | 0.398 | – | – | – | – | – | – |

Validation-subset (4,000 rows) best during training: C3 0.2249 (FT 0.1691, RHT 0.6701); C2 0.2252; C1 0.2102 (FT 0.1497, RHT 0.6936); C4 0.2269 (FT 0.1725, RHT 0.6618).
Full validation tier (10,617 rows) by regime: C3 FT 0.1691 / RHT 0.6701; C2 FT 0.1609 / RHT 0.6928; C1 FT 0.1510 / RHT 0.6877; C4 FT 0.1668 / RHT 0.6773.

Digest statistics (test rows): C1 mean 19.9 source functions, 0.8% empty digests; C2 91.4% of functions with ≥1 resolved caller/callee, mean 5.5 functions, 26.7% empty digests; C3 mean 19.8, 6.1% empty; C4 mean 24.1, 4.1% empty; ±5 mean 9.9, 12.2% empty; ±20 mean 39.2, 2.4% empty. Training rows: C2 87.1% resolved, mean 4.3, 27.4% empty; C1 mean 19.8, 0.1% empty; C4 mean 22.7, 2.7% empty.

## Plan §13 — Paired statistics on the trained heads (test; package-level paired bootstrap, 10,000 resamples over 50 packages)

| Comparison | Δ F1 (fn) | Δ F1 (pkg) | 95% CI (pkg) | packages where first is higher |
|---|---|---|---|---|
| C3 address − C2 callgraph | −0.0013 | −0.0030 | [−0.0109, +0.0051] | 23 / 50 |
| C3 address − C1 random | +0.0198 | +0.0109 | [+0.0022, +0.0192] | 33 / 50 |
| C2 callgraph − C1 random | +0.0211 | +0.0139 | [+0.0049, +0.0227] | 38 / 50 |
| C4 union − C3 address | +0.0040 | +0.0052 | [−0.0006, +0.0106] | 31 / 50 |
| C4 union − C2 callgraph | +0.0027 | +0.0023 | [−0.0043, +0.0082] | 31 / 50 |

Per-function relations between heads (test): C3 and C2 emit the same prediction for 12.1% of functions; F1 = 1 for 7.4% (C3) and 7.4% (C2), both 5.9%.
Oracle max per function: max(C3, C2) 0.2612 (FT 0.1887, RHT 0.6237); max(C3, C2, C4) 0.2825.
Selection by higher generation confidence per function (no fitting): C3/C2 → val 0.2302, test 0.2224 fn / 0.4199 pkg (C2 chosen for 51.8% of test functions; 47.7% on val); C3/C4 → test 0.2197; C2/C4 → 0.2244; C3/C2/C4 → 0.2247. Confidence-margin sweep for C3/C2: val-optimal margin +0.05 → val 0.2308; test-optimal margin −0.05 → test 0.2224.

## Plan §5 (inference-only complement) — Context swapped at inference on the adopted head C3 (LineageBench test)

| Source fed to the adopted head | F1 (fn) | F1 (pkg) | FT | RHT | novel |
|---|---|---|---|---|---|
| ±10 address (reproduction of C3) | 0.2125 | 0.4000 | 0.1444 | 0.5529 | 0.1478 |
| ±20 address | 0.2114 | 0.3957 | 0.1463 | 0.5373 | 0.1494 |
| ±5 address | 0.2078 | 0.3939 | 0.1399 | 0.5474 | 0.1434 |
| ±10 address ∪ callers + callees | 0.2157 | 0.4019 | 0.1481 | 0.5542 | 0.1512 |
| callers + callees | 0.1960 | 0.3585 | 0.1379 | 0.4869 | 0.1413 |
| 20 random (seed 20260925, target excluded) | 0.1836 | 0.3521 | 0.1313 | 0.4452 | 0.1355 |
| 20 random, ±10 window excluded, seed 1 | 0.1825 | 0.3417 | 0.1312 | 0.4396 | 0.1354 |
| 20 random, ±10 window excluded, seed 2 | 0.1827 | 0.3444 | 0.1313 | 0.4396 | 0.1356 |
| empty digest | 0.1703 | 0.3095 | 0.1204 | 0.4197 | 0.1241 |

Package-level paired bootstrap of the swap arms vs the ±10 arm: random −0.0479 [−0.0673, −0.0296] (address higher in 41/50); callers + callees −0.0415 [−0.0552, −0.0282] (42/50); ±5 −0.0061 [−0.0095, −0.0026] (37/50); ±20 −0.0043 [−0.0095, +0.0008] (26/50); empty −0.0905 [−0.1152, −0.0680] (47/50).

## Plan §10 — Layout sensitivity on LineageBench (Δ = F1 of C3 − F1 of C0, same test functions, by build)

| Compiler | Optimisation | n functions | No context | Address context | Δ |
|---|---|---|---|---|---|
| GCC | O0 | 77,854 | 0.1683 | 0.2008 | +0.0325 |
| GCC | O1 | 55,682 | 0.1796 | 0.2057 | +0.0261 |
| GCC | O2 | 59,500 | 0.1875 | 0.2095 | +0.0220 |
| GCC | O3 | 55,758 | 0.1772 | 0.1979 | +0.0207 |
| Clang | O0 | 1,957 | 0.0820 | 0.1145 | +0.0325 |
| Clang | O1 | 8,045 | 0.3271 | 0.3796 | +0.0525 |
| Clang | O2 | 1,358 | 0.0859 | 0.0950 | +0.0091 |
| Clang | O3 | 7,982 | 0.3285 | 0.3730 | +0.0444 |

Joined functions: 268,136 (C0 predictions from a4_codet5p220m_v1, full test tier). Builds: `CFLAGS="-g -O{0..3}"`, default GCC/Clang, no LTO, no -ffunction-sections, no PGO.

## Plan §11 — Translation-unit / source-file locality (addr2line on the unstripped builds)

Share of the source set that lies in the same source file as the target function (scored test functions; source set = functions retained by the protocol for that binary):

| Corpus | ±5 | ±10 | ±20 | ±10: majority same file | ±10: ≥1 same file | callers + callees (mean n) | uniform-random expectation | ±10 unknown (no DWARF) |
|---|---|---|---|---|---|---|---|---|
| LineageBench (390 of 611 test binaries with debug ELF; 103,415 rows; 27 packages) | 0.745 | 0.637 | 0.504 | 0.697 | 0.971 | 0.447 (4.65) | 0.078 | 0.018 |
| Punstrip / Debian (218 of 451 binaries with usable dbgsym; 6,280 rows; 96 packages) | 0.685 | 0.587 | 0.489 | 0.635 | 0.946 | 0.442 (1.55) | 0.338 | 0.096 |

LineageBench ±10 same-file share by build (rows; ±10 same / majority / callers+callees same): GCC O0 (29,002) 0.702 / 0.780 / 0.520; GCC O1 (16,803) 0.627 / 0.687 / 0.468; GCC O2 (24,188) 0.622 / 0.684 / 0.401; GCC O3 (23,203) 0.628 / 0.683 / 0.308; Clang O0 (1,434) 0.484 / 0.395 / 0.622; Clang O1 (4,037) 0.541 / 0.598 / 0.673; Clang O2 (772) 0.434 / 0.316 / 0.512; Clang O3 (3,976) 0.539 / 0.580 / 0.678. Clang ±10 unknown share 0.16–0.31. By ELF type: DYN (58,363 rows) 0.629, EXEC (45,052) 0.648. By regime: FT (81,795) 0.616, RHT (21,620) 0.717.
Coverage of the locality analysis: 210 LineageBench test binaries (raw_wulver harvest) have no surviving debug ELF; 3% of rows skipped for no DWARF at the target. Punstrip: 233 binaries with <20% DWARF coverage skipped, 1,272 rows without DWARF at the target skipped.

## Plan §12 — Evidence-coverage diagnostics (LineageBench test, 267,726 scored functions with ≥1 name sub-token)

| Context source (40-token digest) | ≥1 ground-truth sub-token present | first sub-token (prefix) present | all ground-truth sub-tokens present |
|---|---|---|---|
| ±10 address | 0.4706 | 0.3590 | 0.0312 |
| ±20 address | 0.5063 | 0.3872 | 0.0279 |
| ±5 address | 0.4139 | 0.3105 | 0.0292 |
| callers + callees | 0.3096 | 0.2067 | 0.0319 |
| 20 random | 0.3035 | 0.2199 | 0.0034 |
| empty | 0 | 0 | 0 |

Earlier census (job 1210177, 12K novel-name functions with zero self-evidence): ±10 neighbours ≥1 token 65.2%, prefix 42.2%, mean coverage 0.315; direct callers/callees ≥1 token 36.4%, mean 0.170; whole-binary pool ≥1 token 94.4%, prefix 69.8%, mean 0.727.

## Plan §14 — Punstrip confirmation (Punstrip-trained head a4_punstrip_modctx_v1, 451 Debian binaries, 23,873 test functions, our scorer, context swapped at inference)

| Source fed to the Punstrip head | F1 (fn) | F1 (pkg) | EM |
|---|---|---|---|
| P2: ±10 address (reproduction) | 0.4013 | 0.6012 | 14.24% |
| ±10 address ∪ callers + callees | 0.4042 | 0.6018 | 14.27% |
| ±20 address | 0.3999 | 0.6017 | 13.53% |
| ±5 address | 0.3966 | 0.5983 | 13.75% |
| P3: 20 random (seed 20260925) | 0.3683 | 0.5889 | 11.61% |
| 20 random, window excluded, seed 1 | 0.3629 | 0.5785 | 11.10% |
| 20 random, window excluded, seed 2 | 0.3609 | 0.5760 | 10.94% |
| P1: callers + callees | 0.3428 | 0.5643 | 10.95% |
| P0: empty digest | 0.3265 | 0.5509 | 10.52% |

Punstrip digest statistics: callers + callees resolved for 59.3% of functions, mean 2.0 functions, 47.3% empty digests; ±10 mean 18.0, 1.8% empty; random mean 19.1, 0.1% empty; union mean 19.4, 1.1% empty. (Seen/novel splits are not reported for these arms: the Punstrip row files carry a different flag key; use punstrip_strata.py for strata.) Punstrip retrained variants were not run.

## Plan §14 (continuation) — Punstrip TRAINED context comparison (P3 recipe, Punstrip train split; test 23,873 fns / 451 binaries / 174 packages; our scorer)

| head | val F1 | test F1 (fn) | test F1 (pkg) | EM |
|---|---|---|---|---|
| P0 none | 0.4313 | 0.3726 | 0.5844 | 13.44% |
| P1 20 random same-binary, window-excluded (seed 1) | 0.4412 | 0.3934 | 0.5925 | 13.97% |
| P2 callers + callees | 0.4443 | 0.3862 | 0.6206 | 12.88% |
| P3 ±10 address (existing model) | 0.4901 | 0.4013 | 0.6012 | 14.24% |
| P4 address ∪ callers/callees | 0.4896 | 0.4087 | 0.6036 | 14.27% |

Package-level paired bootstrap (174 packages, 10,000 resamples): P3−P1 +0.0087 [+0.0028, +0.0147] 71/174; P2−P1 +0.0281
[+0.0193, +0.0366] 117/174; P3−P2 −0.0194 [−0.0288, −0.0097] 56/174; P4−P3 +0.0024 [−0.0025, +0.0072] 64/174; P4−P2 −0.0170
[−0.0266, −0.0071] 58/174; P3−P0 +0.0168 [+0.0093, +0.0247] 78/174; P1−P0 +0.0081 [+0.0009, +0.0152] 70/174.
Function-level Δ: P3−P1 +0.0079; P2−P1 −0.0072; P3−P2 +0.0151; P4−P3 +0.0074; P4−P2 +0.0225; P3−P0 +0.0287; P1−P0 +0.0209.
P2/P3 complementarity: same prediction 34.1% (val) / 29.6% (test); exact match P3 24.5%, P2 24.9%, both 21.9% (test); oracle
max(P2,P3) 0.5321 val / 0.4521 test. Max-confidence selection: val 0.4956; test 0.4125 fn / 0.6343 pkg; P2 chosen 40.7% val / 43.7% test.
Digest statistics (Punstrip): callgraph resolved for 61.5% train / 59.3% test functions, mean 2.0, 47% empty digests; random_excl
mean 17.7–18.2, 1.4–1.9% empty; union 18.8–19.4, ~1% empty; ±10 17.6–18.0, 1–2% empty.
Cross-benchmark (F1 fn / pkg): none 0.184/0.360 vs 0.373/0.584; random 0.193/0.389 vs 0.393/0.593; callers+callees 0.214/0.403
vs 0.386/0.621; ±10 address 0.213/0.400 vs 0.401/0.601; union 0.217/0.405 vs 0.409/0.604 (results_table.csv).

## Reproduction checks
- LineageBench builder ±10 mode vs adopted training data (results/a4_modctx): 278,753 val+test rows compared, 0 mismatches.
- Punstrip builder ±10 mode vs punstrip/data/test.jsonl: 23,873 rows compared, 0 mismatches.
- Inference-swap ±10 arm reproduces the adopted head's test scores (0.2125 / 0.4000 / FT 0.1444 / novel 0.1478); Punstrip ±10 arm reproduces 0.4013 / 0.6012.

## Other observations recorded during the study (facts, no interpretation)
- The August 3-tier builder (a4_build_poolctx.py) resolved call targets by string-matching the row key; on PIE binaries (353 of 611 test binaries, 58%) this resolved 0 references (acct_ac_O0: 0/176), on non-PIE binaries it resolved them (cvs_cvs_O0: 6,963/8,156).
- Decompilation files (dh2/symgen_v2/decomp) contain only the protocol's scored functions; the ±K window is therefore over retained functions (e.g. nginx118_O2: 1,159 labelled functions, 397 retained).
- LineageBench data mask the first occurrence of the Ghidra placeholder; Punstrip data mask every occurrence.
- Three random draws at inference on LineageBench span 0.1825–0.1836 F1.

## Jobs
LineageBench: build 1334179; train C2 1334180, C1 1334181, C4 1336225; predict 1334183, 1334184, 1336226; swap arrays 1334182 (±10, ±5, ±20, random, empty, callers+callees) and 1336227 (union, random_excl s1, random_excl s2); build2 1336224.
Punstrip: build + locality 1335916; arrays 1335917 (six arms) and 1336228 (union, random_excl s1, s2).
Diagnostics: scripts/ctx_checks/ctx_diagnostics.py (layout, coverage, boot); tu_locality.py (local + Wulver, merged).
