# Context-source study (advisor checks, 2026-09-25)

Question from Prof. Zhang (2026-09-24): is the address-neighbourhood context real, how much depends on our builds,
and how does it compare with caller/callee context and with random same-binary context?
All runs: CodeT5+ 220M generation head, same recipe as the adopted head (a4_codet5p220m_modctx_dm_v1), same 40-token
digest of string-literal + named-library-call identifiers, same masking, canonical targets; only the SET OF SOURCE
FUNCTIONS changes. Builder: `scripts/ctx_checks/a4_build_ctxvar.py` (reproduces the adopted digests byte-for-byte on all
278,753 val+test rows). Files in this folder are the raw outputs (eval JSONs, stats, locality, diagnostics).
Address+call union head (C4): still running at export time (jobs 1336225/1336226); pending rows marked "pending".

## 1. Trained heads (fair comparison; LineageBench test, 268,136 functions, 50 packages)

| context source the head was trained on | val F1 | test F1 (fn) | test F1 (pkg) | EM | FT | RHT | seen | novel |
|---|---|---|---|---|---|---|---|---|
| none (decompiled text only; Table 7) | 0.200 | 0.184 | 0.360 | | | | | |
| 20 random functions of the same binary | 0.2065 | 0.1927 | 0.3891 | 5.5% | 0.1243 | 0.5346 | 0.6395 | 0.1289 |
| ±10 address neighbours (adopted) | 0.2176 | 0.2125 | 0.4000 | 5.9% | 0.1444 | 0.5530 | 0.6659 | 0.1478 |
| callers + callees only | 0.2158 | 0.2138 | 0.4030 | 6.1% | 0.1451 | 0.5578 | 0.6795 | 0.1474 |
| ±10 address ∪ callers/callees (C4) | pending | pending | pending | | | | | |

Package-level paired bootstrap (10k resamples over 50 packages):
- address − callgraph: −0.003 [−0.011, +0.005], address better in 23/50 → tie
- address − random: +0.011 [+0.002, +0.019], 33/50
- callgraph − random: +0.014 [+0.005, +0.023], 38/50

Per-function complementarity of the two local sources: oracle max(address, callgraph) = 0.2612 fn (+0.049; FT 0.189 vs
0.144; RHT 0.624 vs 0.553); the two heads emit the same name for only 12.1% of functions.
Zero-training selection (per function, take the head with the higher generation confidence): val 0.2302 (+0.013);
test 0.2224 fn / 0.4199 pkg (+0.010 / +0.020); callgraph chosen for 52% of functions. Margin sweep: val-optimal d=+0.05
(0.2308), test-optimal d=−0.05 (0.2224) → plain max-confidence is the val-safe rule.

## 2. Context swapped at inference on the ADOPTED head (no retraining)

LineageBench test (F1 fn / pkg / FT / RHT / novel):

| source fed to the adopted head | fn | pkg | FT | RHT | novel |
|---|---|---|---|---|---|
| ±10 address (reproduction) | 0.2125 | 0.4000 | 0.1444 | 0.5529 | 0.1478 |
| ±20 address | 0.2114 | 0.3957 | 0.1463 | 0.5373 | 0.1494 |
| ±5 address | 0.2078 | 0.3939 | 0.1399 | 0.5474 | 0.1434 |
| ±10 address ∪ callers/callees | 0.2157 | 0.4019 | 0.1481 | 0.5542 | 0.1512 |
| callers + callees | 0.1960 | 0.3585 | 0.1379 | 0.4869 | 0.1413 |
| 20 random (seed 20260925) | 0.1836 | 0.3521 | 0.1313 | 0.4452 | 0.1355 |
| 20 random, window excluded, seed 1 | 0.1825 | 0.3417 | 0.1312 | 0.4396 | 0.1354 |
| 20 random, window excluded, seed 2 | 0.1827 | 0.3444 | 0.1313 | 0.4396 | 0.1356 |
| no tokens | 0.1703 | 0.3095 | 0.1204 | 0.4197 | 0.1241 |

Bootstrap (pkg): address − random +0.048 [0.030, 0.067] 41/50; address − callgraph +0.042 [0.028, 0.055] 42/50;
address − win5 +0.006 [0.003, 0.010]; address − win20 +0.004 [−0.001, 0.010] n.s.; address − empty +0.091 [0.068, 0.115].
Caveat: these swap gaps overstate the source difference, because a head trained on one source reads another source
poorly (compare section 1: trained callgraph head 0.2138 vs swap 0.1960).

Punstrip test (Debian builds, 451 binaries, 23,873 functions, Punstrip-trained head, our scorer, F1 fn / pkg / EM):

| source fed to the Punstrip head | fn | pkg | EM |
|---|---|---|---|
| ±10 address (reproduction) | 0.4013 | 0.6012 | 14.2% |
| ±10 address ∪ callers/callees | 0.4042 | 0.6018 | 14.3% |
| ±20 address | 0.3999 | 0.6017 | 13.5% |
| ±5 address | 0.3966 | 0.5983 | 13.8% |
| 20 random | 0.3683 | 0.5889 | 11.6% |
| 20 random, window excluded, seed 1 / 2 | 0.3629 / 0.3609 | 0.5785 / 0.5760 | 11.1% / 10.9% |
| callers + callees | 0.3428 | 0.5643 | 11.0% |
| no tokens | 0.3265 | 0.5509 | 10.5% |

## 3. Translation-unit locality (addr2line on the unstripped builds; share of the source set in the target's source file)

| corpus | ±5 | ±10 | ±20 | ±10 majority same | ±10 any same | callers+callees (mean n) | uniform random |
|---|---|---|---|---|---|---|---|
| LineageBench (390/611 test binaries, 103,415 fns, 27 pkgs) | 0.745 | 0.637 | 0.504 | 0.70 | 0.97 | 0.447 (4.7) | 0.078 |
| Punstrip / Debian (218/451 binaries, 6,280 fns, 96 pkgs) | 0.685 | 0.587 | 0.489 | 0.64 | 0.95 | 0.442 (1.5) | 0.338 |

LineageBench ±10 by build: GCC O0 0.70, O1 0.63, O2 0.62, O3 0.63; Clang O0 0.48, O1 0.54, O2 0.43, O3 0.54 (Clang
binaries: 16–31% of the window has no DWARF, 12 packages only). PIE 0.63 vs non-PIE 0.65; FT 0.62 vs RHT 0.72.
Note: neighbours are the ±10 nearest functions AMONG THOSE THE PROTOCOL RETAINS (train-duplicate bodies and
linker-visible names removed); the decompilation files hold only those functions. Builds: default GCC/Clang `-g -O{0..3}`,
no LTO, no -ffunction-sections, no PGO; Debian packages are default builds too.

## 4. Layout sensitivity: Δ = F1(adopted ±10 head) − F1(no-context head), same functions, by build

| build | n | no ctx | address ctx | Δ |
|---|---|---|---|---|
| GCC O0 | 77,854 | 0.168 | 0.201 | +0.033 |
| GCC O1 | 55,682 | 0.180 | 0.206 | +0.026 |
| GCC O2 | 59,500 | 0.188 | 0.210 | +0.022 |
| GCC O3 | 55,758 | 0.177 | 0.198 | +0.021 |
| Clang O0 | 1,957 | 0.082 | 0.115 | +0.033 |
| Clang O1 | 8,045 | 0.327 | 0.380 | +0.053 |
| Clang O2 | 1,358 | 0.086 | 0.095 | +0.009 |
| Clang O3 | 7,982 | 0.329 | 0.373 | +0.044 |

## 5. Evidence coverage of the 40-token digest (LineageBench test, 267,726 scored functions)

| source | ≥1 GT sub-token | first sub-token (prefix) | all GT sub-tokens |
|---|---|---|---|
| ±10 address | 0.471 | 0.359 | 0.031 |
| ±20 address | 0.506 | 0.387 | 0.028 |
| ±5 address | 0.414 | 0.311 | 0.029 |
| callers + callees | 0.310 | 0.207 | 0.032 |
| 20 random | 0.304 | 0.220 | 0.003 |

Digest statistics (test): callers+callees resolved for 91% of functions, mean 5.5 functions, 27% empty digests;
address ±10: mean 19.8 functions, 6% empty; random: 19.9, 0.8% empty; union: 24.1, 4% empty.

## 6. Defects found on the way
- `a4_build_poolctx.py` (Table 7 "address + call-graph + binary-pool" row) resolved call targets by string-matching the
  row key, which fails on PIE binaries (58% of the test tier): its call tier was empty there. That row is not a
  call-graph measurement; the callers+callees head above is.
- §4.2 said every placeholder occurrence is masked; the LineageBench data mask the definition occurrence only
  (every occurrence on Punstrip). No name information leaks either way; wording corrected.

## 7. Jobs and paths
LineageBench build 1334179; train callgraph 1334180 / random 1334181 / addrcall 1336225; predict 1334183 / 1334184 / 1336226;
sensitivity arrays 1334182 (6 arms), 1336227 (3 arms); build2 1336224. Punstrip build 1335916, arrays 1335917, 1336228.
Wulver: `dh2/results/a4_ctx_<mode>_dm/`, `dh2/results/a4_codet5p220m_ctx{callgraph,random,addrcall}_dm_v1/`,
`dh2/results/a4_modctx_dm_sens_<mode>/`, `dh2/results/ctx_layout/`, `punstrip/data_ctx/`, `dh2/results/a4_punstrip_modctx_sens_<mode>/`.
