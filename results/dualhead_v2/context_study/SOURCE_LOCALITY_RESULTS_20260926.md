# Source-locality follow-up: results only (organised by the follow-up plan sections)

Population definitions. "Labelled subset" = LineageBench test functions whose binary has a debug build and whose target has a
source-file label from addr2line: 103,397 functions, 390 of 611 binaries, 27 packages (results/ctx_layout/files_map_test.json).
"Matched population" (plan §12–§13) = labelled functions whose source file has ≥1 other function outside the ±10 window AND ≥1
labelled different-file function outside the window; same-file-out (SFO) and different-file (DFF) receive the SAME number of
context functions n = min(20, |same-file pool|, |different-file pool|): train 112,408 functions / 788 binaries (n median 20,
mean 15.2, n=20 for 59%), val 2,146 / 92, test 68,624 / 223 binaries / 24 packages (n mean 14.3, n=20 for 52%).
Source-file identity is used only as an analysis oracle; no deployed input uses it. Terminology: "same source file" (DWARF
line-table paths), not compilation unit. Raw files: this folder (provenance_test.json, provenance_analysis.json,
provenance_extra.json, samefile_population.json, samefile_analysis.json, eval_matched_*.json, sens_lineagebench_su/du.json).

## §5–§6 Context-token provenance (adopted ±10 digest, labelled subset, 103,397 functions)

| token category (contributing neighbours) | all digest tokens | tokens matching a ground-truth sub-token |
|---|---|---|
| same-source-file only | 55.4% | 64.0% |
| both same- and different-file | 9.8% | 23.1% |
| different-source-file only | 32.8% | 12.3% |
| unknown (no DWARF) | 2.1% | 0.6% |

Share of digest tokens with ≥1 same-file contributor: 0.651. First ground-truth sub-token ("prefix") in the digest: from a
same-file neighbour 21.9% of functions, from another file only 3.5%, absent 74.6%.

## §8 Prefix-evidence coverage per context source (40-token digest)

| source | n | first GT sub-token present | any remaining sub-token present | all remaining present |
|---|---|---|---|---|
| ±10 address | 267,726 | 0.359 | 0.326 | 0.052 |
| callers + callees | 267,726 | 0.207 | 0.231 | 0.058 |
| 20 random same-binary | 267,726 | 0.220 | 0.192 | 0.013 |
| address ∪ callers/callees | 267,726 | 0.385 | 0.372 | 0.073 |
| same-file outside ±10 (SU, labelled subset, incl. 48.9% empty digests) | 103,125 | 0.153 | 0.149 | 0.011 |
| different-file outside ±10 (DU, labelled subset, 2.4% empty) | 103,125 | 0.257 | 0.200 | 0.010 |

## §9 Prefix and remainder prediction accuracy

Inference arms on the adopted head, labelled subset (103,397 functions; "none" = no-context head trained on full data):

| arm | prefix recall | remainder F1 |
|---|---|---|
| none | 0.1426 | 0.1275 |
| ±10 address | 0.1829 | 0.1541 |
| random | 0.1599 | 0.1321 |
| same-file outside ±10 | 0.1604 | 0.1369 |
| different-file outside ±10 | 0.1542 | 0.1295 |

Δ(address − none): prefix recall +0.0403, remainder F1 +0.0266.
Matched-trained heads (test, 68,624 functions): see §14 table below (prefix 0.137 / 0.137 / 0.153 / 0.161; remainder 0.129 / 0.131 / 0.142 / 0.143).

## §10–§15 Matched-population trained comparison (plan §14 table; test, 68,624 functions, 223 binaries, 24 packages)

| context condition (all four trained on the same 112,408 rows) | package F1 | function F1 | prefix recall | remainder F1 | val F1 (2,146) |
|---|---|---|---|---|---|
| NONE-M: no context | 0.2444 | 0.1459 | 0.1372 | 0.1292 | 0.2499 |
| DFF: different source file, outside ±10 | 0.2540 | 0.1481 | 0.1370 | 0.1314 | 0.2464 |
| SFO: same source file, outside ±10 | 0.2512 | 0.1614 | 0.1532 | 0.1416 | 0.2444 |
| ADDR-M: ±10 address window | 0.2688 | 0.1659 | 0.1612 | 0.1433 | 0.2563 |
| reference, inference arms of the adopted head on the same functions: random | 0.2750 | 0.1521 | 0.1517 | 0.1310 | |
| reference, inference arm: callers + callees | 0.2898 | 0.1647 | 0.1700 | 0.1397 | |
| reference, full-data heads (190K training rows): none / address / callers+callees / union | 0.2729 / 0.3180 / 0.3125 / 0.3132 | 0.1425 / 0.1760 / 0.1845 / 0.1854 | 0.1356 / 0.1786 / 0.1871 / 0.1906 | 0.1249 / 0.1499 / 0.1562 / 0.1566 | |

By matched context size n (SFO/DFF receive n functions; ADDR-M always the full window), function F1:

| n | functions | NONE-M | DFF | SFO | ADDR-M |
|---|---|---|---|---|---|
| 1–9 | 20,340 | 0.1311 | 0.1344 | 0.1434 | 0.1532 |
| 10–19 | 12,537 | 0.1488 | 0.1465 | 0.1534 | 0.1619 |
| 20 | 35,747 | 0.1534 | 0.1563 | 0.1744 | 0.1744 |

## §22 Statistics (matched population)

Package-level paired bootstrap (24 packages, 10,000 resamples), Δ package F1 [95% CI], packages where first is higher:
- SFO − DFF: −0.0028 [−0.0224, +0.0175], 14/24
- SFO − random (inference arm): −0.0238 [−0.0756, +0.0227], 10/24
- ADDR-M − SFO: +0.0176 [+0.0018, +0.0317], 20/24
- ADDR-M − DFF: +0.0148 [+0.0031, +0.0266], 18/24
- SFO − NONE-M: +0.0068 [−0.0109, +0.0263], 15/24
- DFF − NONE-M: +0.0096 [−0.0004, +0.0201], 16/24
- ADDR-M − NONE-M: +0.0244 [+0.0133, +0.0347], 21/24

Function-level, binary-cluster paired bootstrap (223 binaries, 10,000 resamples), Δ function F1 [95% CI], binaries where first is higher:
- SFO − DFF: +0.0133 [+0.0071, +0.0194], 92/223
- ADDR-M − SFO: +0.0045 [+0.0007, +0.0090], 157/223
- ADDR-M − DFF: +0.0178 [+0.0111, +0.0254], 142/223
- SFO − NONE-M: +0.0154 [+0.0094, +0.0214], 124/223
- DFF − NONE-M: +0.0021 [−0.0019, +0.0069], 161/223
- ADDR-M − NONE-M: +0.0199 [+0.0127, +0.0281], 188/223

Inference-swap version (adopted head, both SU and DU digests non-empty, 51,944 functions, 24 packages): none 0.1746 |
address 0.2132 | SU 0.2067 | callers+callees swap 0.1975 | random 0.1822 | DU 0.1774; package bootstrap SU − DU +0.0468
[+0.0169, +0.0763] 20/24; address − SU +0.0121 [−0.0002, +0.0236] 18/24; SU − random +0.0354 [+0.0077, +0.0656]; DU − random
−0.0114 [−0.0267, +0.0033].

## §16 Preserved locality vs. prediction gain

Adopted head vs. full-data no-context head, labelled subset, by same-file share of the ±10 window (functions):

| same-file share | functions | address F1 | no-context F1 | Δ |
|---|---|---|---|---|
| 0% | 3,038 | 0.2328 | 0.2039 | +0.0290 |
| 1–25% | 14,531 | 0.1619 | 0.1343 | +0.0276 |
| 25–50% | 21,015 | 0.1571 | 0.1285 | +0.0286 |
| 50–75% | 23,504 | 0.1761 | 0.1367 | +0.0394 |
| 75–100% | 41,309 | 0.2057 | 0.1694 | +0.0363 |

By same-file share of the digest tokens: 0 → +0.0080 (25,910); (0, 0.5) → +0.0354 (18,144); [0.5, 1) → +0.0491 (21,384); 1 → +0.0426 (37,959).

Matched-trained heads (ADDR-M vs NONE-M, matched population), by same-file share of the window:

| same-file share | functions | ADDR-M F1 | NONE-M F1 | Δ |
|---|---|---|---|---|
| 0% | 999 | 0.0725 | 0.1015 | −0.0290 |
| 1–25% | 3,308 | 0.0588 | 0.0811 | −0.0223 |
| 25–50% | 7,222 | 0.1254 | 0.1094 | +0.0160 |
| 50–75% | 17,792 | 0.1471 | 0.1244 | +0.0226 |
| 75–100% | 39,303 | 0.1932 | 0.1690 | +0.0242 |

Gain by where the ground-truth prefix in the digest came from (adopted head vs none): same file +0.0771 (n 22,634; 0.332 → 0.409);
other file +0.0849 (n 3,573; 0.141 → 0.226); absent +0.0190 (n 77,190; 0.097 → 0.116).

## §17 Descriptive regression (per-function Δ = address − none, labelled subset, n 103,397, OLS)
intercept 0.0145; same-file share of window −0.0157; number of same-file window functions +0.0010; prefix present in digest
+0.0283; number of ground-truth sub-tokens present +0.0155; Clang build −0.0078; R² 0.0224.

## §18 Deployment cost and coverage (LineageBench test tier, 610 binaries, 268,178 functions)

| | address ±10 | callers + callees |
|---|---|---|
| digest construction, CPU, per function (decompilation excluded) | 0.172 ms | 0.212 ms |
| digest construction, whole test tier | 46.2 s | 56.9 s |
| functions with a non-empty digest | 93.9% | 73.3% |
| mean number of context functions | 19.8 | 5.5 |
| extra analysis required | address ordering | FUN_ reference resolution |

Call resolution: 1,935,979 FUN_ references in the decompiled test functions; 74.4% resolve to a function retained by the protocol
(the remainder point to functions removed by deduplication / linker-visible-name filtering or are unresolved); 99.8% of functions
contain ≥1 FUN_ reference; 91.4% have ≥1 resolved caller or callee; 26.7% end with an empty digest. Punstrip (Debian builds):
59.3% of functions with a resolved relation, mean 2.0, 47.3% empty digests. Both sources require the same Ghidra decompilation.

## Source-file locality of the window (from the locality study, for reference)
±10 window same-file share 0.637 (±5 0.745, ±20 0.504); majority same-file for 69.7% of functions; ≥1 same-file neighbour for 97.1%;
uniform-random expectation 0.078; callers+callees 0.447. GCC O0/O1/O2/O3 0.70/0.63/0.62/0.63; Clang O0–O3 0.48/0.54/0.43/0.54.
Punstrip/Debian: ±10 0.587, random expectation 0.338, callers+callees 0.442.

## Jobs and files
Provenance: 1342155 build → 1342156 [su, du] inference arms → 1342157 analysis; 1342616 extra diagnostics + 5-bin/OLS rerun.
Matched heads: 1342605 build → trains sfo 1342607, dff 1342609, addrm 1342611, nonem 1342613 → predicts 1342608/10/12/14 → 1342615 analysis.
Scripts: scripts/ctx_checks/{provenance_build,provenance_analyze,provenance_extra,samefile_build,samefile_analyze,tu_locality}.py.
Wulver: dh2/results/ctx_layout/*.json; dh2/results/a4_ctx_{su,du,sfo,dff,addrm,nonem}_dm/; dh2/results/a4_codet5p220m_ctx{sfo,dff,addrm,nonem}_dm_v1/.
