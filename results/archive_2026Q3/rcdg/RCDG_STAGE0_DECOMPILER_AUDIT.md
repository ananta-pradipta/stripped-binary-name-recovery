# RCDG Stage 0 — Decompiler Feasibility Audit

Decompiler: Ghidra 11.2.1 headless (JDK 21), stripped-binary input only.
Sample: 4728 functions / 316 binaries (stratified pkg×opt
×size, seed 42; clustered ≤2 binaries/stratum — documented).

## Rates
- decompile success: 4728/4728 = **100.00%** (gate ≥95%)
- AST parse success (tree-sitter C, tolerant): 4728/4728 = **100.00%** (gate ≥90%)
- GT-metadata leakage: **0** (gate 0)

## Size (normalized, parsed functions)
- pseudocode tokens: median 61 · p90 384 · p95 635 · p99 2323
- AST nodes: median 192 · p90 1153 · p95 1962

## Feature coverage (% of parsed)
- array: 37.6%
- branch: 69.5%
- deref: 59.2%
- ext_call: 60.5%
- loop: 33.6%
- return_val: 67.0%
- string: 29.1%

## Failures
- by package (top): {}
- by opt level: {}

## Decision: **PASS**

## Provenance notes (2026-08-16)
- Export ran on HPC (jobs 1182380 / 1182389 / 1182411; general partition, 14 parallel Ghidra instances, ~11 min for 194 binaries). Local WSL run had hit a Ghidra 11.2.1 bug (`AutoAnalysisManager.getTaskTimesString`: `"000".substring(len)` on a negative task time → post-script never runs); zero occurrences on HPC, worker retries 3× anyway.
- **Build-consistency audit** (`experiments_semantic/rcdg_stage0_elfcheck.py`; `results/rcdg/stage0_elfcheck.tsv`, `corpus_elfcheck.tsv`): for 7 sampled binaries the file `data/stripped/<bin>_stripped` is a DIFFERENT build from the `.bir` BAP lifted (curl_curl_O0, libxml2_xmllint_O0/O2, zlib_example_O0/O2, zlib_minigzip_O0/O2). Symptom: 167 `no_function` failures + 124 silently WRONG decompilations (curl). Fixed by exporting from the un-suffixed originals `data/stripped/<bin>` (ratio 0.90–1.00 vs .bir); manifest.tsv updated. Corpus-wide: 927/939 .bir have a consistent stripped ELF somewhere; 15 have a wrong `_stripped` sibling (the 7 above + grep×4 / sed×4 whose consistent copy is under `data/cross_project/stripped/`); 12 (cvs/lighttpd/tinycc) have no local stripped file. Any pipeline resolving `data/stripped/<bin>_stripped` first (e.g. `p1_lex_extract.py`) read the wrong ELF for those 15.
- Strict tree-sitter parse (no ERROR node anywhere): 4676/4728 = 98.90%; the 100% above is the tolerant rate defined in the spec (§10).
- Leak checker fixes made during this audit (all false positives): (i) string regex did not handle escaped chars, (ii) `foo_00` suffix trim collapsed `VAR_10..99` → `VAR` (identity collision, also fixed in normalizer), (iii) C keywords / Ghidra type names, (iv) `.dynsym` data imports (stdout/optarg/errno), (v) decompiler `/* WARNING */` comments. Final: 0 leaks.
