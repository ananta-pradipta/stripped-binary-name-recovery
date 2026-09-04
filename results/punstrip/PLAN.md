# Punstrip / XFL / BLens benchmark evaluation — protocol (v1, 2026-09-03)

Goal: score our single-backbone system on the **BLens cross-project test split** of the Punstrip Debian corpus, head-to-head with the numbers BLens (USENIX Sec'25) published for BLens, XFL, SymLM, AsmDepictor, HexT5 — on the *same functions*, with *their* scorer and with ours. Everything below is gated; a stage does not start until the previous stage's sanity gate passes and is logged in `results/experiment_log.md`.

## 0. Ground truth and splits (fixed, external)
- Corpus manifest: BLens Zenodo artifact (10.5281/zenodo.15119877) `data/xflBlensXProjectData` = [train, val, test] records `(bin_path, addr, raw_name, canonical_name, addr, bin_id, fn_id)`; XFL repo `xfl/evaluation/dataset.txt` (10,047 bins) agrees.
- Sizes: train 394,985 fns / 9,042 bins / 3,112 pkgs; val 18,081 / 367 / 173; test 23,875 / 451 / 174. **Gate 0 (PASSED 2026-09-03): package overlap train∩val = train∩test = val∩test = 0.**
- Strict subset = BLens `strict_setting/projectHashFilterTest.csv` (True = duplicate body, drop) + `forbidden_functions.txt` (GT names dropped) + their forbidden-label list + free-function list (crt stubs). We report **both** the full cross-project test and the strict subset, using their filter files verbatim.
- We never touch the split assignment. Training uses only the train records; model selection uses only val.

## 1. Binary rebuild (Stage 1)
- Source: snapshot.debian.org, binary package `<pkg>` and `<pkg>-dbgsym`, amd64. Selection: newest version with first_seen ≤ 2018-08-30, walking backwards; a version is accepted only if **every** manifest binary of the package matches ≥ 98% of its (address, name) pairs in the dbgsym symtab (pilot: correct versions score 1.000, wrong versions ≤ 0.11).
- Inputs to the model come **only from the shipped Debian binary** (already stripped of .symtab; .dynsym kept). The `.debug` file is used **only** for ground truth and function boundaries. No unstripping is performed on model inputs.
- **Gate 1:** ≥ 95% of test functions recovered in exactly-matched binaries; per-package status logged; unmatched binaries listed and *excluded from every system's score* (baselines re-scored on the same matched subset from BLens's `evaluation/cross-project.csv`).

## 2. Function boundaries and decompilation (Stage 2)
- Same convention as XFL/BLens ("we equip all tools with ground-truth function boundaries"): Ghidra 11.2.1 headless, functions created at the manifest addresses (existing `GhidraExportDecompMasked.py`: `getFunctionAt` → `createFunction`), decompile each, mask the Ghidra placeholder name `FUN_xxxx` → `[MASK]`.
- Module-context digest exactly as in our v2 pipeline (`a4_build_modctx.py`, ±10 address-neighbours, 40 tokens, string literals + named import calls from the stripped binary only). Demangled targets (`dm`) irrelevant here (C corpus) but the canonicaliser is applied identically.
- **Gate 2:** decompile success ≥ 97% of matched functions; `[MASK]` applied to **every** occurrence of the Ghidra name in 100% of kept rows (first-only masking leaked recursive self-calls of exported functions). Name-in-input is audited and *reported as a stratum*, not gated to zero: on this corpus 7.3% of test inputs legitimately contain the GT name (usage/error string literals, single-token names in neighbours' calls) — information every tool sees in the stripped binary. PASSED 2026-09-04: test 99.99%, val 99.9%.

## 3. Leakage controls (all mandatory)
| risk | control |
|---|---|
| package leakage across splits | Gate 0 (verified 0 overlap); our train = BLens train only |
| our existing checkpoints/kNN index (trained on our v2 corpus, GNU-heavy) | **not used** in the fair row: CodeT5p-220m is fine-tuned from the public base checkpoint on Punstrip-train only; kNN index built from Punstrip-train embeddings only |
| overlap between our v2 corpus and Punstrip test packages | measured and reported; only relevant to an optional zero-shot row (our v2 model on Punstrip test), which is labelled and excludes overlapping packages |
| GT names visible in inputs (strings, dynsym) | name-in-input audit; report F1 with/without rows whose GT name is in .dynsym (our stricter view) and on the full key set (their view) |
| body-duplicate functions train↔test | BLens strict filter (their hash list) reported as the strict row; we additionally report our own body-hash dedup rate |
| code-LM pretraining contamination (CodeT5p saw Debian sources on GitHub) | cannot be removed; disclosed; shared with every LLM baseline (SymGen) and with BLens's CLAP pretraining |
| val used for anything but selection | router and abstention thresholds fit on Punstrip-val only; test touched once per system |

## 4. Training (Stage 3) — identical recipe to our v2 heads
- Generation head: `a4_train_codet5p.py`, CodeT5p-220m, max_src 1280, lr 5e-5, 3 epochs, bf16, seed 42, best-on-val; **no** extra corpora. Trained on Punstrip-train modctx rows (~395K).
- Retrieval head: mean-pooled encoder of the same checkpoint, cosine k-NN over Punstrip-train (`c_lmemb_knn.py`).
- Router: 4-feature MLP (`lmemb_router_v2.py`), fit on Punstrip-val. Abstention regressor: fit on val, reported as coverage curve.
- **Gate 3:** training-time val F1 curve logged; prediction uniqueness > 0.1 (collapse check); val system F1 > best single head on val (else report heads only).

## 5. Scoring (Stage 4) — two directions, one key set
- Key set = matched test functions ∩ BLens csv keys. Same keys for every system.
- (a) **Their scorer** (label space verified 2026-09-04: groundtruth = `NLP.tristan_canonical_name` restricted to the original 1024-label vocabulary from the Zenodo tokenizer; our canonicalisation reproduces their groundtruth column on 95.8% of keys, their column is used verbatim as target): `blens/evaluation/evaluator.py` logic (label-set micro-F1 over XFL canonical tokens, forbidden labels removed, free functions handled as they do), with our raw predictions mapped through `NLP.tristan_canonical_name` (their canonicaliser). Validation: re-scoring their csv columns must reproduce their Table 3 (BLens 0.460 cross-project / 0.294 strict, ±0.005) before our numbers are read.
- (b) **Our scorer**: metric v2 sub-token F1 (canon both sides) on raw names for ours; baselines' canonical-token predictions scored against canonical GT (documented as an approximation, since their outputs exist only in the expanded-token space).
- Strata: full / strict / seen-name vs novel-name (relative to Punstrip-train) / per-package macro. Selective-prediction curve for ours.
- **Gate 4:** reproduction of BLens's own number within ±0.005 on their full key set; matched-subset numbers for all baselines reported next to the published ones.

## 6. Order of work
1. Rebuild val+test binaries (347 pkgs, 818 bins) → Gate 1 → decompile → Gate 2. 2. Rebuild train (3,112 pkgs, 9,042 bins) in parallel shards → decompile. 3. Train → Gate 3. 4. Score → Gate 4 → tables. Every stage: job ids, counts and gate results appended to `results/experiment_log.md`.
