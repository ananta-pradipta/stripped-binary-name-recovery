# Related Work — Binary Function Name Recovery Papers

Research compiled 2026-04-05 to identify comparable systems for the paper's related-work and comparison sections.

## Summary table (ranked by reproducibility feasibility)

| # | System | Year/Venue | Input | Output | Code | Feasibility | Key Constraint |
|---|---|---|---|---|---|---|---|
| 1 | **AsmDepictor** | AsiaCCS 2023 | Tokenized x86 assembly (max 300 tokens) | Sub-token sequence (F1) | [agwaBom/AsmDepictor](https://github.com/agwaBom/AsmDepictor) | **HIGH** | Pretrained ckpt + dataset on Zenodo, PyTorch, no IDA. 64GB VRAM to train; inference small |
| 2 | **NERO** | OOPSLA 2020 | IDA-Pro augmented CFG | Name sub-tokens | [tech-srl/Nero](https://github.com/tech-srl/Nero) | **MEDIUM** | Needs IDA Pro 6.95 (paid) + TF 1.13 + Py 3.6 |
| 3 | **DIRTY** | USENIX Sec 2022 | Hex-Rays decompiler output | Variable names + types (function name secondary) | [CMUSTRUDEL/DIRTY](https://github.com/CMUSTRUDEL/DIRTY) | **MEDIUM** | Preprocessed DIRT dataset + ckpt public; task mismatch (variables, not names) |
| 4 | **SymLM** | CCS 2022 | Ghidra ICFG + Trex embeddings | Name sub-tokens (F1/P/R) | [OSUSecLab/SymLM](https://github.com/OSUSecLab/SymLM) | **MEDIUM** | Ghidra-based (free); 2-3 days env setup; Apex install tricky |
| 5 | **XFL** | IEEE S&P 2023 | Static features + call-graph (DEXTER) | Multi-label tags (extreme MLL) | [lmu-plai/xfl](https://github.com/lmu-plai/xfl) | **MEDIUM** | 10K Debian binaries; output is tags, not name strings — metric mismatch |
| 6 | **SymGen** | NDSS 2025 | Ghidra decompiled pseudo-C | Generated name (autoregressive, CodeLlama-7B/34B LoRA) | [OSUSecLab/SymGen](https://github.com/OSUSecLab/SymGen) | **MEDIUM-LOW** | 34B wants A100 80GB; 7B feasible on our 40GB |
| 7 | **BLens** | USENIX Sec 2025 | Ensemble: PalmTree + CLAP + DEXTER + VarCLR | Name sub-tokens (F1) | [lmu-plai/blens](https://github.com/lmu-plai/blens) | **LOW** | 80GB VRAM recommended, 50GB per experiment, 4 upstream embedding models |
| 8 | **NFRE** | ISSTA 2021 | BAP-lifted instructions + CFG | Name sub-tokens | [USTC-TTCN/NFRE](https://github.com/USTC-TTCN/NFRE) | **LOW** | Repo is **dataset-build toolkit only**, no model/checkpoint |
| 9 | **DEBIN** | CCS 2018 | BAP-IR + CRF structured prediction | Names + types (classification) | [eth-sri/debin](https://github.com/eth-sri/debin) | **LOW** | Pinned to Ubuntu 16.04 + gcc 5.4 + OCaml; cite numbers, don't build |
| — | **GenNm** | NDSS 2025 | Decompiled C | **Variable names** (not function names) | arXiv page | **N/A** | Different task — out of scope |
| — | **"FuncNamer"** | — | — | — | — | **N/A** | No paper with this exact title found |

## Per-paper notes

### 1. AsmDepictor — HIGH feasibility
Kim et al., AsiaCCS 2023. Transformer encoder-decoder with per-layer positional embeddings + unique-softmax. Input is plain tokenized x86 assembly, so **no decompiler dependency** — matches closest to our BAP-IR pipeline philosophy. Pretrained model + dataset on [Zenodo 7978756](https://zenodo.org/record/7978756). Reported F1 71.5, Jaccard 75.4. Feasible in ~1 week: pip install, download Zenodo, adapt our binaries into their tokenizer.

### 2. NERO — MEDIUM feasibility
Gluska, David et al., OOPSLA 2020. GNN over IDA-augmented CFG where each node has a (call-target, arg-vector) signature; LSTM decoder with copy mechanism. **Stack is old**: IDA Pro 6.95 required (institutional license or skip entirely), TF 1.13.1, Python 3.6, llvmlite, angr+simuvex. Zenodo has dictionaries + `model_iter495`. Feasible only if IDA license available.

### 3. DIRTY — MEDIUM feasibility
Chen et al., USENIX Sec 2022. Transformer over Hex-Rays decompiler output — predicts variable names + types. Note this is mainly a **variable** renamer; function-name task is secondary. Preprocessed DIRT dataset (1M functions) and `dirty_mt.ckpt` public on S3, so IDA not needed if using their data. **Metric mismatch concern** for direct F1 comparison.

### 4. SymLM — MEDIUM feasibility
Jin, Pei, Won, Lin, CCS 2022. Trex-based execution-aware encoder on Ghidra ICFGs + CodeWordNet for sub-token semantics. Fully Ghidra-based (free). Sample dataset in repo; full 1.4M-function dataset on Zenodo. Setup pain: Apex install, Trex checkpoint compatibility, fairseq version drift. Plausible in 1-2 weeks with 2-3 days for environment alone.

### 5. XFL — MEDIUM feasibility
Patrick-Evans, Dannehl, Kinder, IEEE S&P 2023. Reformulates naming as extreme multi-label tag prediction using the DEXTER static-feature embedding. Trained on 10K+ Debian binaries (82.5% precision). **xfl-r** refactored version recommended for comparison. Caveat: output is multi-label tags, not generated name strings — direct F1 comparison requires recomposing names the way the paper does.

### 6. SymGen — MEDIUM-LOW feasibility
Jiang et al., NDSS 2025. "Beyond Classification" autoregressive generation using CodeLlama-7B/34B fine-tuned with LoRA on Ghidra-decompiled pseudo-C. Dataset: 2.24M functions across 33 projects, x86/x64/ARM/MIPS, O0-O3, on Zenodo. Sample data + LoRA weights in repo. **Hardware blocker**: 34B wants A100 80GB; 7B feasible on our HPC 40GB A100 but with weaker results. Pipeline requires Ghidra headless per binary.

### 7. BLens — LOW feasibility
Benoit et al., USENIX Sec 2025. Ensemble of **DEXTER + CLAP + PalmTree + VarCLR** embeddings fused contrastively with a "LORD" decoder. Very heavy: **80GB VRAM recommended**, 50GB disk per experiment, 200+200 epoch schedule, requires 4 separate pretrained embedding stacks wired together. Dataset + checkpoints on Zenodo. Only feasible with H100/A100-80GB + 2+ weeks.

### 8. NFRE — LOW feasibility
Gao et al., ISSTA 2021. The public repo is **only the dataset-construction toolkit** (apt + dbgsym + BuildID matching) — **no model implementation or checkpoint published**. Would require reimplementing their Bi-GRU + GNN from the paper. Not realistic as a quick baseline.

### 9. DEBIN — LOW feasibility (cite-only)
He et al., CCS 2018. ETH SRI. CRF structured prediction via Nice2Predict (not ExtraTrees as sometimes miscited). Pretrained x86/x64/ARM models exist. Stack is pinned to Ubuntu 16.04 + gcc 5.4 + OCaml plugins + custom BAP build. **Do not build in 2026** — cite numbers from the original paper.

### Out-of-scope
- **GenNm** (NDSS 2025, Xu et al.) — recovers **variable** names inside decompiled functions using CodeGemma/CodeLlama, not function names. Sometimes miscited as a function-naming baseline. Do not compare head-to-head.
- **"FuncNamer"** — no matching paper found. Likely confusion with NFRE, SymLM, or an internal tool name.

## Recommendations for paper comparison

**Train + evaluate on OUR data (realistic within 1-2 weeks):**
1. **AsmDepictor** — highest-value target. Same assembly-level input philosophy as ours, no decompiler, clean PyTorch, public checkpoint. **Primary head-to-head.**
2. **SymLM** — second priority. Free Ghidra pipeline, well-known CCS baseline, strong published numbers. Budget 2-3 days for environment.
3. **DIRTY** (function-name head only) or **XFL-r** — pick one as the decompiled/multi-label contrast point. XFL-r is more directly about function naming; DIRTY is more about variable names.

**Cite paper numbers only (do not re-run):**
- **NERO** (IDA-locked)
- **DEBIN** (2018-era stack)
- **NFRE** (no model code)
- **BLens** (80GB VRAM)
- **SymGen-34B** (80GB VRAM; try 7B variant only if time permits)

**Do not include:**
- **GenNm** (variable names, different task)
- **"FuncNamer"** (paper does not appear to exist)

## Our model's differentiating properties (for related-work table)

1. **BAP-IR front-end** — decompiler-free, works without IDA or Ghidra
2. **25M parameters** vs BLens (0.5B+) and SymGen-34B
3. **k-NN retrieval head** with contrastive fine-tuning (not seq2seq-only)
4. **Explicit cross-project eval on nginx-family** (tengine/angie/nginx-1.18) where SymGen reports ~0.38 F1 and BLens reports ~0.46 F1 — our **0.704 F1** is meaningfully above both **if metric definitions align**
5. **Training from scratch** (no PalmTree / CLAP / CodeLlama foundation)
6. **Joint seq2seq + contrastive (NT-Xent)** objective with cross-optimization-level positive pairs

**Metric caveat to verify before claiming win:** BLens/SymGen report per-token Precision/Recall/F1 with sub-token splitting. Need to confirm:
- Stemming rules (do they lowercase? strip digits?)
- Tokenization (by underscore? camelCase?)
- Whether partial matches count

If our metric definition is stricter or looser, adjust the comparison accordingly.

## Sources
- [NERO GitHub](https://github.com/tech-srl/Nero)
- [XFL GitHub](https://github.com/lmu-plai/xfl), [XFL arXiv 2107.13404](https://arxiv.org/abs/2107.13404)
- [SymLM GitHub](https://github.com/OSUSecLab/SymLM), [CCS 2022 paper](https://dl.acm.org/doi/10.1145/3548606.3560612)
- [AsmDepictor GitHub](https://github.com/agwaBom/AsmDepictor), [AsiaCCS 2023 paper](https://dl.acm.org/doi/10.1145/3579856.3582823)
- [BLens GitHub](https://github.com/lmu-plai/blens), [arXiv 2409.07889](https://arxiv.org/abs/2409.07889), [USENIX Sec 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/benoit)
- [SymGen GitHub](https://github.com/OSUSecLab/SymGen), [NDSS 2025 paper](https://www.ndss-symposium.org/wp-content/uploads/2025-797-paper.pdf)
- [GenNm NDSS 2025](https://www.ndss-symposium.org/ndss-paper/unleashing-the-power-of-generative-model-in-recovering-variable-names-from-stripped-binary/)
- [DIRTY GitHub](https://github.com/CMUSTRUDEL/DIRTY), [USENIX Sec 2022](https://www.usenix.org/conference/usenixsecurity22/presentation/chen-qibin)
- [DEBIN GitHub](https://github.com/eth-sri/debin), [CCS 2018 paper](https://files.sri.inf.ethz.ch/website/papers/ccs18-debin.pdf)
- [NFRE GitHub](https://github.com/USTC-TTCN/NFRE)
- [Awesome-Info-Inferring-Binary survey list](https://github.com/Bin2Own/Awesome-Info-Inferring-Binary)
