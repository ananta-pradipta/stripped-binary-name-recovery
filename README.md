# HyDRA: Hybrid Decoder-Retrieval Architecture for Adaptive Dual-Regime Binary Function Name Recovery

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![BAP 2.5](https://img.shields.io/badge/BAP-2.5.0-green.svg)](https://github.com/BinaryAnalysisPlatform/bap)

> Anonymous artifact for CCS 2026 double-blind review.

---

## Abstract

HyDRA recovers function names from stripped x86-64 binaries by coupling a graph-neural-network encoder with an **adaptive dual-regime inference mechanism**. A cascaded gated fusion over three inter-procedural context sources (external calls, callee signatures, caller signatures) supplies the encoder signal. At inference, a per-binary adaptive gate routes each query to one of two heads: **k-NN retrieval** for near-clone paradigms and a **GRU decoder pretrained as an unconditional sub-token language model** for novel codebases.

**Key Results (25M parameters, `best_model.pt`):**
- **Test Set:** 70.8% Exact Match, 0.770 F1 (13,559 functions)
- **Cross-Project (7 packages, 13,581 functions):** 0.738 aggregate F1
  - **Near-Clone Transfer (NCT, nginx118 / angie / tengine):** **0.833 F1**
  - **Far Transfer (FT, recutils / dash / gettext / psmisc):** **0.606 F1**
- **Efficiency:** 25M parameters, trained from scratch on BAP-IR with no source-code pretraining. $\sim$1,360× smaller than SymGen's CodeLlama-34B and ~40× faster per-function inference.

---

## Architecture

```
Stripped Binary → BAP → BAP-IR (.bir files)
    ↓
Stage 1: Instruction-Type Tokenization (~1,510 semantic types)
         → Embedding(256-dim) + Positional Encoding
         → Transformer (4 layers, 8 heads)
         → Mean Pool over tokens
         → b_i ∈ R^512 per block
    ↓
Stage 2: GAT (3 layers, 8 heads) over CFG edges
         → Attention Pooling over blocks
         → f ∈ R^1024 (function-level embedding)
    ↓
Cascaded Gated Fusion (conditional bypass when no context):
  1. External calls:  gate ⊙ f + (1-gate) ⊙ ext_emb
  2. Callee context:  gate ⊙ z + (1-gate) ⊙ callee_ctx
  3. Caller context:  gate ⊙ z + (1-gate) ⊙ caller_ctx
    ↓
Adaptive Dual-Path Inference (per-binary gate on external-call Jaccard):
  • Path A (k-NN retrieval):  cosine similarity over encoder embeddings
                              + BinFilter (post-retrieval binary-similarity filter)
                              → nearest training function name
  • Path B (LM-pretrained decoder):  GRU + beam search (k=5, repetition penalty)
                                     → Votes sub-token sequence → function name

  Routing: if J_max(binary, training) ≥ τ_bin (0.5): Path A else Path B
```

---

## Key Innovations

| Innovation | Impact |
|---|---|
| Instruction-Type Tokenization (1,510 types) | Raw-token F1 ~0.03 → instruction-type F1 >0.5 (same architecture) |
| Multi-context cascaded fusion (ext + callee + caller) with conditional bypass | Resolves the **ext-call paradox** (ext calls alone hurt cross-project F1 by −0.066; callee/caller context recovers +0.086) |
| Adaptive dual-path inference (k-NN ↔ decoder) | Per-binary gate: retrieval for near-clones, decoder for novel codebases |
| LM-pretrained decoder (XFL 397K corpus) | Enables compositional generation on packages with OOV full names but in-corpus sub-tokens |
| Votes sub-token tokenizer (rule + corpus-frequency) | 95% less OOV vs BPE at comparable vocab size |
| Self-supervised pretraining (MLM + contrastive, 84.4% embedding transfer) | Largest single contribution: +0.13 cross-project F1 |
| Conditional gate bypass | Eliminates 47% mode collapse (xmalloc default) on context-less functions |
| Indirect-jump (ENDBR64) resolution | O0 EM: 1.6% → 41.5% |

---

## Repository Structure

```
stripped-binary-name-recovery/
├── scripts/                        # Pipeline scripts
│   ├── 01_setup_environment.sh     # Install dependencies
│   ├── 02_compile_dataset.sh       # Download, compile, strip binaries
│   ├── 03_preprocess.sh            # BAP lifting, graph extraction
│   ├── 04_train.sh                 # Train model variants
│   ├── predict.py                  # Single-binary prediction
│   ├── eval_test.py                # Test eval with all metrics
│   ├── eval_cross_project.py       # Cross-project eval (decoder + k-NN + BinFilter + adaptive gate)
│   ├── eval_knn_hybrid.py          # k-NN hybrid evaluation utilities
│   ├── eval_knn_ablation.py        # k-NN ablation experiments
│   ├── eval_full.py                # Full evaluation pipeline
│   ├── error_analysis.py           # Prediction error analysis
│   ├── statistical_significance.py # Statistical significance tests
│   ├── build_pretrain_pairs.py     # Build contrastive pairs for pretraining
│   ├── archive/                    # Archived scripts
│   └── *.sbatch                    # Slurm scripts for HPC training/eval
│
├── src/
│   ├── preprocessing/
│   │   ├── parse_bap.py            # BAP-IR → CFG (V3 instruction-type tokenization)
│   │   ├── extract_external.py     # PLT/GOT external call extraction
│   │   ├── build_dataset.py        # PyTorch dataset with callee/caller context
│   │   ├── build_votes.py          # Votes name tokenizer
│   │   └── extract_strings.py      # String reference extraction
│   ├── models/
│   │   ├── block_encoder.py        # Transformer block encoder with mean pooling
│   │   ├── graph_encoder.py        # GAT with attention pooling over CFG
│   │   ├── external_encoder.py     # Bi-GRU encoders for ext/callee/caller context
│   │   ├── gated_fusion.py         # Conditional cascaded gated fusion
│   │   ├── decoder.py              # GRU decoder with beam search
│   │   ├── function_namer.py       # Full model assembly
│   │   └── pretrain_heads.py       # MLM + contrastive pretraining heads
│   ├── training/
│   │   ├── train.py                # Training loop (AMP, scheduled sampling)
│   │   ├── pretrain.py             # Self-supervised pretraining (MLM + contrastive)
│   │   ├── pretrain_decoder.py     # Decoder LM pretraining on name corpus
│   │   ├── pretrain_dataset.py     # Pretraining data loader
│   │   └── contrastive_sampler.py  # Batch sampler for contrastive learning
│   └── evaluation/
│       └── metrics.py              # F1, EM, n-gram similarity, edit distance
│
├── configs/
│   ├── optimized.yaml              # 8M ablation-model config
│   ├── optimized_large.yaml        # 25M config (headline model)
│   ├── pretrain.yaml               # Encoder pretraining config
│   └── ablation_model[2-4].yaml    # Compute-controlled ablation configs
│
├── results/                        # Evaluation outputs (JSON)
├── data/                           # Preprocessed data (not tracked)
├── demo/                           # Cross-project evaluation data
└── checkpoints/                    # Model weights (not tracked)
```

---

## Quick Start

```bash
# Clone the anonymous artifact
# (anonymous.4open.science serves a read-only view during double-blind review)

# Setup environment (installs BAP, PyTorch, dependencies)
bash scripts/01_setup_environment.sh
source activate.sh

# Full pipeline
bash scripts/02_compile_dataset.sh   # Compile binaries at -O0/1/2/3 and strip
bash scripts/03_preprocess.sh        # BAP lift + extract features + build vocabularies
python3 -m src.training.train --config configs/optimized_large.yaml --seed 42

# Predict on any stripped binary
python3 scripts/predict.py --binary /path/to/stripped/binary
```

### Full training pipeline (headline model)

```bash
# 1. Build encoder pretraining pairs
python3 scripts/build_pretrain_pairs.py

# 2. Pretrain encoder (MLM + contrastive)
python3 -m src.training.pretrain --config configs/pretrain.yaml --seed 42

# 3. Pretrain decoder (XFL 397K-name LM)
python3 -m src.training.pretrain_decoder --corpus data/xfl_397k_names.txt

# 4. Fine-tune with pretrained encoder and decoder
python3 -m src.training.train \
  --config configs/optimized_large.yaml \
  --seed 42 \
  --pretrained-encoder checkpoints/pretrained_encoder.pt \
  --pretrained-decoder checkpoints/pretrained_decoder.pt
```

### HPC Training (Slurm)

```bash
# Submit a training job
sbatch scripts/train.sbatch

# Submit the 7-package cross-project evaluation
sbatch scripts/eval_xproj.sbatch
```

---

## Evaluation

### Ablation Study (compute-controlled, 8M parameters per variant)

Each variant trained at matched compute on a 300K-function corpus; evaluated on a fixed cross-project set. F1 values are decoder-only sub-token F1 over 3 random seeds (mean ± std).

| # | Model | Params | Test F1 | Cross-Project F1 (seed mean ± std) |
|---|---|---|---|---|
| 1 | DeBin (ExtraTrees baseline) | — | 0.535 | — |
| 2 | GAT + Decoder (basic) | ~8M | 0.606 | 0.473 ± 0.006 |
| 3 | + External Calls | ~8M | 0.683 | **0.408 ± 0.017** (ext-call paradox) |
| 4 | + Callee/Caller Context | ~8M | 0.781 | 0.493 ± 0.018 |
| 5 | **+ Pretrain + Scale (headline)** | **25M** | **0.770** | **0.738 (adaptive gate)** |

**Ablation findings:**
- **Model 2 → 3 (+ext):** Test F1 +0.077 but cross-project F1 **−0.066 (≈4σ)** — the **ext-call paradox**. External calls alone help in-distribution but hurt cross-project generalization.
- **Model 3 → 4 (+callee/caller):** cross-project F1 **+0.086 (≈5σ)** — multi-context gated fusion disambiguates the library-call shortcuts from step 3.
- **Model 4 → 5 (+pretrain + scale):** +0.13 cross-project F1 on the 7-pkg set — the single largest contribution.

### Cross-Project Results (7 unseen packages, 13,581 functions)

Stratified by per-binary external-call Jaccard similarity (J\_max) to training:

**Near-Clone Transfer (NCT, J\_max ≥ 0.70):**

| Package | N | Retrieval (k-NN) | Decoder | **Adaptive Gate** | %-kNN | %-Dec |
|---|---|---|---|---|---|---|
| nginx118 | 3,470 | 0.878 | 0.773 | **0.878** | 82.8 | 17.2 |
| angie | 3,893 | 0.819 | 0.709 | **0.819** | 76.4 | 23.6 |
| tengine | 554 | 0.531 | 0.630 | 0.645 | 91.3 | 8.7 |
| **NCT subtotal** | **7,917** | 0.825 | 0.731 | **0.833** | 79.2 | 20.8 |

**Far Transfer (FT, J\_max < 0.45):**

| Package | N | Retrieval (k-NN) | Decoder | **Adaptive Gate** | %-kNN | %-Dec |
|---|---|---|---|---|---|---|
| recutils | 2,550 | 0.308 | 0.311 | 0.346 | 56.2 | 43.8 |
| dash | 1,324 | 0.130 | 0.770 | **0.815** | 17.4 | 82.6 |
| gettext | 1,518 | 0.036 | 0.807 | **0.834** | 11.5 | 88.5 |
| psmisc | 272 | 0.167 | 0.745 | **0.761** | 23.2 | 76.8 |
| **FT subtotal** | **5,664** | 0.187 | 0.572 | **0.606** | 27.1 | 72.9 |

**7-pkg aggregate:** 13,581 functions, **F1 = 0.738** (gate) vs. 0.558 (retrieval-only) vs. 0.665 (decoder-only).

**Regime interpretation:** NCT is retrieval-dominant (retrieval alone ≈ gate); FT is decoder-dominant (decoder alone ≪ gate). The adaptive gate's value is largest in the FT regime where retrieval collapses but the LM-pretrained decoder composes novel sub-token sequences.

### Comparison with Published Systems (7-pkg cross-project set)

| System | Venue | Params | Cross-Project F1 |
|---|---|---|---|
| SymLM | CCS'22 | 86M | 0.277 |
| BLens (retrained c+p) | USENIX Sec'25 | ~200M | 0.454 |
| SymGen (released ckpt, zero-shot) | NDSS'25 | 34B (CodeLlama-34B) | 0.450 |
| SymGen + LoRA (fine-tuned on our corpus) | NDSS'25 | 34B | 0.630 |
| StarCoder-3B (zero-shot) | — | 3B | 0.654 |
| **HyDRA (ours)** | — | **25M** | **0.738** |

HyDRA leads the SymGen + LoRA reproduction by **+0.108 F1 overall**, widening to **+0.38 F1** on the three non-recutils far-transfer packages (dash / gettext / psmisc) where CodeLlama has no GNU pretraining advantage.

### Head-to-head with SymGen (matched functions, original 4-pkg subset)

For a strict apples-to-apples comparison against SymGen + LoRA on the original 4-package cross-project set, we restrict both systems to **the same function set**: for every `(package, ground-truth-name)` key that appears in both evaluation runs, we take `min(count_HyDRA, count_SymGen)` predictions from each side. Matched subset: **8,062 functions** (3,428 unique `(package, name)` keys across angie / nginx118 / tengine / recutils).

| Package | N | HyDRA F1 | HyDRA EM | SymGen F1 | SymGen EM | Δ F1 |
|---|---|---|---|---|---|---|
| nginx118 | 2,815 | **0.880** | 71.9% | 0.642 | 30.9% | **+0.238** |
| angie | 3,159 | **0.814** | 62.6% | 0.643 | 29.7% | **+0.171** |
| tengine | 554 | **0.739** | 66.8% | 0.701 | 45.8% | +0.038 |
| recutils | 1,534 | 0.359 | 27.5% | **0.636** | 39.8% | −0.277 |
| **Overall** | **8,062** | **0.745** | **59.5%** | 0.645 | 33.2% | **+0.100** |

On the matched set, HyDRA (25M, from scratch) leads SymGen + LoRA (34B) by **+0.100 F1** and **+26.3 pp EM**. Recutils is the only package where SymGen wins — consistent with CodeLlama-34B's source-code pretraining having already seen recutils on GitHub.

### End-to-End Cost Comparison (single A100-80GB)

| System | Params | Train | Preproc / binary | Inference / function | End-to-end (100-fn binary) |
|---|---|---|---|---|---|
| **HyDRA (ours)** | **25M** | **2.2 GPU-h** | **~60 s (BAP)** | **~5 ms** | **~60 s** |
| BLens (retrained c+p) | ~200M | 8 GPU-h | ~100 s (Ghidra+CLAP) | ~20 ms | ~102 s |
| SymGen + LoRA | 34B | 72 GPU-h (4×A100 DDP) | ~180 s (Ghidra+decomp.) | ~200 ms | ~200 s |
| StarCoder-3B (zero-shot) | 3B | 0 (pretrained) | ~180 s (Ghidra+decomp.) | ~80 ms | ~188 s |

**End-to-end wall-clock reduction:** 41–70% vs LLM-based baselines on a typical 100-function binary; 44–80% on the full 13.6K-function cross-project set.

---

## Dataset

- **Training:** 300,013 functions from 77 open-source packages (851 binaries, optimization levels O0/O1/O2/O3)
  - Distribution: O0 47%, O1 16%, O2 16%, O3 15%, default 6%
- **Test:** 13,559 held-out functions (same-package cross-binary evaluation)
- **Cross-Project:** 13,581 functions from **7 fully held-out packages** (tengine, angie, nginx118, recutils, dash, gettext, psmisc) at all 4 optimization levels. Released at `data/cross_project/`.
- **Compilation:** `gcc -g -O{0,1,2,3}`, stripped with `strip -s`
- **Binary lifter:** BAP 2.5.0 (language-agnostic intermediate representation)
- **Token vocabulary:** 2,279 instruction types (V3 tokenization, deterministic)
- **Name vocabulary:** 7,004 sub-tokens (Votes tokenizer, rule-based splitting with corpus-frequency filter)
- **Decoder LM pretraining corpus:** 397K names (73K from our compiled training set + 329K from the XFL release)

---

## References

- He, J., Ivanov, P., Tsankov, P., Raychev, V., & Vechev, M. (2018). **Debin: Predicting Debug Information in Stripped Binaries.** CCS'18.
- David, Y., Alon, U., & Yahav, E. (2020). **Neural Reverse Engineering of Stripped Binaries using Augmented Control Flow Graphs** (NERO). OOPSLA'20.
- Jin, X., Pei, K., Won, J. Y., & Lin, Z. (2022). **SymLM: Predicting Function Names in Stripped Binaries via Context-Sensitive Execution-Aware Code Embeddings.** CCS'22.
- Patrick-Evans, J., Dannehl, M., & Kinder, J. (2023). **XFL: Naming Functions in Binaries with Extreme Multi-Label Learning.** IEEE S&P'23.
- Benoit, T., Wang, Y., Dannehl, M., & Kinder, J. (2025). **BLens: Contrastive Captioning of Binary Functions using Ensemble Embedding.** USENIX Security'25.
- Jiang, L., Jin, X., & Lin, Z. (2025). **Beyond Classification: Inferring Function Names in Stripped Binaries via Domain Adapted LLMs** (SymGen). NDSS'25.
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). **Graph Attention Networks.** ICLR'18.
- Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., & Lewis, M. (2020). **Generalization through Memorization: Nearest Neighbor Language Models.** ICLR'20.

---

## License

MIT License (see `LICENSE`).
