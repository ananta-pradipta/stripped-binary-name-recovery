# FuncR: Binary Function Name Recovery via Graph-Based Multi-Context Embeddings and Retrieval

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![BAP 2.5](https://img.shields.io/badge/BAP-2.5.0-green.svg)](https://github.com/BinaryAnalysisPlatform/bap)

> NJIT: Ananta Dian Pradipta, Robert Blacha, Zhihao Lin, Haotian Zhang

---

## Abstract

We present a deep learning pipeline that recovers function names from stripped binaries. Our approach combines graph attention networks over control flow graphs with inter-procedural context (external calls, callee/caller signatures) via a cascaded gated fusion mechanism, and generates sub-token names using a GRU decoder with Votes tokenization.

**Key Results (300K training set, `best_model.pt`):**
- **Test Set:** 70.8% Exact Match, 0.770 F1 (13,559 functions, k-NN + P2 binary filter)
- **Cross-Project:** 46.3% Exact Match, **0.704 F1** (9,492 functions from 4 unseen packages: tengine, angie, nginx118, recutils)
- **25M parameters**, trained from scratch on BAP-IR (no source-code pretraining)

---

## Architecture

```
Stripped Binary → BAP → BAP-IR (.bir files)
    ↓
Stage 1: Instruction-Type Tokenization (~1,510 types)
         → Embedding(256-dim) + Positional Encoding
         → Transformer (4 layers, 8 heads)
         → Mean Pool over tokens
         → b_i ∈ R^512 per block
    ↓
Stage 2: GAT (3 layers, 8 heads) over CFG edges
         → Attention Pooling over blocks
         → f ∈ R^1024 (function-level embedding)
    ↓
Cascaded Gated Fusion:
  1. External calls:  gate ⊙ f + (1-gate) ⊙ ext_emb
  2. Callee context:  gate ⊙ z + (1-gate) ⊙ callee_ctx
  3. Caller context:  gate ⊙ z + (1-gate) ⊙ caller_ctx
  (conditional bypass when no context available)
    ↓
Stage 3a: GRU Decoder (1 layer, 1024 hidden)
          + Beam Search (k=5, repetition penalty)
          → Votes sub-tokens → function name

Stage 3b: k-NN Retrieval (alternative to decoder)
          → Cosine similarity over encoder embeddings
          → Retrieve nearest training function name
```

---

## Key Innovations

| Innovation | Impact |
|---|---|
| Instruction-Type Tokenization (1,510 types) | +1,750% F1 over raw tokens |
| Mean pooling over semantically typed tokens (attention pooling tested, no improvement) | Simple, effective token aggregation |
| Inter-procedural context (ext calls + callee + caller) | +8.7% Val F1; ext-call paradox: ext calls alone hurt cross-project EM, callee/caller context required for disambiguation |
| k-NN retrieval over encoder embeddings | +4.7pp cross-project EM over decoder |
| Votes sub-token tokenizer (3-model voting) | 95% less OOV vs BPE |
| Self-supervised pretraining (MLM + contrastive) | +15.9pp cross-project EM |
| Conditional gate bypass | Fixed 47% mode collapse |
| Indirect jump resolution | O0 EM: 1.6% → 41.5% |

---

## Repository Structure

```
cs785-project/
├── scripts/                        # Pipeline scripts
│   ├── 01_setup_environment.sh     # Install dependencies
│   ├── 02_compile_dataset.sh       # Download, compile, strip binaries
│   ├── 03_preprocess.sh            # BAP lifting, graph extraction
│   ├── 04_train.sh                 # Train model variants
│   ├── predict.py                  # Single-binary prediction
│   ├── eval_test.py                # Test eval with all metrics
│   ├── eval_demo_wulver.py         # Cross-project eval (HPC, no BAP)
│   ├── eval_knn_hybrid.py          # k-NN hybrid evaluation (forward/reverse, sweep, demo)
│   ├── eval_knn_ablation.py        # k-NN ablation experiments
│   ├── eval_full.py                # Full evaluation pipeline
│   ├── error_analysis.py           # Prediction error analysis
│   ├── statistical_significance.py # Statistical significance tests
│   ├── eval_stratified.py          # Stratified evaluation
│   ├── build_pretrain_pairs.py     # Build contrastive pairs for pretraining
│   ├── archive/                    # Archived scripts (05_evaluate.sh, 06_demo.sh, etc.)
│   └── wulver_*.sh / *.sbatch     # NJIT Wulver HPC scripts
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
│   │   ├── pretrain.py             # Self-supervised pretraining
│   │   ├── pretrain_dataset.py     # Pretraining data loader
│   │   └── contrastive_sampler.py  # Batch sampler for contrastive learning
│   └── evaluation/
│       └── metrics.py              # F1, EM, n-gram similarity, edit distance
│
├── configs/
│   ├── optimized.yaml              # 8M model config
│   ├── optimized_large.yaml        # 25M model config (current best)
│   ├── pretrain.yaml               # Pretraining config
│   ├── ablation_model[2-4].yaml    # Ablation study configs
│   └── variant[2-4]_*.yaml         # Midterm architecture variants
│
├── docs/                           # Contributor guides
│   ├── preprocessing_contributor_guide.md
│   ├── modeling_contributor_guide.md
│   └── evaluation_contributor_guide.md
│
├── reports/
│   ├── final_report.md
│   └── final_report_presentation.md
├── results/                        # Evaluation outputs
├── data/                           # Preprocessed data (not tracked)
├── demo/                           # Cross-project evaluation data
└── checkpoints/                    # Model weights (not tracked)
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/ananta-pradipta/stripped-binary-name-recovery.git
cd stripped-binary-name-recovery

# Setup environment (installs BAP, PyTorch, dependencies)
bash scripts/01_setup_environment.sh
source ~/cs785-project/activate.sh

# Full pipeline
bash scripts/02_compile_dataset.sh   # Compile binaries
bash scripts/03_preprocess.sh        # BAP lift + extract features
python3 -m src.training.train --config configs/optimized_large.yaml --seed 42

# Predict on any stripped binary
python3 scripts/predict.py --binary /path/to/stripped/binary
```

### Training with Pretrained Encoder

```bash
# Build pretraining pairs
python3 scripts/build_pretrain_pairs.py

# Pretrain encoder (MLM + contrastive)
python3 -m src.training.pretrain --config configs/pretrain.yaml --seed 42

# Finetune with pretrained encoder
python3 -m src.training.train \
  --config configs/optimized_large.yaml \
  --seed 42 \
  --pretrained-encoder checkpoints/pretrained_encoder.pt
```

### HPC Training (NJIT Wulver)

```bash
# Setup SSH multiplexing (one-time)
# Add to ~/.ssh/config:
#   Host wulver
#       HostName wulver.njit.edu
#       User <ucid>
#       ControlMaster auto
#       ControlPath ~/.ssh/sockets/%r@%h-%p
#       ControlPersist 12h

# Upload data (first time)
bash scripts/wulver_upload.sh

# Quick code sync (after changes)
bash scripts/wulver_sync.sh

# Submit training job
ssh wulver "cd <project_dir> && sbatch scripts/wulver_train.sbatch"
```

---

## Evaluation

### Ablation Study (87K dataset, early architecture study)

The ablation below was run on the earlier 87K-function dataset with a 5-package demo set (diffutils, datamash, cppi, csplit2, hello). It shows the **incremental contribution of each architectural component**. The headline Test and Cross-Project numbers above use the later 300K dataset with the 4-package cross-project set (tengine / angie / nginx118 / recutils).

| # | Model | Params | Test F1 | Test EM |
|---|---|---|---|---|
| 1 | DeBin (ExtraTrees) | — | 0.535 | 53.5% |
| 2 | GAT + Decoder | 4.4M | 0.606 | 51.3% |
| 3 | + External Calls | 6.1M | 0.683 | 60.1% |
| 4 | + Callee/Caller Context | 8.0M | 0.781 | 72.3% |
| 5 | + Pretrain + Scale | 25M | **0.795** | **74.3%** |

**Key findings:**
- Stage 2→3 (+Ext calls): Test F1 +0.077, but cross-project EM drops **−4.3pp** — the **ext-call paradox**. Ext calls alone help in-distribution but hurt cross-project generalization.
- Stage 3→4 (+Callee/Caller): cross-project EM +12.8pp — multi-context gated fusion is required to disambiguate similar library call patterns.
- Stage 4→5 (+Pretrain+Scale): cross-project EM +15.9pp — SSL pretraining + capacity scaling gives the best cross-project transfer.

### Cross-Project Results (4 unseen packages, 9,492 functions)

| Package | N | EM | F1 |
|---|---|---|---|
| nginx118 | 3,470 | 53.1% | **0.813** |
| tengine | 554 | 62.3% | **0.761** |
| angie | 3,893 | 45.1% | **0.739** |
| recutils | 1,575 | 28.6% | 0.357 |
| **Overall** | **9,492** | **46.3%** | **0.704** |

Cross-project packages are held out entirely from training.

### Comparison with Published Systems (same 4-pkg cross-project set)

| System | Venue | Params | Cross-Project F1 |
|---|---|---|---|
| SYMGEN (released ckpt) | NDSS'25 | 34B (CodeLlama-34B + LoRA) | 0.450 |
| SYMGEN + LoRA | NDSS'25 | 34B | 0.699 |
| BLens (reported) | USENIX Sec'25 | ~200M | 0.46 |
| LLM Zero-Shot (StarCoder-3B) | - | 3B | 0.654 |
| **FuncR (ours)** | - | **25M** | **0.704** |

Our 25M from-scratch model beats SymGen+LoRA (34B) by **+0.005 F1** and the released SymGen checkpoint by **+0.254 F1**, at 1000× fewer parameters and without the source-code pretraining that risks LLM-contamination on open-source eval packages.

---

## Dataset

- **Training:** 300,013 functions from 77 packages, 851 binaries (O0, O1, O2, O3)
  - Distribution: O0 47%, O1 16%, O2 16%, O3 15%, default 6%
- **Test:** 13,559 held-out functions
- **Cross-Project:** 9,492 scored functions from **4 completely unseen packages** — tengine, angie, nginx118, recutils (all 4 optimization levels each). Released at `data/cross_project/` (46 stripped binaries + 18,595 ground-truth address→name entries).
- **Compilation:** `gcc -g -O{0,1,2,3}`, stripped with `strip -s`
- **Binary lifter:** BAP 2.5.0
- **Token vocabulary:** 2,279 instruction types (V3 tokenization)
- **Name vocabulary:** 7,004 sub-tokens (Votes tokenizer)

---

## References

- He, J., et al. (2018). **Debin: Predicting Debug Information in Stripped Binaries.** CCS'18.
- Jin, M., et al. (2022). **SymLM: Predicting Function Names via Context-Sensitive Execution-Aware Code Embeddings.** CCS'22.
- Wang, H., et al. (2022). **jTrans: Jump-Aware Transformer for Binary Code Similarity.** ISSTA'22.
- Al-Kaswan, A., et al. (2023). **BLens: Contrastive Captioning of Binary Functions.** USENIX Security'25.
- Xu, Z., et al. (2024). **SYMGEN: Generating Function Name from Binary Code.** NDSS'25.
- David, Y., et al. (2020). **NERO: A Neural Rule Grounding Framework for Label-Free Knowledge Transfer.** OOPSLA'20.
- Khandelwal, U., et al. (2020). **Nearest Neighbor Language Models.** ICLR'20.
- Khandelwal, U., et al. (2021). **Nearest Neighbor Machine Translation.** ICLR'21.

---

## License

This project is released under the [MIT License](LICENSE).
