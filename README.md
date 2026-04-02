# FuncR: Binary Function Name Recovery via Graph-Based Multi-Context Embeddings and Retrieval

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![BAP 2.5](https://img.shields.io/badge/BAP-2.5.0-green.svg)](https://github.com/BinaryAnalysisPlatform/bap)

> NJIT: Ananta Dian Pradipta, Robert Blacha, Zhihao Lin, Haotian Zhang

---

## Abstract

We present a deep learning pipeline that recovers function names from stripped binaries. Our approach combines graph attention networks over control flow graphs with inter-procedural context (external calls, callee/caller signatures) via a cascaded gated fusion mechanism, and generates sub-token names using a GRU decoder with Votes tokenization.

**Key Results:**
- **Test Set:** 76.3% Exact Match, 0.804 F1 (8,973 functions, k-NN k=1); decoder-only: 74.3% EM, 0.795 F1
- **Cross-Project:** 46.3% Exact Match, 0.519 F1 (2,256 functions from 5 unseen packages, k-NN k=1)
- **25M parameters**, trained with self-supervised pretrained encoder

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

### Ablation Study

| # | Model | Params | Test F1 | Test EM | Cross-Project EM |
|---|---|---|---|---|---|
| 1 | DeBin (ExtraTrees) | — | 0.535 | 53.5% | — |
| 2 | GAT + Decoder | 4.4M | 0.606 | 51.3% | 24.1% |
| 3 | + External Calls | 6.1M | 0.683 | 60.1% | 19.8% |
| 4 | + Callee/Caller Context | 8.0M | 0.781 | 72.3% | 32.6% |
| 5 | + Pretrain + Scale (ours) | 25M | **0.795** | **74.3%** | **48.5%** |

### Cross-Project Results (5 unseen packages)

| Package | Decoder EM | k-NN EM | k-NN F1 |
|---|---|---|---|
| csplit2 | 35.1% | 42.4% | 0.454 |
| diffutils | 68.5% | 69.9% | 0.713 |
| datamash | 19.2% | 20.4% | 0.303 |
| cppi | 50.0% | 53.6% | 0.641 |
| hello | 49.4% | 57.5% | 0.714 |
| **Overall** | **41.6%** | **46.3%** | **0.519** |

### Comparison with Published Systems

| System | Venue | Cross-Project F1 |
|---|---|---|
| SYMGEN | NDSS'25 | 0.38 |
| BLens | USENIX Sec'25 | 0.46 |
| FuncR (decoder) | - | 0.493 |
| FuncR (k-NN) | - | 0.519 |

---

## Dataset

- **Training:** 87,724 functions from 40 GNU packages (O0 + O2)
- **Test:** 8,973 functions (18 held-out binaries)
- **Cross-Project:** 2,256 functions from 5 completely unseen packages
- **Compilation:** `gcc -g -O0` and `gcc -g -O2`, stripped with `strip -s`
- **Binary lifter:** BAP 2.5.0
- **Token vocabulary:** 2,279 instruction types (V3 tokenization)
- **Name vocabulary:** 2,642 sub-tokens (Votes tokenizer)

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
