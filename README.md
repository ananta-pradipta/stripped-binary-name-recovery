# Revisiting DeBin: Function Name Recovery in Stripped Binaries Using Graph Neural Networks with Gated Multi-Context Fusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![BAP 2.5](https://img.shields.io/badge/BAP-2.5.0-green.svg)](https://github.com/BinaryAnalysisPlatform/bap)

> NJIT:
> Ananta Dian Pradipta, Robert Blacha, Zhihao Lin

---

## Abstract

We present a deep learning pipeline that recovers function names from stripped binaries. Our approach combines graph attention networks over control flow graphs with inter-procedural context (external calls, callee/caller signatures) via a cascaded gated fusion mechanism, and generates sub-token names using a GRU decoder with Votes tokenization.

**Key Results:**
- **Test Set:** 74.3% Exact Match, 0.795 F1 (8,973 functions)
- **Cross-Project:** 48.5% Exact Match, 0.593 F1 (12,688 functions from 11 unseen packages)
- **25M parameters**, trained with self-supervised pretrained encoder

---

## Architecture

```
Stripped Binary → BAP → BAP-IR (.bir files)
    ↓
Stage 1: Instruction-Type Tokenization (~1,510 types)
         → Embedding(256-dim) + Positional Encoding
         → Transformer (4 layers, 8 heads)
         → Self-Attention Pool over tokens
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
Stage 3: GRU Decoder (1 layer, 1024 hidden)
         + Beam Search (k=5, repetition penalty)
         → Votes sub-tokens → function name
```

---

## Key Innovations

| Innovation | Impact |
|---|---|
| Instruction-Type Tokenization (1,510 types) | +1,750% F1 over raw tokens |
| Token-level self-attention in block encoder | Learns importance weights of instruction tokens |
| Inter-procedural context (ext calls + callee + caller) | +8.7% Val F1 |
| Votes sub-token tokenizer (3-model voting) | 95% less OOV vs BPE |
| Self-supervised pretraining (MLM + contrastive) | +2% Val F1, faster convergence |
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
│   ├── 05_evaluate.sh              # Evaluation on test set
│   ├── 06_demo.sh                  # Cross-project evaluation
│   ├── predict.py                  # Single-binary prediction
│   ├── eval_test.py                # Test eval with all metrics
│   ├── eval_demo_expanded.py       # Cross-project eval (local, with BAP)
│   ├── eval_demo_wulver.py         # Cross-project eval (HPC, no BAP)
│   ├── eval_stratified.py          # Stratified evaluation
│   ├── build_pretrain_pairs.py     # Build contrastive pairs for pretraining
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
│   │   ├── block_encoder.py        # Transformer block encoder with attention pooling
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
├── reports/
│   └── midterm_vs_final_comparison.md
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
| 5 | + Pretrain + Scale (ours) | 24.7M | **0.795** | **74.3%** | **48.5%** |

### Cross-Project Results (11 unseen packages)

| Package | EM | F1 | NgSim | EdSim |
|---|---|---|---|---|
| texinfo | 92.7% | 0.969 | 0.974 | 0.976 |
| acct | 72.7% | 0.785 | 0.792 | 0.827 |
| diffutils | 68.5% | 0.704 | 0.709 | 0.749 |
| direvent | 67.7% | 0.754 | 0.762 | 0.794 |
| rush | 62.0% | 0.733 | 0.738 | 0.778 |
| cppi | 55.5% | 0.648 | 0.648 | 0.704 |
| hello | 50.0% | 0.659 | 0.685 | 0.729 |
| strace | 42.7% | 0.584 | 0.586 | 0.653 |
| csplit2 | 37.2% | 0.441 | 0.452 | 0.536 |
| datamash | 18.8% | 0.296 | 0.313 | 0.416 |
| htop | 0.9% | 0.088 | 0.374 | 0.518 |
| **Overall** | **48.5%** | **0.593** | **0.625** | **0.688** |

---

## Dataset

- **Training:** 87,724 functions from 40 GNU packages (O0 + O2)
- **Test:** 8,973 functions (18 held-out binaries)
- **Cross-Project:** 12,688 functions from 11 completely unseen packages
- **Compilation:** `gcc -g -O0` and `gcc -g -O2`, stripped with `strip -s`
- **Binary lifter:** BAP 2.5.0
- **Token vocabulary:** 2,279 instruction types (V3 tokenization)
- **Name vocabulary:** 2,642 sub-tokens (Votes tokenizer)

---

## References

- He, J., et al. (2018). **Debin: Predicting Debug Information in Stripped Binaries.** CCS'18.
- Jin, M., et al. (2022). **SymLM: Predicting Function Names via Context-Sensitive Execution-Aware Code Embeddings.** NDSS'23.
- Wang, H., et al. (2022). **jTrans: Jump-Aware Transformer for Binary Code Similarity.** ISSTA'22.
- Al-Kaswan, A., et al. (2023). **BLens: Contrastive Captioning of Binary Functions.** ICSE'23.
- Xu, Z., et al. (2024). **SYMGEN: Generating Function Name from Binary Code.** ASE'24.
- Xu, Z., et al. (2023). **NERO: Neural-Embeddings-based Function Recognition from Optimized Code.** USENIX'23.

---

## License

This project is released under the [MIT License](LICENSE).
