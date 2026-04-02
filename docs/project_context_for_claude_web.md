# CS785 Binary Function Name Recovery -- Complete Project Context

This document is the sole context for Claude.ai web chat sessions about this project. It compiles all knowledge from the codebase, experiment history, memory files, and research references as of **2026-04-01**.

---

## 1. Project Overview

### Goal
Predict human-readable function names from stripped x86-64 binaries using deep learning. Maximize sub-token F1 and exact match (EM) rate on unseen binaries.

### Team
- **Ananta Dian Pradipta**: Model implementation and training (block encoder, GNN/attention, fusion, decoder)
- **Robert Blacha**: Dataset compilation, BAP extraction pipeline, data preprocessing
- **Zhihao Lin**: NLP evaluation metrics implementation and semantic similarity analysis

### Architecture (3-Stage Pipeline)
```
Stripped Binary --> BAP --> BAP-IR (.bir files)
    |
Stage 1: Instruction-Type Tokenization (V3: ~1510 types)
         -> Embedding + Positional Encoding
         -> Transformer Encoder
         -> Mean Pool over tokens
         -> Linear -> b_i (per-block embedding)
    |
Stage 2: GAT over CFG edges
         -> Attention Pooling over blocks
         -> f (function-level embedding)
    |
Cascaded Gated Fusion (3-stage, conditional bypass):
  1. External calls:  if has_ext: z = gate*f + (1-gate)*ext_emb     else: z = f
  2. Callee context:  if has_callees: z = gate*z + (1-gate)*callee_ctx  else: z = z
  3. Caller context:  if has_callers: z = gate*z + (1-gate)*caller_ctx  else: z = z
    |
Stage 3: GRU Decoder (beam search k=5, repetition penalty)
         -> Votes sub-tokens -> function name

Optional: k-NN hybrid inference (no retraining needed)
  - Build FAISS index of 87K training function embeddings
  - Forward hybrid: decoder default, k-NN fallback on low beam score
  - Threshold=-0.02 -> 87% k-NN, 13% decoder
```

### Model Configurations

#### 8M Model (configs/optimized.yaml -- for ablation)
- Block Encoder: `nn.Embedding(1510, 128)` -> `TransformerEncoder(2L, 4H, ff=256)` -> mean pool -> `Linear(128, 256)`
- Graph Encoder: 2-layer GAT, 4 heads, input 256d -> output 512d, attention pooling
- External Call Encoder: `nn.Embedding(683, 256)` -> `Bi-GRU(256)` -> `Linear(512, 512)`
- Callee/Caller Context Encoder: `nn.Embedding(1510, 64)` -> `Bi-GRU(128)` -> `Linear(256, 512)`
- Gated Fusion: `gate = sigmoid(Linear(1024, 512))`, conditional bypass
- Decoder: GRU, 256 embed, 512 hidden, 1 layer
- Training: LR=0.001, batch=128, epochs=50, patience=10, warmup=5, cosine LR
- Teacher forcing: 1.0 -> 0.3, label smoothing: 0.1

#### 25M Model (configs/optimized_large.yaml -- current best, Exp 36)
- Block Encoder: `nn.Embedding(2279, 256)` -> `TransformerEncoder(4L, 8H, ff=512)` -> mean pool -> `Linear(256, 512)`
- Graph Encoder: 3-layer GAT, 8 heads, input 512d -> output 1024d, attention pooling
- External Call Encoder: `nn.Embedding(656, 256)` -> `Bi-GRU(256)` -> `Linear(512, 1024)`
- Callee/Caller Encoders: `nn.Embedding(2279, 128)` -> `Bi-GRU(256)` -> `Linear(512, 1024)`
- Decoder: GRU, 512 embed, 1024 hidden, 1 layer
- Training: LR=0.0003, batch=32, epochs=50, patience=10, warmup=5, cosine LR
- Pretrained encoder: `checkpoints/pretrained_encoder.pt` (MLM + contrastive, 10 epochs)

### Preprocessing Pipeline
```bash
scripts/03_preprocess.sh   # Full pipeline:
  # 3.1: Extract ground truth labels from debug binaries (nm -> addr->name)
  # 3.2: Lift stripped binaries with BAP -> .bir files
  # 3.3: Parse BAP-IR into CFG graphs (parse_bap.py -> JSON per function)
  # 3.4: Extract external calls per function
  # 3.5: Address matching (BAP sub_XXXX <-> debug nm addresses)
  # 3.6: Build BPE vocabulary (legacy, Votes used instead)
  # 3.7: Build external call vocabulary
```

### Instruction-Type Tokenization (V3)
Raw BAP-IR instructions classified into ~1510 semantic types:
- Calls: `CALL_malloc`, `CALL_printf`, `CALL_INTERNAL`, `CALL_INDIRECT`
- Memory: `MEM_READ_32`, `MEM_WRITE_ARG_64`, `MEM_READ_GLOBAL_32`
- Stack: `STACK_LOAD_64`, `STACK_STORE_32`
- Flags: `FLAG_CF`, `FLAG_ZF`, `COND_BRANCH_CF`
- Registers: `ARG_SETUP`, `ARG_LOAD_ADDR`, `RETVAL`
- Arithmetic: `ARITH_ADD`, `ARITH_SHIFT`, `ARITH_XOR`
- Constants (V3): `ASSIGN_ZERO`, `ASSIGN_POW2`, `ASSIGN_ADDR`
- Control: `BRANCH`, `RETURN`, `COMPARE`

Reduces 32K raw BAP-IR tokens to ~1510 types -- +1750% F1 gain over raw tokens.

### Votes Name Tokenizer
Replaces BPE (SentencePiece). Splits function names on `_` and camelCase boundaries using 3-model voting on sub-token boundaries. 2,642 sub-token vocabulary. 95% less OOV than BPE.

### Key Files
- `src/models/block_encoder.py` -- Stage 1: Transformer + mean pool
- `src/models/graph_encoder.py` -- Stage 2: GAT + attention pooling
- `src/models/external_encoder.py` -- Bi-GRU for ext calls + callee/caller + strings
- `src/models/gated_fusion.py` -- Conditional gated fusion
- `src/models/decoder.py` -- GRU decoder with beam search
- `src/models/function_namer.py` -- Full model assembly
- `src/preprocessing/parse_bap.py` -- BAP-IR parser with V3 tokenization
- `src/preprocessing/build_dataset.py` -- Dataset loader + thunk resolution
- `src/preprocessing/build_votes.py` -- Votes name tokenizer
- `src/preprocessing/extract_external.py` -- External call extractor
- `src/training/train.py` -- Training loop
- `src/evaluation/metrics.py` -- F1, EM, edit similarity, n-gram similarity
- `scripts/predict.py` -- End-to-end inference on stripped binaries
- `scripts/eval_test.py` -- Val/test evaluation
- `scripts/eval_demo_wulver.py` -- Demo evaluation on Wulver (no BAP)
- `scripts/eval_knn_hybrid.py` -- k-NN hybrid evaluation
- `configs/optimized.yaml` -- 8M model config
- `configs/optimized_large.yaml` -- 25M model config

### Build and Run Commands
```bash
source ~/cs785-project/activate.sh
# Train 25M model:
python3 -m src.training.train --config configs/optimized_large.yaml --seed 42
# Train 8M model (ablation):
python3 -m src.training.train --config configs/optimized.yaml --seed 42
# Evaluate on test set:
python3 scripts/eval_test.py checkpoints/best_model_wulver.pt
# Demo evaluation (Wulver):
python3 scripts/eval_demo_wulver.py checkpoints/best_model_wulver.pt
# k-NN hybrid evaluation:
python3 scripts/eval_knn_hybrid.py checkpoints/best_model_wulver.pt
# Predict on any binary (local, needs BAP):
python3 scripts/predict.py --binary /path/to/stripped/binary
```

---

## 2. Current Best Results

### Exp 36 (25M params, pretrained encoder) + k-NN Hybrid Inference

**Decoder-only results:**
- **Val F1: 0.7341** (epoch 46, greedy decode on 17 val binaries)
- **Test: EM=74.3%, F1=0.795, NgSim=0.798, EdSim=0.829** (8,973 functions)
- **Demo: EM=48.5%, F1=0.593, NgSim=0.625, EdSim=0.688** (12,688 functions, 11 unseen packages)
- **Diffutils: 68.5% EM** (300/438, completely unseen package)

**k-NN hybrid results (no retraining, inference-only improvement):**
- **Test: EM=76.3%, F1=0.804** (8,973 functions)
- **Demo: EM=53.1%, F1=0.625, NgSim=0.658, EdSim=0.716** (12,688 functions)
- **Diffutils: 69.9% EM** (with k-NN, was 68.5% decoder-only)
- Strategy: forward hybrid, threshold=-0.02 (87% k-NN, 13% decoder)

**Dataset:** 87,724 functions from 40 packages, 363 binaries (O0+O2)
**Token vocab:** 2,279 types | **Name vocab:** 2,642 tokens (Votes)
**Checkpoint:** `checkpoints/best_model_wulver.pt`

### Per-Package Demo Breakdown (k-NN hybrid, t=-0.02)
```
texinfo:   95.6% EM, F1=0.982   (was 92.7% decoder-only)
acct:      80.5% EM, F1=0.845   (was 72.8%)
rush:      75.1% EM, F1=0.786   (was 62.0%)   <- +13.1pp biggest gain
direvent:  73.4% EM, F1=0.770   (was 67.7%)
diffutils: 69.9% EM, F1=0.713   (was 68.5%)
hello:     57.5% EM, F1=0.714   (was 50.0%)
cppi:      53.6% EM, F1=0.641   (was 55.5%)   <- only regression
strace:    50.8% EM, F1=0.627   (was 42.7%)
csplit2:   42.4% EM, F1=0.454   (was 37.2%)
datamash:  20.4% EM, F1=0.303   (was 18.8%)
htop:       2.2% EM, F1=0.099   (was 0.9%)
```

### Comparison with Published Systems
| System | Venue | F1 (cross-project) | Notes |
|--------|-------|---------------------|-------|
| **Ours (k-NN hybrid)** | CS785 | **0.804 (test)** | 87K training, BAP pipeline |
| BLens | USENIX Sec 2025 | 0.46 | Ensemble + contrastive captioning |
| SYMGEN | NDSS 2025 | 0.38 | LLM fine-tuning (CodeLlama + LoRA) |
| AsmDepictor | AsiaCCS 2023 | 0.80+ | Inflated by no dedup per SYMGEN |
| SymLM | CCS 2022 | 0.585 | Pre-trained via Trex micro-traces |
| NERO | OOPSLA 2020 | 0.455 | Pointer-aware slicing |

Our F1=0.804 on test is competitive. Note: direct comparison is imperfect due to different datasets, dedup policies, and evaluation protocols.

---

## 3. Key Findings & Analysis

### Finding 1: Model is a Pure Recognizer
ALL correct demo predictions are function names seen in training (shared gnulib functions). **0/1,873 unseen function names were correctly predicted.** The k-NN hybrid confirms this -- retrieval outperforms generation because the model is fundamentally doing pattern matching, not composing novel names.

### Finding 2: k-NN > Decoder (Novel Finding)
k-NN retrieval over encoder embeddings outperforms autoregressive decoding on both test and demo:
- Test: k-NN EM=76.3% vs decoder EM=74.3% (+2.0pp)
- Demo: k-NN hybrid EM=53.1% vs decoder EM=48.5% (+4.6pp)

Why:
1. **Phantom predictions eliminated**: Decoder composes plausible but non-existent names (43.8% phantom rate). k-NN always returns real training names.
2. **Autoregressive error cascade**: One wrong sub-token derails the whole sequence. k-NN makes one holistic decision.
3. **Search space constrained**: 2,642^20 possible sequences but only ~3,000 real names. k-NN constrains to real names.
4. **Decoder changes won't help**: Transformer decoder, pointer networks, CTC all expected neutral-to-negative.

**Paper implication**: The encoder pipeline is the core contribution, not the decoder. k-NN hybrid is free improvement at inference time.

### Finding 3: External Call Paradox
External calls alone improve test F1 (+0.077) but **HURT demo EM (-4.3pp)**. Without callee/caller context, the model overfits to library call patterns. Adding callee/caller context resolves this (+12.8pp demo EM). This shows why naive feature addition fails and principled multi-context fusion matters.

### Finding 4: Multi-Context Gated Fusion
The 3-stage cascaded fusion (ext calls -> callee -> caller) with conditional bypass is the key architectural innovation. Each stage builds on the previous:
- Ext calls provide library-level signal
- Callee context disambiguates similar call patterns
- Caller context adds calling-site structural information
- Conditional bypass prevents mode collapse for functions without context (critical -- eliminated 48% mode collapse)

Total gain from ablation (Model 2->4): Test F1 +0.175, Demo EM +8.5pp.

### Finding 5: ENDBR64 Indirect Jump Wrappers
96.4% of O0 functions were single-token wrapper functions (ENDBR64 stubs). BAP lifts `endbr64; jmp addr+4` as separate functions. Resolved by following wrappers to their callee code in preprocessing. **This single fix: O0 EM 1.6% -> 41.5%.** No published paper found this because IDA/Ghidra auto-resolve these wrappers.

### Finding 6: Mode Collapse Root Cause
The baseline's 45.7% test EM was achieved WITH 47% mode collapse -- the collapse target happened to be correct for many binutils functions. Every attempt to improve 0-ext functions (UL loss, contrastive loss, expanded data) caused mode collapse to a different name. The fix was: (a) fixed random seed for reproducibility, (b) conditional gate bypass, (c) callee/caller context to break input degeneracy.

### Finding 7: Val F1 Paradox
Higher Val F1 doesn't always mean better demo EM. Val is dominated by small GNU packages; demo tests on diverse unseen packages. Val F1 can improve while test collapses.

### Finding 8: Capacity-Generalization Tradeoff
At 8M params, adding non-GNU packages helps those packages but hurts gnulib demo performance (cross-contamination). Scaling to 25M params resolves this -- the larger model holds both naming patterns simultaneously. Every demo package improved when going from 8M to 25M.

### Finding 9: Pretrained Encoder Helps
MLM + contrastive pretraining with partial embedding transfer (84.4% of rows) gives best Val F1 and demo EM. Transfer learning from related task validates SymLM/CALLEE findings in our domain.

### Finding 10: O0/O2 Generalization Gap
O0 binaries are fundamentally harder -- no inlining/optimization produces wrapper functions with similar token signatures. After ENDBR64 fix: O0 ~41.5% EM vs O2 ~70%+ EM. This is a novel finding about optimization level impact on name recovery.

---

## 4. Experiment History

### Summary Table (All Experiments)

| Exp | Date | Change | Val F1 | Test EM | Demo EM | Outcome |
|-----|------|--------|--------|---------|---------|---------|
| -- | 03-18 | Baseline (mean pool, no features) | 0.5808 | 40.0% | 51.7% | Starting point |
| 1 | 03-18 | Diverse beam search | -- | -- | -- | WORSE |
| 2 | 03-18 | Decoder cross-attention | 0.5517 | 40.4% | -- | WORSE |
| 3 | 03-18 | Token dropout (0.15) | 0.4920 | -- | -- | WORSE |
| 4 | 03-18 | Fix scheduled sampling | ~0.50 | -- | -- | INCONCLUSIVE |
| 5 | 03-19 | V2 tokens (per-flag, sizes) | 0.5825 | 41.8% | -- | BETTER |
| 6 | 03-19 | V3 tokens (imm buckets, addr) | 0.5927 | 42.6% | -- | BETTER |
| 7 | 03-19 | Multi-task aux loss | 0.5784 | -- | -- | WORSE |
| 8 | 03-19 | Longer training (120 epochs) | 0.6122 | 43.1% | -- | BETTER |
| 9 | 03-19 | Callee context encoder | 0.6192 | 43.4% | -- | BETTER |
| 10 | 03-19 | + Caller context | 0.6677 | 46.6% | -- | BETTER |
| 11 | 03-19 | Token attention pooling | 0.6674 | -- | -- | WASH |
| 12 | 03-19 | Votes tokenization | 0.6691 | 45.6% | 66.9% | BETTER |
| 13 | 03-20 | Supervised contrastive loss | 0.6563 | -- | -- | WORSE |
| 14 | 03-20 | String reference encoder | 0.6683 | 45.7% | 64.4% | WASH |
| 15 | 03-20 | Dataset expansion (O0-O3) | 0.6582 | OOM | 66.9% | INCONCLUSIVE |
| 16 | 03-21 | Binary context embedding | 0.6718 | 10.8% | 62.7% | COLLAPSE |
| 17 | 03-21 | Weighted sampling | 0.6102 | 5.1% | 53.1% | COLLAPSE |
| 16b | 03-22 | V4 tokenization | 0.6682 | -- | 71/118 | WASH |
| 17b | 03-22 | 80-epoch training | 0.6723 | 5.1% | 73/118 | COLLAPSE |
| 18 | 03-22 | Pretrained embeddings research | -- | -- | -- | NOT PURSUED |
| 19 | 03-22 | Callee name enrichment + 2-pass | 0.6686 | -- | +1/201 | MARGINAL |
| 20 | 03-22 | Expanded dataset + fixed splits | 0.6793 | 6.2% | 63.6% | COLLAPSE |
| 21 | 03-23 | Unlikelihood loss (0.1) | 0.6716 | 5.5% | 62.7% | COLLAPSE |
| 22 | 03-23 | Contrastive loss (NT-Xent, 0.5) | 0.6516 | 3.2% | 63.6% | COLLAPSE |
| 23 | 03-23 | String encoder v2 | 0.6680 | 4.1% | 56.8% | COLLAPSE |
| 25 | 03-23 | ENDBR64 thunk resolution | 0.7319 | ~10% | 62.7% | BETTER (Val) |
| 33 | 03-24 | 105K + non-GNU packages | 0.7475 | -- | 59.3% | MARGINAL |
| 34 | 03-25 | Restructured 87K + unified demo | 0.7221 | 8.8% | 57.8% | CAPACITY ISSUE |
| 35 | 03-25 | 25M model | 0.7103 | -- | 65.1% diff | BETTER (demo) |
| 36 | 03-27 | + Pretrained encoder | **0.7341** | **74.3%** | **68.5% diff** | **BEST** |
| -- | 03-30 | + k-NN hybrid (inference-only) | -- | **76.3%** | **53.1% all** | **BEST** |

### Ablation Study Results (5-model progression, all on 87K dataset)

| # | Model | Params | Val F1 | Test EM | Test F1 | Demo EM | Demo F1 |
|---|-------|--------|--------|---------|---------|---------|---------|
| 1 | DeBin (ExtraTrees) | -- | -- | -- | -- | -- | -- |
| 2 | GAT + Decoder (basic) | 4.4M | 0.374 | 51.3% | 0.606 | 24.1% | 0.364 |
| 3 | + Ext Call Encoder | 6.1M | 0.545 | 60.1% | 0.683 | 19.8% | 0.320 |
| 4 | + Callee/Caller Context | 8.0M | 0.720 | 72.3% | 0.781 | 32.6% | 0.460 |
| 5 | + Pretrain + Scale (25M) | 25M | 0.734 | 74.3% | 0.795 | 48.5% | 0.593 |

**Incremental gains:**
- 2->3 (+Ext Calls): Test F1 +0.077, Test EM +8.8pp -- but Demo EM **-4.3pp** (ext-call paradox)
- 3->4 (+Callee/Caller): Test F1 +0.098, Demo EM +12.8pp -- resolves ext-call paradox
- 4->5 (+Pretrain+Scale): Test F1 +0.014, Demo EM +15.9pp -- best generalization

### What Worked (keep these)
- V3 instruction-type tokenization (+1,750% F1 over raw tokens)
- Callee + Caller context encoders (+8.7% Val F1)
- Votes name tokenizer (95% less OOV than BPE)
- Conditional gate bypass (eliminated 48% mode collapse)
- ENDBR64 indirect jump wrapper resolution for O0 binaries
- Larger model (8M->25M) -- reduces cross-contamination
- Pretrained encoder (MLM+contrastive) + partial embedding init
- Expanded diverse training data (when combined with larger model)
- k-NN hybrid inference (+6.7pp demo EM, no retraining)

### What Failed (DO NOT RETRY)
- Token attention pooling, block statistics, copy mechanism
- Data deduplication, binary context embedding, weighted sampling, package cap
- Expanding to O1/O3 (gnulib duplication), 80-epoch training
- Unlikelihood loss, contrastive loss, multi-label aux loss, string encoder (v1 and v2)
- Diverse beam search, decoder cross-attention, token dropout 0.15
- Non-GNU packages with 8M model (causes cross-contamination)
- Cosine LR restart for extended training (patience runs out)
- Pretrained embeddings (PalmTree, CLAP) -- require raw x86 (not BAP-IR), and Maier et al. showed no benefit at 64K+ labeled data

---

## 5. Research Context

### Related Work Papers

| Paper | Venue | Key Technique | F1 | Notes |
|-------|-------|---------------|-----|-------|
| NERO | OOPSLA 2020 | Pointer-aware slicing, per-call-site decoder | 0.455 | Only works for functions with calls |
| XFL | IEEE S&P 2023 | DEXTER embedding + multi-label classification | Prec 82-83% | Multi-label avoids seq2seq errors |
| SymLM | CCS 2022 | Pre-trained via Trex micro-traces, caller/callee fusion | 0.585 | Pretraining is key differentiator |
| AsmDepictor | AsiaCCS 2023 | Standard Transformer seq2seq | 0.80+ | Inflated by no dedup (per SYMGEN) |
| SYMGEN | NDSS 2025 | LLM fine-tuning (CodeLlama + LoRA), callee normalization | 0.38 | Showed prior results inflated 1.2-2.9x |
| BLens | USENIX Sec 2025 | Ensemble + contrastive captioning | 0.79/0.46 | 0.79 per-binary, 0.46 cross-project |
| GenNm | NDSS 2025 | Fine-tunes CodeGemma + SymPO preference optimization | 168% on unseen | Novel approach for unseen names |
| CALLEE | IEEE S&P 2023 | Contrastive learning for call graph recovery | F1 94.6% | Transfer learning +28% validates our pretraining |

### Key Takeaways from Literature
- Our F1=0.804 is competitive after proper deduplication
- The 0-ext-call problem is universal across all systems
- Pre-training helps across multiple papers (SymLM, CALLEE, our Exp 36)
- Cross-project generalization is the real challenge (BLens: 0.79 -> 0.46)
- Our ext-call paradox finding is novel (no prior work reports this)

### Pretrained Binary Code Models Evaluated
- **PalmTree**: Per-instruction 128-dim embeddings, x86 only
- **CLAP**: Per-function 768-dim (HuggingFace), easiest integration
- **jTrans**: Per-function 768-dim with jump-aware positional encoding
- **Trex**: Per-function 512-dim, multi-architecture

**All require raw x86 assembly (not BAP-IR).** Maier et al. (AsiaCCS 2024) showed pretrained embeddings provide NO benefit when labeled data >= 64K functions. NOT pursued.

---

## 6. Infrastructure

### Wulver HPC Cluster
- **UCID**: adp232 | **Host**: wulver.njit.edu
- **Account**: 2026-spring-cs-785-hz79-adp232
- **Project dir**: /course/2026/spring/cs/785/hz79/adp232/cs785
- **Partition**: course_gpu (GPU) | **GPU**: A100 40GB
- **Python**: 3.9.21 via `module load bright && module load python3`
- **Virtualenv**: /course/2026/spring/cs/785/hz79/adp232/cs785-env
- **Packages**: torch 2.5.1+cu121, torch-geometric, torch-scatter/sparse/cluster

### SSH Multiplexing
- Config with `ControlPersist 12h` -- user SSHs once (Duo 2FA), then agent can run commands
- Always use `ssh wulver` (not full hostname) for multiplexing
- Stale socket fix: delete `~/.ssh/sockets/adp232@wulver.njit.edu-22`

### Workflow
1. User opens `ssh wulver` (authenticates with Duo once)
2. Make code changes locally
3. Run `bash scripts/wulver_sync.sh` to rsync src/configs/scripts to Wulver
4. Submit jobs via `ssh wulver "cd ... && sbatch ..."`
5. Monitor via `ssh wulver "cat cs785-train.*.out"`

### Evaluation Pipeline
- **Training + Test eval**: Run on Wulver (no BAP needed)
- **Demo eval**: Run on Wulver via `eval_demo_wulver.py` (loads pre-processed graphs)
- **k-NN eval**: Run on Wulver via `eval_knn_hybrid.py`
- **BAP preprocessing**: Local only (BAP not installed on Wulver)
- **Ground truth**: Always use labels.json (nm has alias duplicates that inflate denominators)

### Key Checkpoints
```
checkpoints/best_model_wulver.pt  -- Exp 36 (25M, Val F1=0.7341, CURRENT BEST)
checkpoints/best_model.pt         -- Latest local copy
checkpoints/exp32_val7508.pt      -- Exp 32 (best Val F1 on old 64K dataset)
checkpoints/pretrained_encoder.pt -- Pretrained block+graph encoder (MLM+contrastive)
checkpoints/baseline_v1_val0668.pt -- Original baseline
checkpoints/knn_index.bin         -- FAISS index (87,724 vectors, 1024-dim)
checkpoints/knn_index_names.json  -- Names mapping (24,177 unique names)
```

### Dataset
- **Training**: 87,724 functions, 40 packages, 363 binaries (O0+O2)
- **Validation**: 4,695 functions, 17 binaries
- **Test**: 8,973 functions, 18 binaries
- **Demo**: 11 packages, ~12,688 matched functions
- **Split file**: `data/split_assignments.json` (137/17/18 train/val/test binary assignments)
- All demo data pre-processed (graphs, labels, ext calls) -- no BAP needed on Wulver

---

## 7. Remaining Work

### Timeline
- **Deadline**: ~April 22, 2026

### Tasks
- [x] Ablation study: All 5 models trained and evaluated
- [x] k-NN hybrid inference: Implemented and evaluated
- [ ] **Write paper** -- ablation + k-NN results complete, all data ready
- [ ] **Final presentation slides**
- [ ] Confidence-based selective prediction (optional)
- [ ] Clean up code and prepare final submission
- [ ] Update CLAUDE.md with k-NN hybrid results

### Paper Strategy
1. **Ablation study**: 5-model ablation (DeBin -> basic -> +ext -> +context -> +pretrain)
2. **k-NN hybrid as novel finding**: Retrieval > generation for this domain
3. **Ext-call paradox**: Core finding about multi-context fusion necessity
4. **Recognition vs composition analysis**: Show model memorizes, doesn't compose
5. **O0/O2 generalization gap**: Novel finding about optimization level impact
6. **Comparison with published systems**: BLens, SYMGEN, AsmDepictor, XFL

---

## 8. Critical Rules

1. **NEVER overwrite `data/external_calls/external_vocab.json` during inference/evaluation.** The ext_vocab is saved inside the checkpoint and loaded from there.
2. **Use deterministic sort `(-count, name)` for token vocab building** to ensure consistent token IDs. Non-deterministic sort caused 1,232 token ID mismatches (bug found 2026-03-23).
3. **Validation F1 uses greedy decode** (argmax), but evaluation/demo uses beam search (k=5). Beam search gives ~3-5% higher F1.
4. **Use `data/split_assignments.json` for splits** -- contains fixed 137/17/18 train/val/test binary assignments.
5. **Demo EM is the primary generalization metric** -- stable across experiments, no collapse.
6. **`predict.py` tokenizes correctly** (uses checkpoint token_vocab directly). Eval scripts must load token_vocab from checkpoint.
7. **Always set `torch.manual_seed(seed)`** for reproducible training. Use `--seed 42`.
8. **Val F1 is unreliable for generalization** -- can improve while test/demo regresses.
9. **O0 binaries need ENDBR64 thunk resolution** -- without it, 96.4% are degenerate.
10. **Test set is 80.8% binutils+bison** -- pathologically hard, dominated by mode collapse.
11. **Use `ssh wulver`** (not full hostname) for SSH multiplexing.
12. **Use generic "cs785-eval" for SLURM job names**.

---

## 9. User Preferences & Collaboration Style

- Max training time: 30-45 min (~50 epochs) for 8M model. Larger models need more time.
- Compare experiments at same epoch count for fairness.
- Prefers fundamental/innovative changes over hyperparameter tuning.
- Wants deep research with literature references.
- Report unified demo numbers by default -- no O0/O2 or in-training/unseen breakdown unless asked.
- Make code changes WITHOUT asking permission (auto-apply).
- Only ask permission before: architectural changes, hyperparameter modifications, or file deletions.
- After making changes, briefly state what was changed and why.
- Always log experiment results to `results/experiment_log.md`.
- For long-running processes: update Discord + terminal every 30 minutes.
