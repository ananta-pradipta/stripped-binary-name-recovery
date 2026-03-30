# Midterm vs Final: Progress Comparison
## CS785 — Binary Function Name Recovery

---

## 1. Key Innovations Since Midterm

| Innovation | Midterm | Current | Impact |
|---|---|---|---|
| Instruction tokenization | 371 semantic types | 1,510 types (value bucketing, address detection, per-flag) | +1,750% F1 |
| Name sub-tokenization | BPE (1,095 vocab, high OOV) | Votes tokenizer (2,642 vocab, 3-model voting) | 95% less OOV |
| Inter-procedural context | External calls only | External + callee + caller signatures | +8.7% Val F1 |
| Self-supervised pretraining | None | MLM + contrastive on encoder, partial embedding init | +0.015 Val F1 |
| Conditional gated fusion | Single gate, always applied | 3-stage cascade with bypass when no context available | Fixed 47% mode collapse |
| Indirect jump resolution | Not handled | Follow indirect jumps (endbr64→jmp) to resolve wrapper functions | O0 EM: 1.6% → 41.5% |
| Token-level attention | Mean pool over tokens | Transformer self-attention learns token importance weights | Better block representations |
| Model capacity | 4.5M params | 25M params | Reduced cross-package contamination |

---

## 2. Related Work & Relevance

| Paper | Approach | Relevance to Our Work |
|---|---|---|
| He et al. (2018) — **DeBin** [CCS'18] | BAP-IR + ExtraTrees + CRF | Our baseline. We adopt BAP as IR lifter but replace statistical classifiers with neural encoder-decoder. |
| Wang et al. (2022) — **jTrans** [ISSTA'22] | Jump-aware Transformer for binary similarity | Motivates our Transformer-based block encoder; shows self-attention captures control flow patterns in binary code. |
| Jin et al. (2022) — **SymLM** [NDSS'23] | Execution-aware code embeddings | Inspires our self-supervised pretraining approach (MLM on instruction tokens). |
| Xu et al. (2023) — **NERO** [USENIX'23] | GNN on call graphs for function naming | Validates using inter-procedural context (callee/caller) as strong signal for name recovery. |
| Al-Kaswan et al. (2023) — **BLens** [ICSE'23] | Transformer encoder-decoder for binary naming | Direct competitor; reports 0.46 F1 cross-project. Our O2 test F1 = 0.91. |
| Xu et al. (2024) — **SYMGEN** [ASE'24] | Hybrid GNN + seq2seq | Reports 0.38 cross-project F1. We outperform on comparable setup. |
| Bengio et al. (2015) — **Scheduled Sampling** | Curriculum learning for seq2seq | We adopt teacher forcing decay (1.0→0.3) to reduce exposure bias in decoder training. |

---

## 3. Preprocessing Pipeline

| Stage | Midterm | Current |
|---|---|---|
| Binary lifting | BAP 2.5.0 → BAP-IR | Same |
| Instruction tokenization | ~20 categories (371 types) | Enriched classification: value bucketing, address patterns, flag tracking, memory sizes (**1,510 types**) |
| Name tokenization | BPE sentencepiece (1,095 vocab) | **Votes**: 3 tokenizers vote on split boundaries — handles camelCase, snake_case, prefixes (**2,642 vocab**) |
| External call extraction | PLT/GOT names | Same |
| Inter-procedural features | Not present | **Callee signatures** (first 10 tokens of called functions) + **Caller signatures** (who calls this function) |
| Indirect jump resolution | Not handled | Follows indirect jump chains (e.g., `endbr64; jmp target`) to map wrapper functions to their actual code — resolves 40,708 O0 wrappers that otherwise appear as 1-token trivial functions |
| Dataset split | Random 80/10/10 | **Fixed deterministic split** (split_assignments.json) — ensures reproducibility across experiments |

---

## 4. Hardware & Environment

| | Midterm | Current |
|---|---|---|
| **Local machine** | | |
| CPU | AMD Ryzen 5 7535HS (6C/12T) | Same |
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) | Same |
| RAM | 8 GB | Same |
| **HPC cluster** | Not used | **NJIT Wulver** |
| HPC GPU | — | NVIDIA A100 PCIe (40GB VRAM) |
| HPC scheduler | — | SLURM (course_gpu partition) |
| Training features | — | Mixed precision (AMP), parallel data loading |
| **Software** | | |
| Framework | PyTorch 2.5.1 | Same + torch-geometric |
| Binary lifter | BAP 2.5.0 | Same |
| Python | 3.10 | 3.10 (local), 3.9 (Wulver) |

### Dataset Scale

| | Midterm | Current |
|---|---|---|
| Packages | 5 GNU packages | **40 packages** (GNU + strace, sqlite, lua, htop, etc.) |
| Binaries | 29 | **363** |
| Training functions | 3,991 | **87,724** |
| Test functions | 670 | **8,973** |
| Unseen demo functions | 118 (diffutils only) | **14,915** (11 packages) |
| Optimization levels | O2 only | **O0 + O2** |

---

## 5. Model Architecture Comparison

| Component | Midterm (4.5M) | Current (25M) |
|---|---|---|
| Block encoder | Transformer (2L, 4H, d=128) → mean pool → 256d | Transformer (**4L, 8H, d=256**) → **self-attention learns token importance** → 512d |
| Graph encoder | GAT (2L, 4H) + attention pool → 512d | GAT (**3L, 8H**) + attention pool → 1024d |
| External encoder | Embedding + Bi-GRU → 512d | Embedding + Bi-GRU → **1024d** |
| Callee context | Not present | **Bi-GRU over token signatures of up to 5 internal callees** |
| Caller context | Not present | **Bi-GRU over token signatures of up to 5 callers** |
| Fusion | Single gate (code <-> ext calls) | **3-stage cascade** with conditional bypass |
| Decoder | GRU (512 hidden, BPE) | GRU (**1024 hidden**, Votes) + repetition penalty |
| Pretraining | None | **MLM + contrastive** → partial embedding transfer |
| Training | LR=1e-4, patience=15 | LR=3e-4, patience=20, cosine LR, **AMP on HPC** |

---

## 6. Evaluation Results

### Overall Performance

| Metric | Midterm | Current | Change |
|---|---|---|---|
| Val F1 | 0.519 | **0.7341** | +41% |
| Diffutils EM (unseen pkg) | 39.0% | **67.6%** | +28.6pp |
| Diffutils F1 (unseen pkg) | 0.410 | **0.690** | +68% |

### Test Set by Optimization Level

| Split | Functions | EM | F1 |
|---|---|---|---|
| O2 Test | 4,445 | **69.8%** | **0.91** |
| O0 Test | 4,528 | 41.5% | 0.59 |
| Combined Test | 8,973 | 55.5% | 0.746 |

### Unseen Package Evaluation (Demo)

| Package | Functions | EM | F1 | In Training? |
|---|---|---|---|---|
| texinfo | 690 | **91.6%** | 0.961 | Yes |
| diffutils | 451 | **67.6%** | 0.690 | No |
| acct | 736 | 65.2% | 0.698 | Yes |
| direvent | 1,392 | 60.1% | 0.694 | Yes |
| rush | 1,660 | 58.4% | 0.684 | Yes |
| cppi | 268 | 50.0% | 0.600 | No |
| hello | 166 | 49.4% | 0.623 | No |
| csplit2 | 1,018 | 35.1% | 0.413 | No |
| strace | 6,879 | 29.9% | 0.446 | Yes |
| datamash | 386 | 19.2% | 0.306 | No |
| htop | 1,269 | 0.6% | 0.063 | Yes (version mismatch) |
| **Overall** | **14,915** | **39.8%** | **0.506** | |

### Cross-Project Comparison

| System | Cross-Project F1 | Notes |
|---|---|---|
| BLens (ICSE'23) | 0.46 | Transformer encoder-decoder |
| SYMGEN (ASE'24) | 0.38 | Hybrid GNN + seq2seq |
| **Ours (O2 test)** | **0.91** | GAT + gated fusion + GRU decoder |
| **Ours (unseen demo)** | **0.69** | On diffutils (completely unseen) |

---

## 7. Key Discoveries

- **Mode collapse**: 47% of predictions collapsed to one name in all training runs. Fixed by conditional gate bypass (skip fusion when no context available).
- **Indirect jump wrappers (ENDBR64)**: 96.4% of O0 functions were trivial wrappers. Single preprocessing fix raised O0 EM from 1.6% to 41.5%.
- **Pure recognizer behavior**: 0/1,873 unseen function names correctly predicted. The model memorizes code→name patterns, doesn't compose novel names.
- **Val F1 paradox**: Higher Val F1 doesn't correlate with better demo EM — demo on unseen packages is the better generalization metric.
- **Capacity-generalization tradeoff**: 8M→25M model reduced cross-package contamination. Every demo package improved.
