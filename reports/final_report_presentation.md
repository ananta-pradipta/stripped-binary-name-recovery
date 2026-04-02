# Binary Function Name Recovery Using GNN with Gated Multi-Context Fusion
## CS785 Final Presentation

**Team:** Ananta Dian Pradipta, Robert Blacha, Zhihao Lin
**NJIT, Spring 2026**

---

## Slide 1: Problem

- Stripped binaries have **no function names** — critical for reverse engineering, malware analysis, vulnerability research
- A typical binary has hundreds of unnamed functions (`sub_4a30`, `sub_5c10`, ...)
- **Goal:** Automatically predict human-readable function names from binary code

---

## Slide 2: Our Approach

3-stage encoder-decoder pipeline:

```
Stripped Binary → BAP (IR lifter) → Instruction-Type Tokens
    → Transformer (per-block) → GAT (per-function CFG)
    → Multi-Context Gated Fusion (ext calls + callee + caller)
    → GRU Decoder → Function Name
```

- **Input:** Control flow graph with semantically-typed instructions (~1,510 types)
- **Context:** External library calls + inter-procedural signatures (callee/caller)
- **Output:** Sub-token sequence decoded into function name

---

## Slide 3: Key Innovation — Multi-Context Gated Fusion

3-stage cascaded fusion with **conditional bypass**:

| Stage | Context Source | Signal | Bypass When |
|-------|--------------|--------|-------------|
| 1 | External calls (malloc, printf, ...) | Library-level behavior | No ext calls |
| 2 | Callee signatures (what this function calls) | Inter-procedural specificity | No callees |
| 3 | Caller signatures (who calls this function) | Call-site differentiation | No callers |

Each stage: `z = gate * z_prev + (1-gate) * context`

**Conditional bypass is critical:** 70% of functions have no ext calls. Without bypass → 47% mode collapse (all predict same name).

---

## Slide 4: Novel Finding — Ext-Call Paradox

| Model | Test F1 | Demo EM (unseen pkgs) |
|-------|---------|----------------------|
| Code only (GAT + Decoder) | 0.606 | 24.1% |
| + External calls | 0.683 (+0.077) | **19.8% (-4.3pp)** |
| + Callee/Caller context | 0.781 (+0.098) | **32.6% (+12.8pp)** |

- Ext calls **improve** test but **hurt** generalization when used alone
- Many unrelated functions share similar library call patterns
- Callee/caller context provides the missing **disambiguation**
- **Takeaway:** Naive feature addition can degrade generalization; principled fusion is required

---

## Slide 5: Self-Supervised Pretraining

Two-stage training:

1. **Pretrain encoder** (unsupervised):
   - MLM: mask 15% of instruction tokens, predict originals
   - Contrastive: same function at O0 vs O2 should be similar
   - 84.4% of token embeddings transferred to downstream model

2. **Fine-tune full model** (supervised):
   - 25M params, 50 epochs on A100 GPU
   - Cosine LR with warmup, label smoothing 0.1

**Impact:** +15.9pp demo EM (biggest generalization gain)

---

## Slide 6: Ablation Study

| # | Model | Params | Test F1 | Demo EM |
|---|-------|--------|---------|---------|
| 1 | DeBin baseline (ExtraTrees) | — | — | — |
| 2 | GAT + GRU Decoder | 4.4M | 0.606 | 24.1% |
| 3 | + External Calls | 6.1M | 0.683 | 19.8% |
| 4 | + Callee/Caller Context | 8.0M | 0.781 | 32.6% |
| 5 | + Pretrain + Scale | 25M | **0.795** | **48.5%** |

Each component builds on the previous:
- Multi-context fusion (2→4): **+0.175 Test F1**
- Pretraining + scaling (4→5): **+15.9pp Demo EM**

---

## Slide 7: Results

### Test Set (8,973 functions, 18 held-out binaries)

| Metric | Decoder | k-NN Hybrid |
|--------|---------|-------------|
| F1 | 0.795 | **0.804** |
| Exact Match | 74.3% | **76.3%** |
| Edit Similarity | 0.829 | — |

### Cross-Project (12,688 functions, 11 unseen packages)

| Metric | Decoder | k-NN Hybrid |
|--------|---------|-------------|
| F1 | 0.593 | **0.625** |
| Exact Match | 48.5% | **53.1%** |
| Diffutils EM | 68.5% | **69.9%** |

### vs Published Systems

| System | Cross-Project F1 |
|--------|-----------------|
| BLens (USENIX Security '25) | 0.46 |
| SYMGEN (NDSS '25) | 0.38 |
| **Ours (decoder)** | **0.593** |
| **Ours (k-NN hybrid)** | **0.625** |

---

## Slide 8: Per-Package Demo Breakdown (k-NN Hybrid)

| Package | Decoder EM | k-NN EM | In Training? |
|---------|-----------|---------|-------------|
| texinfo | 91.6% | **95.6%** | Yes |
| acct | 65.2% | **80.5%** | Yes |
| rush | 58.4% | **75.1%** | Yes |
| direvent | 60.1% | **73.4%** | Yes |
| diffutils | 68.5% | **69.9%** | **No** |
| hello | 49.4% | **57.5%** | **No** |
| cppi | 50.0% | 53.6% | **No** |
| strace | 29.9% | **50.8%** | Yes |
| datamash | 19.2% | **20.4%** | **No** |
| htop | 0.6% | **2.2%** | Yes* |

*htop: different version than training → demonstrates recognizer limitation
k-NN improves 10/11 packages; cppi is the only slight regression

---

## Slide 9: Novel Finding — k-NN > Decoder

k-NN retrieval over encoder embeddings **outperforms autoregressive decoding** with zero retraining:

| Method | Test EM | Demo EM | Demo F1 |
|--------|---------|---------|---------|
| Decoder only | 74.3% | 48.5% | 0.593 |
| k-NN hybrid | **76.3%** | **53.1%** | **0.625** |

**Why k-NN wins:**
- Decoder composes plausible but nonexistent names (43.8% "phantom" rate)
- One wrong sub-token cascades through the entire sequence
- k-NN constrains to real training names — the model is a recognizer, not a composer

**Implication:** The **encoder pipeline** (Transformer + GAT + gated fusion) is the core contribution. It produces embeddings good enough for direct retrieval.

---

## Slide 10: Key Discoveries

1. **Ext-Call Paradox:** Ext calls help test (+0.077 F1) but hurt generalization (-4.3pp demo EM) without inter-procedural context

2. **k-NN > Decoder:** Retrieval outperforms generation (+4.6pp demo EM, no retraining). Encoder embeddings are the real contribution.

3. **Pure Recognizer:** 0/1,873 unseen function names predicted correctly. Model memorizes code→name patterns, doesn't compose novel names.

4. **ENDBR64 Wrapper Resolution:** 96.4% of O0 functions were trivial indirect jump wrappers. Resolving them: O0 EM 1.6% → 41.5%.

5. **Capacity vs Contamination:** 8M model suffers cross-package naming pollution with diverse training data. 25M model resolves this — every package improves.

---

## Slide 11: Preprocessing Innovation — V3 Instruction-Type Tokenization

Raw BAP-IR has 32K+ unique tokens → reduced to **~1,510 semantic types**

| Category | Examples |
|----------|---------|
| Calls | `CALL_malloc`, `CALL_INTERNAL`, `CALL_INDIRECT` |
| Memory | `MEM_READ_32`, `MEM_WRITE_ARG_64` |
| Stack | `STACK_LOAD_64`, `STACK_STORE_32` |
| Flags | `COND_BRANCH_ZF`, `FLAG_CF` |
| Constants | `ASSIGN_ZERO`, `ASSIGN_POW2`, `ASSIGN_ADDR` |

**+1,750% F1 gain** over raw tokens

---

## Slide 12: Dataset & Training

| Metric | Value |
|--------|-------|
| Training functions | 87,724 |
| Packages | 40 (GNU + non-GNU) |
| Binaries | 363 (O0 + O2) |
| Test functions | 8,973 |
| Demo functions (unseen) | 12,688 |
| Instruction token vocab | 2,279 types |
| Name token vocab | 2,642 (Votes tokenizer) |

**Hardware:**
### Local Machine (preprocessing + development)

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 5 7535HS (6 cores / 12 threads, 3.3GHz) |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB GDDR6 VRAM) |
| RAM | 8 GB DDR5 |
| Storage | 512GB NVMe SSD |
| OS | Ubuntu 22.04 (WSL2 on Windows 11) |
| Role | BAP preprocessing, development, debugging |

### NJIT Wulver HPC Cluster (training + evaluation)

| Component | Specification |
|---|---|
| CPU | Intel Xeon Sapphire Rapids (48 cores per node) |
| GPU | NVIDIA A100 PCIe (40GB HBM2e VRAM) |
| RAM | 384 GB per node |
| Nodes | 4 GPU nodes (n0801, n0802, n0807, n0808), 8 GPUs each |
| GPU config | A100 with MIG (Multi-Instance GPU) 10GB slices, but allocated full 40GB card |
| Scheduler | SLURM (course_gpu partition, course QoS) |
| Interconnect | InfiniBand HDR |
| Storage | MMFS1 parallel filesystem |
| Role | Model training (AMP), evaluation, ablation study |

### Software Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10 (local), 3.9.21 (Wulver) | Runtime |
| PyTorch | 2.5.1 + CUDA 12.1 | Deep learning framework |
| PyTorch Geometric | 2.6.1 | Graph neural network layers |
| BAP | 2.5.0 | Binary Analysis Platform (local only) |
| SentencePiece | 0.2.0 | BPE tokenization (legacy) |
| NVIDIA Driver | 535.183 (Wulver) | GPU acceleration |
| Mixed Precision | torch.cuda.amp (FP16) | 2x training speedup |

### Training Time

| Model | Params | Batch | Epochs | Time | Hardware |
|---|---|---|---|---|---|
| 8M (ablation models 2-4) | 4.4-8.0M | 128 | 50 | ~1.5 hours | Wulver A100 |
| 25M (final model) | 25M | 32 | 50 | ~3.75 hours | Wulver A100 |
| Pretraining (encoder) | 12M | 128 | 10 | ~45 minutes | Wulver A100 |

---

## Slide 13: Midterm → Final Progress

| Metric | Midterm | Final | Change |
|--------|---------|-------|--------|
| Val F1 | 0.519 | **0.734** | +41% |
| Test F1 | — | **0.804** (k-NN) | — |
| Demo EM | — | **53.1%** (k-NN) | — |
| Diffutils EM | 39.0% | **69.9%** (k-NN) | +30.9pp |
| Model params | 4.5M | **25M** | 5.6x |
| Training data | 3,991 functions | **87,724** | 22x |
| Packages | 5 | **40** | 8x |

---

## Slide 14: Limitations & Future Work

**Limitations:**
- Pure recognizer — cannot compose novel names (0/1,873 unseen names correct)
- O0/O2 gap persists (41.5% vs 69.8% EM)
- Version-sensitive (htop: 0.6% EM despite being in training)

**Future directions:**
- Confidence-based selective prediction (abstain when uncertain)
- LLM integration (use embeddings as features for CodeLlama to compose novel names)
- Cross-architecture extension (ARM, MIPS)
- Iterative context propagation (use predicted names as input for second pass)

---

## Slide 15: Contributions Summary

1. **Multi-context gated fusion** with conditional bypass — 3-stage cascade providing +0.175 Test F1 over code-only baseline

2. **Ext-call paradox** — novel finding that external calls hurt generalization without inter-procedural disambiguation

3. **k-NN > Decoder** — retrieval over learned embeddings outperforms autoregressive generation (+4.6pp demo EM, no retraining)

4. **Self-supervised pretraining** with partial embedding transfer — +15.9pp demo EM

5. **V3 instruction-type tokenization** — reduces 32K tokens to 1,510 types (+1,750% F1)

6. **ENDBR64 wrapper resolution** — preprocessing fix raising O0 EM from 1.6% to 41.5%

**Results:** 0.804 F1 on test (k-NN), 0.625 F1 on 11 unseen packages — competitive with BLens (0.46) and SYMGEN (0.38)
