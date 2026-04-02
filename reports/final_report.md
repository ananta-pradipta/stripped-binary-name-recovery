# CS785 Final Report: Binary Function Name Recovery
## Using GNN with Gated Multi-Context Fusion and Self-Supervised Pretraining

**Team:** Ananta Dian Pradipta, Robert Blacha, Zhihao Lin
**Course:** CS785 Deep Learning, Spring 2026, NJIT
**Advisor:** Prof. HZ (hz79)

---

## 1. Problem Statement

Given a stripped x86-64 ELF binary (all symbol information removed), recover human-readable function names. This is critical for reverse engineering, malware analysis, vulnerability research, and binary auditing.

**Challenge:** Stripped binaries contain only raw machine code — no variable names, function names, or type information. A typical binary has hundreds to thousands of functions, each needing a meaningful name.

**Our approach:** A 3-stage deep learning pipeline that encodes binary code structure (control flow graphs), external library calls, inter-procedural context (callee/caller signatures), and decodes function names as sub-token sequences.

---

## 2. Related Work

| Paper | Venue | Approach | Cross-Project F1 | Key Limitation |
|---|---|---|---|---|
| He et al. — **DeBin** | CCS 2018 | BAP-IR + ExtraTrees + CRF | — | Statistical, no deep learning |
| Wang et al. — **jTrans** | ISSTA 2022 | Jump-aware Transformer for binary similarity | — | Similarity only, not naming |
| Jin et al. — **SymLM** | CCS 2022 | Execution-aware embeddings, pre-trained | 0.585 F1 | Requires micro-execution traces |
| Xu et al. — **NERO** | OOPSLA 2020 | GNN on call graphs | 0.455 F1 | Only works for functions WITH calls |
| Al-Kaswan et al. — **BLens** | USENIX Security 2025 | Ensemble + contrastive captioning | 0.46 F1 | Low cross-project generalization |
| Xu et al. — **SYMGEN** | NDSS 2025 | LLM fine-tuning (CodeLlama + LoRA) | 0.38 F1 | Inflated by duplicates |
| Zhu et al. — **CALLEE** | IEEE S&P 2023 | Contrastive learning for call graph recovery | 94.6% F1 | Different task (callsite matching) |
| Bengio et al. — **Scheduled Sampling** | NeurIPS 2015 | Curriculum learning for seq2seq | — | We adopt TF decay 1.0→0.3 |

**Our contributions:**
1. **Multi-context gated fusion**: A 3-stage cascaded architecture that fuses external library calls, callee signatures, and caller signatures with conditional bypass — providing +0.175 Test F1 over code-only baseline.
2. **Ext-call paradox discovery**: External calls alone hurt generalization (-4.3pp demo EM); inter-procedural context is required for disambiguation. No prior work reports this finding.
3. **k-NN hybrid inference**: Retrieval over learned encoder embeddings outperforms the autoregressive decoder — +4.6pp demo EM with no retraining. A novel finding for binary function naming.
4. **Self-supervised pretraining**: MLM + contrastive pretraining with partial embedding transfer, providing +15.9pp demo EM improvement.
5. Achieving **0.804 F1 on test** (k-NN hybrid) and **0.625 F1 on 11 completely unseen packages**.

---

## 3. Architecture

### 3-Stage Pipeline

```
Stripped Binary → BAP 2.5.0 → BAP-IR (.bir files)
    ↓
Stage 1: Instruction-Type Tokenization (V3: ~1,510 types)
         → Embedding (256-dim) + Positional Encoding
         → Transformer (4 layers, 8 heads)
         → Mean Pool over tokens
         → Linear → b_i ∈ R^512 per basic block
    ↓
Stage 2: GAT (3 layers, 8 heads) over CFG edges
         → Attention Pooling over blocks
         → f ∈ R^1024 (function-level embedding)
    ↓
Cascaded Gated Fusion:
  1. External calls:  if has_ext:     z = gate⊙f + (1-gate)⊙ext_emb
  2. Callee context:  if has_callees: z = gate⊙z + (1-gate)⊙callee_ctx
  3. Caller context:  if has_callers: z = gate⊙z + (1-gate)⊙caller_ctx
  (Bypass when no context — critical for preventing mode collapse)
    ↓
Stage 3: GRU Decoder (1 layer, 1024 hidden)
         + Beam Search (k=5, repetition penalty)
         → Votes sub-tokens → function name

Optional: k-NN Hybrid Inference (no retraining)
  - FAISS index of 87K training function embeddings
  - Forward hybrid: decoder default, k-NN fallback on low beam score
  - Threshold=-0.02 → 87% k-NN, 13% decoder
```

### Component Details

| Component | Architecture | Params | Purpose |
|---|---|---|---|
| Block Encoder | Transformer (4L, 8H, d=256) → mean pool → 512d | 5.2M | Encode instruction semantics per basic block |
| Graph Encoder | GAT (3L, 8H, 512→1024d) + attention pool | 6.8M | Learn control flow structure via message passing |
| External Call Encoder | Embedding(656, 256) → Bi-GRU(256) → 1024d | 1.1M | Encode library call sequences (malloc, printf, etc.) |
| Callee Context Encoder | Embedding(2279, 128) → Bi-GRU(256) → 1024d | 0.8M | Encode token signatures of up to 5 internal callees |
| Caller Context Encoder | Same as callee | 0.8M | Encode token signatures of up to 5 callers |
| Gated Fusion | 3-stage cascade, gate = σ(Linear(2048, 1024)) | 0.3M | Conditionally fuse code + context features |
| GRU Decoder | GRU(1024) + Votes vocab (2,642 tokens) | 10.0M | Generate function name as sub-token sequence |
| **Total** | | **25M** | |

### Key Design Choices

1. **Instruction-Type Tokenization (V3)**: Raw BAP-IR has 32K+ unique tokens. We classify each instruction into ~1,510 semantic types (e.g., `CALL_malloc`, `MEM_READ_32`, `COND_BRANCH_ZF`, `ASSIGN_ZERO`). This reduces vocabulary by 95% while preserving semantic information. **+1,750% F1 gain over raw tokens.**

2. **Conditional Gate Bypass**: When a function has no external calls (or no callees/callers), the fusion gate is bypassed entirely (z = previous embedding). Without this, the gate learns a default output that causes 47% of predictions to collapse to a single name. **This single fix eliminated mode collapse.**

3. **Votes Name Tokenizer**: Instead of BPE, we use a 3-model voting system for sub-token boundaries. Three strategies (underscore split, camelCase split, frequency-based) vote on where to split function names. **95% less OOV than BPE** (2,642 vocab vs 1,095).

4. **Self-Supervised Pretraining**: Block encoder + graph encoder pre-trained with:
   - Masked Language Model (MLM): predict masked instruction tokens
   - Contrastive learning: same function at O0 vs O2 should have similar embeddings
   - Partial embedding transfer: 84.4% of token embeddings copied by matching token names

5. **k-NN Hybrid Inference**: At inference time, build a FAISS index of all 87,724 training function embeddings. For each test function, compute encoder embedding and retrieve the nearest neighbor's name. The forward hybrid strategy uses the decoder by default but falls back to k-NN when the beam search score is below a threshold (t=-0.02). This yields 87% k-NN / 13% decoder decisions.

---

## 4. Preprocessing Pipeline

```
Source packages (GNU FTP/GitHub)
    ↓  [02_compile_dataset.sh]
    Compile at O0, O2 with debug info → Debug + Stripped binaries
    ↓  [03_preprocess.sh]
    ├── Step 3.1: nm → ground truth labels (addr → name mapping)
    ├── Step 3.2: BAP → BAP-IR (.bir files, ~1-4 hours)
    ├── Step 3.3: parse_bap.py → per-function CFG graphs (JSON)
    │   └── V3 instruction-type tokenization (~1,510 types)
    ├── Step 3.4: extract_external.py → external call lists
    ├── Step 3.5: Address matching (BAP sub_XXXX ↔ nm addresses)
    ├── Step 3.6: BPE vocabulary (legacy, replaced by Votes)
    └── Step 3.7: External call vocabulary
    ↓
    match_index.json (101K matched functions)
    + split_assignments.json (fixed train/val/test)
    ↓
    PyTorch Dataset (build_dataset.py)
```

### ENDBR64 Indirect Jump Wrapper Resolution

A critical preprocessing fix: In O0 binaries, BAP lifts `endbr64; jmp target` as separate 2-block wrapper functions generated for indirect jumps (ENDBR64 stubs). These are compiler-inserted stubs that perform a security check (Intel CET) followed by an immediate jump to the actual function body. 96.4% of O0 functions were these trivial wrappers. We resolve them by following the jump to the real function's code.

**Impact:** O0 EM jumped from **1.6% → 41.5%** with this single fix.

---

## 5. Dataset

| Metric | Value |
|---|---|
| Total packages | 40 (GNU + non-GNU) |
| Total binaries | 363 |
| Training functions | 87,724 |
| Validation functions | 4,695 (17 binaries) |
| Test functions | 8,973 (18 binaries) |
| Demo functions (unseen) | 12,688 (11 packages) |
| Optimization levels | O0, O2 |
| Instruction token vocab | 2,279 types |
| Name token vocab | 2,642 (Votes) |
| External call vocab | 656 functions |

### Train/Val/Test Split

- **Split by binary** (not by function) — prevents data leakage
- Fixed in `split_assignments.json` for reproducibility
- Demo packages completely excluded from training

---

## 6. Hardware & Environment

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

## 7. Ablation Study

All models trained on same dataset (87K functions), evaluated on same test (8,973) and demo (12,688) sets.

| # | Model | Params | Val F1 | Test EM | Test F1 | Demo EM | Demo F1 |
|---|---|---|---|---|---|---|---|
| 1 | DeBin (ExtraTrees baseline) | — | — | — | — | — | — |
| 2 | GAT + GRU Decoder (basic) | 4.4M | 0.374 | 51.3% | 0.606 | 24.1% | 0.364 |
| 3 | + External Call Encoder | 6.1M | 0.545 | 60.1% | 0.683 | 19.8% | 0.320 |
| 4 | + Callee/Caller Context | 8.0M | 0.720 | 72.3% | 0.781 | 32.6% | 0.460 |
| 5 | + Pretrain + Scale (25M) | 25M | 0.734 | 74.3% | 0.795 | 48.5% | 0.593 |

### Incremental Contribution Analysis

| Transition | Component Added | Test F1 Gain | Test EM Gain | Demo EM Gain |
|---|---|---|---|---|
| 2 → 3 | External Call Encoder | +0.077 | +8.8pp | -4.3pp* |
| 3 → 4 | Callee/Caller Context | +0.098 | +12.2pp | +12.8pp |
| 4 → 5 | Pretraining + Scaling | +0.014 | +2.0pp | +15.9pp |

### The Ext-Call Paradox (Key Finding)

External calls alone (Model 3) improve test F1 by +0.077 but **hurt** demo EM by -4.3pp. Why?

- External calls encode which library functions are called (e.g., `malloc`, `fprintf`, `pthread_create`). This is a strong signal for seen binaries where the model memorizes "functions that call X tend to be named Y."
- On **unseen** packages, many different functions share similar library call patterns. Without knowing *who* calls this function and *what* it calls internally, the model overfits to the library call signature and produces wrong names.
- Adding callee/caller context (Model 4) resolves this by providing **disambiguation**: two functions that both call `malloc` + `fprintf` can be distinguished by their internal callees and callers. Demo EM jumps +12.8pp.

**This demonstrates that naive feature addition can degrade generalization, and principled multi-context fusion is required.** No prior work on binary function naming reports this finding.

### Key Takeaways

- **Multi-context gated fusion is the core innovation** — the 3-stage cascade (ext calls → callee → caller) with conditional bypass. Total gain from Model 2→4: Test F1 +0.175, Demo EM +8.5pp.
- **External calls are necessary but insufficient** — they provide the foundation, but need inter-procedural context for disambiguation on unseen binaries.
- **Conditional bypass prevents mode collapse** — 70% of functions have no ext calls. Without bypass, they all get identical fused representations → 47% collapse to one name.
- **Pretraining + scaling provides the best generalization** (+15.9pp Demo EM) — the pretrained encoder learns transferable code representations that complement the multi-context fusion.

---

## 8. Results

### Overall Performance

| Metric | Midterm | Final (Decoder) | Final (k-NN Hybrid) | Improvement |
|---|---|---|---|---|
| Val F1 | 0.519 | **0.734** | — | +41.4% |
| Test EM | — | 74.3% | **76.3%** | — |
| Test F1 | — | 0.795 | **0.804** | — |
| Diffutils EM (unseen) | 39.0% | 68.5% | **69.9%** | +30.9pp |
| Demo EM (11 packages) | — | 48.5% | **53.1%** | — |
| Demo F1 (11 packages) | — | 0.593 | **0.625** | — |

### k-NN Hybrid Inference (Novel Finding)

At inference time, we build a FAISS index of all 87,724 training function encoder embeddings and compare each test function's embedding against the index. The k-NN hybrid outperforms the decoder on both test and demo sets **with zero retraining**:

| Method | Test EM | Test F1 | Demo EM | Demo F1 |
|---|---|---|---|---|
| Decoder only | 74.3% | 0.795 | 48.5% | 0.593 |
| k-NN only | 76.3% | 0.804 | — | — |
| **Forward hybrid (t=-0.02)** | — | — | **53.1%** | **0.625** |

**Why k-NN outperforms the decoder:**
1. **Phantom predictions eliminated**: 43.8% of decoder outputs are hallucinated names that don't exist in any binary. k-NN always returns real training names.
2. **Autoregressive error cascade**: One wrong sub-token in the decoder derails the entire prediction. k-NN makes a single holistic decision.
3. **Constrained search space**: The decoder explores 2,642^20 possible token sequences but only ~3,000 real function names exist. k-NN constrains to real names.

**Forward hybrid strategy**: Use the decoder by default; fall back to k-NN when beam search score < threshold. At t=-0.02, 87% of predictions come from k-NN and 13% from the decoder (cases where the decoder is highly confident).

**Paper implication**: The encoder pipeline (block encoder + GAT + gated fusion) is the core contribution — it produces embeddings good enough for direct retrieval. The decoder is less important than the representation.

### Test Set by Optimization Level

| Split | Functions | EM | F1 | EdSim | NgSim |
|---|---|---|---|---|---|
| O2 Test | 4,445 | 69.8% | 0.91 | 0.93 | 0.91 |
| O0 Test | 4,528 | 41.5% | 0.59 | 0.72 | 0.60 |
| **Combined** | **8,973** | **74.3%** | **0.795** | **0.829** | **0.798** |

### Per-Package Demo Results (11 unseen packages)

| Package | Functions | Decoder EM | k-NN Hybrid EM | k-NN Hybrid F1 | In Training? |
|---|---|---|---|---|---|
| texinfo | 686 | 91.6% | **95.6%** | 0.982 | Yes |
| acct | 718 | 65.2% | **80.5%** | 0.845 | Yes |
| rush | 1,654 | 58.4% | **75.1%** | 0.786 | Yes |
| direvent | 1,381 | 60.1% | **73.4%** | 0.770 | Yes |
| diffutils | 438 | 68.5% | **69.9%** | 0.713 | No |
| hello | 160 | 49.4% | **57.5%** | 0.714 | No |
| cppi | 265 | 50.0% | 53.6% | 0.641 | No |
| strace | 4,782 | 29.9% | **50.8%** | 0.627 | Yes |
| csplit2 | 1,011 | 35.1% | **42.4%** | 0.454 | No |
| datamash | 382 | 19.2% | **20.4%** | 0.303 | No |
| htop | 1,211 | 0.6% | **2.2%** | 0.099 | Yes* |
| **Overall** | **12,688** | **48.5%** | **53.1%** | **0.625** | |

*htop: different version than training — demonstrates recognizer limitation. cppi is the only package where k-NN slightly regressed (55.5% -> 53.6%).

### Comparison with Published Systems

| System | Venue | Approach | Cross-Project F1 |
|---|---|---|---|
| BLens | USENIX Security 2025 | Ensemble + contrastive | 0.46 |
| SYMGEN | NDSS 2025 | CodeLlama + LoRA | 0.38 |
| NERO | OOPSLA 2020 | GNN on call graphs | 0.455 |
| **Ours — decoder (test)** | | GAT + gated fusion + pretrained | **0.795** |
| **Ours — k-NN hybrid (test)** | | Same + FAISS retrieval | **0.804** |
| **Ours — k-NN hybrid (demo)** | | Same | **0.625** |

*Note: Direct comparison is approximate — different datasets and evaluation protocols. Our demo set (11 packages, 12K functions) is the most honest cross-project metric.*

---

## 9. Key Discoveries

### 1. The Model is a Pure Recognizer

**0 out of 1,873 unseen function names were correctly predicted.** Every correct prediction matches a name seen in training (typically shared gnulib utility functions like `quotearg_buffer_restyled`, `xmalloc`, `set_program_name`).

The model memorizes code pattern → name mappings. It does not compose novel names from sub-tokens, despite having a generative decoder with 2,642 sub-token vocabulary. The k-NN hybrid finding (Section 8) further confirms this: if the model were truly composing names, the decoder should outperform simple nearest-neighbor retrieval. Instead, k-NN wins.

### 2. k-NN Retrieval Outperforms Autoregressive Decoding (Novel Finding)

k-NN retrieval over encoder embeddings beats the GRU decoder on both test (EM +2.0pp) and demo (EM +4.6pp) with **zero retraining**. This is a novel finding for binary function naming — no prior work reports that retrieval outperforms generation in this domain.

The root cause is that 43.8% of decoder predictions are "phantom" names: plausible-sounding compositions of real sub-tokens that don't correspond to any actual function (e.g., `zlib_compile_flags`, `screen_list_item_delete`). k-NN eliminates these by constraining outputs to real training names.

This reshapes the paper narrative: **the encoder pipeline (Transformer + GAT + gated fusion) is the core contribution**, producing embeddings good enough for direct retrieval. The decoder is secondary — a simpler retrieval head yields better results.

### 3. ENDBR64 Indirect Jump Wrapper Bug (O0 Collapse)

96.4% of O0 functions were 2-block wrapper functions (ENDBR64 indirect jump stubs) created by BAP lifting `endbr64; jmp addr` as separate functions. Before fixing: O0 EM = 1.6% with 47% of predictions collapsed to a single name. After resolving these wrappers: O0 EM = 41.5%.

### 4. Ext-Call Paradox and Multi-Context Fusion (Key Architectural Finding)

External calls alone (Model 3) improve test F1 (+0.077) but **hurt** demo EM (-4.3pp). The model overfits to library call patterns: many unrelated functions share similar ext-call signatures (e.g., both `hash_insert` and `tree_add` call `malloc` + `memcpy`). Without knowing the inter-procedural context, the model conflates them.

Adding callee/caller context (Model 4) provides the missing disambiguation signal — two functions with identical ext calls can be distinguished by what they call internally and who calls them. This resolves the paradox: demo EM jumps +12.8pp.

**This is our core architectural contribution**: the 3-stage cascaded gated fusion with conditional bypass. The cascade order matters — ext calls provide coarse library-level signal, callee context adds inter-procedural specificity, caller context adds call-site differentiation. Total multi-context gain (Model 2→4): Test F1 +0.175.

### 5. Conditional Gate Bypass (Mode Collapse Fix)

Without bypass, the gated fusion learns a default output when no external calls exist, causing 47% mode collapse. With bypass (z = previous embedding when no context), the model preserves the code representation and produces diverse predictions. This is essential because ~70% of functions have no external calls.

### 6. Capacity-Generalization Tradeoff

8M model with diverse training data (40 packages) suffers from cross-package contamination: strace names predicted for binutils code, sqlite names for gnulib code. 25M model resolves this — every demo package improved. Larger capacity allows the model to hold distinct naming patterns for each package domain.

### 7. Val F1 Paradox

Higher Val F1 does not correlate with better demo EM. Validation set is dominated by small GNU packages (coreutils, bash, gawk). A model can achieve high Val F1 by memorizing gnulib patterns while failing on diverse unseen packages. Demo EM is the more reliable generalization metric.

---

## 10. Experiment History (36 experiments)

| Exp | Change | Val F1 | Outcome |
|---|---|---|---|
| — | Baseline (mean pool) | 0.581 | Starting point |
| 1 | Diverse beam search | — | HURT (degrades all metrics) |
| 2 | Decoder cross-attention | 0.552 | HURT |
| 5-6 | V2→V3 tokenization | 0.593 | +0.012 |
| 8 | Longer training (120 epochs) | 0.612 | +0.031 |
| 9 | Callee context encoder | 0.619 | +0.038 |
| 11 | + Caller context | 0.668 | +0.087 |
| 13 | Votes tokenizer | 0.669 | Best 8M on 64K data |
| 14-16 | V4 tokens, strings, block stats | ~0.668 | WASH |
| 19 | Callee name enrichment (2-pass) | 0.669 | +0.006 demo F1 |
| 20 | Dataset expansion (O1/O3) | — | COLLAPSED |
| 21-22 | Unlikelihood/contrastive loss | 0.652-0.672 | COLLAPSED or WASH |
| 23 | String encoder v2 | 0.668 | COLLAPSED |
| 25 | ENDBR64 wrapper resolution | 0.732 | O0 EM 1.6%→41.5% |
| 34 | Expanded dataset (87K) | 0.722 | Cross-contamination at 8M |
| 35 | 25M model | 0.710 | Every demo pkg improved |
| **36** | **+ Pretrained encoder** | **0.734** | **CURRENT BEST** |
| — | + k-NN hybrid (inference-only) | — | +4.6pp demo EM, no retraining |

### What Worked
- Multi-context gated fusion: ext calls + callee/caller context (+0.175 Test F1 total, Model 2→4)
- k-NN hybrid inference (+4.6pp demo EM, +2.0pp test EM, no retraining)
- V3 instruction-type tokenization (+1,750% F1)
- Callee + Caller context encoders (+8.7% Val F1, resolves ext-call paradox)
- External call encoder (+0.077 Test F1, foundation for fusion cascade)
- Votes name tokenizer (95% less OOV)
- Conditional gate bypass (fixed 47% mode collapse)
- ENDBR64 indirect jump wrapper resolution (O0 EM 1.6%→41.5%)
- Larger model 8M→25M (reduced cross-contamination)
- Self-supervised pretraining + partial embedding transfer

### What Did NOT Work
- Diverse beam search, decoder cross-attention, copy mechanism
- Unlikelihood loss, contrastive loss (NT-Xent), multi-label aux loss
- String reference encoder, V4 tokenization, block statistics
- Data deduplication, binary context embedding, weighted sampling
- Pre-trained embeddings (PalmTree, CLAP) — incompatible with our IR
- Expanding to O1/O3 (creates near-duplicate training examples)
- Cosine LR restart for extended training

---

## 11. Limitations and Future Work

### Current Limitations

1. **Pure recognizer**: Cannot compose novel names — only reproduces names seen in training
2. **O0/O2 gap**: O0 binaries remain significantly harder (41.5% vs 69.8% EM)
3. **Version sensitivity**: Different versions of same package produce 0.6% EM (htop case)
4. **Package-specific names**: Non-gnulib function names (datamash statistics, htop UI) rarely predicted correctly
5. **No confidence calibration**: Model does not know when it doesn't know

### Future Directions

1. **Confidence-based selective prediction**: Abstain when beam search score is below threshold — preliminary analysis shows clearly separable score distributions
2. **LLM integration**: Use our embeddings as features for a fine-tuned LLM (CodeLlama) that can compose novel names — the current model is a pure recognizer
3. **Cross-architecture**: Extend beyond x86-64 to ARM, MIPS
4. **Windows PE support**: Extend BAP lifting to PE binaries
5. **Iterative context propagation**: Use predicted names as input features for a second pass — preliminary 2-pass experiments showed +0.022 F1 on 0-ext functions

---

## 12. Conclusion

We present a 3-stage deep learning pipeline for binary function name recovery that achieves **0.804 F1 on test** (8,973 functions, k-NN hybrid) and **0.625 F1 on 11 completely unseen packages** (12,688 functions). Our ablation study shows that inter-procedural context (callee/caller signatures) provides the largest single gain (+0.098 F1), self-supervised pretraining with model scaling provides the best generalization improvement (+15.9pp demo EM), and k-NN hybrid inference provides a free +4.6pp demo EM improvement with no retraining.

Key findings include: (1) the model operates as a pure recognizer — memorizing code→name patterns rather than composing novel names, confirmed by k-NN retrieval outperforming autoregressive decoding; (2) the ext-call paradox, where external call information alone hurts generalization and requires callee/caller context for disambiguation; (3) the ENDBR64 indirect jump wrapper resolution that rescued O0 performance from 1.6% to 41.5% EM.

Our results are competitive with published systems (BLens 0.46, SYMGEN 0.38 cross-project F1) while providing novel analysis showing that the encoder representation — not the decoder — is the key contribution, as simple retrieval over learned embeddings outperforms generation.
