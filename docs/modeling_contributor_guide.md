# Model Architecture & Training Contributor Guide

**For:** Ananta Dian Pradipta (model implementation, training, block encoder, GNN/attention variants)
**Project:** CS785 Binary Function Name Recovery
**Last updated:** 2026-04-02

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Architecture Overview](#2-architecture-overview)
3. [Model Components in Detail](#3-model-components-in-detail)
4. [Forward Pass Flow](#4-forward-pass-flow)
5. [Training Pipeline](#5-training-pipeline)
6. [Loss Functions](#6-loss-functions)
7. [Self-Supervised Pretraining](#7-self-supervised-pretraining)
8. [Configurations](#8-configurations)
9. [Running Training](#9-running-training)
10. [Wulver HPC Workflow](#10-wulver-hpc-workflow)
11. [Checkpoint Format](#11-checkpoint-format)
12. [Common Tasks](#12-common-tasks)
13. [Debugging & Troubleshooting](#13-debugging--troubleshooting)
14. [Critical Design Decisions](#14-critical-design-decisions)

---

## 1. Environment Setup

### Local (development + small experiments)

```bash
source ~/cs785-project/activate.sh
# GPU: RTX 4060 Laptop (8GB VRAM) — batch_size=128 for 8M model, batch_size=32 for 25M
```

### Wulver HPC (full training)

```bash
# Step 1: User opens SSH connection (Duo 2FA required once)
ssh wulver

# Step 2: Sync code changes to Wulver
bash scripts/wulver_sync.sh

# Step 3: Submit training job
ssh wulver "cd /course/2026/spring/cs/785/hz79/adp232/cs785 && sbatch scripts/wulver_train.sbatch"

# Step 4: Monitor
ssh wulver "tail -f /course/2026/spring/cs/785/hz79/adp232/cs785/cs785-train.*.out"
```

---

## 2. Architecture Overview

```
Stage 1: Block Encoder — per-block instruction encoding
  Input: [B, N, T] instruction-type token IDs (N blocks, T tokens each)
  → Embedding(vocab, 128-256) + Positional Encoding
  → Transformer(2-4 layers, 4-8 heads)
  → Mean Pool over tokens
  → Linear → b_i ∈ R^{256-512} per block

Stage 2: Graph Encoder — CFG-level message passing
  Input: [B*N, D] block embeddings + edge_index (CFG edges)
  → GAT(2-3 layers, 4-8 heads)
  → Attention Pooling over blocks
  → f ∈ R^{512-1024} (function embedding)

Stage 3: Context Encoders + Gated Fusion
  → External Call Encoder: Embedding → Bi-GRU → c_ext
  → Callee Context Encoder: Embedding → Bi-GRU → c_callee
  → Caller Context Encoder: Embedding → Bi-GRU → c_caller
  → 3-stage cascaded gated fusion with conditional bypass

Stage 4: GRU Decoder — autoregressive name generation
  Input: fused embedding z
  → GRU(1 layer, 512-1024 hidden)
  → Beam search (k=5) with length normalization + repetition penalty
  → Votes sub-tokens → function name
```

---

## 3. Model Components in Detail

### Block Encoder (`src/models/block_encoder.py`)

Two variants available:

**TransformerBlockEncoder** (used in all experiments):
```python
Embedding(vocab_size, embed_dim)     # 3000×128 or 3000×256
+ Positional Encoding                # [1, max_tokens, embed_dim]
→ TransformerEncoder(d_model, nhead, num_layers, dim_feedforward)
→ Mean Pool over token dimension     # [B, N, T, D] → [B, N, D]
→ Linear(embed_dim, output_dim)      # Project to block embedding size
```

Config flags:
- `use_block_features: false` — 11 numeric features (call density, degree, etc.). Experiments show minimal gain.
- `token_dropout: 0.0` — Random token masking during training. Did not help.

### Graph Encoder (`src/models/graph_encoder.py`)

**GATGraphEncoder** (used in all experiments):
```python
GATConv layers × num_layers
  - in_channels: output_dim from block encoder
  - hidden: hidden_dim (256-512)
  - heads: num_heads (4-8)
  - Each layer: GATConv → ELU → Dropout

AttentionPooling:
  - Linear(hidden_dim, 1) → softmax over nodes → weighted sum
  - Produces single function-level embedding

Output: Linear(hidden_dim, output_dim)  # → 512 or 1024
```

Uses PyTorch Geometric `GATConv`. Edge index represents CFG edges (basic block transitions).

### External Call Encoder (`src/models/external_encoder.py`)

```python
Embedding(ext_vocab_size, embed_dim)   # 656×256
→ Bi-GRU(embed_dim, hidden_dim)        # hidden_dim=256
→ Take final hidden states (fwd + bwd) # 512-dim
→ Linear(hidden_dim*2, output_dim)     # → 512 or 1024
```

Also contains:
- **CalleeContextEncoder**: Encodes token signatures of up to 5 internal callees
- **CallerContextEncoder**: Same architecture, encodes up to 5 callers
- **StringReferenceEncoder**: Encodes tokenized .rodata strings (disabled — WASH/HURT in experiments)

### Gated Fusion (`src/models/gated_fusion.py`)

The most critical component for preventing mode collapse:

```python
# 3-stage cascaded fusion
def forward(self, f, ext_emb, has_ext, callee_ctx, has_callees, caller_ctx, has_callers):

    # Stage 1: External calls
    if has_ext:
        gate = sigmoid(Linear([f; ext_emb]))  # [B, D]
        z = gate * f + (1 - gate) * ext_emb
    else:
        z = f  # BYPASS — critical for preventing mode collapse

    # Stage 2: Callee context
    if has_callees:
        gate = sigmoid(Linear([z; callee_ctx]))
        z = gate * z + (1 - gate) * callee_ctx
    # else: z unchanged

    # Stage 3: Caller context
    if has_callers:
        gate = sigmoid(Linear([z; caller_ctx]))
        z = gate * z + (1 - gate) * caller_ctx

    return z
```

**Why bypass matters:** ~70% of functions have 0 external calls. Without bypass, all these functions get the same `no_context_emb` fused through the gate, producing identical decoder inputs → 47% mode collapse.

### Decoder (`src/models/decoder.py`)

```python
GRUDecoder:
  embedding = Embedding(vocab_size, embed_dim)   # 2642×256 or 2642×512
  gru = GRU(embed_dim, hidden_dim, num_layers=1) # hidden=512 or 1024
  output_proj = Linear(hidden_dim, vocab_size)    # → logits over sub-tokens

  # Training: teacher forcing with scheduled sampling
  # Inference: beam search (width=5)
```

Beam search features:
- **Length normalization**: `score / length^0.7` — prevents bias toward short names
- **Repetition penalty**: 1.2x reduction for already-generated tokens
- **Max length**: 20 sub-tokens

### Full Model Assembly (`src/models/function_namer.py`)

`FunctionNamer` wraps all components and handles:
- Conditional encoder enablement (based on config flags)
- Batch construction with variable-size graphs
- Teacher forcing ratio during training
- Gate value tracking for analysis

---

## 4. Forward Pass Flow

```python
def forward(self, block_tokens, edge_index, ext_call_ids, decoder_input, ...):

    # Stage 1: Block encoding
    block_embs = self.block_encoder(block_tokens, block_features)
    # [B, N, 256] → flattened to [B*N, 256]

    # Stage 2: Graph encoding
    f, block_embs_out, attn_weights = self.graph_encoder(
        block_embs_flat, edge_index, batch_vec
    )
    # f: [B, 512]

    # Stage 3: Context fusion (cascaded)
    z = f
    if self.ext_encoder_enabled:
        c_ext = self.ext_encoder(ext_call_ids)
        has_ext = (ext_call_ids > 0).any(dim=1)
        z, gate = self.fusion(z, c_ext, has_ext)

    if self.callee_encoder_enabled:
        c_callee, has_callees = self.callee_encoder(callee_tokens)
        g = sigmoid(self.callee_gate([z; c_callee]))
        z = has_callees * (g*z + (1-g)*c_callee) + ~has_callees * z

    # Similar for caller...

    # Stage 4: Decode
    logits = self.decoder(z, decoder_input, teacher_forcing_ratio)
    # [B, T, vocab_size]

    return logits, gate_values, aux_logits, z, ml_logits
```

---

## 5. Training Pipeline

### Main training loop (`src/training/train.py`)

```python
for epoch in range(num_epochs):
    # Scheduled sampling: TF ratio decays from 1.0 → 0.3
    tf_ratio = max(0.3, 1.0 - (epoch / num_epochs) * 0.7)

    for batch in train_loader:
        optimizer.zero_grad()

        logits, gates, aux, z, ml = model(
            block_tokens, edge_index, ext_ids, decoder_input,
            teacher_forcing_ratio=tf_ratio, ...
        )

        loss = ce_loss(logits, targets)
        # + optional: ul_loss, contrastive_loss, ml_loss, aux_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Validation (greedy decode, no teacher forcing)
    val_f1 = evaluate(model, val_loader)

    # Save best checkpoint
    if val_f1 > best_val_f1:
        save_checkpoint(model, optimizer, epoch, val_f1, config, vocabs)

    # Early stopping
    if no_improvement_for > patience:
        break

    scheduler.step()  # Cosine LR with warmup
```

### Learning Rate Schedule

```
Warmup: linear ramp from 0 → lr over min(5, epochs//10) epochs
Cosine: lr * 0.5 * (1 + cos(π * progress)) for remaining epochs
```

---

## 6. Loss Functions

### Primary: Cross-Entropy with Label Smoothing

```python
nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
# ignore_index=0: PAD token
# label_smoothing=0.1: prevent overconfident predictions
```

### Optional Losses (all tested, most did NOT help)

| Loss | Flag | Weight | Result |
|------|------|--------|--------|
| Unlikelihood | `--ul-weight` | 0.1-2.0 | WASH/COLLAPSED |
| Contrastive (NT-Xent) | `--contrastive-weight` | 0.1-1.0 | COLLAPSED |
| Multi-label BCE | `--ml-weight` | 0.5-2.0 | WASH |
| Auxiliary (num_blocks) | config `aux_loss_weight` | 0.1 | WASH |

---

## 7. Self-Supervised Pretraining

### Two-stage approach

**Stage 1: Pretrain encoder** (`src/training/pretrain.py`)
```
MLM objective: Mask 15% of instruction tokens, predict originals
  → Uses per-token Transformer output
  → Head: Linear → GELU → LayerNorm → Linear

Contrastive objective: Same function at O0 vs O2 should be close
  → Uses graph-level embedding
  → Head: Linear → ReLU → Linear → L2 normalize
  → NT-Xent loss, temperature=0.07
```

**Stage 2: Fine-tune full model** (`src/training/train.py`)
```python
# Load pretrained encoder weights
--pretrained-encoder checkpoints/pretrained_encoder.pt

# Partial embedding transfer: match tokens by name
# 84.4% of tokens transfer (1,923/2,279)
# Remaining 356 randomly initialized
```

### Impact: Val F1 +0.024, Demo EM +15.9pp (4→5 in ablation)

---

## 8. Configurations

### 8M Model (`configs/optimized.yaml`)

```yaml
block_encoder:
  type: transformer
  token_vocab_size: 3000
  token_embed_dim: 128
  num_layers: 2
  num_heads: 4
  output_dim: 256
  dropout: 0.15

graph_encoder:
  type: gat
  hidden_dim: 256
  num_layers: 2
  num_heads: 4
  output_dim: 512

external_encoder:
  enabled: true
  vocab_size: 1000
  embed_dim: 256
  hidden_dim: 256
  output_dim: 512

callee_encoder:
  enabled: true
  embed_dim: 64
  hidden_dim: 128

caller_encoder:
  enabled: true
  embed_dim: 64
  hidden_dim: 128

decoder:
  bpe_vocab_size: 3000
  embed_dim: 256
  hidden_dim: 512
  beam_width: 5
  max_length: 20

training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  patience: 10
  label_smoothing: 0.1
```

### 25M Model (`configs/optimized_large.yaml`)

Key differences from 8M:
- Block encoder: 256 embed, 4 layers, 8 heads, output_dim=512
- Graph encoder: 512 hidden, 3 layers, 8 heads, output_dim=1024
- Decoder: 512 embed, 1024 hidden
- Training: batch_size=32, lr=0.0003

### Ablation Configs

| Config | Ext Encoder | Callee/Caller | Purpose |
|--------|------------|---------------|---------|
| `ablation_model2.yaml` | disabled | disabled | Code-only baseline |
| `ablation_model3.yaml` | enabled | disabled | + External calls |
| `ablation_model4.yaml` | enabled | enabled | + Inter-procedural context |

---

## 9. Running Training

### Local (small experiments)

```bash
source ~/cs785-project/activate.sh

# 8M model
python3 -m src.training.train --config configs/optimized.yaml --seed 42

# 25M model (needs more VRAM)
python3 -m src.training.train --config configs/optimized_large.yaml --seed 42 --amp

# With pretrained encoder
python3 -m src.training.train --config configs/optimized_large.yaml --seed 42 \
  --pretrained-encoder checkpoints/pretrained_encoder.pt --amp

# Resume from checkpoint
python3 -m src.training.train --config configs/optimized.yaml --seed 42 \
  --resume checkpoints/best_model.pt
```

### Key CLI flags

| Flag | Purpose |
|------|---------|
| `--seed 42` | Reproducible training (always use) |
| `--amp` | Mixed precision (2x faster, less VRAM) |
| `--resume PATH` | Resume from checkpoint |
| `--pretrained-encoder PATH` | Initialize from pretrained encoder |
| `--ul-weight 0.1` | Unlikelihood loss (not recommended) |
| `--contrastive-weight 0.5` | Contrastive loss (not recommended) |
| `--ml-weight 1.0` | Multi-label loss (not recommended) |
| `--num-workers 4` | Data loading parallelism |

---

## 10. Wulver HPC Workflow

### Project location on Wulver

```
/course/2026/spring/cs/785/hz79/adp232/cs785/
├── src/              # Code (synced from local)
├── configs/          # Configs (synced from local)
├── scripts/          # Scripts (synced from local)
├── data/             # Pre-processed data (uploaded once)
├── checkpoints/      # Model checkpoints (generated on Wulver)
├── results/          # Evaluation results
└── cs785-train.*.out # SLURM job output logs
```

### Step-by-step

```bash
# 1. Make code changes locally

# 2. Sync code to Wulver (seconds, rsync)
bash scripts/wulver_sync.sh

# 3. Submit job
ssh wulver "cd /course/2026/spring/cs/785/hz79/adp232/cs785 && sbatch scripts/wulver_train.sbatch"

# 4. Check job status
ssh wulver "squeue -u adp232"

# 5. Monitor training output
ssh wulver "tail -50 /course/2026/spring/cs/785/hz79/adp232/cs785/cs785-train.*.out"

# 6. Copy checkpoint back to local
scp wulver:/course/2026/spring/cs/785/hz79/adp232/cs785/checkpoints/best_model.pt \
    checkpoints/best_model_wulver.pt
```

### SLURM job settings (`wulver_train.sbatch`)

```bash
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --gres=gpu:a100_10g:1    # Gets full A100 40GB
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
```

### Training times on Wulver A100

| Model | Params | Batch | Epochs | Time |
|-------|--------|-------|--------|------|
| 8M (ablation) | 4.4-8.0M | 128 | 50 | ~1.5 hours |
| 25M (full) | 25M | 32 | 50 | ~3.75 hours |
| Pretraining | 12M | 128 | 10 | ~45 minutes |

---

## 11. Checkpoint Format

```python
# What's saved in best_model.pt
{
    'epoch': int,
    'model_state_dict': OrderedDict,   # All model weights
    'optimizer_state_dict': OrderedDict, # AdamW state
    'val_f1': float,
    'config': dict,                     # Full YAML config
    'token_vocab': dict,                # {token_name: int_id}
    'ext_vocab': dict,                  # {ext_func_name: int_id}
}

# What's saved in pretrained_encoder.pt
{
    'encoder_state_dict': OrderedDict,  # Block + graph encoder only
    'token_vocab': dict,
    'mask_token_id': int,
    'epoch': int,
    'best_loss': float,
    'config': dict,
}
```

---

## 12. Common Tasks

### Add a new encoder component

1. Create `src/models/new_encoder.py` with forward method returning `[B, output_dim]`
2. Add config section in YAML (`new_encoder: {enabled: true, ...}`)
3. Register in `src/models/function_namer.py`:
   - Instantiate in `__init__`
   - Add fusion gate: `self.new_gate = nn.Linear(2*dim, dim)`
   - Add to `forward()` with conditional bypass
4. Update checkpoint saving/loading if new vocab needed

### Modify the decoder

Key files:
- `src/models/decoder.py` — GRU cell, beam search, teacher forcing
- Beam search: `generate()` method (lines ~85-175)
- Teacher forcing: handled in `forward()` with `teacher_forcing_ratio`

### Run an ablation experiment

```bash
# 1. Create config (copy + modify)
cp configs/optimized.yaml configs/ablation_new.yaml
# Edit: disable/enable components

# 2. Sync + submit
bash scripts/wulver_sync.sh
ssh wulver "cd /course/.../cs785 && sbatch scripts/wulver_train.sbatch"

# 3. Evaluate
ssh wulver "cd /course/.../cs785 && python3 scripts/eval_test.py checkpoints/new/best_model.pt"
```

### Inspect gate values

```python
# During evaluation, gate_values are returned by model.forward()
# Average gate per ext-call bucket:
#   0 ext calls: gate = 1.0 (bypass, uses code only)
#   1-3 ext calls: gate ≈ 0.4-0.6 (balanced)
#   4+ ext calls: gate ≈ 0.3-0.4 (trusts ext calls more)
```

---

## 13. Debugging & Troubleshooting

### OOM on GPU

- Reduce `batch_size` (128 → 64 → 32)
- Enable AMP (`--amp` flag)
- Reduce `max_blocks_per_function` (30 → 20)

### Mode collapse (many predictions = same name)

Check:
1. Is conditional gate bypass working? (`gated_fusion.py` line 64)
2. Are `has_ext_calls` masks correct? (should be boolean per sample)
3. Is teacher forcing decaying? (should go 1.0 → 0.3)

### Val F1 plateaus early

- Check learning rate schedule (cosine should decay smoothly)
- Try increasing warmup steps
- Check if gradient clipping is too aggressive (default 1.0 is fine)

### Pretrained encoder loading fails

- Check vocab sizes match between pretrain and finetune configs
- Partial embedding transfer handles size mismatches automatically
- Look for "Transferred X/Y tokens" message in training output

### Training output format

```
Epoch  3/50 | TF=0.99 | LR=0.000900 | Train Loss: 3.21 | Val Loss: 3.12 | Val F1: 0.623
  ↓       ↓           ↓              ↓                    ↓                  ↓
epoch   teacher_force learning_rate  train_CE_loss      val_CE_loss        key metric
```

---

## 14. Critical Design Decisions

### Why Conditional Gate Bypass (not learned default)

Without bypass, functions with 0 ext calls all get `no_context_emb` (a learned vector) as context input. The gate then produces the same fused output for all of them → identical decoder input → mode collapse (47% predictions = one name). With bypass, `z = f` directly, preserving unique code representations.

### Why GRU Decoder (not Transformer)

GRU is simpler and works well for short sequences (function names are 3-5 sub-tokens). Transformer decoder was not tried, but the bottleneck is encoder representation quality, not decoder capacity.

### Why Mean Pool (not Attention Pool) for Token Aggregation

Experiments showed token attention pooling did not improve over mean pool. The Transformer self-attention already captures token importance; a second attention layer on top is redundant.

### Why Votes Tokenizer (not BPE)

BPE fragments rare sub-tokens into characters, making sub-token F1 unreliable. Votes preserves whole semantic units (`malloc` stays as one token, not `mal` + `loc`). 95% less OOV, better F1 alignment.

---

## Files You'll Work With Most

| File | Purpose |
|------|---------|
| `src/models/function_namer.py` | Full model assembly |
| `src/models/block_encoder.py` | Transformer block encoder |
| `src/models/graph_encoder.py` | GAT graph encoder |
| `src/models/gated_fusion.py` | Conditional gated fusion |
| `src/models/decoder.py` | GRU decoder + beam search |
| `src/models/external_encoder.py` | All context encoders |
| `src/training/train.py` | Training loop |
| `configs/optimized_large.yaml` | Current best config (25M) |
| `configs/optimized.yaml` | 8M config (for ablation) |
| `scripts/wulver_train.sbatch` | HPC job submission |
| `scripts/wulver_sync.sh` | Code sync to Wulver |
