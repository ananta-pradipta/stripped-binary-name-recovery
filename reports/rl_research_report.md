# Reinforcement Learning for Binary Function Name Recovery — Deep Research Report

Generated 2026-04-06.

## Executive Summary

RL has been applied extensively to sequence generation (captioning, summarization, code generation) but **never directly to binary function name prediction**. The core opportunity: our model trains with cross-entropy (token-level MLE) but evaluates with sub-token F1 (non-differentiable). RL bridges this gap by directly optimizing the evaluation metric as reward.

**Critical caveat:** Our model is a pure recognizer (0% on unseen names). RL cannot teach compositional generalization — it can only help the model better select among names it already knows. The biggest gains come from RL-based routing (decoder vs k-NN), not from RL-based generation.

## 7 Approaches Ranked by Feasibility × Impact

### #1 — SCST with Sub-Token F1 Reward ⭐ RECOMMENDED

**What:** Self-Critical Sequence Training. After cross-entropy pretraining, fine-tune decoder with REINFORCE where reward = sub-token F1. Baseline = greedy decode reward.

**Gradient:** `∇J = -E[(R(y_sample) - R(y_greedy)) · ∇log p(y_sample|x)]`

**Implementation:**
1. Add `sample()` method to GRU decoder (sample from softmax instead of argmax)
2. Compute log-probabilities along sampled path
3. Use `compute_subtoken_f1()` from `src/evaluation/metrics.py` as reward
4. Mixed loss: `L = λ_RL · L_RL + (1-λ_RL) · L_XE` to prevent catastrophic forgetting

**Papers:** Rennie et al. CVPR 2017 (SCST), Ranzato et al. ICLR 2016 (MIXER)

| Feasibility | Impact | Risk | Time |
|---|---|---|---|
| HIGH | +1-3% F1 on test | LOW | 3-5 days |

### #2 — RL-Learned Decoder vs k-NN Router ⭐ RECOMMENDED

**What:** Learn a gating network that decides per-function: use decoder OR k-NN. Train with REINFORCE where reward = F1 of chosen prediction.

**Architecture:**
```python
class PredictionRouter(nn.Module):
    def __init__(self, hidden_dim):
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 2, 128), nn.ReLU(), nn.Linear(128, 2))
    def forward(self, z, decoder_conf, knn_dist):
        return F.softmax(self.gate(torch.cat([z, decoder_conf, knn_dist], -1)), -1)
```

**Oracle upper bound:** If k-NN gets 70% right and decoder 74% right on DIFFERENT functions, oracle router reaches 80%+.

**Papers:** SELF-RAG (Asai et al., ICLR 2024), SmartRAG (ICLR 2025), RouteRAG (2025)

| Feasibility | Impact | Risk | Time |
|---|---|---|---|
| MEDIUM | +2-4% EM | LOW | 5-7 days |

### #3 — DPO on Decoder Output Pairs

**What:** Direct Preference Optimization. Generate diverse beam candidates, score by F1, form (better, worse) pairs, train with DPO loss. No sampling during training.

**Loss:** `L = -log σ(β · (log p(y_w) - log p(y_l)))`

**Papers:** ACL 2024 "Improving Code Generation with RL" (recommends DPO over PPO for small models)

| Feasibility | Impact | Risk | Time |
|---|---|---|---|
| HIGH | LOW-MOD | LOW | 3-5 days |

### #4 — GRPO Fine-Tuning (DeepSeek-R1 style)

**What:** Group Relative Policy Optimization. Sample K=8 names per function, score by F1, normalize rewards within group, update with clipped PPO.

**Key precedent: D-LiFT** (2025) uses GRPO directly for binary DECOMPILATION — closest paper to our setting. Improved 68.2% of functions.

**Papers:** D-LiFT (arxiv 2506.10125), DeepSeek-R1 GRPO

| Feasibility | Impact | Risk | Time |
|---|---|---|---|
| MEDIUM | +1-3% F1 | LOW-MED | 5-7 days |

### #5 — Token-Level Continuous Reward

**What:** Per-token reward: +1 if generated sub-token is in ground truth set, -1 otherwise. Denser signal than sequence-level.

**Papers:** TLCR (arxiv 2407.16574), Fine-Grained RLHF, Process-Supervised RL for Code (EMNLP 2025)

| Feasibility | Impact | Risk | Time |
|---|---|---|---|
| MEDIUM | LOW-MOD | LOW-MED | 5-7 days |

### #6 — Reward-Shaped Exploration for Novel Compositions

**What:** Intrinsic reward for generating names with common sub-tokens in novel combinations. Encourages compositional generalization.

**Reward:** `R = α·token_commonality + β·name_novelty + γ·F1`

| Feasibility | Impact | Risk | Time |
|---|---|---|---|
| LOW-MED | LOW-MOD | MED-HIGH | 7-10 days |

### #7 — Contrastive RL for Embedding Optimization

**What:** RL to improve encoder embeddings for k-NN. Reward = F1 of nearest neighbor.

| Feasibility | Impact | Risk | Time |
|---|---|---|---|
| LOW | LOW | MEDIUM | 10+ days |

## Critical Assessment: Will RL Actually Help?

**Honestly: marginally on test/val, NOT on cross-project.**

1. **Bottleneck is representation, not generation.** 0% EM on unseen names = encoder can't distinguish unseen function types. RL fine-tunes the decoder, but the encoder is the bottleneck.

2. **RL optimizes training-distribution metric, not generalization.** SCST with F1 reward teaches better outputs on SEEN functions. Won't help unseen packages.

3. **Our names are short (~3 sub-tokens).** RL shines for long sequences where MLE accumulates errors (exposure bias). For 3-token names, the MLE-RL gap is small.

4. **Strongest gain is Approach #2 (Router)** — purely additive, combines two complementary strategies without modifying either.

## Recommended Plan for Paper Deadline

1. **Implement SCST (#1)** — 3-5 days. Even +1-2% F1 is publishable as "RL fine-tuning with metric reward."
2. **If time permits, implement Router (#2)** — 5-7 days. Stronger result, novel contribution.
3. **Frame honestly in paper:** "RL provides modest gains on the generation stage but does not address the fundamental recognition bottleneck."

## Key Papers to Cite

1. SCST (Rennie et al., CVPR 2017) — foundational REINFORCE for captioning
2. MIXER (Ranzato et al., ICLR 2016) — sequence-level training with RNNs
3. D-LiFT (2025) — GRPO for binary decompilation (closest related work)
4. SELF-RAG (Asai et al., ICLR 2024) — retrieve vs generate routing
5. Enhancing Code LLMs with RL Survey (arxiv 2412.20367)
6. DPO for Code (ACL 2024) — recommends DPO for resource-constrained settings
