"""
Fusion modules for combining code and external call representations.

Option A: Node features (inject into block embeddings before GNN)
Option B: Gated fusion with conditional bypass [RECOMMENDED]
Option C: Cross-attention (decoder attends to ext calls per step)

FIX: Conditional bypass — when a function has NO external calls,
skip fusion entirely and use code embedding directly. This prevents
mode collapse where 70% of functions get identical embeddings via
no_context_emb → gate → same decoder input → same prediction.
"""
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Option B: Conditional gated fusion.

    When ext calls exist:
        z = g ⊙ f + (1 - g) ⊙ c
        where g = σ(Wg · [f; c] + bg)

    When NO ext calls:
        z = f   (bypass fusion, use code embedding directly)

    This eliminates mode collapse from no_context_emb being fed
    through the gate for 70% of functions.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        )
        # FIX: Initialize gate to start balanced (g ≈ 0.5)
        # Without this, gate collapses to ~0.28 (over-trusting ext calls)
        gate_linear = self.gate[0]
        nn.init.zeros_(gate_linear.bias)
        nn.init.xavier_uniform_(gate_linear.weight, gain=0.1)
        # Initialize gate to start balanced (g ≈ 0.5)
        gate_linear = self.gate[0]
        nn.init.zeros_(gate_linear.bias)
        nn.init.xavier_uniform_(gate_linear.weight, gain=0.1)

    def forward(self, f, c, has_ext_calls=None):
        """
        Args:
            f: [batch, dim] function embedding from graph encoder
            c: [batch, dim] calling context from external encoder
            has_ext_calls: [batch] bool — True if function has ext calls
                          If None, use old behavior (fuse all)
        Returns:
            z: [batch, dim] fused representation
            gate_values: [batch, dim] gate values for analysis
        """
        concat = torch.cat([f, c], dim=1)
        g = self.gate(concat)
        z_fused = g * f + (1 - g) * c

        if has_ext_calls is not None:
            # Conditional: use f directly when no ext calls
            mask = has_ext_calls.unsqueeze(1).float()  # [batch, 1]
            z = mask * z_fused + (1 - mask) * f
            # Gate values: set to 1.0 (trust code) for no-ext functions
            g = mask * g + (1 - mask) * torch.ones_like(g)
        else:
            z = z_fused

        return z, g


class NodeFeatureFusion(nn.Module):
    """
    Option A: Enrich block embeddings with external call info before GNN.
    """

    def __init__(self, block_dim: int, ext_embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(block_dim + ext_embed_dim, block_dim)

    def forward(self, block_embs, ext_embs_per_block):
        concat = torch.cat([block_embs, ext_embs_per_block], dim=1)
        return self.proj(concat)


class CrossAttentionFusion(nn.Module):
    """
    Option C: Decoder cross-attends to external call embeddings.
    Used inside the decoder at each step.
    """

    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=key_dim,
            batch_first=True,
        )

    def forward(self, query, keys, key_padding_mask=None):
        context, attn_weights = self.attn(
            query, keys, keys,
            key_padding_mask=key_padding_mask
        )
        return context, attn_weights
