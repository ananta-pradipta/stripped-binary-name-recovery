"""
Stage 1: Block Encoder — encodes each basic block into a vector.

Supports:
  - TransformerBlockEncoder: Transformer + attention pooling + block stats
  - MeanBlockEncoder: embedding + mean pooling (for GCN baseline)

Two improvements over the original:
  1. Token Attention Pooling: learned weights instead of mean averaging
  2. Block Statistics Fusion: numerical features (call density, degree, etc.)
     concatenated with the pooled token embedding before output projection

Architecture:
  tokens → Embedding → Transformer → AttentionPool → [pooled; stats] → Linear → b_i
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Must match NUM_BLOCK_FEATURES in build_dataset.py
NUM_BLOCK_FEATURES = 11


class TokenAttentionPooling(nn.Module):
    """Learned attention pooling over tokens within a block.

    Architecture matches graph-level AttentionPooling (Stage 2) for consistency:
    Linear → Tanh → Linear → softmax.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x:    [batch, seq_len, dim] — token embeddings after Transformer
            mask: [batch, seq_len] — True for real tokens, False for padding
        Returns:
            pooled: [batch, dim] — attention-weighted sum
        """
        scores = self.attention(x).squeeze(-1)  # [batch, seq_len]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)  # [batch, seq_len]
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # [batch, dim]
        return pooled


class TransformerBlockEncoder(nn.Module):
    """Encode block tokens with Transformer + attention pooling + block stats.

    Flow:
        Tokens: [B, N, T] → embed → Transformer → AttentionPool → [B*N, embed_dim]
        Stats:  [B, N, 11] → flatten → [B*N, 11]
        Combined: [B*N, embed_dim + 11] → Linear → [B*N, output_dim]
        Reshape: → [B, N, output_dim]
    """

    def __init__(self, vocab_size, embed_dim=128, num_layers=2, num_heads=4,
                 dropout=0.1, output_dim=256, pooling='attention',
                 use_block_features=True, num_block_features=NUM_BLOCK_FEATURES,
                 token_dropout=0.0, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.pooling_type = pooling
        self.use_block_features = use_block_features
        self.num_block_features = num_block_features
        self.token_dropout = token_dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Token pooling
        if pooling == 'attention':
            self.token_pool = TokenAttentionPooling(embed_dim)
        else:
            self.token_pool = None  # mean pooling fallback

        # Block statistics fusion
        if use_block_features:
            # Project stats to a small embedding, then combine with token embedding
            proj_input_dim = embed_dim + num_block_features
            self.output_proj = nn.Linear(proj_input_dim, output_dim)
        else:
            self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, block_tokens, block_features=None):
        """
        Args:
            block_tokens:   [B, N, T] token IDs per block
            block_features: [B, N, NUM_BLOCK_FEATURES] statistics per block (optional)
        Returns:
            block_embs: [B, N, output_dim]
        """
        B, N, T = block_tokens.shape
        tokens_flat = block_tokens.view(B * N, T)

        # Token dropout: randomly zero out tokens during training
        if self.training and self.token_dropout > 0:
            real_mask = (tokens_flat != 0)
            drop_mask = torch.rand_like(tokens_flat, dtype=torch.float) < self.token_dropout
            tokens_flat = tokens_flat.clone()
            tokens_flat[real_mask & drop_mask] = 0

        # Embed tokens
        emb = self.embedding(tokens_flat)  # [B*N, T, embed_dim]
        T_actual = min(T, self.pos_encoding.shape[1])
        emb[:, :T_actual] = emb[:, :T_actual] + self.pos_encoding[:, :T_actual]

        # Transformer: self-attention within each block
        padding_mask = (tokens_flat == 0)
        encoded = self.transformer(emb, src_key_padding_mask=padding_mask)

        # Pool tokens into single block vector
        real_token_mask = ~padding_mask  # [B*N, T]

        if self.token_pool is not None:
            pooled = self.token_pool(encoded, mask=real_token_mask)  # [B*N, embed_dim]
        else:
            mask_float = real_token_mask.unsqueeze(-1).float()
            pooled = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

        # Fuse with block statistics
        if self.use_block_features and block_features is not None:
            stats_flat = block_features.view(B * N, -1)  # [B*N, num_features]
            pooled = torch.cat([pooled, stats_flat], dim=-1)  # [B*N, embed_dim + num_features]

        pooled = pooled.view(B, N, -1)  # [B, N, embed_dim (+ features)]
        return self.output_proj(pooled)  # [B, N, output_dim]

    def forward_token_embeddings(self, block_tokens):
        """Return per-token Transformer output BEFORE pooling (for MLM pre-training).

        Args:
            block_tokens: [B, N, T] token IDs per block
        Returns:
            encoded: [B*N, T, embed_dim] per-token Transformer output
        """
        B, N, T = block_tokens.shape
        tokens_flat = block_tokens.view(B * N, T)

        # No token dropout during MLM — we mask explicitly
        emb = self.embedding(tokens_flat)  # [B*N, T, embed_dim]
        T_actual = min(T, self.pos_encoding.shape[1])
        emb[:, :T_actual] = emb[:, :T_actual] + self.pos_encoding[:, :T_actual]

        padding_mask = (tokens_flat == 0)
        encoded = self.transformer(emb, src_key_padding_mask=padding_mask)
        return encoded  # [B*N, T, embed_dim]


class MeanBlockEncoder(nn.Module):
    """Simple mean pooling encoder (for GCN baseline)."""

    def __init__(self, vocab_size, embed_dim=128, output_dim=256,
                 dropout=0.1, use_block_features=True,
                 num_block_features=NUM_BLOCK_FEATURES, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.use_block_features = use_block_features

        if use_block_features:
            self.output_proj = nn.Linear(embed_dim + num_block_features, output_dim)
        else:
            self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, block_tokens, block_features=None):
        B, N, T = block_tokens.shape
        emb = self.embedding(block_tokens)
        mask = (block_tokens != 0).unsqueeze(-1).float()
        pooled = (emb * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # [B, N, embed_dim]

        if self.use_block_features and block_features is not None:
            pooled = torch.cat([pooled, block_features], dim=-1)  # [B, N, embed_dim + features]

        return self.output_proj(pooled)


def build_block_encoder(cfg):
    """Factory function for block encoders."""
    enc_type = cfg.get('type', 'transformer')
    use_features = cfg.get('use_block_features', True)

    if enc_type == 'transformer':
        return TransformerBlockEncoder(
            vocab_size=cfg['token_vocab_size'],
            embed_dim=cfg['token_embed_dim'],
            num_layers=cfg['num_layers'],
            num_heads=cfg['num_heads'],
            dropout=cfg['dropout'],
            output_dim=cfg['output_dim'],
            pooling=cfg.get('token_pooling', 'attention'),
            use_block_features=use_features,
            token_dropout=cfg.get('token_dropout', 0.0),
        )
    elif enc_type == 'mean':
        return MeanBlockEncoder(
            vocab_size=cfg['token_vocab_size'],
            embed_dim=cfg['token_embed_dim'],
            output_dim=cfg['output_dim'],
            dropout=cfg['dropout'],
            use_block_features=use_features,
        )
    else:
        raise ValueError(f"Unknown block encoder type: {enc_type}")
