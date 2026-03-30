"""
Pre-training heads for the binary function name recovery model.

Two self-supervised objectives:
  1. MLM (Masked Language Model): predict masked instruction-type tokens
     within basic blocks, teaching the block encoder fine-grained token semantics.
  2. Contrastive (NT-Xent / InfoNCE): pull together graph-level embeddings
     of the same function compiled at different optimization levels, teaching
     the graph encoder optimization-invariant representations.

PretrainModel wraps block_encoder + graph_encoder + both heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_encoder import build_block_encoder
from .graph_encoder import build_graph_encoder


class MLMHead(nn.Module):
    """Masked Language Model head for instruction-type token prediction.

    Operates on per-token Transformer output (before mean pooling).
    Architecture: Linear -> GELU -> LayerNorm -> Linear -> vocab logits
    Loss: CrossEntropyLoss on masked positions only.
    """

    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: [*, embed_dim] per-token Transformer output
        Returns:
            logits: [*, vocab_size]
        """
        x = self.dense(token_embeddings)
        x = self.act(x)
        x = self.norm(x)
        return self.proj(x)


class ContrastiveHead(nn.Module):
    """Contrastive projection head for NT-Xent loss.

    Projects graph-level embedding to a lower-dimensional L2-normalized space.
    Architecture: Linear -> ReLU -> Linear -> L2 normalize
    """

    def __init__(self, input_dim: int, projection_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_embedding: [B, input_dim] from GAT pooling
        Returns:
            projected: [B, projection_dim] L2-normalized
        """
        z = self.proj(graph_embedding)
        return F.normalize(z, dim=-1)


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent (InfoNCE) loss for paired embeddings.

    z_i[k] and z_j[k] are a positive pair (same function, different opt level).
    All other combinations within the batch are negatives.

    Args:
        z_i: [N, D] L2-normalized projections of anchor samples
        z_j: [N, D] L2-normalized projections of positive samples
        temperature: scaling factor (lower = sharper distribution)
    Returns:
        scalar loss
    """
    N = z_i.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=z_i.device)

    # Concatenate: [z_i; z_j] -> [2N, D]
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

    # Full similarity matrix [2N, 2N]
    sim = torch.mm(z, z.t()) / temperature  # [2N, 2N]

    # Mask out self-similarity
    mask_self = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask_self, -1e9)

    # Positive pairs: (i, i+N) and (i+N, i)
    # For row i (0..N-1), positive is column i+N
    # For row i+N (N..2N-1), positive is column i
    pos_idx = torch.cat([
        torch.arange(N, 2 * N, device=z.device),  # positives for first N
        torch.arange(0, N, device=z.device),        # positives for second N
    ])  # [2N]

    # InfoNCE: -log(exp(sim_pos) / sum_j exp(sim_j))
    log_softmax = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -log_softmax[torch.arange(2 * N, device=z.device), pos_idx].mean()
    return loss


class PretrainModel(nn.Module):
    """Wraps block_encoder + graph_encoder + MLM head + contrastive head.

    During pre-training:
      - Block encoder runs with return_token_embeddings=True for MLM
      - Graph encoder produces function-level embeddings for contrastive learning
      - Both losses are computed and combined
    """

    def __init__(self, cfg: dict):
        super().__init__()

        # Build encoders from config (same architecture as fine-tuning)
        self.block_encoder = build_block_encoder(cfg['block_encoder'])

        gcfg = cfg['graph_encoder'].copy()
        gcfg['input_dim'] = cfg['block_encoder']['output_dim']
        self.graph_encoder = build_graph_encoder(gcfg)

        # MLM head: predicts original token from Transformer output
        # vocab_size includes the [MASK] token (added at token_vocab_size index)
        embed_dim = cfg['block_encoder']['token_embed_dim']
        mlm_vocab_size = cfg['block_encoder']['token_vocab_size']  # includes MASK
        self.mlm_head = MLMHead(embed_dim, mlm_vocab_size)

        # Contrastive head: projects graph embedding to normalized space
        graph_output_dim = cfg['graph_encoder']['output_dim']
        contrastive_cfg = cfg.get('contrastive', {})
        projection_dim = contrastive_cfg.get('projection_dim', 256)
        self.contrastive_head = ContrastiveHead(graph_output_dim, projection_dim)

        self.temperature = contrastive_cfg.get('temperature', 0.07)

    def forward(self, block_tokens, edge_index, batch_vec=None,
                mask_positions=None, original_tokens=None,
                pair_block_tokens=None, pair_edge_index=None,
                pair_batch_vec=None):
        """
        Args:
            block_tokens:    [B, N, T] (with some tokens replaced by MASK id)
            edge_index:      [2, E] batched edge index
            batch_vec:       [B*N] batch assignment for graph encoder
            mask_positions:  [B, N, T] bool — True at masked positions
            original_tokens: [B, N, T] original token ids before masking (MLM targets)
            pair_block_tokens: [B', N, T] positive pair samples (for contrastive)
            pair_edge_index:   [2, E'] batched edge index for pairs
            pair_batch_vec:    [B'*N] batch assignment for pairs

        Returns:
            dict with:
              'mlm_loss': scalar, masked language model loss
              'contrastive_loss': scalar, NT-Xent loss (0 if no pairs)
              'mlm_logits': [num_masked, vocab] for monitoring
        """
        B, N, T = block_tokens.shape
        device = block_tokens.device

        # --- MLM forward ---
        # Get per-token embeddings from block encoder (before pooling)
        token_embs = self.block_encoder.forward_token_embeddings(block_tokens)
        # token_embs: [B*N, T, embed_dim]

        mlm_loss = torch.tensor(0.0, device=device)
        mlm_logits = None
        if mask_positions is not None and original_tokens is not None:
            mask_flat = mask_positions.view(B * N, T)  # [B*N, T]
            targets_flat = original_tokens.view(B * N, T)  # [B*N, T]

            # Extract embeddings at masked positions
            masked_embs = token_embs[mask_flat]  # [num_masked, embed_dim]

            if masked_embs.shape[0] > 0:
                mlm_logits = self.mlm_head(masked_embs)  # [num_masked, vocab]
                mlm_targets = targets_flat[mask_flat]  # [num_masked]
                mlm_loss = F.cross_entropy(mlm_logits, mlm_targets)

        # --- Graph forward (for contrastive) ---
        # Get pooled block embeddings via normal forward
        block_embs = self.block_encoder(block_tokens)  # [B, N, output_dim]
        x = block_embs.view(B * N, -1)

        if batch_vec is None:
            batch_vec = torch.arange(B, device=device).repeat_interleave(N)

        f, _, _ = self.graph_encoder(x, edge_index, batch_vec)  # [B, graph_out_dim]

        # Project for contrastive learning
        z_anchor = self.contrastive_head(f)  # [B, projection_dim]

        contrastive_loss = torch.tensor(0.0, device=device)
        if pair_block_tokens is not None:
            B2, N2, T2 = pair_block_tokens.shape
            pair_block_embs = self.block_encoder(pair_block_tokens)
            x2 = pair_block_embs.view(B2 * N2, -1)

            if pair_batch_vec is None:
                pair_batch_vec = torch.arange(B2, device=device).repeat_interleave(N2)

            f2, _, _ = self.graph_encoder(x2, pair_edge_index, pair_batch_vec)
            z_pair = self.contrastive_head(f2)

            contrastive_loss = nt_xent_loss(z_anchor, z_pair, self.temperature)

        return {
            'mlm_loss': mlm_loss,
            'contrastive_loss': contrastive_loss,
            'mlm_logits': mlm_logits,
        }

    def get_encoder_state_dict(self):
        """Extract encoder weights for loading into FunctionNamer.

        Returns a dict with keys prefixed by 'block_encoder.' and 'graph_encoder.'
        matching FunctionNamer's state dict naming.
        """
        state = {}
        for name, param in self.block_encoder.named_parameters():
            state[f'block_encoder.{name}'] = param.data.clone()
        for name, buf in self.block_encoder.named_buffers():
            state[f'block_encoder.{name}'] = buf.data.clone()
        for name, param in self.graph_encoder.named_parameters():
            state[f'graph_encoder.{name}'] = param.data.clone()
        for name, buf in self.graph_encoder.named_buffers():
            state[f'graph_encoder.{name}'] = buf.data.clone()
        return state
