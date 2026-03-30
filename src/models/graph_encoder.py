"""
Stage 2: Graph Encoder — GAT / GCN over the CFG with attention pooling.

Input:  block embeddings [batch, num_blocks, dim] + edge_index
Output: function embedding f [batch, output_dim]
        (also returns per-block embeddings for Option C)
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class AttentionPooling(nn.Module):
    """Learned attention pooling over graph nodes."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x, batch=None, num_nodes=None):
        """
        Args:
            x: [total_nodes, dim]  (flattened across batch)
            batch: [total_nodes] batch assignment
        Returns:
            pooled: [batch_size, dim]
            weights: [total_nodes, 1] attention weights
        """
        scores = self.attention(x)  # [N, 1]

        if batch is not None:
            # Scatter softmax per graph
            from torch_geometric.utils import softmax
            weights = softmax(scores.squeeze(-1), batch).unsqueeze(-1)
        else:
            weights = torch.softmax(scores, dim=0)

        # Weighted sum
        if batch is not None:
            from torch_geometric.nn import global_add_pool
            pooled = global_add_pool(x * weights, batch)
        else:
            pooled = (x * weights).sum(dim=0, keepdim=True)

        return pooled, weights


class GATGraphEncoder(nn.Module):
    """Delta: GAT + attention pooling."""

    def __init__(self, input_dim, hidden_dim=256, output_dim=512,
                 num_layers=2, num_heads=4, dropout=0.1, pooling='attention'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads,
                                   heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads,
                                       heads=num_heads, dropout=dropout))

        if pooling == 'attention':
            self.pool = AttentionPooling(hidden_dim)
        else:
            self.pool = None

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        """
        Returns:
            f: [batch, output_dim] pooled function embedding
            block_embs: [total_nodes, hidden_dim] per-block (for Option C)
            attn_weights: [total_nodes, 1] pooling weights
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        block_embs = x  # Keep for Option C

        if self.pool is not None:
            f, attn_weights = self.pool(x, batch)
        else:
            f = global_mean_pool(x, batch)
            attn_weights = None

        f = self.output_proj(f)
        return f, block_embs, attn_weights


class GCNGraphEncoder(nn.Module):
    """Baseline: GCN + mean pooling."""

    def __init__(self, input_dim, hidden_dim=256, output_dim=512,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        block_embs = x
        f = global_mean_pool(x, batch)
        f = self.output_proj(f)
        return f, block_embs, None


def build_graph_encoder(cfg: dict) -> nn.Module:
    """Factory function."""
    input_dim = cfg.get('input_dim', 256)
    if cfg['type'] == 'gat':
        return GATGraphEncoder(
            input_dim=input_dim,
            hidden_dim=cfg['hidden_dim'],
            output_dim=cfg['output_dim'],
            num_layers=cfg['num_layers'],
            num_heads=cfg.get('num_heads', 4),
            dropout=cfg['dropout'],
            pooling=cfg.get('pooling', 'attention'),
        )
    else:
        return GCNGraphEncoder(
            input_dim=input_dim,
            hidden_dim=cfg['hidden_dim'],
            output_dim=cfg['output_dim'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
        )
