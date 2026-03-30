"""
External Call Encoder — Bi-GRU over ordered external call sequence.
Callee Context Encoder — encodes internal callee token signatures.
String Reference Encoder — encodes string constants referenced by a function.

Input:  external call IDs [batch, max_ext_calls]
Output: calling context embedding c [batch, output_dim]
"""
import torch
import torch.nn as nn


class CalleeContextEncoder(nn.Module):
    """Encode internal callee function signatures as context.

    Each callee is represented by its first N tokens (a signature of what it does).
    We encode each callee signature with a shared encoder, then mean-pool across callees.
    """

    def __init__(self, token_vocab_size, embed_dim=64, hidden_dim=128,
                 output_dim=512, max_sig_tokens=10, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(token_vocab_size, embed_dim, padding_idx=0)
        # Small GRU to encode each callee's token signature
        self.sig_gru = nn.GRU(
            embed_dim, hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # Project from bi-GRU hidden to output dim
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        self.no_callee_emb = nn.Parameter(torch.randn(output_dim) * 0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, callee_tokens):
        """
        Args:
            callee_tokens: [batch, max_callees, max_sig_tokens] token IDs

        Returns:
            callee_ctx: [batch, output_dim] callee context embedding
            has_callees: [batch] bool — True if function has internal callees
        """
        B, K, T = callee_tokens.shape

        # Check which samples have callees (any non-zero token)
        has_callees = (callee_tokens > 0).any(dim=-1).any(dim=-1)  # [B]

        # Reshape to encode all callee signatures in one batch
        flat = callee_tokens.view(B * K, T)  # [B*K, T]
        emb = self.embedding(flat)  # [B*K, T, embed_dim]
        emb = self.dropout(emb)

        _, hidden = self.sig_gru(emb)  # hidden: [2, B*K, hidden_dim]
        sig_emb = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B*K, hidden_dim*2]
        sig_emb = self.output_proj(sig_emb)  # [B*K, output_dim]
        sig_emb = sig_emb.view(B, K, -1)  # [B, K, output_dim]

        # Mean pool across callees (mask padding callees)
        callee_mask = (callee_tokens > 0).any(dim=-1).float().unsqueeze(-1)  # [B, K, 1]
        num_callees = callee_mask.sum(dim=1).clamp(min=1)  # [B, 1]
        callee_ctx = (sig_emb * callee_mask).sum(dim=1) / num_callees  # [B, output_dim]

        # Replace with no_callee_emb for functions without callees
        no_ctx = self.no_callee_emb.unsqueeze(0).expand(B, -1)
        callee_ctx = torch.where(has_callees.unsqueeze(1), callee_ctx, no_ctx)

        return callee_ctx, has_callees


class ExternalCallEncoder(nn.Module):
    """Bi-GRU encoder for external function call sequences."""

    def __init__(self, vocab_size, embed_dim=512, hidden_dim=256,
                 num_layers=1, dropout=0.1, output_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # Bi-GRU gives hidden_dim * 2
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)

        # Learned fallback for functions with no external calls
        self.no_context_emb = nn.Parameter(torch.randn(output_dim) * 0.01)

    def forward(self, ext_call_ids):
        """
        Args:
            ext_call_ids: [batch, max_ext_calls] — 0 means <NO_EXT> or padding

        Returns:
            c: [batch, output_dim] calling context embedding
        """
        B = ext_call_ids.size(0)

        # Check which samples have no external calls (all zeros or only sentinel)
        has_calls = (ext_call_ids > 0).any(dim=1)  # [batch]

        # Embed
        x = self.embedding(ext_call_ids)  # [B, L, embed_dim]

        # Run Bi-GRU
        _, hidden = self.gru(x)  # hidden: [2*layers, B, hidden_dim]
        # Concatenate forward and backward final hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, hidden_dim*2]
        c = self.output_proj(hidden)  # [B, output_dim]

        # Replace with no_context_emb for functions without external calls
        no_ctx = self.no_context_emb.unsqueeze(0).expand(B, -1)
        c = torch.where(has_calls.unsqueeze(1), c, no_ctx)

        return c


class StringReferenceEncoder(nn.Module):
    """Encode string constants referenced by a function.

    String references (e.g., "memory exhausted", "Usage: %s") are tokenized
    into semantic words and encoded as a bag-of-tokens with learned embeddings.
    Uses a small BiGRU over the token sequence to capture token ordering.
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                 output_dim=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        # No more no_string_emb — use conditional bypass instead
        self.dropout = nn.Dropout(dropout)

    def forward(self, string_token_ids):
        """
        Args:
            string_token_ids: [batch, max_string_tokens] — token IDs from string vocab

        Returns:
            string_ctx: [batch, output_dim] — string context embedding
            has_strings: [batch] bool — True if function references strings
        """
        has_strings = (string_token_ids > 0).any(dim=1)  # [B]

        emb = self.embedding(string_token_ids)  # [B, L, embed_dim]
        emb = self.dropout(emb)

        _, hidden = self.gru(emb)  # hidden: [2, B, hidden_dim]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, hidden_dim*2]
        ctx = self.output_proj(hidden)  # [B, output_dim]

        return ctx, has_strings
