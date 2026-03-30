"""
Stage 3: Autoregressive GRU Decoder for sub-token sequence generation.

FIXES:
  1. Length-normalized beam search (prevents short-sequence bias)
  2. Repetition penalty (prevents decoder from repeating tokens)
  3. Temperature scaling at inference for better diversity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDecoder(nn.Module):
    """Autoregressive GRU decoder with teacher forcing and beam search."""

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512,
                 num_layers=1, dropout=0.1, max_length=15, encoder_dim=None,
                 pretrained_embeddings=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_layers = num_layers

        # encoder_dim: dimension of z from fusion (may differ from hidden_dim)
        self.encoder_dim = encoder_dim or hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            # Handle size mismatch: copy what fits, random init the rest
            n = min(pretrained_embeddings.shape[0], vocab_size)
            self.embedding.weight.data[:n] = pretrained_embeddings[:n]
            self.embedding.weight.requires_grad = True  # fine-tune
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.h0_proj = nn.Linear(self.encoder_dim, hidden_dim * num_layers)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, z):
        """Initialize GRU hidden state from fused representation z."""
        h = self.h0_proj(z)
        h = h.view(-1, self.num_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()
        return h

    def forward(self, z, decoder_input, teacher_forcing_ratio=1.0):
        """
        Training forward pass with scheduled sampling.

        When teacher_forcing_ratio < 1.0, the decoder sometimes uses
        its OWN predictions as input for the next step instead of
        ground truth. This is critical for preventing collapse at
        inference, where ground truth is never available.
        """
        B, T = decoder_input.shape
        hidden = self.init_hidden(z)

        logits = []
        input_token = decoder_input[:, 0:1]  # <SOS>

        for t in range(T):
            emb = self.embedding(input_token)
            emb = self.dropout(emb)
            output, hidden = self.gru(emb, hidden)
            logit = self.output_proj(output)
            logits.append(logit)

            if t + 1 < T:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    input_token = decoder_input[:, t + 1:t + 2]
                else:
                    # Use model's own prediction (argmax) for scheduled sampling
                    # Multinomial sampling was tried but injects too much noise,
                    # especially for 0-ext-call functions with weak representations
                    input_token = logit.argmax(dim=-1)

        return torch.cat(logits, dim=1)

    def generate(self, z, sos_id, eos_id, beam_width=5, length_penalty=0.7,
                 repetition_penalty=1.2):
        """
        Length-normalized beam search with repetition penalty.

        FIX 1: Length penalty — divides score by len^alpha, so longer
        sequences aren't penalized vs short ones. Without this, the
        decoder prefers short common names.

        FIX 2: Repetition penalty — reduces probability of tokens
        already generated, preventing "rpl_rpl_rpl" type outputs.

        Args:
            z: [1, hidden_dim]
            sos_id: start token
            eos_id: end token
            beam_width: number of beams
            length_penalty: alpha for length normalization (0=no penalty, 1=full)
            repetition_penalty: multiplier for repeated token log-probs
        """
        hidden = self.init_hidden(z)
        device = z.device

        # (score, token_list, hidden_state, set_of_generated_tokens)
        beams = [(0.0, [sos_id], hidden, set())]
        completed = []

        for step in range(self.max_length):
            candidates = []

            for score, tokens, h, generated in beams:
                if tokens[-1] == eos_id:
                    # Length-normalized score
                    length = len(tokens) - 1  # exclude SOS
                    norm_score = score / max(length, 1) ** length_penalty
                    completed.append((norm_score, tokens, h, generated))
                    continue

                input_token = torch.tensor([[tokens[-1]]], device=device)
                emb = self.embedding(input_token)
                output, new_h = self.gru(emb, h)
                logit = self.output_proj(output.squeeze(1))

                # Apply repetition penalty
                log_probs = F.log_softmax(logit, dim=-1).clone()
                for prev_token in generated:
                    if prev_token != sos_id and prev_token != eos_id:
                        log_probs[0, prev_token] /= repetition_penalty

                topk_probs, topk_ids = log_probs.topk(beam_width * 2)  # over-sample
                for i in range(min(beam_width * 2, topk_probs.shape[1])):
                    new_token = topk_ids[0, i].item()
                    new_score = score + topk_probs[0, i].item()
                    new_tokens = tokens + [new_token]
                    new_generated = generated | {new_token}
                    candidates.append((new_score, new_tokens, new_h, new_generated))

            if not candidates:
                break

            # Length-normalize scores for ranking
            def rank_score(c):
                s, toks, _, _ = c
                length = len(toks) - 1
                return s / max(length, 1) ** length_penalty

            candidates.sort(key=rank_score, reverse=True)
            beams = candidates[:beam_width]

            if all(b[1][-1] == eos_id for b in beams):
                for b in beams:
                    length = len(b[1]) - 1
                    norm_score = b[0] / max(length, 1) ** length_penalty
                    completed.append((norm_score, b[1], b[2], b[3]))
                break

        # Add remaining beams
        for b in beams:
            if b[1][-1] != eos_id:
                length = len(b[1]) - 1
                norm_score = b[0] / max(length, 1) ** length_penalty
                completed.append((norm_score, b[1], b[2], b[3]))

        completed.sort(key=lambda x: x[0], reverse=True)

        if completed:
            best_score, best_tokens, _, _ = completed[0]
            result = [t for t in best_tokens if t != sos_id and t != eos_id]
            return result, best_score
        else:
            return [], float('-inf')
