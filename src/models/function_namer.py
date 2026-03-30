"""
Full model: assembles block encoder, graph encoder, external encoder,
callee context encoder, fusion module, and decoder into a single nn.Module.

V2: Added callee context encoder for inter-procedural context.
"""
import os
import torch
import torch.nn as nn

from .block_encoder import build_block_encoder
from .graph_encoder import build_graph_encoder
from .external_encoder import ExternalCallEncoder, CalleeContextEncoder, StringReferenceEncoder
from .gated_fusion import GatedFusion, NodeFeatureFusion, CrossAttentionFusion
from .decoder import GRUDecoder


class FunctionNamer(nn.Module):
    """End-to-end function name prediction model."""

    def __init__(self, cfg: dict):
        super().__init__()

        # Stage 1: Block Encoder
        self.block_encoder = build_block_encoder(cfg['block_encoder'])

        # Stage 2: Graph Encoder
        gcfg = cfg['graph_encoder'].copy()
        gcfg['input_dim'] = cfg['block_encoder']['output_dim']
        self.graph_encoder = build_graph_encoder(gcfg)

        # External Call Encoder
        ecfg = cfg['external_encoder']
        self.ext_encoder_enabled = ecfg.get('enabled', True)
        if self.ext_encoder_enabled:
            self.ext_encoder = ExternalCallEncoder(
                vocab_size=ecfg['vocab_size'],
                embed_dim=ecfg['embed_dim'],
                hidden_dim=ecfg['hidden_dim'],
                output_dim=ecfg['output_dim'],
                dropout=ecfg['dropout'],
            )

        # Callee Context Encoder (inter-procedural: what this function calls)
        ccfg = cfg.get('callee_encoder', {})
        self.callee_encoder_enabled = ccfg.get('enabled', False)
        if self.callee_encoder_enabled:
            self.callee_encoder = CalleeContextEncoder(
                token_vocab_size=cfg['block_encoder']['token_vocab_size'],
                embed_dim=ccfg.get('embed_dim', 64),
                hidden_dim=ccfg.get('hidden_dim', 128),
                output_dim=cfg['fusion']['input_dim'],
                dropout=ccfg.get('dropout', 0.1),
            )

        # Caller Context Encoder (inter-procedural: what calls this function)
        crcfg = cfg.get('caller_encoder', {})
        self.caller_encoder_enabled = crcfg.get('enabled', False)
        if self.caller_encoder_enabled:
            self.caller_encoder = CalleeContextEncoder(  # reuse same architecture
                token_vocab_size=cfg['block_encoder']['token_vocab_size'],
                embed_dim=crcfg.get('embed_dim', 64),
                hidden_dim=crcfg.get('hidden_dim', 128),
                output_dim=cfg['fusion']['input_dim'],
                dropout=crcfg.get('dropout', 0.1),
            )

        # String Reference Encoder
        scfg = cfg.get('string_encoder', {})
        self.string_encoder_enabled = scfg.get('enabled', False)
        if self.string_encoder_enabled:
            self.string_encoder = StringReferenceEncoder(
                vocab_size=scfg.get('vocab_size', 503),
                embed_dim=scfg.get('embed_dim', 64),
                hidden_dim=scfg.get('hidden_dim', 128),
                output_dim=cfg['fusion']['input_dim'],
                dropout=scfg.get('dropout', 0.1),
            )

        # Binary PLT Fingerprint Encoder
        bfcfg = cfg.get('binary_fingerprint', {})
        self.binary_fp_enabled = bfcfg.get('enabled', False)
        if self.binary_fp_enabled:
            fp_vocab = cfg['external_encoder']['vocab_size']
            fp_embed = bfcfg.get('embed_dim', 64)
            fp_out = cfg['fusion']['input_dim']
            self.binary_fp_embedding = nn.Embedding(fp_vocab, fp_embed, padding_idx=0)
            self.binary_fp_proj = nn.Linear(fp_embed, fp_out)

        # Fusion
        fusion_type = cfg['fusion']['type']
        fusion_dim = cfg['fusion']['input_dim']
        self.fusion_type = fusion_type
        if fusion_type == 'gated' and self.ext_encoder_enabled:
            self.fusion = GatedFusion(fusion_dim)
        else:
            self.fusion = None

        # Callee/caller fusion gates
        if self.callee_encoder_enabled:
            self.callee_gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Sigmoid(),
            )
        if self.caller_encoder_enabled:
            self.caller_gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Sigmoid(),
            )
        if self.string_encoder_enabled:
            self.string_gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Sigmoid(),
            )
        if self.binary_fp_enabled:
            self.binary_fp_gate = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Sigmoid(),
            )

        # Stage 3: Decoder
        dcfg = cfg['decoder']
        pretrained_emb = None
        pretrained_path = dcfg.get('pretrained_embeddings')
        if pretrained_path and os.path.exists(pretrained_path):
            import torch as _torch
            pretrained_emb = _torch.load(pretrained_path, map_location='cpu', weights_only=True)
            print(f"Loaded pretrained decoder embeddings: {pretrained_emb.shape}")
        self.decoder = GRUDecoder(
            vocab_size=dcfg['bpe_vocab_size'],
            embed_dim=dcfg['embed_dim'],
            hidden_dim=dcfg['hidden_dim'],
            num_layers=dcfg.get('num_layers', 1),
            dropout=dcfg['dropout'],
            max_length=dcfg['max_length'],
            encoder_dim=cfg['fusion']['input_dim'],
            pretrained_embeddings=pretrained_emb,
        )

        # Auxiliary head (disabled by default)
        aux_buckets = cfg.get('training', {}).get('aux_num_blocks_buckets', 7)
        self.aux_head = nn.Linear(fusion_dim, aux_buckets) if aux_buckets > 0 else None

        # Multi-label classification head (XFL-style)
        # Predicts which sub-tokens appear in the function name (independent BCE)
        # Used as primary decoder for 0-ext functions at inference
        self.multilabel_enabled = cfg.get('multilabel', {}).get('enabled', False)
        if self.multilabel_enabled:
            ml_hidden = cfg.get('multilabel', {}).get('hidden_dim', 512)
            self.multilabel_head = nn.Sequential(
                nn.Linear(fusion_dim, ml_hidden),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(ml_hidden, dcfg['bpe_vocab_size']),
            )

    def _compute_has_ext_calls(self, ext_call_ids):
        return (ext_call_ids > 0).any(dim=1)

    def forward(self, block_tokens, edge_index, ext_call_ids,
                decoder_input, batch=None, teacher_forcing_ratio=1.0,
                block_features=None, callee_tokens=None, caller_tokens=None,
                string_tokens=None, binary_ext_ids=None):
        """Training forward pass."""
        # Stage 1: encode blocks
        block_embs = self.block_encoder(block_tokens, block_features=block_features)
        B, N, D = block_embs.shape
        x = block_embs.view(B * N, D)

        if batch is None:
            batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
        else:
            batch_vec = batch

        # Stage 2: graph encode
        f, block_embs_updated, attn_weights = self.graph_encoder(x, edge_index, batch_vec)

        # External encoding + conditional fusion
        gate_values = None
        if self.ext_encoder_enabled and self.fusion is not None:
            c = self.ext_encoder(ext_call_ids)
            has_ext = self._compute_has_ext_calls(ext_call_ids)
            z, gate_values = self.fusion(f, c, has_ext_calls=has_ext)
        else:
            z = f

        # Callee context fusion
        if self.callee_encoder_enabled and callee_tokens is not None:
            callee_ctx, has_callees = self.callee_encoder(callee_tokens)
            g = self.callee_gate(torch.cat([z, callee_ctx], dim=1))
            z_fused = g * z + (1 - g) * callee_ctx
            mask = has_callees.unsqueeze(1).float()
            z = mask * z_fused + (1 - mask) * z

        # Caller context fusion
        if self.caller_encoder_enabled and caller_tokens is not None:
            caller_ctx, has_callers = self.caller_encoder(caller_tokens)
            g = self.caller_gate(torch.cat([z, caller_ctx], dim=1))
            z_fused = g * z + (1 - g) * caller_ctx
            mask = has_callers.unsqueeze(1).float()
            z = mask * z_fused + (1 - mask) * z

        # String reference fusion (conditional bypass — skip when no strings)
        if self.string_encoder_enabled and string_tokens is not None:
            string_ctx, has_strings = self.string_encoder(string_tokens)
            g = self.string_gate(torch.cat([z, string_ctx], dim=1))
            z_fused = g * z + (1 - g) * string_ctx
            mask = has_strings.unsqueeze(1).float()
            z = mask * z_fused + (1 - mask) * z

        # Binary PLT fingerprint fusion (always available — conditional on having PLT calls)
        if self.binary_fp_enabled and binary_ext_ids is not None:
            fp_emb = self.binary_fp_embedding(binary_ext_ids)  # [B, 30, 64]
            # Mean pool over non-zero entries
            mask = (binary_ext_ids > 0).float().unsqueeze(-1)  # [B, 30, 1]
            fp_pooled = (fp_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, 64]
            fp_ctx = self.binary_fp_proj(fp_pooled)  # [B, 512]
            has_fp = (binary_ext_ids > 0).any(dim=1)  # [B]
            g = self.binary_fp_gate(torch.cat([z, fp_ctx], dim=1))
            z_fused = g * z + (1 - g) * fp_ctx
            fp_mask = has_fp.unsqueeze(1).float()
            z = fp_mask * z_fused + (1 - fp_mask) * z

        # Stage 3: decode
        logits = self.decoder(z, decoder_input, teacher_forcing_ratio)
        aux_logits = self.aux_head(z) if self.aux_head is not None else None

        # Multi-label head: predict sub-token set
        ml_logits = self.multilabel_head(z) if self.multilabel_enabled else None

        return logits, gate_values, aux_logits, z, ml_logits

    def predict(self, block_tokens, edge_index, ext_call_ids,
                sos_id, eos_id, batch=None, beam_width=5,
                block_features=None, callee_tokens=None, caller_tokens=None,
                string_tokens=None, binary_ext_ids=None):
        """Inference: generate function name."""
        with torch.no_grad():
            block_embs = self.block_encoder(block_tokens, block_features=block_features)
            B, N, D = block_embs.shape
            x = block_embs.view(B * N, D)

            if batch is None:
                batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
            else:
                batch_vec = batch

            f, _, _ = self.graph_encoder(x, edge_index, batch_vec)

            if self.ext_encoder_enabled and self.fusion is not None:
                c = self.ext_encoder(ext_call_ids)
                has_ext = self._compute_has_ext_calls(ext_call_ids)
                z, _ = self.fusion(f, c, has_ext_calls=has_ext)
            else:
                z = f

            # Callee context fusion
            if self.callee_encoder_enabled and callee_tokens is not None:
                callee_ctx, has_callees = self.callee_encoder(callee_tokens)
                g = self.callee_gate(torch.cat([z, callee_ctx], dim=1))
                z_fused = g * z + (1 - g) * callee_ctx
                mask = has_callees.unsqueeze(1).float()
                z = mask * z_fused + (1 - mask) * z

            # Caller context fusion
            if self.caller_encoder_enabled and caller_tokens is not None:
                caller_ctx, has_callers = self.caller_encoder(caller_tokens)
                g = self.caller_gate(torch.cat([z, caller_ctx], dim=1))
                z_fused = g * z + (1 - g) * caller_ctx
                mask = has_callers.unsqueeze(1).float()
                z = mask * z_fused + (1 - mask) * z

            # String reference fusion (conditional bypass)
            if self.string_encoder_enabled and string_tokens is not None:
                string_ctx, has_strings = self.string_encoder(string_tokens)
                g = self.string_gate(torch.cat([z, string_ctx], dim=1))
                z_fused = g * z + (1 - g) * string_ctx
                mask = has_strings.unsqueeze(1).float()
                z = mask * z_fused + (1 - mask) * z

            # Binary PLT fingerprint fusion
            if self.binary_fp_enabled and binary_ext_ids is not None:
                fp_emb = self.binary_fp_embedding(binary_ext_ids)
                mask = (binary_ext_ids > 0).float().unsqueeze(-1)
                fp_pooled = (fp_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                fp_ctx = self.binary_fp_proj(fp_pooled)
                has_fp = (binary_ext_ids > 0).any(dim=1)
                g = self.binary_fp_gate(torch.cat([z, fp_ctx], dim=1))
                z_fused = g * z + (1 - g) * fp_ctx
                fp_mask = has_fp.unsqueeze(1).float()
                z = fp_mask * z_fused + (1 - fp_mask) * z

            results = []
            for i in range(B):
                tokens, score = self.decoder.generate(
                    z[i:i+1], sos_id, eos_id, beam_width
                )
                results.append((tokens, score))

            return results
