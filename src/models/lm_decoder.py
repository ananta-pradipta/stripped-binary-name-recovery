"""
Pre-trained Language Model Decoder for function name generation.

Replaces the GRU decoder with a pre-trained code LM (CodeGen/GPT-2) that
has learned function naming patterns from source code pre-training.

The key insight: the encoder produces a "function behavior embedding" z,
and the LM decoder translates this embedding into a function name using
its pre-trained knowledge of naming conventions.

Architecture:
  z (from encoder/fusion) → projection → LM prefix embedding
  LM generates: <bos> function_name <eos>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LMDecoder(nn.Module):
    """Pre-trained LM decoder conditioned on encoder embedding."""

    def __init__(self, encoder_dim=1024, model_name="Salesforce/codegen-350M-mono",
                 max_length=20, dropout=0.1, freeze_lm=False, num_prefix_tokens=4):
        super().__init__()
        self.max_length = max_length
        self.num_prefix_tokens = num_prefix_tokens
        self.model_name = model_name

        # Load pre-trained LM and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        self.lm_hidden_dim = self.lm.config.hidden_size  # 1024 for codegen-350M

        # Optionally freeze LM weights (train only projection)
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

        # Project encoder embedding z into LM prefix tokens
        # We create `num_prefix_tokens` virtual tokens that serve as
        # a "soft prompt" for the LM, encoding the binary function's behavior
        self.prefix_proj = nn.Sequential(
            nn.Linear(encoder_dim, self.lm_hidden_dim * num_prefix_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Special tokens for function name generation
        self.name_prefix = "Function name: "  # prompt format

    def _get_prefix_embeds(self, z):
        """Project encoder embedding into LM prefix token embeddings.

        Args:
            z: (B, encoder_dim) fused function embedding

        Returns:
            prefix_embeds: (B, num_prefix_tokens, lm_hidden_dim)
        """
        B = z.shape[0]
        # Project and reshape into prefix tokens
        prefix = self.prefix_proj(z.float())  # (B, lm_hidden_dim * num_prefix_tokens)
        prefix = prefix.view(B, self.num_prefix_tokens, self.lm_hidden_dim)
        return prefix.half()  # match LM dtype

    def forward(self, z, target_names=None, teacher_forcing_ratio=1.0):
        """Training forward pass.

        Args:
            z: (B, encoder_dim) encoder output
            target_names: list of str, ground truth function names
            teacher_forcing_ratio: unused (kept for API compatibility)

        Returns:
            loss: scalar training loss
            logits: (B, seq_len, vocab_size) for analysis
        """
        B = z.shape[0]
        device = z.device

        # Get prefix embeddings from encoder output
        prefix_embeds = self._get_prefix_embeds(z)  # (B, num_prefix, hidden)

        # Tokenize target names
        prompt_texts = [self.name_prefix + name for name in target_names]
        targets = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length + len(self.tokenizer.encode(self.name_prefix)),
        ).to(device)

        input_ids = targets["input_ids"]
        attention_mask = targets["attention_mask"]

        # Get token embeddings from LM
        token_embeds = self.lm.transformer.wte(input_ids)  # (B, seq_len, hidden)

        # Prepend prefix embeddings
        # combined = [prefix_embeds, token_embeds]
        combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        # Extend attention mask for prefix tokens
        prefix_mask = torch.ones(B, self.num_prefix_tokens, device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Forward through LM
        outputs = self.lm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )

        # Get logits (skip prefix positions)
        logits = outputs.logits[:, self.num_prefix_tokens:, :]  # (B, seq_len, vocab)

        # Compute loss on the function name part (after the prompt prefix)
        prompt_len = len(self.tokenizer.encode(self.name_prefix))
        # Shift for autoregressive loss
        shift_logits = logits[:, prompt_len-1:-1, :].contiguous()
        shift_labels = input_ids[:, prompt_len:].contiguous()

        # Mask padding
        loss_mask = attention_mask[:, prompt_len:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        )
        loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()

        return loss, logits

    @torch.no_grad()
    def generate(self, z, max_length=None, temperature=0.1, top_k=50):
        """Generate function name from encoder embedding.

        Args:
            z: (1, encoder_dim) single function embedding

        Returns:
            name: str, predicted function name
            score: float, generation score
        """
        if max_length is None:
            max_length = self.max_length

        device = z.device

        # Get prefix embeddings
        prefix_embeds = self._get_prefix_embeds(z)  # (1, num_prefix, hidden)

        # Encode the prompt "Function name: "
        prompt_ids = self.tokenizer.encode(self.name_prefix, return_tensors="pt").to(device)
        prompt_embeds = self.lm.transformer.wte(prompt_ids)  # (1, prompt_len, hidden)

        # Combined input: prefix + prompt
        input_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

        # Generate autoregressively
        generated_ids = []
        past_key_values = None
        total_score = 0.0

        # First forward: process prefix + prompt
        outputs = self.lm(
            inputs_embeds=input_embeds,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :] / max(temperature, 0.01)

        for step in range(max_length):
            # Sample or greedy
            if temperature <= 0.01:
                next_token = next_logits.argmax(dim=-1)
            else:
                # Top-k sampling
                top_k_logits, top_k_indices = next_logits.topk(top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                sampled_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(-1, sampled_idx).squeeze(-1)

            score = F.log_softmax(next_logits, dim=-1)
            total_score += score[0, next_token[0]].item()

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token.item())

            # Next step
            next_embeds = self.lm.transformer.wte(next_token.unsqueeze(0))
            outputs = self.lm(
                inputs_embeds=next_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :] / max(temperature, 0.01)

        # Decode generated tokens
        name = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Clean up: remove any trailing whitespace, newlines, etc.
        name = name.split('\n')[0].strip()
        # Remove common artifacts
        if name.startswith('('):
            name = name.split('(')[0]

        norm_score = total_score / max(len(generated_ids), 1)
        return name, norm_score


class LMDecoderLight(nn.Module):
    """Lightweight LM decoder using GPT-2 small (124M params).

    V2 fixes: unfreeze top layers, larger projection, repetition penalty.
    """

    def __init__(self, encoder_dim=1024, model_name="gpt2",
                 max_length=20, dropout=0.1, num_prefix_tokens=16,
                 unfreeze_layers=2):
        super().__init__()
        self.max_length = max_length
        self.num_prefix_tokens = num_prefix_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        self.lm_hidden_dim = self.lm.config.hidden_size  # 768 for gpt2

        # Freeze most of LM, unfreeze top N layers + lm_head
        for param in self.lm.parameters():
            param.requires_grad = False
        # Unfreeze top layers
        n_layers = len(self.lm.transformer.h)
        for layer in self.lm.transformer.h[n_layers - unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        # Unfreeze final layer norm and lm_head
        for param in self.lm.transformer.ln_f.parameters():
            param.requires_grad = True
        for param in self.lm.lm_head.parameters():
            param.requires_grad = True

        # Larger projection: encoder_dim → prefix tokens
        self.prefix_proj = nn.Sequential(
            nn.Linear(encoder_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.lm_hidden_dim * num_prefix_tokens),
        )

        self.name_prefix = ""

    def _get_prefix_embeds(self, z):
        B = z.shape[0]
        prefix = self.prefix_proj(z)
        return prefix.view(B, self.num_prefix_tokens, self.lm_hidden_dim)

    def forward(self, z, target_names=None, teacher_forcing_ratio=1.0):
        B = z.shape[0]
        device = z.device

        prefix_embeds = self._get_prefix_embeds(z)

        # Tokenize targets: just the function name directly
        targets = self.tokenizer(
            target_names,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)

        input_ids = targets["input_ids"]
        attention_mask = targets["attention_mask"]
        token_embeds = self.lm.transformer.wte(input_ids)

        combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        prefix_mask = torch.ones(B, self.num_prefix_tokens, device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.lm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )

        logits = outputs.logits[:, self.num_prefix_tokens:, :]

        # Autoregressive loss on function name tokens
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_mask = attention_mask[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        )
        loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()

        return loss, logits

    @torch.no_grad()
    def generate(self, z, max_length=None, temperature=0.1,
                 repetition_penalty=2.0, **kwargs):
        if max_length is None:
            max_length = self.max_length

        device = z.device
        prefix_embeds = self._get_prefix_embeds(z)

        generated_ids = []
        past_key_values = None
        total_score = 0.0

        outputs = self.lm(inputs_embeds=prefix_embeds, use_cache=True)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        for step in range(max_length):
            # Apply repetition penalty
            if generated_ids and repetition_penalty != 1.0:
                for prev_id in set(generated_ids):
                    if next_logits[0, prev_id] > 0:
                        next_logits[0, prev_id] /= repetition_penalty
                    else:
                        next_logits[0, prev_id] *= repetition_penalty

            # Temperature
            scaled_logits = next_logits / max(temperature, 0.01)
            next_token = scaled_logits.argmax(dim=-1)

            score = F.log_softmax(scaled_logits, dim=-1)
            total_score += score[0, next_token[0]].item()

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Stop on newline, delimiters, or space after underscore (name boundary)
            decoded = self.tokenizer.decode([next_token.item()])
            if '\n' in decoded or '(' in decoded or '{' in decoded or ';' in decoded:
                break
            # Stop if generating spaces (function names don't have spaces)
            if decoded.strip() == '' and step > 0:
                break

            generated_ids.append(next_token.item())

            next_embeds = self.lm.transformer.wte(next_token.unsqueeze(0))
            outputs = self.lm(
                inputs_embeds=next_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        name = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        name = name.split('\n')[0].split('(')[0].split('{')[0].strip()
        # Remove trailing dots, numbers, spaces
        name = name.rstrip('. 0123456789')
        norm_score = total_score / max(len(generated_ids), 1)
        return name, norm_score
