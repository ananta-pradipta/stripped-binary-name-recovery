"""
LoRA-adapted LM Decoder for function name generation.

Instead of prefix tuning (which failed), we use LoRA to adapt CodeGen's
attention layers. The encoder embedding z is projected into a sequence
of virtual tokens that serve as input, and LoRA modifies how CodeGen
processes these tokens at EVERY layer.

Key difference from prefix tuning:
- Prefix tuning: only modifies input, LM processes normally
- LoRA: modifies how the LM processes at every attention layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class LoRADecoder(nn.Module):
    """LoRA-adapted CodeGen decoder conditioned on encoder embedding."""

    def __init__(self, encoder_dim=1024, model_name="Salesforce/codegen-350M-mono",
                 max_length=20, dropout=0.1, num_prefix_tokens=16,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.max_length = max_length
        self.num_prefix_tokens = num_prefix_tokens
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model and apply LoRA
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.lm_hidden_dim = base_model.config.hidden_size

        # Apply LoRA to attention layers
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["qkv_proj"],  # CodeGen uses fused qkv
            bias="none",
        )
        self.lm = get_peft_model(base_model, lora_config)
        self.lm.print_trainable_parameters()

        # Project encoder embedding to prefix tokens
        self.prefix_proj = nn.Sequential(
            nn.Linear(encoder_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.lm_hidden_dim * num_prefix_tokens),
        )

    def _get_prefix_embeds(self, z):
        B = z.shape[0]
        prefix = self.prefix_proj(z)
        return prefix.view(B, self.num_prefix_tokens, self.lm_hidden_dim)

    def forward(self, z, target_names=None, teacher_forcing_ratio=1.0):
        """Training forward pass."""
        B = z.shape[0]
        device = z.device

        prefix_embeds = self._get_prefix_embeds(z)

        # Tokenize target names
        targets = self.tokenizer(
            target_names,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)

        input_ids = targets["input_ids"]
        attention_mask = targets["attention_mask"]

        # Get token embeddings
        # CodeGen uses transformer.wte for token embeddings
        wte = self.lm.base_model.model.transformer.wte
        token_embeds = wte(input_ids)

        # Combine: [prefix_embeds, token_embeds]
        combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        prefix_mask = torch.ones(B, self.num_prefix_tokens, device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Forward through LoRA-adapted LM
        outputs = self.lm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )

        # Get logits for token positions (skip prefix)
        logits = outputs.logits[:, self.num_prefix_tokens:, :]

        # Autoregressive loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_mask = attention_mask[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        )
        loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum().clamp(min=1)

        return loss, logits

    @torch.no_grad()
    def generate(self, z, max_length=None, temperature=0.1,
                 repetition_penalty=1.5):
        if max_length is None:
            max_length = self.max_length

        device = z.device
        prefix_embeds = self._get_prefix_embeds(z)

        generated_ids = []
        past_key_values = None
        total_score = 0.0

        # First forward: process prefix
        outputs = self.lm(inputs_embeds=prefix_embeds, use_cache=True)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        for step in range(max_length):
            # Repetition penalty
            if generated_ids and repetition_penalty != 1.0:
                for prev_id in set(generated_ids):
                    if next_logits[0, prev_id] > 0:
                        next_logits[0, prev_id] /= repetition_penalty
                    else:
                        next_logits[0, prev_id] *= repetition_penalty

            scaled_logits = next_logits / max(temperature, 0.01)
            next_token = scaled_logits.argmax(dim=-1)

            score = F.log_softmax(scaled_logits, dim=-1)
            total_score += score[0, next_token[0]].item()

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            decoded = self.tokenizer.decode([next_token.item()])
            if '\n' in decoded or '(' in decoded or '{' in decoded or ';' in decoded:
                break
            if decoded.strip() == '' and step > 0:
                break

            generated_ids.append(next_token.item())

            # Get token embedding for next step
            wte = self.lm.base_model.model.transformer.wte
            next_embeds = wte(next_token.unsqueeze(0))
            outputs = self.lm(
                inputs_embeds=next_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        name = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        name = name.split('\n')[0].split('(')[0].split('{')[0].strip()
        name = name.rstrip('. 0123456789')
        norm_score = total_score / max(len(generated_ids), 1)
        return name, norm_score
