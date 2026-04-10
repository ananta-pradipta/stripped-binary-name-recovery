"""
Train CodeGen with LoRA on text descriptions of binary functions.

No encoder needed — input is structured text describing the function's
behavior (ext calls, CFG structure, instruction patterns, callee/caller context).
The LM reads this text and generates the function name.

This is the approach closest to SymGen (text in → name out) but using
our BAP-extracted features as text instead of Ghidra decompiled code.
"""
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import time
import random


class TextFunctionDataset(Dataset):
    """Dataset of text prompts → function names."""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # Combine prompt + name for causal LM training
        text = entry['prompt'] + " " + entry['name'] + self.tokenizer.eos_token
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        # Find where the name starts (after "Function name: ")
        prompt_text = entry['prompt'] + " "
        prompt_tokens = self.tokenizer(prompt_text, truncation=True,
                                        max_length=self.max_length)
        prompt_len = len(prompt_tokens['input_ids'])

        # Create labels: -100 for prompt tokens (don't compute loss on prompt)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def main():
    parser = argparse.ArgumentParser(description='Train text-based LM for function naming')
    parser.add_argument('--data', default='data/text_dataset.json')
    parser.add_argument('--model', default='Salesforce/codegen-350M-mono')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', default='checkpoints/text_lm.pt')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data) as f:
        all_data = json.load(f)
    print(f"  {len(all_data)} functions")

    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["qkv_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # Split data
    val_size = int(len(all_data) * args.val_split)
    train_size = len(all_data) - val_size

    # Split by binary (no binary in both train and val)
    binaries = list(set(d['binary'] for d in all_data))
    random.shuffle(binaries)
    val_binaries = set(binaries[:max(1, int(len(binaries) * args.val_split))])

    train_data = [d for d in all_data if d['binary'] not in val_binaries]
    val_data = [d for d in all_data if d['binary'] in val_binaries]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = TextFunctionDataset(train_data, tokenizer, args.max_length)
    val_dataset = TextFunctionDataset(val_data, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    print(f"\nTraining: {args.epochs} epochs")
    print(f"Batches/epoch: {len(train_loader)}")

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        start = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - start

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with torch.amp.autocast('cuda', enabled=bool(scaler)):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                val_loss += outputs.loss.item()
                val_batches += 1

            # Quick generation check on a few samples
            for entry in val_data[:10]:
                prompt = entry['prompt'] + " "
                inputs = tokenizer(prompt, return_tensors='pt',
                                   truncation=True, max_length=args.max_length - 30).to(device)

                with torch.amp.autocast('cuda', enabled=bool(scaler)):
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=25,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.5,
                    )

                # Decode only the generated part
                gen_text = tokenizer.decode(gen_ids[0][inputs['input_ids'].shape[1]:],
                                            skip_special_tokens=True).strip()
                # Clean up
                gen_text = gen_text.split('\n')[0].split('(')[0].strip()

                if gen_text == entry['name']:
                    val_correct += 1
                val_total += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_em = val_correct / max(val_total, 1)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val EM (sample): {val_em:.1%} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s")

        # Print a few predictions
        if val_total > 0:
            for entry in val_data[:3]:
                prompt = entry['prompt'] + " "
                inputs = tokenizer(prompt, return_tensors='pt',
                                   truncation=True, max_length=args.max_length - 30).to(device)
                with torch.amp.autocast('cuda', enabled=bool(scaler)):
                    gen_ids = model.generate(
                        **inputs, max_new_tokens=25, temperature=0.1,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.5,
                    )
                gen_text = tokenizer.decode(gen_ids[0][inputs['input_ids'].shape[1]:],
                                            skip_special_tokens=True).strip().split('\n')[0].split('(')[0].strip()
                match = "✓" if gen_text == entry['name'] else "✗"
                print(f"  {match} true=\"{entry['name']}\" pred=\"{gen_text}\"")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(args.save)
            tokenizer.save_pretrained(args.save)
            print(f"  ★ New best! Saved to {args.save}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
