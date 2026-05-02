"""Pretrain the GRU decoder as an unconditional sub-token language model
on a large corpus of function names. After pretraining, the decoder's
embedding, GRU, and output_proj weights can be used to initialize the
full model's decoder in finetune.

Usage:
  python3 -m src.training.pretrain_decoder \
      --corpus data/name_corpus.txt \
      --votes-vocab data/votes_vocab.json \
      --output checkpoints/decoder_pretrained.pt \
      --epochs 20 --batch-size 256
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.decoder import GRUDecoder
from src.preprocessing.build_votes import VotesTokenizer


class NameDataset(Dataset):
    """Dataset of function names → token sequences."""

    def __init__(self, names, tokenizer, max_len=20):
        self.names = names
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()
        self.pad_id = 0  # PAD_ID from build_votes.py

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        tokens = self.tokenizer.encode(name)[: self.max_len]
        # input:  <sos> tok1 tok2 ... tokN
        # target: tok1 tok2 ... tokN <eos>
        input_ids = [self.sos_id] + tokens
        target_ids = tokens + [self.eos_id]
        # Pad to same length
        pad_to = self.max_len + 1
        input_ids = input_ids[:pad_to] + [self.pad_id] * (pad_to - len(input_ids))
        target_ids = target_ids[:pad_to] + [self.pad_id] * (pad_to - len(target_ids))
        return torch.tensor(input_ids), torch.tensor(target_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to name corpus (one name per line)")
    parser.add_argument("--votes-vocab", default="data/votes_vocab.json")
    parser.add_argument("--output", required=True, help="Where to save pretrained decoder weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-dim", type=int, default=1024,
                        help="Dimension of z the full model expects (used by h0_proj; kept consistent with finetune config).")
    parser.add_argument("--max-length", type=int, default=15)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = VotesTokenizer(vocab_path=args.votes_vocab)
    vocab_size = tokenizer.get_piece_size()
    print(f"Votes vocab size: {vocab_size}")

    # Load corpus
    with open(args.corpus) as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"Corpus size: {len(names):,} names")

    # Train/val split
    random.shuffle(names)
    n_val = int(len(names) * args.val_frac)
    val_names = names[:n_val]
    train_names = names[n_val:]
    print(f"Train: {len(train_names):,}  Val: {len(val_names):,}")

    train_ds = NameDataset(train_names, tokenizer, max_len=args.max_length)
    val_ds = NameDataset(val_names, tokenizer, max_len=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model — match the finetune config
    model = GRUDecoder(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=1,
        dropout=0.1,
        max_length=args.max_length,
        encoder_dim=args.encoder_dim,
    ).to(device)
    print(f"Decoder params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss ignores padding
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Unconditional: use zero vector as "z". GRU's h0_proj will learn to output
    # a sensible initial hidden state from this zero vector (essentially learns a BOS
    # hidden state for LM use).
    z_zero = torch.zeros(args.batch_size, args.encoder_dim, device=device)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for inp, tgt in pbar:
            inp = inp.to(device)
            tgt = tgt.to(device)
            B = inp.size(0)
            z = z_zero[:B] if B <= z_zero.size(0) else torch.zeros(B, args.encoder_dim, device=device)

            # Forward through decoder with teacher forcing = 1.0
            logits = model(z, inp, teacher_forcing_ratio=1.0)
            # logits: (B, T, vocab)
            # target: (B, T)
            loss = criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=total_loss / n_batches)

        avg_train_loss = total_loss / max(1, n_batches)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device); tgt = tgt.to(device)
                B = inp.size(0)
                z = torch.zeros(B, args.encoder_dim, device=device)
                logits = model(z, inp, teacher_forcing_ratio=1.0)
                val_loss += criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1)).item()
                n_val_batches += 1
        avg_val_loss = val_loss / max(1, n_val_batches)
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  ppl={torch.exp(torch.tensor(avg_val_loss)).item():.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "vocab_size": vocab_size,
            }, args.output)
            print(f"  Saved best checkpoint to {args.output}")

    print(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
