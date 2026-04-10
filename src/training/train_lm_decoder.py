"""
Train the LM decoder on top of a frozen encoder.

Strategy:
  1. Load pre-trained encoder checkpoint (block encoder + GAT + fusion)
  2. Freeze encoder weights
  3. Attach LM decoder (GPT-2 or CodeGen) with projection layer
  4. Train only the projection layer (LM is also frozen)
  5. The projection learns to map binary embeddings → LM's "understanding space"

This is essentially "prefix tuning" — we train soft prompts that condition
the LM to generate function names based on binary code embeddings.
"""
import argparse
import json
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.function_namer import FunctionNamer
from src.models.lm_decoder import LMDecoderLight
from src.preprocessing.build_dataset import FunctionDataset

def build_decoder(args, fusion_dim, device):
    """Build the appropriate LM decoder based on args."""
    if args.use_lora:
        from src.models.lora_decoder import LoRADecoder
        decoder = LoRADecoder(
            encoder_dim=fusion_dim,
            model_name=args.lm_model,
            max_length=20,
            num_prefix_tokens=args.num_prefix_tokens,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r * 2,
        ).to(device)
    else:
        decoder = LMDecoderLight(
            encoder_dim=fusion_dim,
            model_name=args.lm_model,
            max_length=20,
            num_prefix_tokens=args.num_prefix_tokens,
        ).to(device)
    return decoder


def collate_fn(batch):
    """Custom collate for variable-length sequences."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            if k == 'edge_index':
                edge_lists, offset = [], 0
                for sample in batch:
                    ei = sample[k].clone() + offset
                    edge_lists.append(ei)
                    offset += sample['num_blocks']
                result[k] = torch.cat(edge_lists, dim=1)
            elif k in ('decoder_input', 'decoder_target', 'ext_call_ids'):
                max_len = max(s[k].size(0) for s in batch)
                padded = torch.zeros(len(batch), max_len, dtype=batch[0][k].dtype)
                for i, s in enumerate(batch):
                    padded[i, :s[k].size(0)] = s[k]
                result[k] = padded
            else:
                result[k] = torch.stack([s[k] for s in batch])
        elif k == 'block_features':
            result[k] = torch.stack([s[k] for s in batch])
        elif k == 'num_blocks':
            result[k] = [s[k] for s in batch]
        else:
            result[k] = [s[k] for s in batch]
    return result


def extract_embeddings(model, batch, device, use_amp=False, with_grad=False):
    """Extract encoder embeddings z from a batch.

    Args:
        with_grad: If True, keep gradients (for unfrozen encoder training)
    """
    block_tokens = batch['block_tokens'].to(device)
    edge_index = batch['edge_index'].to(device)
    ext_call_ids = batch['ext_call_ids'].to(device)
    block_features = batch.get("block_features")
    if block_features is not None:
        block_features = block_features.to(device)
    callee_tokens = batch.get('callee_tokens')
    if callee_tokens is not None:
        callee_tokens = callee_tokens.to(device)
    caller_tokens = batch.get('caller_tokens')
    if caller_tokens is not None:
        caller_tokens = caller_tokens.to(device)

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        with torch.amp.autocast('cuda', enabled=use_amp):
            block_embs = model.block_encoder(block_tokens, block_features=block_features)
            B, N, D = block_embs.shape
            x = block_embs.view(B * N, D)
            batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
            f, _, _ = model.graph_encoder(x, edge_index, batch_vec)

            z = f
            if model.ext_encoder_enabled and model.fusion is not None:
                c = model.ext_encoder(ext_call_ids)
                has_ext = model._compute_has_ext_calls(ext_call_ids)
                z, _ = model.fusion(z, c, has_ext_calls=has_ext)

            if model.callee_encoder_enabled and callee_tokens is not None:
                callee_ctx, has_callees = model.callee_encoder(callee_tokens)
                g = model.callee_gate(torch.cat([z, callee_ctx], dim=1))
                z_fused = g * z + (1 - g) * callee_ctx
                mask = has_callees.unsqueeze(1).float()
                z = mask * z_fused + (1 - mask) * z

            if model.caller_encoder_enabled and caller_tokens is not None:
                caller_ctx, has_callers = model.caller_encoder(caller_tokens)
                g = model.caller_gate(torch.cat([z, caller_ctx], dim=1))
                z_fused = g * z + (1 - g) * caller_ctx
                mask = has_callers.unsqueeze(1).float()
                z = mask * z_fused + (1 - mask) * z

    return z  # (B, fusion_dim)


def get_true_names(batch, sp_model):
    """Extract ground truth function names from batch."""
    decoder_target = batch['decoder_target']
    eos_id = sp_model.eos_id()
    names = []
    for i in range(decoder_target.shape[0]):
        tokens = []
        for t in decoder_target[i]:
            tid = t.item()
            if tid == eos_id:
                break
            if tid != 0:
                tokens.append(tid)
        name = sp_model.decode(tokens)
        names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(description='Train LM Decoder')
    parser.add_argument('encoder_checkpoint', help='Path to pre-trained encoder checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--lm-model', default='gpt2', help='Pre-trained LM (gpt2 or Salesforce/codegen-350M-mono)')
    parser.add_argument('--num-prefix-tokens', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--encoder-lr', type=float, default=None,
                        help='LR for encoder (if unfreezing). Default: None (frozen)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unfreeze-encoder', action='store_true',
                        help='Unfreeze encoder for joint training')
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA-adapted LM instead of prefix tuning')
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--save', default='checkpoints/lm_decoder.pt')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load encoder checkpoint
    print(f"Loading encoder: {args.encoder_checkpoint}")
    ckpt = torch.load(args.encoder_checkpoint, map_location=device)
    token_vocab = ckpt.get('token_vocab', None)
    ext_vocab = ckpt.get('ext_vocab', None)

    # Load name tokenizer (for extracting ground truth names)
    votes_vocab_path = cfg['data'].get('votes_vocab_path', 'data/votes_vocab.json')
    if votes_vocab_path and os.path.exists(votes_vocab_path):
        from src.preprocessing.build_votes import VotesTokenizer
        sp_model = VotesTokenizer(vocab_path=votes_vocab_path)
        print(f"Votes vocab size: {sp_model.get_piece_size()}")
    else:
        import sentencepiece as spm
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(cfg['data']['bpe_model_path'])

    # Load dataset
    dataset = FunctionDataset(
        graphs_dir=cfg['data']['graphs_dir'],
        labels_dir=cfg['data']['labels_dir'],
        external_calls_dir=cfg['data']['external_calls_dir'],
        bpe_model_path=cfg['data']['bpe_model_path'],
        external_vocab_path=cfg['data']['external_vocab_path'],
        max_blocks=cfg['data']['max_blocks_per_function'],
        max_tokens=cfg['data']['max_tokens_per_block'],
        max_name_len=cfg['data']['max_name_length'],
        votes_vocab_path=votes_vocab_path,
        token_vocab=token_vocab,
        ext_vocab_override=ext_vocab,
    )
    train_idx, val_idx, test_idx = dataset.get_splits(
        cfg['data']['train_split'], cfg['data']['val_split']
    )

    # Build encoder model
    ckpt_state = ckpt['model_state_dict']
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    if 'ext_encoder.embedding.weight' in ckpt_state:
        cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = ckpt_state['decoder.embedding.weight'].shape[0]

    encoder = FunctionNamer(cfg).to(device)
    encoder.load_state_dict(ckpt['model_state_dict'])

    if args.unfreeze_encoder:
        # Unfreeze encoder for joint training (slow LR)
        encoder.train()
        print(f"Encoder loaded (UNFROZEN): {sum(p.numel() for p in encoder.parameters()):,} params")
    else:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        print(f"Encoder loaded (frozen): {sum(p.numel() for p in encoder.parameters()):,} params")

    fusion_dim = cfg['fusion']['input_dim']

    # Build LM decoder
    print(f"Loading LM decoder: {args.lm_model} (LoRA={args.use_lora})")
    lm_decoder = build_decoder(args, fusion_dim, device)

    trainable_params = sum(p.numel() for p in lm_decoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lm_decoder.parameters())
    print(f"LM decoder: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer — train projection + LM layers, optionally encoder
    param_groups = [
        {'params': [p for p in lm_decoder.parameters() if p.requires_grad],
         'lr': args.lr},
    ]
    if args.unfreeze_encoder:
        encoder_lr = args.encoder_lr or args.lr * 0.01  # 100x slower for encoder
        param_groups.append({
            'params': [p for p in encoder.parameters() if p.requires_grad],
            'lr': encoder_lr,
        })
        print(f"Encoder LR: {encoder_lr}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Data loaders
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # Training loop
    best_val_loss = float('inf')
    print(f"\nTraining: {args.epochs} epochs")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    print(f"Batches/epoch: {len(train_loader)}")

    for epoch in range(args.epochs):
        lm_decoder.train()
        epoch_loss = 0
        n_batches = 0
        start = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            # Extract encoder embeddings
            z = extract_embeddings(encoder, batch, device, use_amp=args.amp,
                                   with_grad=args.unfreeze_encoder)

            # Get ground truth names
            true_names = get_true_names(batch, sp_model)

            # Forward through LM decoder
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    z_input = z if args.unfreeze_encoder else z.detach()
                    loss, _ = lm_decoder(z_input, target_names=true_names)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                all_params = list(lm_decoder.parameters())
                if args.unfreeze_encoder:
                    all_params += list(encoder.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                z_input = z if args.unfreeze_encoder else z.detach()
                loss, _ = lm_decoder(z_input, target_names=true_names)
                loss.backward()
                all_params = list(lm_decoder.parameters())
                if args.unfreeze_encoder:
                    all_params += list(encoder.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - start

        # Validation
        lm_decoder.eval()
        val_loss = 0
        val_batches = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                z = extract_embeddings(encoder, batch, device, use_amp=args.amp)
                true_names = get_true_names(batch, sp_model)

                with torch.amp.autocast('cuda', enabled=bool(scaler)):
                    loss, _ = lm_decoder(z.detach(), target_names=true_names)

                val_loss += loss.item()
                val_batches += 1

                # Quick EM check on a few samples
                if val_batches <= 5:
                    for i in range(min(4, z.shape[0])):
                        pred_name, _ = lm_decoder.generate(z[i:i+1].detach())
                        if pred_name == true_names[i]:
                            val_correct += 1
                        val_total += 1

        avg_val_loss = val_loss / val_batches
        val_em = val_correct / max(val_total, 1)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val EM (sample): {val_em:.1%} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'lm_decoder_state': lm_decoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': {
                    'lm_model': args.lm_model,
                    'num_prefix_tokens': args.num_prefix_tokens,
                    'encoder_dim': fusion_dim,
                },
                'encoder_checkpoint': args.encoder_checkpoint,
            }, args.save)
            print(f"  ★ New best! Saved to {args.save}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
