"""
Pre-training script for block encoder + graph encoder.

Self-supervised objectives:
  1. MLM: reconstruct masked instruction-type tokens (teaches token semantics)
  2. Contrastive (NT-Xent): align embeddings of same function across opt levels
     (teaches optimization-invariant representations)

Combined loss: L = L_mlm + contrastive_weight * L_contrastive

Saves encoder weights to checkpoints/pretrained_encoder.pt for loading into
the full FunctionNamer model via --pretrained-encoder flag in train.py.

Usage:
    python3 -m src.training.pretrain --config configs/pretrain.yaml [--seed 42]
"""
import argparse
import json
import math
import os
import random
import time
from functools import partial

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.pretrain_heads import PretrainModel
from src.training.pretrain_dataset import PretrainDataset, pretrain_collate_fn


def train_epoch(model, loader, optimizer, device, contrastive_weight=0.5):
    """Run one pre-training epoch.

    Returns:
        dict with 'total_loss', 'mlm_loss', 'contrastive_loss', 'mlm_accuracy'
    """
    model.train()
    total_loss = 0
    total_mlm = 0
    total_cl = 0
    total_mlm_correct = 0
    total_mlm_count = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Pre-train", leave=False):
        block_tokens = batch['block_tokens'].to(device)
        original_tokens = batch['original_tokens'].to(device)
        mask_positions = batch['mask_positions'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_vec = batch['batch_vec'].to(device)

        # Contrastive pair data (may be None)
        pair_block_tokens = batch['pair_block_tokens']
        pair_edge_index = batch['pair_edge_index']
        pair_batch_vec = batch['pair_batch_vec']
        anchor_indices = batch['anchor_indices']

        # For contrastive: select only anchor samples that have pairs
        # and re-index them to match pair samples
        kwargs = {
            'block_tokens': block_tokens,
            'edge_index': edge_index,
            'batch_vec': batch_vec,
            'mask_positions': mask_positions,
            'original_tokens': original_tokens,
        }

        if pair_block_tokens is not None and len(anchor_indices) > 0:
            # We need to pass the full batch for MLM, but for contrastive
            # we need to extract the anchor embeddings after the forward pass.
            # Instead, we pass pair data and let the model handle it.
            # The model will compute contrastive loss between the anchor
            # graph embeddings and pair graph embeddings.
            #
            # However, the model's forward() expects paired inputs of same batch size.
            # We need to select anchor samples from the full batch to match pairs.
            #
            # Solution: extract anchor block_tokens and edge_index for just the
            # anchors that have pairs, then run contrastive on those.
            anchor_bt = block_tokens[anchor_indices]  # [P, N, T]
            B_full = block_tokens.shape[0]
            N = block_tokens.shape[1]

            # Rebuild edge_index for anchor subset
            anchor_edge_lists = []
            for new_idx, orig_idx in enumerate(anchor_indices):
                # Extract edges for this sample from the batched edge_index
                lo = orig_idx * N
                hi = lo + N
                mask = (edge_index[0] >= lo) & (edge_index[0] < hi)
                ei = edge_index[:, mask] - lo + new_idx * N
                anchor_edge_lists.append(ei)
            if anchor_edge_lists:
                anchor_ei = torch.cat(anchor_edge_lists, dim=1)
            else:
                anchor_ei = torch.zeros(2, 0, dtype=torch.long, device=device)

            P = len(anchor_indices)
            anchor_bv = torch.arange(P, device=device).repeat_interleave(N)

            kwargs['pair_block_tokens'] = pair_block_tokens.to(device)
            kwargs['pair_edge_index'] = pair_edge_index.to(device)
            kwargs['pair_batch_vec'] = pair_batch_vec.to(device)

            # Override the anchor block_tokens for contrastive
            # But we still want MLM on the full batch — so we do it in two parts.
            # Actually, the model forward already handles both MLM and contrastive.
            # The issue is the contrastive head needs anchors that correspond 1:1
            # with pairs. Let's do a simpler approach: run full forward for MLM,
            # then run a separate contrastive forward on anchors + pairs.

            # Full forward for MLM (no contrastive pairs)
            outputs = model(
                block_tokens=block_tokens,
                edge_index=edge_index,
                batch_vec=batch_vec,
                mask_positions=mask_positions,
                original_tokens=original_tokens,
            )
            mlm_loss = outputs['mlm_loss']

            # Separate contrastive forward on anchor subset
            # We need graph embeddings for anchors and their pairs
            with torch.no_grad():
                # We'll compute contrastive inline to avoid double-encoding
                pass

            # Actually, let's just use the model's contrastive path directly
            # by calling forward with the anchor subset
            cl_outputs = model(
                block_tokens=anchor_bt,
                edge_index=anchor_ei,
                batch_vec=anchor_bv,
                pair_block_tokens=pair_block_tokens.to(device),
                pair_edge_index=pair_edge_index.to(device),
                pair_batch_vec=pair_batch_vec.to(device),
            )
            cl_loss = cl_outputs['contrastive_loss']

        else:
            outputs = model(**kwargs)
            mlm_loss = outputs['mlm_loss']
            cl_loss = outputs['contrastive_loss']

        loss = mlm_loss + contrastive_weight * cl_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mlm += mlm_loss.item()
        total_cl += cl_loss.item()

        # MLM accuracy tracking
        if outputs.get('mlm_logits') is not None and outputs['mlm_logits'] is not None:
            mlm_logits = outputs['mlm_logits']
            mlm_preds = mlm_logits.argmax(dim=-1)
            B_tok, N_tok, T_tok = original_tokens.shape
            targets_flat = original_tokens.view(B_tok * N_tok, T_tok)
            mask_flat = mask_positions.view(B_tok * N_tok, T_tok)
            mlm_targets = targets_flat[mask_flat]
            total_mlm_correct += (mlm_preds == mlm_targets).sum().item()
            total_mlm_count += mlm_targets.shape[0]

        num_batches += 1

    n = max(num_batches, 1)
    mlm_acc = total_mlm_correct / max(total_mlm_count, 1)
    return {
        'total_loss': total_loss / n,
        'mlm_loss': total_mlm / n,
        'contrastive_loss': total_cl / n,
        'mlm_accuracy': mlm_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='Pre-train block + graph encoders')
    parser.add_argument('--config', default='configs/pretrain.yaml',
                        help='Path to pre-training config')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output path for pretrained encoder weights')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    torch.backends.cudnn.deterministic = True
    print(f"Random seed: {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build token vocabulary from match_index (same logic as build_dataset.py)
    # We need to scan all graphs to build the vocab
    match_index_path = cfg['data'].get('match_index_path', 'data/match_index.json')
    graphs_dir = cfg['data']['graphs_dir']

    token_vocab = _build_token_vocab(match_index_path, cfg)
    print(f"Token vocab: {len(token_vocab)} types")

    # Override config with actual vocab size (+ MASK token)
    mask_token_id = max(token_vocab.values()) + 1
    cfg['block_encoder']['token_vocab_size'] = mask_token_id + 1

    # Pre-training dataset
    mlm_cfg = cfg.get('mlm', {})
    pairs_path = cfg['data'].get('pretrain_pairs_path', 'data/pretrain_pairs.json')

    dataset = PretrainDataset(
        graphs_dir=graphs_dir,
        match_index_path=match_index_path,
        token_vocab=token_vocab,
        max_blocks=cfg['data'].get('max_blocks_per_function', 30),
        max_tokens=cfg['data'].get('max_tokens_per_block', 20),
        mask_prob=mlm_cfg.get('mask_prob', 0.15),
        mask_token_ratio=mlm_cfg.get('mask_token_ratio', 0.8),
        random_token_ratio=mlm_cfg.get('random_token_ratio', 0.1),
        pairs_path=pairs_path if os.path.exists(pairs_path) else None,
    )

    # Create collate function with dataset reference
    collate = partial(pretrain_collate_fn, dataset=dataset)

    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=True,
    )

    # Build model
    model = PretrainModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PretrainModel parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training'].get('weight_decay', 0.01),
    )

    # Cosine LR scheduler with warmup
    num_epochs = cfg['training']['epochs']
    warmup_steps = cfg['training'].get('warmup_steps', 500)
    total_steps = len(loader) * num_epochs
    warmup_steps = min(warmup_steps, total_steps // 4)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    contrastive_cfg = cfg.get('contrastive', {})
    contrastive_weight = contrastive_cfg.get('weight', 0.5)

    output_path = args.output or cfg['training'].get(
        'output_path', 'checkpoints/pretrained_encoder.pt')

    print(f"\nPre-training config:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {cfg['training']['batch_size']}")
    print(f"  LR: {cfg['training']['learning_rate']}")
    print(f"  MLM mask prob: {mlm_cfg.get('mask_prob', 0.15)}")
    print(f"  Contrastive weight: {contrastive_weight}")
    print(f"  Contrastive temperature: {contrastive_cfg.get('temperature', 0.07)}")
    print(f"  Output: {output_path}")
    print()

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        metrics = train_epoch(model, loader, optimizer, device, contrastive_weight)

        # Step scheduler per batch (approximate: step once per epoch * batches)
        for _ in range(len(loader)):
            scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} "
              f"[{elapsed/60:.1f}min] "
              f"loss={metrics['total_loss']:.4f} "
              f"mlm={metrics['mlm_loss']:.4f} "
              f"cl={metrics['contrastive_loss']:.4f} "
              f"mlm_acc={metrics['mlm_accuracy']:.3f} "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # Save best checkpoint
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            encoder_state = model.get_encoder_state_dict()
            save_dict = {
                'encoder_state_dict': encoder_state,
                'token_vocab': token_vocab,
                'mask_token_id': mask_token_id,
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'mlm_accuracy': metrics['mlm_accuracy'],
                'config': cfg,
            }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(save_dict, output_path)
            print(f"  -> Saved best encoder (loss={best_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\nPre-training complete in {total_time/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Encoder saved to: {output_path}")


def _build_token_vocab(match_index_path: str, cfg: dict) -> dict:
    """Build token vocabulary by scanning all graph files.

    Replicates the vocab-building logic from build_dataset.py for consistency.
    Uses deterministic sort (-count, name) to ensure reproducible token IDs.
    """
    from collections import Counter

    with open(match_index_path) as f:
        match_index = json.load(f)

    token_counter = Counter()
    scanned = 0
    for graph_path, info in match_index.items():
        if not os.path.exists(graph_path):
            continue
        with open(graph_path) as f:
            graph = json.load(f)
        for block in graph.get('blocks', []):
            for tok in block.get('tokens', []):
                token_counter[tok] += 1
        scanned += 1

    print(f"Scanned {scanned} graph files for token vocab")

    # Build vocab with deterministic sort (same as build_dataset.py)
    max_vocab = cfg['block_encoder'].get('token_vocab_size', 3000)
    token_vocab = {'<PAD>': 0, '<UNK>': 1}
    for tok, count in sorted(token_counter.items(), key=lambda x: (-x[1], x[0])):
        if len(token_vocab) >= max_vocab:
            break
        token_vocab[tok] = len(token_vocab)

    return token_vocab


if __name__ == '__main__':
    main()
