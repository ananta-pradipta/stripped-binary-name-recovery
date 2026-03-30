"""
Training loop for the FunctionNamer model.

Features:
  - Scheduled sampling (teacher forcing 1.0 → 0.5)
  - Label smoothing (0.1)
  - Cosine LR with warmup
"""
import argparse
import json
import os
import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import FunctionDataset
from src.evaluation.metrics import compute_subtoken_f1
from src.training.contrastive_sampler import ContrastiveBatchSampler


def collate_fn(batch):
    """Custom collate for variable-length sequences."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            if k == 'edge_index':
                edge_lists = []
                offset = 0
                for sample in batch:
                    ei = sample[k].clone()
                    ei = ei + offset
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


def get_teacher_forcing_ratio(epoch, num_epochs, start=1.0, end=0.5):
    """Scheduled sampling: linearly decay tf from start to end."""
    progress = epoch / max(num_epochs - 1, 1)
    return start - (start - end) * progress


def bucket_num_blocks(num_blocks_list, device):
    """Map num_blocks to 7 buckets: [1, 2, 3-4, 5-8, 9-16, 17-32, 33+]."""
    targets = []
    for n in num_blocks_list:
        if n <= 1: targets.append(0)
        elif n <= 2: targets.append(1)
        elif n <= 4: targets.append(2)
        elif n <= 8: targets.append(3)
        elif n <= 16: targets.append(4)
        elif n <= 32: targets.append(5)
        else: targets.append(6)
    return torch.tensor(targets, dtype=torch.long, device=device)


def compute_unlikelihood_loss(logits, targets, pad_id=0):
    """Compute token-level unlikelihood loss (Welleck et al. 2020).

    For each position, tokens that the model predicts (argmax) but that
    are NOT the ground truth target become "negative candidates".
    The loss penalizes high probability on these candidates:
        UL = -log(1 - p(negative_candidate))

    This directly combats mode collapse by pushing down probabilities
    of incorrectly over-predicted tokens.
    """
    B, T, V = logits.shape
    probs = torch.softmax(logits, dim=-1)  # (B, T, V)

    # Get model's greedy predictions
    preds = logits.argmax(dim=-1)  # (B, T)

    # Negative candidates: positions where pred != target and target != pad
    is_negative = (preds != targets) & (targets != pad_id)  # (B, T)

    if not is_negative.any():
        return torch.tensor(0.0, device=logits.device)

    # Gather the probability the model assigns to its own (wrong) predictions
    # at negative positions
    neg_pred_probs = probs.gather(2, preds.unsqueeze(-1)).squeeze(-1)  # (B, T)

    # Clamp probability to avoid numerical issues
    # Cap at 0.95 to prevent extreme gradients when model is very confident
    neg_pred_probs = neg_pred_probs.clamp(min=1e-8, max=0.95)

    # UL loss: -log(1 - p(wrong_prediction)) at negative positions
    ul_loss = -torch.log(1 - neg_pred_probs)

    # Only apply at negative positions
    ul_loss = (ul_loss * is_negative.float()).sum() / is_negative.float().sum()

    return ul_loss


def compute_contrastive_loss(z, names, temperature=0.1):
    """Compute NT-Xent contrastive loss on encoder embeddings.

    Functions with the same name (e.g., O0 and O2 of the same function)
    are positive pairs. All other functions in the batch are negatives.

    This teaches the encoder to produce similar embeddings for the same
    function at different optimization levels, and distinct embeddings
    for different functions — directly attacking the 0-ext-call degeneracy.

    Args:
        z: (B, D) encoder embeddings after fusion
        names: list of B function name strings
        temperature: scaling factor for similarity scores
    """
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    # L2-normalize embeddings
    z_norm = torch.nn.functional.normalize(z, dim=1)

    # Cosine similarity matrix (B, B)
    sim = torch.mm(z_norm, z_norm.t()) / temperature

    # Build positive mask: (i, j) is positive if names[i] == names[j] and i != j
    positive_mask = torch.zeros(B, B, dtype=torch.bool, device=z.device)
    name_to_indices = {}
    for i, name in enumerate(names):
        if name not in name_to_indices:
            name_to_indices[name] = []
        name_to_indices[name].append(i)

    has_positives = False
    for indices in name_to_indices.values():
        if len(indices) > 1:
            has_positives = True
            for i in indices:
                for j in indices:
                    if i != j:
                        positive_mask[i, j] = True

    if not has_positives:
        return torch.tensor(0.0, device=z.device)

    # For each sample with at least one positive, compute InfoNCE loss
    # L_i = -log(exp(sim(i, pos)) / sum_j!=i exp(sim(i, j)))
    # Mask out self-similarity
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)

    # For each row, compute log-softmax over all non-self entries
    log_softmax = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # Average the log-softmax values at positive positions
    pos_log_probs = (log_softmax * positive_mask.float()).sum()
    num_positives = positive_mask.float().sum()

    loss = -pos_log_probs / num_positives
    return loss


def train_epoch(model, loader, optimizer, criterion, device, tf_ratio=1.0,
                aux_loss_weight=0.0, ul_weight=0.0, contrastive_weight=0.0,
                ml_weight=0.0, scaler=None):
    model.train()
    total_loss = 0
    total_ul_loss = 0
    total_cl_loss = 0
    num_batches = 0
    use_amp = scaler is not None

    for batch in tqdm(loader, desc=f"Train (tf={tf_ratio:.2f})", leave=False):
        block_tokens = batch['block_tokens'].to(device)
        edge_index = batch['edge_index'].to(device)
        ext_call_ids = batch['ext_call_ids'].to(device)
        block_features = batch.get("block_features")
        if block_features is not None:
            block_features = block_features.to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)

        callee_tokens = batch.get('callee_tokens')
        if callee_tokens is not None:
            callee_tokens = callee_tokens.to(device)
        caller_tokens = batch.get('caller_tokens')
        if caller_tokens is not None:
            caller_tokens = caller_tokens.to(device)
        string_tokens = batch.get('string_tokens')
        if string_tokens is not None:
            string_tokens = string_tokens.to(device)
        binary_ext_ids = batch.get('binary_ext_ids')
        if binary_ext_ids is not None:
            binary_ext_ids = binary_ext_ids.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, gate_values, aux_logits, z_enc, ml_logits = model(
                block_tokens, edge_index, ext_call_ids,
                decoder_input, teacher_forcing_ratio=tf_ratio,
                block_features=block_features,
                callee_tokens=callee_tokens,
                caller_tokens=caller_tokens,
                string_tokens=string_tokens,
                binary_ext_ids=binary_ext_ids,
            )

            B, T, V = logits.shape
            target_T = decoder_target.shape[1]
            min_T = min(T, target_T)
            logits_trimmed = logits[:, :min_T, :]
            target_trimmed = decoder_target[:, :min_T]

            loss = criterion(
                logits_trimmed.contiguous().view(-1, V),
                target_trimmed.contiguous().view(-1)
            )

            # Unlikelihood loss: penalize over-predicted tokens
            if ul_weight > 0:
                ul_loss = compute_unlikelihood_loss(logits_trimmed, target_trimmed)
                loss = loss + ul_weight * ul_loss
                total_ul_loss += ul_loss.item()

            # Contrastive loss: pull same-name embeddings together
            if contrastive_weight > 0:
                names = batch['name']  # list of function name strings
                cl_loss = compute_contrastive_loss(z_enc, names)
                loss = loss + contrastive_weight * cl_loss
                total_cl_loss += cl_loss.item()

            # Multi-label loss: predict sub-token set (independent BCE)
            if ml_logits is not None and ml_weight > 0:
                # Build multi-hot target from decoder_target (vectorized)
                V_ml = ml_logits.shape[1]
                target_flat = target_trimmed.clamp(0, V_ml - 1)  # [B, T]
                ml_target = torch.zeros(B, V_ml, device=device)
                ml_target.scatter_(1, target_flat, 1.0)
                ml_target[:, 0] = 0  # zero out PAD position
                ml_loss = nn.functional.binary_cross_entropy_with_logits(
                    ml_logits, ml_target)
                loss = loss + ml_weight * ml_loss

            # Auxiliary loss: predict num_blocks bucket
            if aux_logits is not None and aux_loss_weight > 0:
                num_blocks = batch['num_blocks']
                aux_targets = bucket_num_blocks(num_blocks, device)
                aux_loss = nn.functional.cross_entropy(aux_logits, aux_targets)
                loss = loss + aux_loss_weight * aux_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, sp_model, use_amp=False):
    model.eval()
    total_loss = 0
    all_f1 = []
    num_batches = 0

    sos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()

    for batch in tqdm(loader, desc="Validating", leave=False):
        block_tokens = batch['block_tokens'].to(device)
        edge_index = batch['edge_index'].to(device)
        ext_call_ids = batch['ext_call_ids'].to(device)
        block_features = batch.get("block_features")
        if block_features is not None:
            block_features = block_features.to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)

        callee_tokens = batch.get('callee_tokens')
        if callee_tokens is not None:
            callee_tokens = callee_tokens.to(device)
        caller_tokens = batch.get('caller_tokens')
        if caller_tokens is not None:
            caller_tokens = caller_tokens.to(device)
        string_tokens = batch.get('string_tokens')
        if string_tokens is not None:
            string_tokens = string_tokens.to(device)
        binary_ext_ids = batch.get('binary_ext_ids')
        if binary_ext_ids is not None:
            binary_ext_ids = binary_ext_ids.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _, _, _, _ = model(
                block_tokens, edge_index, ext_call_ids,
                decoder_input, teacher_forcing_ratio=0.0,
                block_features=block_features,
                callee_tokens=callee_tokens,
                caller_tokens=caller_tokens,
                string_tokens=string_tokens,
                binary_ext_ids=binary_ext_ids,
            )

            B, T, V = logits.shape
            target_T = decoder_target.shape[1]
            min_T = min(T, target_T)
            logits_trimmed = logits[:, :min_T, :]
            target_trimmed = decoder_target[:, :min_T]

            loss = criterion(
                logits_trimmed.contiguous().view(-1, V),
                target_trimmed.contiguous().view(-1)
            )
        total_loss += loss.item()
        num_batches += 1

        pred_ids = logits.argmax(dim=-1)
        for i in range(B):
            pred_tokens = []
            for t in pred_ids[i]:
                tid = t.item()
                if tid == eos_id:
                    break
                if tid != sos_id and tid != 0:
                    pred_tokens.append(tid)

            target_tokens = []
            for t in decoder_target[i]:
                tid = t.item()
                if tid == eos_id:
                    break
                if tid != 0:
                    target_tokens.append(tid)

            pred_name = sp_model.decode(pred_tokens)
            true_name = sp_model.decode(target_tokens)
            f1 = compute_subtoken_f1(pred_name, true_name)
            all_f1.append(f1)

    avg_loss = total_loss / max(num_batches, 1)
    avg_f1 = sum(all_f1) / max(len(all_f1), 1)
    return avg_loss, avg_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--enrich-callees', action='store_true',
                        help='Enable callee name enrichment (Phase 2 training)')
    parser.add_argument('--callee-dropout', type=float, default=0.1,
                        help='Fraction of callee enrichments to drop for robustness')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint path')
    parser.add_argument('--ul-weight', type=float, default=0.0,
                        help='Unlikelihood loss weight (0=disabled, try 0.5-2.0)')
    parser.add_argument('--contrastive-weight', type=float, default=0.0,
                        help='Contrastive loss weight (0=disabled, try 0.1-1.0)')
    parser.add_argument('--ml-weight', type=float, default=0.0,
                        help='Multi-label loss weight (0=disabled, try 0.5-2.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--pretrained-encoder', type=str, default=None,
                        help='Path to pretrained encoder weights (from pretrain.py)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training (AMP) for ~2x speedup on A100')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers (0=main thread, try 4 on HPC)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config (e.g. 64 or 128 on A100 40GB)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override batch size from CLI if specified
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size

    # Set random seed for reproducibility
    if args.seed is not None:
        import random
        import numpy as np
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        print(f"Random seed: {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load name tokenizer (votes or BPE)
    votes_vocab_path = cfg['data'].get('votes_vocab_path')
    if votes_vocab_path and os.path.exists(votes_vocab_path):
        from src.preprocessing.build_votes import VotesTokenizer
        sp_model = VotesTokenizer(vocab_path=votes_vocab_path)
        print(f"Votes vocab size (actual): {sp_model.get_piece_size()}")
    else:
        import sentencepiece as spm
        sp_model = spm.SentencePieceProcessor(model_file=cfg['data']['bpe_model_path'])
        print(f"BPE vocab size (actual): {sp_model.get_piece_size()}")
    actual_name_vocab_size = sp_model.get_piece_size()

    with open(cfg['data']['external_vocab_path']) as f:
        ext_vocab_data = json.load(f)
    actual_ext_vocab_size = ext_vocab_data['vocab_size']
    print(f"External vocab size (actual): {actual_ext_vocab_size}")

    # String refs (enabled via config)
    string_refs_dir = None
    string_vocab_path = None
    if cfg.get('string_encoder', {}).get('enabled', False):
        string_refs_dir = 'data/string_refs'
        string_vocab_path = 'data/string_refs/string_vocab.json'

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
        enrich_callees=args.enrich_callees,
        callee_dropout=args.callee_dropout,
        string_refs_dir=string_refs_dir,
        string_vocab_path=string_vocab_path,
    )

    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return

    train_idx, val_idx, test_idx = dataset.get_splits(
        cfg['data']['train_split'], cfg['data']['val_split']
    )

    if len(train_idx) == 0 or len(val_idx) == 0:
        print(f"ERROR: Empty split! Train: {len(train_idx)}, Val: {len(val_idx)}")
        return

    if args.contrastive_weight > 0:
        # Use contrastive batch sampler for dense positive pairs
        contrastive_sampler = ContrastiveBatchSampler(
            dataset, train_idx,
            batch_size=cfg['training']['batch_size'],
            pair_count=32,
        )
        train_loader = DataLoader(
            dataset, batch_sampler=contrastive_sampler,
            collate_fn=collate_fn, num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        print(f"Using contrastive sampler: ~32 pairs/batch, "
              f"{len(contrastive_sampler.pair_names)} pairable names")
    else:
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=cfg['training']['batch_size'],
            shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=cfg['training']['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    # Override config with actual vocab sizes
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['external_encoder']['vocab_size'] = actual_ext_vocab_size
    cfg['decoder']['bpe_vocab_size'] = actual_name_vocab_size
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']

    print(f"Token vocab: {len(dataset.token_vocab)}")
    print(f"Ext vocab: {actual_ext_vocab_size}")
    print(f"Name vocab: {actual_name_vocab_size}")

    model = FunctionNamer(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Load pre-trained encoder weights (block_encoder + graph_encoder only)
    if args.pretrained_encoder:
        pretrain_ckpt = torch.load(args.pretrained_encoder, map_location=device,
                                   weights_only=True)
        encoder_state = pretrain_ckpt['encoder_state_dict']
        model_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []
        partial_keys = []
        for k, v in encoder_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded_keys.append(k)
            elif k in model_state and 'embedding.weight' in k and v.ndim == 2:
                # Partial embedding transfer: copy rows for overlapping tokens
                pretrain_vocab = pretrain_ckpt.get('token_vocab', None)
                if pretrain_vocab is not None:
                    current_vocab = dataset.token_vocab
                    # Build reverse map: token_name -> pretrain_id
                    pretrain_name2id = pretrain_vocab
                    copied = 0
                    for token_name, current_id in current_vocab.items():
                        if token_name in pretrain_name2id:
                            pretrain_id = pretrain_name2id[token_name]
                            if pretrain_id < v.shape[0] and current_id < model_state[k].shape[0]:
                                model_state[k][current_id] = v[pretrain_id]
                                copied += 1
                    partial_keys.append(k)
                    print(f"  Partial embedding transfer for {k}: "
                          f"copied {copied}/{model_state[k].shape[0]} rows "
                          f"({100*copied/model_state[k].shape[0]:.1f}%)")
                else:
                    skipped_keys.append(k)
            else:
                skipped_keys.append(k)
        model.load_state_dict(model_state)
        print(f"Loaded pretrained encoder: {len(loaded_keys)} exact + "
              f"{len(partial_keys)} partial, {len(skipped_keys)} skipped")
        if skipped_keys:
            print(f"  Skipped: {skipped_keys[:5]}{'...' if len(skipped_keys) > 5 else ''}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=0.1,
    )

    num_epochs = cfg['training']['epochs']
    warmup_epochs = min(5, num_epochs // 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1 = 0
    patience_counter = 0
    start_epoch = 0
    ckpt_dir = cfg['training']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    patience = cfg['training']['early_stopping_patience']

    # Resume from checkpoint if specified
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt['val_f1']
        # Advance scheduler to the correct epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}, best Val F1: {best_f1:.4f}")

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if args.amp:
        print(f"Mixed precision (AMP): ENABLED")
    if args.num_workers > 0:
        print(f"Data loading workers: {args.num_workers}")
    if args.batch_size is not None:
        print(f"Batch size override: {args.batch_size}")

    if args.enrich_callees:
        print(f"Callee enrichment: ENABLED (dropout={args.callee_dropout})")

    print(f"\nTraining: {num_epochs} epochs, patience={patience}")
    print(f"Batch size: {cfg['training']['batch_size']}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"Train batches/epoch: {len(train_idx) // cfg['training']['batch_size'] + 1}")
    ul_weight = args.ul_weight
    cl_weight = args.contrastive_weight
    print(f"Scheduled sampling: tf 1.0 → 0.5 over {num_epochs} epochs")
    print(f"Label smoothing: 0.1")
    if ul_weight > 0:
        print(f"Unlikelihood loss weight: {ul_weight}")
    if cl_weight > 0:
        print(f"Contrastive loss weight: {cl_weight}")
    ml_weight = args.ml_weight
    if ml_weight > 0:
        print(f"Multi-label loss weight: {ml_weight}")
    print()

    import time as _time
    training_start = _time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = _time.time()
        tf_ratio = get_teacher_forcing_ratio(epoch, num_epochs, start=1.0, end=0.3)
        current_lr = optimizer.param_groups[0]['lr']

        aux_weight = cfg['training'].get('aux_loss_weight', 0.0)
        dataset.training_mode = True  # Enable callee enrichment for training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, tf_ratio,
                                 aux_loss_weight=aux_weight, ul_weight=ul_weight,
                                 contrastive_weight=cl_weight, ml_weight=ml_weight,
                                 scaler=scaler)
        dataset.training_mode = False  # Disable enrichment for validation
        val_loss, val_f1 = validate(model, val_loader, criterion, device, sp_model,
                                    use_amp=args.amp)

        scheduler.step()

        epoch_time = _time.time() - epoch_start
        elapsed = _time.time() - training_start
        remaining = epoch_time * (num_epochs - epoch - 1)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"TF={tf_ratio:.2f} | LR={current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
              f"{epoch_time:.0f}s/ep | "
              f"Elapsed: {elapsed/60:.0f}m | ETA: {remaining/60:.0f}m", end="")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': cfg,
                'token_vocab': dataset.token_vocab,
                'ext_vocab': dataset.ext_vocab,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f" ★ New best!")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*60}")
    print(f"Training complete. Best Val F1: {best_f1:.4f}")
    print(f"Checkpoint: {os.path.join(ckpt_dir, 'best_model.pt')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
