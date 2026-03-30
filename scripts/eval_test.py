#!/usr/bin/env python3
"""
Evaluate the best checkpoint on train/val/test splits with full metrics:
Sub-token F1, Exact Match, N-gram Similarity, Edit Distance Similarity.

Uses greedy decode (argmax) for speed. Beam search would give ~3-5% higher scores.
"""
import argparse
import json
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import FunctionDataset
from src.evaluation.metrics import (
    compute_subtoken_f1,
    compute_char_ngram_similarity,
    compute_edit_distance_similarity,
)


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


def evaluate_split(model, loader, device, sp_model, split_name, use_amp=False):
    model.eval()
    results = []

    sos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {split_name}", leave=False):
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

            B = logits.shape[0]
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
                em = 1.0 if pred_name == true_name else 0.0
                ngsim = compute_char_ngram_similarity(pred_name, true_name)
                edsim = compute_edit_distance_similarity(pred_name, true_name)

                results.append({
                    'pred': pred_name,
                    'true': true_name,
                    'f1': f1,
                    'em': em,
                    'ngsim': ngsim,
                    'edsim': edsim,
                })

    n = len(results)
    avg_f1 = sum(r['f1'] for r in results) / max(n, 1)
    avg_em = sum(r['em'] for r in results) / max(n, 1)
    avg_ngsim = sum(r['ngsim'] for r in results) / max(n, 1)
    avg_edsim = sum(r['edsim'] for r in results) / max(n, 1)

    return {
        'split': split_name,
        'count': n,
        'f1': avg_f1,
        'em': avg_em,
        'ngsim': avg_ngsim,
        'edsim': avg_edsim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                        help='Splits to evaluate (default: val test)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    token_vocab = ckpt.get('token_vocab', None)
    ext_vocab = ckpt.get('ext_vocab', None)

    # Load name tokenizer (votes or BPE)
    votes_vocab_path = cfg['data'].get('votes_vocab_path', 'data/votes_vocab.json')
    if votes_vocab_path and os.path.exists(votes_vocab_path):
        from src.preprocessing.build_votes import VotesTokenizer
        sp_model = VotesTokenizer(vocab_path=votes_vocab_path)
        print(f"Votes vocab size: {sp_model.get_piece_size()}")
    else:
        import sentencepiece as spm
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(cfg['data']['bpe_model_path'])

    # String refs
    string_refs_dir = None
    string_vocab_path = None
    if cfg.get('string_encoder', {}).get('enabled', False):
        string_refs_dir = 'data/string_refs'
        string_vocab_path = 'data/string_refs/string_vocab.json'

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
        string_refs_dir=string_refs_dir,
        string_vocab_path=string_vocab_path,
    )

    # Get splits
    train_idx, val_idx, test_idx = dataset.get_splits(
        cfg['data']['train_split'], cfg['data']['val_split']
    )

    split_map = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
    }

    # Build model — use checkpoint sizes to match saved weights
    ckpt_state = ckpt['model_state_dict']
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    # Infer vocab sizes from checkpoint (handle disabled encoders)
    if 'ext_encoder.embedding.weight' in ckpt_state:
        cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = ckpt_state['decoder.embedding.weight'].shape[0]

    model = FunctionNamer(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Val F1: {ckpt.get('val_f1', '?'):.4f}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Evaluate each split
    all_results = {}
    for split_name in args.splits:
        if split_name not in split_map:
            print(f"Unknown split: {split_name}")
            continue
        idx = split_map[split_name]
        loader = DataLoader(
            Subset(dataset, idx),
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        result = evaluate_split(model, loader, device, sp_model, split_name, use_amp=args.amp)
        all_results[split_name] = result

    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Split':>8s} | {'Count':>6s} | {'F1':>6s} | {'EM':>6s} | {'NgSim':>6s} | {'EdSim':>6s}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for name, r in all_results.items():
        print(f"{r['split']:>8s} | {r['count']:>6d} | {r['f1']:>6.4f} | {r['em']:>6.4f} | {r['ngsim']:>6.4f} | {r['edsim']:>6.4f}")
    print(f"{'='*70}")

    # Save
    save_path = 'results/test_eval_metrics.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
