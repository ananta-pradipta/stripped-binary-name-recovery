#!/usr/bin/env python3
"""
Error Analysis: Categorize prediction failures for decoder and k-NN (k=1).

Categories:
  1. Correct      — exact match
  2. Close        — wrong but EdSim > 0.7
  3. Phantom      — predicted name not in any training function name
  4. Wrong-seen   — predicted name exists in training but wrong
  5. Wrong-unseen — true name never appeared in training (unsolvable)

Usage:
  python3 scripts/error_analysis.py checkpoints/best_model.pt --config configs/optimized_large.yaml --amp --demo
"""
import argparse
import json
import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import Counter, defaultdict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'scripts'))

from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import (
    FunctionDataset, compute_block_features, compute_block_degrees, NUM_BLOCK_FEATURES
)
from src.evaluation.metrics import (
    compute_subtoken_f1,
    compute_exact_match,
    compute_char_ngram_similarity,
    compute_edit_distance_similarity,
)

# Reuse collate, embedding extraction, demo loading, and k-NN from ablation script
from eval_knn_ablation import (
    collate_fn,
    extract_embeddings_and_predict,
    build_knn_index,
    batch_knn_lookup,
    extract_demo_data,
    DEMO_PACKAGES,
)


# ---------------------------------------------------------------------------
# Error categorization
# ---------------------------------------------------------------------------
CATEGORIES = ['correct', 'close', 'phantom', 'wrong_seen', 'wrong_unseen_target']


def categorize_prediction(pred_name, true_name, train_name_set):
    """Categorize a single prediction into one of 5 categories.

    Returns (category, edsim).
    """
    edsim = compute_edit_distance_similarity(pred_name, true_name)

    if pred_name == true_name:
        return 'correct', edsim

    # True name was never in training — unsolvable
    if true_name not in train_name_set:
        return 'wrong_unseen_target', edsim

    # Close: wrong but EdSim > 0.7
    if edsim > 0.7:
        return 'close', edsim

    # Phantom: predicted name doesn't exist in any training function
    if pred_name not in train_name_set:
        return 'phantom', edsim

    # Wrong-seen: predicted name is a real training name, just wrong
    return 'wrong_seen', edsim


def count_ext_calls(ext_call_ids):
    """Count non-padding external calls in a tensor."""
    if isinstance(ext_call_ids, torch.Tensor):
        return int((ext_call_ids > 0).sum().item())
    return 0


def ext_bucket(n_ext):
    """Map ext call count to bucket label."""
    if n_ext == 0:
        return '0 ext'
    elif n_ext <= 3:
        return '1-3 ext'
    else:
        return '4+ ext'


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def run_analysis(preds, embeddings, train_normed, train_names, train_name_set,
                 ext_counts, split_name):
    """Run error analysis for both decoder and k-NN on a single split.

    Args:
        preds: list of dicts with 'pred_name', 'true_name', 'beam_score'
        embeddings: numpy array [N, D]
        train_normed: normalized training embeddings
        train_names: list of training function names
        train_name_set: set of all unique training function names
        ext_counts: list of ext call counts per function (or None for demo)
        split_name: 'test' or 'demo'

    Returns:
        dict with analysis results
    """
    n = len(preds)
    print(f"\n{'='*80}")
    print(f"ERROR ANALYSIS: {split_name.upper()} ({n} functions)")
    print(f"{'='*80}")

    # k-NN k=1 predictions
    knn_names = batch_knn_lookup(embeddings, train_normed, train_names, k=1)

    # Categorize each prediction
    decoder_cats = []
    knn_cats = []
    decoder_details = []
    knn_details = []

    for i in range(n):
        pred = preds[i]
        true_name = pred['true_name']
        dec_name = pred['pred_name']
        knn_name = knn_names[i]
        n_ext = ext_counts[i] if ext_counts is not None else -1
        bucket = ext_bucket(n_ext) if n_ext >= 0 else 'unknown'

        dec_cat, dec_edsim = categorize_prediction(dec_name, true_name, train_name_set)
        knn_cat, knn_edsim = categorize_prediction(knn_name, true_name, train_name_set)

        decoder_cats.append(dec_cat)
        knn_cats.append(knn_cat)

        decoder_details.append({
            'true_name': true_name,
            'pred_name': dec_name,
            'category': dec_cat,
            'edsim': dec_edsim,
            'ext_bucket': bucket,
        })
        knn_details.append({
            'true_name': true_name,
            'pred_name': knn_name,
            'category': knn_cat,
            'edsim': knn_edsim,
            'ext_bucket': bucket,
        })

    # --- Category counts ---
    dec_counts = Counter(decoder_cats)
    knn_counts = Counter(knn_cats)

    print(f"\n## Category Breakdown\n")
    print(f"| {'Category':<22} | {'Decoder':>10} | {'Decoder%':>9} | {'k-NN(k=1)':>10} | {'k-NN%':>9} |")
    print(f"|{'-'*24}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*11}|")
    for cat in CATEGORIES:
        dc = dec_counts.get(cat, 0)
        kc = knn_counts.get(cat, 0)
        print(f"| {cat:<22} | {dc:>10} | {100*dc/n:>8.1f}% | {kc:>10} | {100*kc/n:>8.1f}% |")
    print(f"| {'TOTAL':<22} | {n:>10} | {'100.0':>8}% | {n:>10} | {'100.0':>8}% |")

    # --- Error-only breakdown (exclude correct) ---
    dec_errors = n - dec_counts.get('correct', 0)
    knn_errors = n - knn_counts.get('correct', 0)
    if dec_errors > 0:
        print(f"\n## Error Distribution (decoder, {dec_errors} errors)\n")
        for cat in CATEGORIES:
            if cat == 'correct':
                continue
            c = dec_counts.get(cat, 0)
            if dec_errors > 0:
                print(f"  {cat:<22}: {c:>6} ({100*c/dec_errors:.1f}% of errors)")

    if knn_errors > 0:
        print(f"\n## Error Distribution (k-NN k=1, {knn_errors} errors)\n")
        for cat in CATEGORIES:
            if cat == 'correct':
                continue
            c = knn_counts.get(cat, 0)
            if knn_errors > 0:
                print(f"  {cat:<22}: {c:>6} ({100*c/knn_errors:.1f}% of errors)")

    # --- Phantom analysis (decoder) ---
    dec_phantoms = [d for d in decoder_details if d['category'] == 'phantom']
    if dec_phantoms:
        phantom_names = Counter(d['pred_name'] for d in dec_phantoms)
        print(f"\n## Top-20 Phantom Names (decoder, {len(dec_phantoms)} total)\n")
        print(f"| {'Rank':>4} | {'Phantom Name':<40} | {'Count':>6} |")
        print(f"|{'-'*6}|{'-'*42}|{'-'*8}|")
        for rank, (name, cnt) in enumerate(phantom_names.most_common(20), 1):
            print(f"| {rank:>4} | {name:<40} | {cnt:>6} |")

    # --- Close predictions (highest EdSim examples) ---
    dec_close = [d for d in decoder_details if d['category'] == 'close']
    if dec_close:
        dec_close_sorted = sorted(dec_close, key=lambda x: x['edsim'], reverse=True)
        print(f"\n## Top-10 Close Predictions (decoder, EdSim > 0.7, {len(dec_close)} total)\n")
        print(f"| {'True Name':<35} | {'Predicted':<35} | {'EdSim':>6} |")
        print(f"|{'-'*37}|{'-'*37}|{'-'*8}|")
        for d in dec_close_sorted[:10]:
            print(f"| {d['true_name']:<35} | {d['pred_name']:<35} | {d['edsim']:>6.3f} |")

    # --- Per ext-call bucket breakdown ---
    if ext_counts is not None and any(e >= 0 for e in ext_counts):
        buckets = ['0 ext', '1-3 ext', '4+ ext']
        print(f"\n## Per Ext-Call Bucket (decoder)\n")
        header = f"| {'Bucket':<10} | {'N':>6} |"
        for cat in CATEGORIES:
            header += f" {cat:<12} |"
        print(header)
        print(f"|{'-'*12}|{'-'*8}|" + f"{'-'*14}|" * len(CATEGORIES))
        for bucket in buckets:
            bucket_details = [d for d in decoder_details if d['ext_bucket'] == bucket]
            nb = len(bucket_details)
            if nb == 0:
                continue
            row = f"| {bucket:<10} | {nb:>6} |"
            bucket_cats = Counter(d['category'] for d in bucket_details)
            for cat in CATEGORIES:
                c = bucket_cats.get(cat, 0)
                row += f" {100*c/nb:>5.1f}% ({c:>4}) |"
            print(row)

        print(f"\n## Per Ext-Call Bucket (k-NN k=1)\n")
        print(header)
        print(f"|{'-'*12}|{'-'*8}|" + f"{'-'*14}|" * len(CATEGORIES))
        for bucket in buckets:
            bucket_details = [d for d in knn_details if d['ext_bucket'] == bucket]
            nb = len(bucket_details)
            if nb == 0:
                continue
            row = f"| {bucket:<10} | {nb:>6} |"
            bucket_cats = Counter(d['category'] for d in bucket_details)
            for cat in CATEGORIES:
                c = bucket_cats.get(cat, 0)
                row += f" {100*c/nb:>5.1f}% ({c:>4}) |"
            print(row)

    # --- Summary line ---
    dec_phantom_pct = 100 * dec_counts.get('phantom', 0) / max(dec_errors, 1)
    dec_wrong_seen_pct = 100 * dec_counts.get('wrong_seen', 0) / max(dec_errors, 1)
    dec_unseen_pct = 100 * dec_counts.get('wrong_unseen_target', 0) / max(dec_errors, 1)
    knn_phantom_pct = 100 * knn_counts.get('phantom', 0) / max(knn_errors, 1)

    print(f"\n## Summary ({split_name})")
    print(f"  Decoder: {dec_phantom_pct:.1f}% of errors are phantoms "
          f"(eliminated by k-NN), {dec_wrong_seen_pct:.1f}% are wrong-seen, "
          f"{dec_unseen_pct:.1f}% have unseen targets")
    print(f"  k-NN:    {knn_phantom_pct:.1f}% of errors are phantoms "
          f"(k-NN can still produce names not in training set: 0.0%), "
          f"{100*knn_counts.get('wrong_seen',0)/max(knn_errors,1):.1f}% are wrong-seen")

    return {
        'split': split_name,
        'n': n,
        'decoder': {
            'counts': {cat: dec_counts.get(cat, 0) for cat in CATEGORIES},
            'phantoms_top20': [
                {'name': name, 'count': cnt}
                for name, cnt in Counter(
                    d['pred_name'] for d in decoder_details if d['category'] == 'phantom'
                ).most_common(20)
            ],
            'close_top10': [
                {'true': d['true_name'], 'pred': d['pred_name'], 'edsim': round(d['edsim'], 4)}
                for d in sorted(
                    [d for d in decoder_details if d['category'] == 'close'],
                    key=lambda x: x['edsim'], reverse=True
                )[:10]
            ],
        },
        'knn_k1': {
            'counts': {cat: knn_counts.get(cat, 0) for cat in CATEGORIES},
        },
        'ext_bucket_decoder': {
            bucket: {cat: 0 for cat in CATEGORIES}
            for bucket in ['0 ext', '1-3 ext', '4+ ext']
        },
        'ext_bucket_knn': {
            bucket: {cat: 0 for cat in CATEGORIES}
            for bucket in ['0 ext', '1-3 ext', '4+ ext']
        },
    }


# ---------------------------------------------------------------------------
# Extract ext call counts for test set functions
# ---------------------------------------------------------------------------
def get_ext_counts_from_dataset(dataset, indices):
    """Get ext call count for each function by index."""
    counts = []
    for idx in indices:
        sample = dataset.samples[idx]
        ext = sample.get('ext_calls', [])
        counts.append(len(ext))
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Error Analysis')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--amp', action='store_true', help='Use AMP for inference')
    parser.add_argument('--demo', action='store_true', help='Also run on demo set')
    parser.add_argument('--save', default='results/error_analysis.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    token_vocab = ckpt.get('token_vocab', None)
    ext_vocab = ckpt.get('ext_vocab', None)

    # Load name tokenizer
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

    # --- Load dataset ---
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

    train_idx, val_idx, test_idx = dataset.get_splits(
        cfg['data']['train_split'], cfg['data']['val_split']
    )

    # --- Build model ---
    ckpt_state = ckpt['model_state_dict']
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    if 'ext_encoder.embedding.weight' in ckpt_state:
        cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = ckpt_state['decoder.embedding.weight'].shape[0]

    model = FunctionNamer(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Val F1: {ckpt.get('val_f1', '?'):.4f}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Build training name set ---
    print("\nBuilding training name set...")
    train_name_set = set()
    for idx in train_idx:
        name = dataset.samples[idx]['name']
        train_name_set.add(name)
    print(f"  Unique training names: {len(train_name_set)}")

    # --- Extract training embeddings for k-NN ---
    print(f"\nExtracting training embeddings ({len(train_idx)} functions)...")
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    train_embeddings, train_preds = extract_embeddings_and_predict(
        model, train_loader, device, sp_model,
        desc="Train embeddings", beam_width=1, use_amp=args.amp
    )
    train_names = [p['true_name'] for p in train_preds]
    train_normed = build_knn_index(train_embeddings)
    print(f"  Train embeddings shape: {train_embeddings.shape}")

    all_results = {}

    # --- Test set analysis ---
    print(f"\nExtracting test predictions ({len(test_idx)} functions)...")
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_embeddings, test_preds = extract_embeddings_and_predict(
        model, test_loader, device, sp_model,
        desc="Test predict", beam_width=args.beam_width, use_amp=args.amp
    )
    test_ext_counts = get_ext_counts_from_dataset(dataset, test_idx)

    test_results = run_analysis(
        test_preds, test_embeddings, train_normed, train_names, train_name_set,
        test_ext_counts, 'test'
    )
    all_results['test'] = test_results

    # --- Demo set analysis ---
    if args.demo:
        print(f"\nExtracting demo predictions...")
        demo_cfg = ckpt.get('config', cfg)
        demo_preds, demo_embeddings, demo_pkg_labels = extract_demo_data(
            model, demo_cfg, token_vocab, ext_vocab, sp_model, device, args
        )
        if demo_preds:
            # Demo doesn't have ext counts from dataset — use None
            demo_results = run_analysis(
                demo_preds, demo_embeddings, train_normed, train_names, train_name_set,
                None, 'demo'
            )
            all_results['demo'] = demo_results

    # --- Save ---
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.save}")


if __name__ == '__main__':
    main()
