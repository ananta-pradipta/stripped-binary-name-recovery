#!/usr/bin/env python3
"""
Statistical Significance: Bootstrap CIs, McNemar's test, per-binary variance.

Tests:
  1. Bootstrap 95% CI on decoder/k-NN EM and F1 (test + demo)
  2. McNemar's test: decoder vs k-NN paired comparison
  3. Per-binary variance: F1 mean +/- std across binaries

Usage:
  python3 scripts/statistical_significance.py checkpoints/best_model.pt --config configs/optimized_large.yaml --amp --demo
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
from collections import defaultdict
from scipy import stats

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

from eval_knn_ablation import (
    collate_fn,
    extract_embeddings_and_predict,
    build_knn_index,
    batch_knn_lookup,
    extract_demo_data,
    DEMO_PACKAGES,
)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean of values.

    Args:
        values: array-like of per-sample metric values
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (default 0.95)
        seed: random seed

    Returns:
        (mean, lower, upper)
    """
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)

    alpha = 1 - ci
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(values)), float(lower), float(upper)


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------
def mcnemar_test(decoder_correct, knn_correct):
    """McNemar's test for paired binary outcomes.

    Args:
        decoder_correct: list/array of booleans (decoder got it right)
        knn_correct: list/array of booleans (k-NN got it right)

    Returns:
        dict with b, c (discordant counts), chi2, p_value
    """
    decoder_correct = np.array(decoder_correct, dtype=bool)
    knn_correct = np.array(knn_correct, dtype=bool)

    # b: decoder correct, k-NN wrong
    b = int(np.sum(decoder_correct & ~knn_correct))
    # c: decoder wrong, k-NN correct
    c = int(np.sum(~decoder_correct & knn_correct))

    # McNemar's chi-squared with continuity correction
    if b + c == 0:
        return {'b': b, 'c': c, 'chi2': 0.0, 'p_value': 1.0}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(stats.chi2.sf(chi2, df=1))

    return {'b': b, 'c': c, 'chi2': float(chi2), 'p_value': p_value}


# ---------------------------------------------------------------------------
# Per-binary analysis
# ---------------------------------------------------------------------------
def per_binary_stats(f1_scores, binary_labels):
    """Compute per-binary F1 mean and std.

    Args:
        f1_scores: list of per-function F1 scores
        binary_labels: list of binary name for each function

    Returns:
        dict with per_binary results and overall mean/std
    """
    by_binary = defaultdict(list)
    for f1, binary in zip(f1_scores, binary_labels):
        by_binary[binary].append(f1)

    per_binary = {}
    binary_means = []
    for binary in sorted(by_binary.keys()):
        scores = by_binary[binary]
        mean_f1 = np.mean(scores)
        per_binary[binary] = {
            'n': len(scores),
            'mean_f1': float(mean_f1),
            'std_f1': float(np.std(scores)),
        }
        binary_means.append(mean_f1)

    return {
        'per_binary': per_binary,
        'n_binaries': len(per_binary),
        'macro_mean_f1': float(np.mean(binary_means)),
        'macro_std_f1': float(np.std(binary_means)),
    }


# ---------------------------------------------------------------------------
# Run statistical tests on a split
# ---------------------------------------------------------------------------
def run_significance_tests(preds, embeddings, train_normed, train_names,
                           binary_labels, split_name, n_bootstrap=1000):
    """Run all statistical tests for a split.

    Args:
        preds: list of dicts with 'pred_name', 'true_name', 'beam_score'
        embeddings: numpy array [N, D]
        train_normed: normalized training embeddings
        train_names: list of training function names
        binary_labels: list of binary name per function
        split_name: 'test' or 'demo'
        n_bootstrap: number of bootstrap resamples

    Returns:
        dict with all results
    """
    n = len(preds)
    print(f"\n{'='*80}")
    print(f"STATISTICAL SIGNIFICANCE: {split_name.upper()} ({n} functions)")
    print(f"{'='*80}")

    # k-NN k=1 predictions
    knn_names = batch_knn_lookup(embeddings, train_normed, train_names, k=1)

    # Per-function metrics
    dec_em = []
    dec_f1 = []
    knn_em = []
    knn_f1 = []
    for i in range(n):
        true = preds[i]['true_name']
        dec = preds[i]['pred_name']
        knn = knn_names[i]

        dec_em.append(1.0 if dec == true else 0.0)
        dec_f1.append(compute_subtoken_f1(dec, true))
        knn_em.append(1.0 if knn == true else 0.0)
        knn_f1.append(compute_subtoken_f1(knn, true))

    # --- 1. Bootstrap CIs ---
    print(f"\n## Bootstrap 95% CIs (1000 resamples)\n")
    metrics_to_test = [
        ('Decoder EM', dec_em),
        ('Decoder F1', dec_f1),
        ('k-NN EM', knn_em),
        ('k-NN F1', knn_f1),
    ]

    print(f"| {'Metric':<15} | {'Mean':>8} | {'95% CI Lower':>13} | {'95% CI Upper':>13} | {'Width':>7} |")
    print(f"|{'-'*17}|{'-'*10}|{'-'*15}|{'-'*15}|{'-'*9}|")

    ci_results = {}
    for name, values in metrics_to_test:
        mean, lower, upper = bootstrap_ci(values, n_bootstrap=n_bootstrap)
        width = upper - lower
        is_pct = 'EM' in name
        if is_pct:
            print(f"| {name:<15} | {100*mean:>7.2f}% | {100*lower:>12.2f}% | {100*upper:>12.2f}% | {100*width:>6.2f}% |")
        else:
            print(f"| {name:<15} | {mean:>8.4f} | {lower:>13.4f} | {upper:>13.4f} | {width:>7.4f} |")
        ci_results[name] = {
            'mean': round(mean, 6),
            'ci_lower': round(lower, 6),
            'ci_upper': round(upper, 6),
        }

    # --- 2. McNemar's test ---
    print(f"\n## McNemar's Test (Decoder vs k-NN k=1)\n")
    dec_correct = [e == 1.0 for e in dec_em]
    knn_correct = [e == 1.0 for e in knn_em]

    mcnemar = mcnemar_test(dec_correct, knn_correct)
    both_correct = sum(1 for d, k in zip(dec_correct, knn_correct) if d and k)
    both_wrong = sum(1 for d, k in zip(dec_correct, knn_correct) if not d and not k)

    print(f"| {'':20} | {'k-NN Correct':>13} | {'k-NN Wrong':>11} |")
    print(f"|{'-'*22}|{'-'*15}|{'-'*13}|")
    print(f"| {'Decoder Correct':<20} | {both_correct:>13} | {mcnemar['b']:>11} |")
    print(f"| {'Decoder Wrong':<20} | {mcnemar['c']:>13} | {both_wrong:>11} |")
    print(f"\n  b (dec right, knn wrong): {mcnemar['b']}")
    print(f"  c (dec wrong, knn right): {mcnemar['c']}")
    print(f"  Chi-squared: {mcnemar['chi2']:.4f}")
    print(f"  p-value: {mcnemar['p_value']:.6f}")
    sig = "YES" if mcnemar['p_value'] < 0.05 else "NO"
    print(f"  Significant at alpha=0.05: {sig}")

    # --- 3. Per-binary variance ---
    print(f"\n## Per-Binary F1 Variance\n")

    dec_binary = per_binary_stats(dec_f1, binary_labels)
    knn_binary = per_binary_stats(knn_f1, binary_labels)

    print(f"### Decoder\n")
    print(f"| {'Binary':<40} | {'N':>6} | {'Mean F1':>8} | {'Std F1':>8} |")
    print(f"|{'-'*42}|{'-'*8}|{'-'*10}|{'-'*10}|")
    for binary, info in sorted(dec_binary['per_binary'].items(),
                                key=lambda x: x[1]['mean_f1'], reverse=True):
        print(f"| {binary:<40} | {info['n']:>6} | {info['mean_f1']:>8.4f} | {info['std_f1']:>8.4f} |")
    print(f"\n  Across {dec_binary['n_binaries']} binaries: "
          f"macro F1 = {dec_binary['macro_mean_f1']:.4f} +/- {dec_binary['macro_std_f1']:.4f}")

    print(f"\n### k-NN (k=1)\n")
    print(f"| {'Binary':<40} | {'N':>6} | {'Mean F1':>8} | {'Std F1':>8} |")
    print(f"|{'-'*42}|{'-'*8}|{'-'*10}|{'-'*10}|")
    for binary, info in sorted(knn_binary['per_binary'].items(),
                                key=lambda x: x[1]['mean_f1'], reverse=True):
        print(f"| {binary:<40} | {info['n']:>6} | {info['mean_f1']:>8.4f} | {info['std_f1']:>8.4f} |")
    print(f"\n  Across {knn_binary['n_binaries']} binaries: "
          f"macro F1 = {knn_binary['macro_mean_f1']:.4f} +/- {knn_binary['macro_std_f1']:.4f}")

    return {
        'split': split_name,
        'n': n,
        'bootstrap_ci': ci_results,
        'mcnemar': mcnemar,
        'contingency': {
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'dec_only': mcnemar['b'],
            'knn_only': mcnemar['c'],
        },
        'per_binary_decoder': {
            'macro_mean_f1': dec_binary['macro_mean_f1'],
            'macro_std_f1': dec_binary['macro_std_f1'],
            'n_binaries': dec_binary['n_binaries'],
            'per_binary': {k: v for k, v in dec_binary['per_binary'].items()},
        },
        'per_binary_knn': {
            'macro_mean_f1': knn_binary['macro_mean_f1'],
            'macro_std_f1': knn_binary['macro_std_f1'],
            'n_binaries': knn_binary['n_binaries'],
            'per_binary': {k: v for k, v in knn_binary['per_binary'].items()},
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Statistical Significance Tests')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--amp', action='store_true', help='Use AMP for inference')
    parser.add_argument('--demo', action='store_true', help='Also run on demo set')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--save', default='results/statistical_significance.json')
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

    # --- Test set ---
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

    # Get binary labels for per-binary analysis
    test_binary_labels = [dataset.samples[idx]['binary'] for idx in test_idx]

    test_results = run_significance_tests(
        test_preds, test_embeddings, train_normed, train_names,
        test_binary_labels, 'test', n_bootstrap=args.n_bootstrap
    )
    all_results['test'] = test_results

    # --- Demo set ---
    if args.demo:
        print(f"\nExtracting demo predictions...")
        demo_cfg = ckpt.get('config', cfg)
        demo_preds, demo_embeddings, demo_pkg_labels = extract_demo_data(
            model, demo_cfg, token_vocab, ext_vocab, sp_model, device, args
        )
        if demo_preds:
            # Use package labels as binary labels for demo
            demo_results = run_significance_tests(
                demo_preds, demo_embeddings, train_normed, train_names,
                demo_pkg_labels, 'demo', n_bootstrap=args.n_bootstrap
            )
            all_results['demo'] = demo_results

    # --- Save ---
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.save}")


if __name__ == '__main__':
    main()
