#!/usr/bin/env python3
"""
Cross-project evaluation with Top-k retrieval + re-ranking + binary fingerprint filtering
+ decoder fallback (reverse hybrid) swept over sigma thresholds.

Extends rag_xproj_rerank.py with a 5th mode: P1+P2 + decoder fallback at sigma
thresholds in [0.50..0.95].

Rule:
  if top1_sim_after_rerank >= sigma -> use retrieved name (k-NN)
  else                              -> use decoder's generated name
"""
import argparse
import json
import os
import sys
import yaml
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import FunctionDataset
from src.evaluation.metrics import compute_subtoken_f1

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from eval_cross_project import (
    collate_fn, predict_binary_with_embeddings,
    load_functions_from_graphs, load_external_calls, load_ground_truth,
    resolve_thunks, DEMO_BINARIES, CROSS_PROJECT_PACKAGES
)
from rag_xproj import extract_train_embeddings


def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def retrieve_topk_with_rerank(query_emb, query_ext_calls, query_num_blocks,
                               train_normed, train_names, train_binaries,
                               train_ext_calls_by_idx, train_num_blocks_by_idx,
                               target_binary_fp, binary_fingerprints,
                               k_retrieve=20, binary_sim_threshold=0.2,
                               use_binary_filter=True, use_rerank=True,
                               train_binary_idx=None, unique_binaries=None,
                               unique_binary_fps=None):
    """Retrieve top-k and re-rank. Returns (pred_name, top_k_with_scores, top1_sim).

    top1_sim is the cosine similarity of the chosen top-1 prediction (after rerank),
    used as the confidence signal for decoder fallback.
    """
    sims = query_emb @ train_normed.T

    if use_binary_filter and target_binary_fp and unique_binaries is not None:
        bin_pass = np.array([
            jaccard_similarity(target_binary_fp, fp) >= binary_sim_threshold
            for fp in unique_binary_fps
        ], dtype=bool)
        binary_mask = bin_pass[train_binary_idx]
        if binary_mask.sum() >= k_retrieve:
            sims = np.where(binary_mask, sims, -1.0)

    top_k_idx = np.argsort(sims)[-k_retrieve:][::-1]
    candidates = [(train_names[idx], sims[idx], idx) for idx in top_k_idx]

    if not use_rerank:
        return candidates[0][0], candidates, float(candidates[0][1])

    name_counts = Counter(name for name, _, _ in candidates)
    reranked = []
    for name, sim, idx in candidates:
        score = float(sim)
        freq_boost = 0.05 * (name_counts[name] - 1)
        score += freq_boost
        candidate_calls = train_ext_calls_by_idx.get(idx, set())
        ext_overlap = jaccard_similarity(query_ext_calls, candidate_calls)
        score += 0.1 * ext_overlap
        candidate_blocks = train_num_blocks_by_idx.get(idx, 0)
        if query_num_blocks > 0 and candidate_blocks > 0:
            block_ratio = min(query_num_blocks, candidate_blocks) / max(query_num_blocks, candidate_blocks)
            score += 0.05 * block_ratio
        reranked.append((name, score, sim, idx))
    reranked.sort(key=lambda x: -x[1])
    # top1_sim is the RAW cosine sim (not the reranked score) — matches eval_cross_project.py's sim_threshold semantics
    return reranked[0][0], reranked, float(reranked[0][2])


def main():
    parser = argparse.ArgumentParser(description='Cross-project RAG with re-rank + binary filter + decoder fallback sweep')
    parser.add_argument('checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--text-dataset', default='data/text_dataset_v2.json')
    parser.add_argument('--k-retrieve', type=int, default=20)
    parser.add_argument('--binary-sim-threshold', type=float, default=0.5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--samples-per-package', type=int, default=10000)
    parser.add_argument('--save', default='results/rag_xproj_rerank_hybrid.json')
    parser.add_argument('--sigma-grid', type=str, default='0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95',
                        help='Comma-separated sigma thresholds for decoder fallback sweep')
    args = parser.parse_args()

    sigma_grid = [float(s) for s in args.sigma_grid.split(',')]

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Sigma grid: {sigma_grid}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading encoder: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    token_vocab = ckpt.get('token_vocab')
    ext_vocab = ckpt.get('ext_vocab')

    votes_vocab_path = cfg['data'].get('votes_vocab_path', 'data/votes_vocab.json')
    from src.preprocessing.build_votes import VotesTokenizer
    sp_model = VotesTokenizer(vocab_path=votes_vocab_path)

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

    xproj_prefixes = tuple(pkg + '_' for pkg in CROSS_PROJECT_PACKAGES)
    clean_train_idx = [
        idx for idx in train_idx
        if not dataset.samples[idx]['binary'].startswith(xproj_prefixes)
    ]
    print(f"Clean train: {len(clean_train_idx)} (removed {len(train_idx) - len(clean_train_idx)} xproj leaks)")

    ckpt_state = ckpt['model_state_dict']
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    if 'ext_encoder.embedding.weight' in ckpt_state:
        cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = ckpt_state['decoder.embedding.weight'].shape[0]

    encoder = FunctionNamer(cfg).to(device)
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()

    print("Extracting clean training embeddings...")
    train_embs, train_names = extract_train_embeddings(
        encoder, dataset, clean_train_idx, device, sp_model, use_amp=args.amp
    )
    print(f"Train: {len(train_embs)} embeddings")

    train_binaries = [dataset.samples[idx]['binary'] for idx in clean_train_idx]

    # Metadata lookup for re-ranking
    print("Loading text prompts for metadata...")
    with open(args.text_dataset) as f:
        text_data = json.load(f)
    prompt_lookup = {(e['binary'], e['bap_name']): e['prompt'] for e in text_data}

    train_ext_calls_by_idx = {}
    train_num_blocks_by_idx = {}
    import re
    for i, idx in enumerate(clean_train_idx):
        sample = dataset.samples[idx]
        key = (sample['binary'], sample['bap_name'])
        prompt = prompt_lookup.get(key, '')
        ext_calls = set()
        for line in prompt.split('\n'):
            if 'Library calls:' in line:
                parts = line.split('Library calls:', 1)[1]
                calls = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\(\)', parts)
                ext_calls = set(calls)
                break
        train_ext_calls_by_idx[i] = ext_calls
        for line in prompt.split('\n'):
            if 'Size:' in line:
                m = re.search(r'(\d+) blocks', line)
                if m:
                    train_num_blocks_by_idx[i] = int(m.group(1))
                break

    # Binary fingerprints
    print("Computing binary fingerprints...")
    import glob
    all_fingerprints = {}
    ext_files = glob.glob('data/external_calls/*_external.json') + glob.glob('demo/external_calls/*_external.json')
    for ef in ext_files:
        binary = os.path.basename(ef).replace('_external.json', '')
        try:
            with open(ef) as f:
                ed = json.load(f)
            calls = set()
            for fn in ed.get('functions', []):
                for c in fn.get('external_calls', []):
                    calls.add(c['name'])
            all_fingerprints[binary] = calls
        except Exception:
            continue
    print(f"Loaded {len(all_fingerprints)} binary fingerprints")

    train_normed = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)

    unique_binaries = sorted(set(train_binaries))
    binary_to_uidx = {b: i for i, b in enumerate(unique_binaries)}
    train_binary_idx = np.array([binary_to_uidx[b] for b in train_binaries], dtype=np.int32)
    unique_binary_fps = [all_fingerprints.get(b, set()) for b in unique_binaries]
    print(f"Precomputed binary lookup: {len(unique_binaries)} unique binaries")

    # Extract cross-project queries — NOW also capture decoder predictions
    print("\nExtracting cross-project demo data...")
    demo_cfg = ckpt.get('config', cfg)
    graphs_dirs = ["data/graphs", "demo/graphs"]
    labels_dirs = ["data/labels", "demo/labels"]
    ext_dirs = ["data/external_calls", "demo/external_calls"]

    all_preds = []

    for pkg, binary, opt_levels in DEMO_BINARIES:
        for opt in opt_levels:
            bin_name = f"{pkg}_{binary}_{opt}" if opt else f"{pkg}_{binary}"
            functions = load_functions_from_graphs(bin_name, graphs_dirs)
            if not functions:
                continue
            ext_by_func = load_external_calls(bin_name, ext_dirs)
            gt = load_ground_truth(bin_name, labels_dirs)
            if not gt:
                continue
            resolve_thunks(functions, ext_by_func)

            print(f"  {bin_name} ({len(gt)} GT)...", end="", flush=True)
            preds, embs = predict_binary_with_embeddings(
                functions, ext_by_func, encoder, token_vocab, ext_vocab,
                sp_model, demo_cfg, device, beam_width=5, use_amp=args.amp
            )

            matched = 0
            for i, pred_func in enumerate(preds):
                addr = pred_func['address']
                if not addr.startswith('0x'):
                    addr = '0x' + addr
                true_name = gt.get(addr)
                if not true_name:
                    try:
                        addr_m4 = hex(int(addr, 16) - 4)
                        true_name = gt.get(addr_m4)
                    except ValueError:
                        pass
                if not true_name:
                    continue
                func_data = functions.get(pred_func['bap_name'])
                if func_data is None:
                    continue
                query_ext_calls = set(ext_by_func.get(pred_func['bap_name'], []))
                query_num_blocks = func_data.get('num_blocks', 0)

                all_preds.append({
                    'package': pkg,
                    'binary': bin_name,
                    'address': addr,
                    'true_name': true_name,
                    'embedding': embs[i:i+1],
                    'query_ext_calls': query_ext_calls,
                    'query_num_blocks': query_num_blocks,
                    'decoder_name': pred_func['predicted_name'],  # NEW: decoder's output
                })
                matched += 1
            print(f" {matched} matched")

    print(f"\nTotal: {len(all_preds)} cross-project functions")

    # Stratified sample
    by_pkg = defaultdict(list)
    for p in all_preds:
        by_pkg[p['package']].append(p)

    sampled = []
    for pkg, items in by_pkg.items():
        random.shuffle(items)
        sampled.extend(items[:args.samples_per_package])
    print(f"Stratified sample: {len(sampled)} functions")

    # Run P1+P2_both mode with sim tracking, and also capture decoder_name
    # This gives us: knn_name, top1_sim, decoder_name per query → sweep sigma
    print(f"\n=== Running P1+P2 with decoder fallback — sigma sweep ===")
    knn_names = []
    top_sims = []
    decoder_names = [p['decoder_name'] for p in sampled]

    for p in tqdm(sampled, desc='P1+P2', leave=False):
        q = p['embedding'].squeeze() / (np.linalg.norm(p['embedding']) + 1e-8)
        target_fp = all_fingerprints.get(p['binary'], set())
        pred_name, top_k, top1_sim = retrieve_topk_with_rerank(
            q, p['query_ext_calls'], p['query_num_blocks'],
            train_normed, train_names, train_binaries,
            train_ext_calls_by_idx, train_num_blocks_by_idx,
            target_fp, all_fingerprints,
            k_retrieve=args.k_retrieve,
            binary_sim_threshold=args.binary_sim_threshold,
            use_binary_filter=True, use_rerank=True,
            train_binary_idx=train_binary_idx,
            unique_binaries=unique_binaries,
            unique_binary_fps=unique_binary_fps,
        )
        knn_names.append(pred_name)
        top_sims.append(top1_sim)

    true_names = [p['true_name'] for p in sampled]

    # Baseline: pure k-NN (no fallback) — σ = 0
    em_knn = sum(1 for p, t in zip(knn_names, true_names) if p == t) / len(knn_names)
    f1_knn = sum(compute_subtoken_f1(p, t) for p, t in zip(knn_names, true_names)) / len(knn_names)

    # Decoder-only baseline
    em_dec = sum(1 for p, t in zip(decoder_names, true_names) if p == t) / len(decoder_names)
    f1_dec = sum(compute_subtoken_f1(p, t) for p, t in zip(decoder_names, true_names)) / len(decoder_names)

    print(f"\n=== Pure k-NN (P1+P2, no fallback) ===")
    print(f"  F1 = {f1_knn:.4f}  EM = {em_knn:.1%}")
    print(f"\n=== Pure Decoder ===")
    print(f"  F1 = {f1_dec:.4f}  EM = {em_dec:.1%}")

    # Sweep σ: if sim >= σ → k-NN, else → decoder
    sweep_results = {}
    print(f"\n=== Sigma sweep (k=20 + rerank + BinFilter + decoder fallback) ===")
    print(f"{'sigma':>8} {'F1':>8} {'EM':>8} {'%kNN':>8} {'%decoder':>10}")
    for sigma in sigma_grid:
        hybrid = []
        n_knn = 0
        n_dec = 0
        for kn, sim, dn in zip(knn_names, top_sims, decoder_names):
            if sim >= sigma:
                hybrid.append(kn)
                n_knn += 1
            else:
                hybrid.append(dn)
                n_dec += 1
        em = sum(1 for p, t in zip(hybrid, true_names) if p == t) / len(hybrid)
        f1 = sum(compute_subtoken_f1(p, t) for p, t in zip(hybrid, true_names)) / len(hybrid)
        pct_knn = 100 * n_knn / len(hybrid)
        pct_dec = 100 * n_dec / len(hybrid)
        sweep_results[sigma] = {'f1': f1, 'em': em, 'pct_knn': pct_knn, 'pct_dec': pct_dec,
                                'predictions': hybrid}
        print(f"{sigma:>8.2f} {f1:>8.4f} {em:>8.1%} {pct_knn:>7.1f}% {pct_dec:>9.1f}%")

    # Per-package F1 at each sigma
    print(f"\n=== Per-package F1 ===")
    sigmas_to_show = [0.0] + sigma_grid  # 0.0 = pure k-NN
    header = f"{'Package':<12} {'N':>5}"
    for s in sigmas_to_show:
        header += f"  σ={s:.2f}"
    print(header)

    per_pkg_f1 = {}
    for pkg in sorted(by_pkg.keys()):
        pkg_indices = [i for i, p in enumerate(sampled) if p['package'] == pkg]
        if not pkg_indices:
            continue
        n = len(pkg_indices)
        row = f"{pkg:<12} {n:>5}"
        pkg_data = {'n': n}
        for s in sigmas_to_show:
            if s == 0.0:
                pkg_preds = knn_names
            else:
                pkg_preds = sweep_results[s]['predictions']
            f1 = sum(compute_subtoken_f1(pkg_preds[i], sampled[i]['true_name']) for i in pkg_indices) / n
            row += f"  {f1:>6.4f}"
            pkg_data[f'sigma_{s:.2f}'] = f1
        per_pkg_f1[pkg] = pkg_data
        print(row)

    # Save
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    out = {
        'n_eval': len(sampled),
        'sigma_grid': sigma_grid,
        'pure_knn_p1p2': {'f1': f1_knn, 'em': em_knn},
        'pure_decoder': {'f1': f1_dec, 'em': em_dec},
        'sweep': {f'{s:.2f}': {k: v for k, v in r.items() if k != 'predictions'}
                  for s, r in sweep_results.items()},
        'per_package_f1': per_pkg_f1,
    }
    with open(args.save, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.save}")

    # Summary
    best_sigma = max(sweep_results, key=lambda s: sweep_results[s]['f1'])
    best_f1 = sweep_results[best_sigma]['f1']
    print(f"\n=== Summary ===")
    print(f"  Pure k-NN (P1+P2):    F1 = {f1_knn:.4f}")
    print(f"  Pure decoder:         F1 = {f1_dec:.4f}")
    print(f"  Best hybrid σ={best_sigma:.2f}: F1 = {best_f1:.4f}  (Δ vs pure k-NN: {best_f1-f1_knn:+.4f})")


if __name__ == '__main__':
    main()
