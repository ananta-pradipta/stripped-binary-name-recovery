#!/usr/bin/env python3
"""
Cross-project evaluation with Top-k retrieval + re-ranking + binary fingerprint filtering.

Priority 1: Top-k retrieval (k=20) + re-rank by:
  - Name frequency in top-k
  - External call overlap with target
  - Block count match

Priority 2: Binary fingerprint filtering
  - Compute ext-call set per binary
  - Filter k-NN to only return functions from binaries with similar fingerprint
  - Prevents cross-package confusion (ngx function → lua function)
"""
import argparse
import json
import os
import sys
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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


# ============================================================
# P2: Binary Fingerprint
# ============================================================

def compute_binary_fingerprint(binary_name, ext_calls_by_binary):
    """Compute a bag-of-ext-calls fingerprint for a binary.

    Fingerprint = set of all external library functions called anywhere in
    the binary. This captures the 'type' of software (web server, crypto, etc).
    """
    calls = ext_calls_by_binary.get(binary_name, set())
    return calls


def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def load_ext_calls_per_binary(match_index, ext_calls_dir):
    """Load the bag of external calls per binary from ext_calls JSONs."""
    import glob
    bin_calls = defaultdict(set)
    for ext_file in glob.glob(os.path.join(ext_calls_dir, '*_external.json')):
        binary = os.path.basename(ext_file).replace('_external.json', '')
        with open(ext_file) as f:
            ext_data = json.load(f)
        for func in ext_data.get('functions', []):
            for call in func.get('external_calls', []):
                bin_calls[binary].add(call['name'])
    return bin_calls


# ============================================================
# P1: Top-k retrieval with re-ranking
# ============================================================

def retrieve_topk_with_rerank(query_emb, query_ext_calls, query_num_blocks,
                               train_normed, train_names, train_binaries,
                               train_ext_calls_by_idx, train_num_blocks_by_idx,
                               target_binary_fp, binary_fingerprints,
                               k_retrieve=20, binary_sim_threshold=0.2,
                               use_binary_filter=True, use_rerank=True,
                               train_binary_idx=None, unique_binaries=None,
                               unique_binary_fps=None):
    """Retrieve top-k and re-rank.

    Args:
        query_emb: target function embedding (normalized)
        query_ext_calls: set of external calls for target
        query_num_blocks: number of blocks in target
        target_binary_fp: fingerprint of target binary (set of ext calls)
        binary_fingerprints: {binary_name: fingerprint_set}
        binary_sim_threshold: minimum jaccard sim for binary to be considered
        use_binary_filter: if True, apply P2 binary filtering
        use_rerank: if True, apply P1 re-ranking

    Returns:
        predicted_name, top_k_with_scores
    """
    # Compute all cosine similarities
    sims = query_emb @ train_normed.T  # (N_train,)

    # P2: Binary fingerprint filter — mask out functions from binaries with
    # very different fingerprints. Vectorized: compute jaccard only for the
    # ~488 unique training binaries, then gather the per-function mask via
    # precomputed train_binary_idx (length N_train, integer index into
    # unique_binaries).
    if use_binary_filter and target_binary_fp and unique_binaries is not None:
        # per-unique-binary pass/fail (length ~488)
        bin_pass = np.array([
            jaccard_similarity(target_binary_fp, fp) >= binary_sim_threshold
            for fp in unique_binary_fps
        ], dtype=bool)
        # expand to per-function mask via index (O(N_train) numpy gather)
        binary_mask = bin_pass[train_binary_idx]
        if binary_mask.sum() >= k_retrieve:
            sims = np.where(binary_mask, sims, -1.0)

    # Top-k
    top_k_idx = np.argsort(sims)[-k_retrieve:][::-1]
    candidates = [(train_names[idx], sims[idx], idx) for idx in top_k_idx]

    if not use_rerank:
        # Just return top-1
        return candidates[0][0], candidates

    # P1: Re-rank top-k
    # Score each candidate by:
    #   - cosine similarity (base score)
    #   - name frequency in top-k (higher freq = more reliable)
    #   - ext call overlap with query
    #   - block count match

    name_counts = Counter(name for name, _, _ in candidates)

    # Compute re-rank score for each candidate
    reranked = []
    for name, sim, idx in candidates:
        # Base: cosine similarity
        score = float(sim)

        # Boost: frequency in top-k (name appears multiple times = more reliable)
        freq_boost = 0.05 * (name_counts[name] - 1)
        score += freq_boost

        # Boost: ext call overlap
        candidate_calls = train_ext_calls_by_idx.get(idx, set())
        ext_overlap = jaccard_similarity(query_ext_calls, candidate_calls)
        score += 0.1 * ext_overlap

        # Boost: block count match (relative)
        candidate_blocks = train_num_blocks_by_idx.get(idx, 0)
        if query_num_blocks > 0 and candidate_blocks > 0:
            block_ratio = min(query_num_blocks, candidate_blocks) / max(query_num_blocks, candidate_blocks)
            score += 0.05 * block_ratio

        reranked.append((name, score, sim, idx))

    # Sort by re-ranked score (descending)
    reranked.sort(key=lambda x: -x[1])

    return reranked[0][0], reranked


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-project RAG with re-rank + binary filter')
    parser.add_argument('checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--text-dataset', default='data/text_dataset_v2.json')
    parser.add_argument('--k-retrieve', type=int, default=20)
    parser.add_argument('--binary-sim-threshold', type=float, default=0.2)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--samples-per-package', type=int, default=50)
    parser.add_argument('--save', default='results/rag_xproj_rerank.json')
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load encoder
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

    # Exclude cross-project from k-NN index
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

    # Extract clean train embeddings
    print("Extracting clean training embeddings...")
    train_embs, train_names = extract_train_embeddings(
        encoder, dataset, clean_train_idx, device, sp_model, use_amp=args.amp
    )
    print(f"Train: {len(train_embs)} embeddings")

    # Per-train-sample metadata for re-ranking
    train_binaries = [dataset.samples[idx]['binary'] for idx in clean_train_idx]
    train_bap_names = [dataset.samples[idx]['bap_name'] for idx in clean_train_idx]

    # Load per-function ext calls and num_blocks (for re-ranking)
    # ext calls per function from text_dataset_v2
    print("Loading text prompts for metadata...")
    with open(args.text_dataset) as f:
        text_data = json.load(f)
    # Build lookup: (binary, bap_name) → prompt metadata
    prompt_lookup = {(e['binary'], e['bap_name']): e['prompt'] for e in text_data}

    train_ext_calls_by_idx = {}
    train_num_blocks_by_idx = {}
    for i, idx in enumerate(clean_train_idx):
        sample = dataset.samples[idx]
        key = (sample['binary'], sample['bap_name'])
        prompt = prompt_lookup.get(key, '')
        # Parse ext calls from prompt
        ext_calls = set()
        for line in prompt.split('\n'):
            if 'Library calls:' in line:
                # Extract call names between "calls:" and "()"
                parts = line.split('Library calls:', 1)[1]
                import re
                calls = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\(\)', parts)
                ext_calls = set(calls)
                break
        train_ext_calls_by_idx[i] = ext_calls

        # Parse block count
        for line in prompt.split('\n'):
            if 'Size:' in line:
                match = re.search(r'(\d+) blocks', line)
                if match:
                    train_num_blocks_by_idx[i] = int(match.group(1))
                break

    print(f"Loaded metadata for {len(train_ext_calls_by_idx)} training functions")

    # Compute binary fingerprints from all ext_calls files
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

    # Normalize train embeddings
    train_normed = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)

    # Precompute unique-binary table for vectorized P2 filtering
    # (previous per-function loop was the 50x slowdown on full eval)
    unique_binaries = sorted(set(train_binaries))
    binary_to_uidx = {b: i for i, b in enumerate(unique_binaries)}
    train_binary_idx = np.array([binary_to_uidx[b] for b in train_binaries], dtype=np.int32)
    unique_binary_fps = [all_fingerprints.get(b, set()) for b in unique_binaries]
    print(f"Precomputed binary lookup: {len(unique_binaries)} unique binaries over {len(train_binaries)} train functions")

    # Extract cross-project demo functions
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

    # Run all 4 modes: baseline, P1 only, P2 only, P1+P2
    modes = [
        ('baseline_top1', False, False),
        ('P1_rerank_only', True, False),
        ('P2_binfilter_only', False, True),
        ('P1+P2_both', True, True),
    ]

    all_results = {}
    for mode_name, use_rerank, use_binary_filter in modes:
        print(f"\n=== Running mode: {mode_name} (rerank={use_rerank}, binfilter={use_binary_filter}) ===")
        preds = []
        for p in tqdm(sampled, desc=mode_name, leave=False):
            q = p['embedding'].squeeze() / (np.linalg.norm(p['embedding']) + 1e-8)
            target_fp = all_fingerprints.get(p['binary'], set())
            pred_name, top_k = retrieve_topk_with_rerank(
                q, p['query_ext_calls'], p['query_num_blocks'],
                train_normed, train_names, train_binaries,
                train_ext_calls_by_idx, train_num_blocks_by_idx,
                target_fp, all_fingerprints,
                k_retrieve=args.k_retrieve,
                binary_sim_threshold=args.binary_sim_threshold,
                use_binary_filter=use_binary_filter,
                use_rerank=use_rerank,
                train_binary_idx=train_binary_idx,
                unique_binaries=unique_binaries,
                unique_binary_fps=unique_binary_fps,
            )
            preds.append(pred_name)

        true_names = [p['true_name'] for p in sampled]
        em = sum(1 for p, t in zip(preds, true_names) if p == t) / len(preds)
        f1 = sum(compute_subtoken_f1(p, t) for p, t in zip(preds, true_names)) / len(preds)
        all_results[mode_name] = {'em': em, 'f1': f1, 'predictions': preds}
        print(f"  {mode_name}: F1 = {f1:.4f} | EM = {em:.1%}")

    # Per-package breakdown (F1 primary, EM secondary)
    print(f"\n=== Per-Package F1 ===")
    print(f"{'Package':<12} {'N':>5} {'baseline':>10} {'P1_only':>10} {'P2_only':>10} {'P1+P2':>10}")
    for pkg in sorted(by_pkg.keys()):
        pkg_indices = [i for i, p in enumerate(sampled) if p['package'] == pkg]
        if not pkg_indices:
            continue
        n = len(pkg_indices)
        row = f"{pkg:<12} {n:>5}"
        for mode_name, _, _ in modes:
            pred = all_results[mode_name]['predictions']
            f1 = sum(compute_subtoken_f1(pred[i], sampled[i]['true_name']) for i in pkg_indices) / n
            row += f" {f1:>10.4f}"
        print(row)

    print(f"\n=== Per-Package EM ===")
    print(f"{'Package':<12} {'N':>5} {'baseline':>10} {'P1_only':>10} {'P2_only':>10} {'P1+P2':>10}")
    for pkg in sorted(by_pkg.keys()):
        pkg_indices = [i for i, p in enumerate(sampled) if p['package'] == pkg]
        if not pkg_indices:
            continue
        n = len(pkg_indices)
        row = f"{pkg:<12} {n:>5}"
        for mode_name, _, _ in modes:
            pred = all_results[mode_name]['predictions']
            em = sum(1 for i in pkg_indices if pred[i] == sampled[i]['true_name']) / n
            row += f" {em:>9.1%}"
        print(row)

    # Summary
    print(f"\n=== Summary ===")
    for mode_name, _, _ in modes:
        f1 = all_results[mode_name]['f1']
        em = all_results[mode_name]['em']
        print(f"  {mode_name}: F1 = {f1:.4f} | EM = {em:.1%}")

    # Save
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    with open(args.save, 'w') as f:
        true_names = [p['true_name'] for p in sampled]
        json.dump({
            'n_eval': len(sampled),
            'modes': {mode: {'em': all_results[mode]['em'], 'f1': all_results[mode]['f1']} for mode, _, _ in modes},
            'predictions': [
                {
                    'pkg': sampled[i]['package'],
                    'true': true_names[i],
                    **{mode: all_results[mode]['predictions'][i] for mode, _, _ in modes},
                }
                for i in range(len(sampled))
            ],
        }, f, indent=2)
    print(f"\nSaved to {args.save}")


if __name__ == '__main__':
    main()
