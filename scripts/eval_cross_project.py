#!/usr/bin/env python3
"""
Cross-project evaluation with proper data separation.

Fixes the data leakage bug where cross-project binaries were included in the
k-NN training index (they defaulted to train since they weren't in split_assignments.json).

This script:
  1. Loads the dataset and explicitly excludes cross-project binaries from the training k-NN index
  2. Evaluates test set (decoder + k-NN) with per-binary breakdown
  3. Evaluates cross-project set (decoder + k-NN) with per-package breakdown
  4. Adds sanity checks to verify zero overlap between k-NN index and eval set

Usage:
  python3 scripts/eval_cross_project.py checkpoints/best_model.pt --config configs/optimized_large.yaml --amp
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
import glob as glob_mod
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hextester(s):
    """True iff s is a valid hex integer. Used to distinguish BAP placeholder
    names (sub_HEX, e.g. sub_4a30) from user-defined sub_* names
    (e.g. sub_append_string)."""
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# P2: Binary Fingerprint Filter
# ---------------------------------------------------------------------------
def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def load_binary_fingerprints(ext_calls_dirs):
    """Load bag-of-external-calls per binary from ext_calls JSON files."""
    bin_calls = defaultdict(set)
    for ext_dir in ext_calls_dirs:
        if not os.path.isdir(ext_dir):
            continue
        for ext_file in glob.glob(os.path.join(ext_dir, '*_external.json')):
            binary = os.path.basename(ext_file).replace('_external.json', '')
            with open(ext_file) as f:
                ext_data = json.load(f)
            for func in ext_data.get('functions', []):
                for call in func.get('external_calls', []):
                    bin_calls[binary].add(call['name'])
    return dict(bin_calls)


def batch_knn_lookup_with_p2(query_embeddings, train_normed, train_names,
                              train_binary_idx, unique_binaries, unique_binary_fps,
                              target_binary_fp, binary_sim_threshold=0.2, k=1):
    """k-NN lookup with P2 binary fingerprint filtering.

    Masks out training functions from binaries with low Jaccard similarity
    to the target binary's external call fingerprint.
    """
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_normed = query_embeddings / (norms + 1e-8)

    # Precompute binary mask: which training binaries pass the filter
    if target_binary_fp and unique_binaries is not None:
        bin_pass = np.array([
            jaccard_similarity(target_binary_fp, fp) >= binary_sim_threshold
            for fp in unique_binary_fps
        ], dtype=bool)
        binary_mask = bin_pass[train_binary_idx]
        # Only apply if enough candidates pass
        if binary_mask.sum() < k:
            binary_mask = None
    else:
        binary_mask = None

    chunk_size = 500
    results = []
    for start in range(0, len(query_normed), chunk_size):
        end = min(start + chunk_size, len(query_normed))
        sims = query_normed[start:end] @ train_normed.T
        if binary_mask is not None:
            sims = np.where(binary_mask, sims, -1.0)
        best_idx = np.argmax(sims, axis=1)
        for idx in best_idx:
            results.append(train_names[idx])
    return results


# Cross-project packages — completely absent from training
# Hub-dependent packages that USE libraries we trained on
CROSS_PROJECT_PACKAGES = [
    'tengine', 'angie', 'nginx118', 'recutils', 'curl',
]

DEMO_BINARIES = [
    # (package, binary_prefix, opt_levels)
    ("tengine", "nginx", ["O0", "O2"]),  # tengine_O2 has preprocessing bug, kept for historical continuity
    ("angie", "angie", ["O0", "O1", "O2", "O3"]),
    ("nginx118", "nginx118", ["O0", "O1", "O2", "O3"]),  # LTS, Apr 2020
    ("recutils", "recfix", ["O0", "O1", "O2", "O3"]),
    ("recutils", "recdel", ["O0", "O1", "O2", "O3"]),
    ("recutils", "recfmt", ["O0", "O1", "O2", "O3"]),
    ("recutils", "recinf", ["O0", "O1", "O2", "O3"]),
    ("recutils", "recins", ["O0", "O1", "O2", "O3"]),
    ("recutils", "recsel", ["O0", "O1", "O2", "O3"]),
    ("curl", "curl", ["O0", "O1", "O2", "O3"]),
]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(batch):
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


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embeddings_and_predict(model, loader, device, sp_model, desc="",
                                   beam_width=5, use_amp=False):
    model.eval()
    all_embeddings = []
    all_predictions = []
    all_binaries = []

    sos_id = sp_model.bos_id()
    eos_id = sp_model.eos_id()

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            block_tokens = batch['block_tokens'].to(device)
            edge_index = batch['edge_index'].to(device)
            ext_call_ids = batch['ext_call_ids'].to(device)
            block_features = batch.get("block_features")
            if block_features is not None:
                block_features = block_features.to(device)
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

            all_embeddings.append(z.cpu().numpy())

            B = z.shape[0]
            for i in range(B):
                tokens, score = model.decoder.generate(
                    z[i:i+1], sos_id, eos_id, beam_width
                )
                pred_name = sp_model.decode(tokens) if tokens else ""

                target_tokens = []
                for t in decoder_target[i]:
                    tid = t.item()
                    if tid == eos_id:
                        break
                    if tid != 0:
                        target_tokens.append(tid)
                true_name = sp_model.decode(target_tokens)

                all_predictions.append({
                    'pred_name': pred_name,
                    'true_name': true_name,
                    'beam_score': score,
                })

            # Track binary names
            if 'binary' in batch:
                all_binaries.extend(batch['binary'])

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_predictions, all_binaries


# ---------------------------------------------------------------------------
# k-NN
# ---------------------------------------------------------------------------
def build_knn_index(train_embeddings):
    norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    return train_embeddings / (norms + 1e-8)


def batch_knn_lookup(query_embeddings, train_normed, train_names, k=1,
                     return_similarities=False):
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_normed = query_embeddings / (norms + 1e-8)

    chunk_size = 500
    results = []
    similarities = []

    for start in range(0, len(query_normed), chunk_size):
        end = min(start + chunk_size, len(query_normed))
        sims = query_normed[start:end] @ train_normed.T

        if k == 1:
            best_idx = np.argmax(sims, axis=1)
            best_sims = np.max(sims, axis=1)
            for idx in best_idx:
                results.append(train_names[idx])
            similarities.extend(best_sims.tolist())
        else:
            top_k_idx = np.argsort(sims, axis=1)[:, -k:][:, ::-1]
            for row_idx, row in enumerate(top_k_idx):
                candidate_names = [train_names[idx] for idx in row]
                candidate_sims = [sims[row_idx, idx] for idx in row]
                name_counts = Counter(candidate_names)
                max_votes = max(name_counts.values())
                best_name, best_sim = None, -999
                for name, sim in zip(candidate_names, candidate_sims):
                    if name_counts[name] == max_votes and sim > best_sim:
                        best_name = name
                        best_sim = sim
                results.append(best_name)
                similarities.append(candidate_sims[0])

    if return_similarities:
        return results, similarities
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(final_names, true_names):
    n = len(final_names)
    if n == 0:
        return {'n': 0, 'em': 0, 'f1': 0, 'ngsim': 0, 'edsim': 0}
    em = sum(1 for p, t in zip(final_names, true_names) if p == t)
    f1_sum = sum(compute_subtoken_f1(p, t) for p, t in zip(final_names, true_names))
    ngsim_sum = sum(compute_char_ngram_similarity(p, t) for p, t in zip(final_names, true_names))
    edsim_sum = sum(compute_edit_distance_similarity(p, t) for p, t in zip(final_names, true_names))
    return {
        'n': n,
        'em': em / n,
        'f1': f1_sum / n,
        'ngsim': ngsim_sum / n,
        'edsim': edsim_sum / n,
    }


# ---------------------------------------------------------------------------
# Demo data loading (for cross-project)
# ---------------------------------------------------------------------------
def load_functions_from_graphs(bin_name, graphs_dirs):
    functions = {}
    for gdir in graphs_dirs:
        pattern = os.path.join(gdir, f"{bin_name}_*.json")
        for path in glob_mod.glob(pattern):
            with open(path) as f:
                data = json.load(f)
            fname = data.get('function_name', '')
            if fname:
                functions[fname] = data
    return functions


def load_external_calls(bin_name, ext_dirs):
    ext_by_func = {}
    for edir in ext_dirs:
        ext_file = os.path.join(edir, f"{bin_name}_external.json")
        if os.path.exists(ext_file):
            with open(ext_file) as f:
                ext_data = json.load(f)
            for func in ext_data.get('functions', []):
                key = func.get('function_name', '')
                calls = [c['name'] for c in func.get('external_calls', [])]
                ext_by_func[key] = calls
            return ext_by_func
    return ext_by_func


def load_ground_truth(bin_name, labels_dirs):
    for ldir in labels_dirs:
        label_file = os.path.join(ldir, f"{bin_name}_labels.json")
        if os.path.exists(label_file):
            with open(label_file) as f:
                labels = json.load(f)
            gt = {}
            funcs = labels.get('functions', labels)
            if isinstance(funcs, dict):
                for name, addr in funcs.items():
                    if isinstance(addr, str):
                        if addr.startswith('0x'):
                            addr_norm = '0x' + addr[2:].lstrip('0')
                            if addr_norm == '0x':
                                addr_norm = '0x0'
                        else:
                            addr_norm = '0x' + addr.lstrip('0')
                            if addr_norm == '0x':
                                addr_norm = '0x0'
                        gt[addr_norm] = name
                        gt[addr] = name
            return gt
    return {}


def resolve_thunks(functions, ext_by_func):
    resolved = 0
    for func_name, func_data in list(functions.items()):
        all_tokens = [t for b in func_data['blocks'] for t in b['tokens']]
        if len(all_tokens) <= 2 and 'CALL_INTERNAL' in all_tokens:
            callees = func_data.get('internal_callees', [])
            if callees:
                callee_name = callees[0]
                callee_data = functions.get(callee_name)
                if callee_data:
                    func_data['blocks'] = callee_data['blocks']
                    func_data['edges'] = callee_data['edges']
                    func_data['num_blocks'] = callee_data['num_blocks']
                    func_data['internal_callees'] = callee_data.get('internal_callees', [])
                    callee_ext = ext_by_func.get(callee_name, [])
                    if callee_ext:
                        ext_by_func[func_name] = callee_ext
                    resolved += 1
    return resolved


def predict_binary_with_embeddings(functions, ext_by_func, model, token_vocab, ext_vocab,
                                    sp, cfg, device, beam_width=5, use_amp=False):
    sos_id = sp.bos_id()
    eos_id = sp.eos_id()

    func_by_addr = {}
    for fname, fdata in functions.items():
        addr = fdata.get('address', '')
        if addr:
            func_by_addr[addr] = fdata

    callers_of = {}
    for fname, fdata in functions.items():
        for callee_name in fdata.get('internal_callees', []):
            if callee_name not in callers_of:
                callers_of[callee_name] = []
            callers_of[callee_name].append(fname)

    callee_enabled = cfg.get('callee_encoder', {}).get('enabled', False)
    caller_enabled = cfg.get('caller_encoder', {}).get('enabled', False)
    max_blocks = cfg['data']['max_blocks_per_function']
    max_tokens = cfg['data']['max_tokens_per_block']

    def get_context_tokens(target_names, max_ctx=5, max_sig_tokens=10):
        sigs = []
        for name in target_names[:max_ctx]:
            graph = functions.get(name)
            if graph is None and name.startswith('sub_') and hextester(name[4:]):
                addr = '0x' + name[4:]
                graph = func_by_addr.get(addr)
            if graph is None:
                sigs.append([0] * max_sig_tokens)
                continue
            sig = []
            for block in graph['blocks'][:3]:
                sig.extend(block['tokens'][:5])
                if len(sig) >= max_sig_tokens:
                    break
            sig_ids = [token_vocab.get(t, token_vocab.get('<UNK>', 1))
                       for t in sig[:max_sig_tokens]]
            sig_ids += [0] * (max_sig_tokens - len(sig_ids))
            sigs.append(sig_ids)
        while len(sigs) < max_ctx:
            sigs.append([0] * max_sig_tokens)
        return sigs

    targets = {name: data for name, data in functions.items()
               if name.startswith('sub_') and data['num_blocks'] >= 2}

    results = []
    embeddings = []

    for func_name, func_data in tqdm(targets.items(), desc="Demo predict", leave=False):
        blocks = func_data['blocks'][:max_blocks]

        block_token_ids = []
        for block in blocks:
            ids = [token_vocab.get(t, token_vocab.get('<UNK>', 1))
                   for t in block['tokens'][:max_tokens]]
            ids += [0] * (max_tokens - len(ids))
            block_token_ids.append(ids)

        num_blocks = len(block_token_ids)
        edges_raw = func_data['edges']
        in_degs, out_degs = compute_block_degrees(edges_raw, num_blocks)
        block_feats = []
        for bi, block in enumerate(blocks):
            feats = compute_block_features(
                block['tokens'], bi, num_blocks,
                in_degs[bi] if bi < len(in_degs) else 0,
                out_degs[bi] if bi < len(out_degs) else 0,
            )
            block_feats.append(feats)

        while len(block_token_ids) < max_blocks:
            block_token_ids.append([0] * max_tokens)
        while len(block_feats) < max_blocks:
            block_feats.append([0.0] * NUM_BLOCK_FEATURES)

        ext = ext_by_func.get(func_name, [])
        ext_ids = [ext_vocab.get(name, ext_vocab.get('<NO_EXT>', 0)) for name in ext]
        if not ext_ids:
            ext_ids = [0]

        bt = torch.tensor([block_token_ids], dtype=torch.long, device=device)
        bf = torch.tensor([block_feats], dtype=torch.float32, device=device)
        ei_filtered = [[s, d] for s, d in edges_raw if s < num_blocks and d < num_blocks]
        if not ei_filtered:
            ei_filtered = [[0, 0]]
        ei = torch.tensor(ei_filtered, dtype=torch.long, device=device).t().contiguous()
        ec = torch.tensor([ext_ids], dtype=torch.long, device=device)

        ct = None
        if callee_enabled:
            callee_sigs = get_context_tokens(func_data.get('internal_callees', []))
            ct = torch.tensor([callee_sigs], dtype=torch.long, device=device)

        crt = None
        if caller_enabled:
            caller_sigs = get_context_tokens(callers_of.get(func_name, []))
            crt = torch.tensor([caller_sigs], dtype=torch.long, device=device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                block_embs = model.block_encoder(bt, block_features=bf)
                B, N, D = block_embs.shape
                x = block_embs.view(B * N, D)
                batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
                f, _, _ = model.graph_encoder(x, ei, batch_vec)

                z = f
                if model.ext_encoder_enabled and model.fusion is not None:
                    c = model.ext_encoder(ec)
                    has_ext = model._compute_has_ext_calls(ec)
                    z, _ = model.fusion(z, c, has_ext_calls=has_ext)

                if model.callee_encoder_enabled and ct is not None:
                    callee_ctx, has_callees = model.callee_encoder(ct)
                    g = model.callee_gate(torch.cat([z, callee_ctx], dim=1))
                    z_fused = g * z + (1 - g) * callee_ctx
                    mask = has_callees.unsqueeze(1).float()
                    z = mask * z_fused + (1 - mask) * z

                if model.caller_encoder_enabled and crt is not None:
                    caller_ctx, has_callers = model.caller_encoder(crt)
                    g = model.caller_gate(torch.cat([z, caller_ctx], dim=1))
                    z_fused = g * z + (1 - g) * caller_ctx
                    mask = has_callers.unsqueeze(1).float()
                    z = mask * z_fused + (1 - mask) * z

            pred_tokens, score = model.decoder.generate(z, sos_id, eos_id, beam_width)
            pred_name = sp.decode(pred_tokens).strip() if pred_tokens else ""
            norm_score = score / max(len(pred_tokens), 1) if pred_tokens else -999

        embeddings.append(z.cpu().numpy())
        results.append({
            'address': func_data['address'],
            'bap_name': func_name,
            'predicted_name': pred_name,
            'beam_score': norm_score,
        })

    return results, np.concatenate(embeddings, axis=0) if embeddings else np.array([])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Cross-Project Evaluation (leak-free)')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--amp', action='store_true', help='Use AMP for inference')
    parser.add_argument('--save', default='results/cross_project_eval_v2.json',
                        help='Path to save JSON results')
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

    # =====================================================================
    # CRITICAL FIX: Remove cross-project binaries from training index
    # =====================================================================
    xproj_prefixes = tuple(pkg + '_' for pkg in CROSS_PROJECT_PACKAGES)

    # Filter train_idx to exclude cross-project binaries
    original_train_size = len(train_idx)
    clean_train_idx = []
    excluded_train_idx = []
    for idx in train_idx:
        binary = dataset.samples[idx]['binary']
        if binary.startswith(xproj_prefixes):
            excluded_train_idx.append(idx)
        else:
            clean_train_idx.append(idx)

    print(f"\n{'='*80}")
    print(f"DATA LEAK FIX: Removed {len(excluded_train_idx)} cross-project functions from training index")
    print(f"  Original train: {original_train_size} → Clean train: {len(clean_train_idx)}")
    excluded_bins = set(dataset.samples[i]['binary'] for i in excluded_train_idx)
    print(f"  Excluded binaries ({len(excluded_bins)}): {sorted(excluded_bins)}")
    print(f"{'='*80}")

    # Sanity check: verify no cross-project binaries in val/test either
    val_xproj = [i for i in val_idx if dataset.samples[i]['binary'].startswith(xproj_prefixes)]
    test_xproj = [i for i in test_idx if dataset.samples[i]['binary'].startswith(xproj_prefixes)]
    if val_xproj:
        print(f"  WARNING: {len(val_xproj)} cross-project functions in val set!")
    if test_xproj:
        print(f"  WARNING: {len(test_xproj)} cross-project functions in test set!")

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
    print(f"\nModel: Epoch {ckpt.get('epoch', '?')}, Val F1: {ckpt.get('val_f1', '?'):.4f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # =====================================================================
    # STEP 1: Extract CLEAN training embeddings (no cross-project)
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 1: Extract CLEAN training embeddings ({len(clean_train_idx)} functions)")
    print(f"{'='*80}")
    train_loader = DataLoader(
        Subset(dataset, clean_train_idx),
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    train_embeddings, train_preds, _ = extract_embeddings_and_predict(
        model, train_loader, device, sp_model,
        desc="Train embeddings", beam_width=1, use_amp=args.amp
    )
    train_names = [p['true_name'] for p in train_preds]
    train_binaries_list = [dataset.samples[i]['binary'] for i in clean_train_idx]
    print(f"  Train embeddings shape: {train_embeddings.shape}")
    print(f"  Unique train names: {len(set(train_names))}")

    print("  Building k-NN index (normalizing)...")
    train_normed = build_knn_index(train_embeddings)

    # --- P2: Build binary fingerprint index ---
    ext_calls_dirs = [cfg['data']['external_calls_dir'], 'demo/external_calls']
    all_fingerprints = load_binary_fingerprints(ext_calls_dirs)
    print(f"  Loaded {len(all_fingerprints)} binary fingerprints for P2 filter")

    # Precompute per-unique-binary index for vectorized P2 filtering
    unique_binaries = sorted(set(train_binaries_list))
    bin_to_idx = {b: i for i, b in enumerate(unique_binaries)}
    train_binary_idx = np.array([bin_to_idx[b] for b in train_binaries_list])
    unique_binary_fps = [all_fingerprints.get(b, set()) for b in unique_binaries]
    print(f"  Unique training binaries for P2: {len(unique_binaries)}")

    save_data = {
        'checkpoint': args.checkpoint,
        'beam_width': args.beam_width,
        'clean_train_size': len(clean_train_idx),
        'excluded_xproj_size': len(excluded_train_idx),
    }

    # =====================================================================
    # STEP 2: Test set evaluation with per-binary breakdown
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 2: Test set evaluation ({len(test_idx)} functions)")
    print(f"{'='*80}")
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_embeddings, test_preds, _ = extract_embeddings_and_predict(
        model, test_loader, device, sp_model,
        desc="Test predict", beam_width=args.beam_width, use_amp=args.amp
    )

    # Get binary names for per-binary breakdown
    test_binaries = [dataset.samples[i]['binary'] for i in test_idx]

    # Decoder-only results
    decoder_names = [p['pred_name'] for p in test_preds]
    true_names = [p['true_name'] for p in test_preds]
    test_decoder = compute_metrics(decoder_names, true_names)
    print(f"  Decoder: EM={test_decoder['em']:.1%}, F1={test_decoder['f1']:.4f}")

    # k-NN k=1 results
    knn_names = batch_knn_lookup(test_embeddings, train_normed, train_names, k=1)
    test_knn = compute_metrics(knn_names, true_names)
    print(f"  k-NN k=1: EM={test_knn['em']:.1%}, F1={test_knn['f1']:.4f}")

    # k-NN k=1 + P2 binary filter (batched per unique binary)
    test_knn_p2_names = [''] * len(test_embeddings)
    test_bins_unique = sorted(set(test_binaries))
    for tbin in test_bins_unique:
        bin_indices = [i for i, b in enumerate(test_binaries) if b == tbin]
        target_fp = all_fingerprints.get(tbin, set())
        bin_embs = test_embeddings[bin_indices]
        names = batch_knn_lookup_with_p2(
            bin_embs, train_normed, train_names,
            train_binary_idx, unique_binaries, unique_binary_fps,
            target_fp, binary_sim_threshold=0.2, k=1
        )
        for j, idx in enumerate(bin_indices):
            test_knn_p2_names[idx] = names[j]
    test_knn_p2 = compute_metrics(test_knn_p2_names, true_names)
    print(f"  k-NN k=1 + P2: EM={test_knn_p2['em']:.1%}, F1={test_knn_p2['f1']:.4f}")

    # Per-binary breakdown
    print(f"\n  Per-binary test results (k-NN k=1):")
    binary_results = defaultdict(lambda: {'pred': [], 'true': []})
    for i, (knn, true, binary) in enumerate(zip(knn_names, true_names, test_binaries)):
        pkg = binary.split('_')[0]
        binary_results[binary]['pred'].append(knn)
        binary_results[binary]['true'].append(true)

    test_per_binary = {}
    for binary in sorted(binary_results.keys()):
        br = binary_results[binary]
        m = compute_metrics(br['pred'], br['true'])
        test_per_binary[binary] = m
        print(f"    {binary}: EM={m['em']:.1%}, F1={m['f1']:.4f} ({m['n']} functions)")

    save_data['test'] = {
        'decoder': test_decoder,
        'knn_k1': test_knn,
        'knn_p2': test_knn_p2,
        'per_binary': test_per_binary,
    }

    # =====================================================================
    # STEP 3: Cross-project evaluation with per-package breakdown
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"STEP 3: Cross-project evaluation")
    print(f"{'='*80}")

    demo_cfg = ckpt.get('config', cfg)
    graphs_dirs = ["data/graphs", "demo/graphs"]
    labels_dirs = ["data/labels", "demo/labels"]
    ext_dirs = ["data/external_calls", "demo/external_calls"]

    all_preds = []
    all_embeddings = []
    all_pkg_labels = []
    per_pkg_preds = defaultdict(list)
    per_pkg_embeddings = defaultdict(list)

    for pkg, binary, opt_levels in DEMO_BINARIES:
        for opt in opt_levels:
            bin_name = f"{pkg}_{binary}_{opt}" if opt else f"{pkg}_{binary}"

            functions = load_functions_from_graphs(bin_name, graphs_dirs)
            if not functions:
                alt_name = f"diffutils_{binary}"
                functions = load_functions_from_graphs(alt_name, graphs_dirs)
                if functions:
                    bin_name = alt_name
            if not functions:
                print(f"  SKIP {bin_name}: no graph files")
                continue

            ext_by_func = load_external_calls(bin_name, ext_dirs)
            gt = load_ground_truth(bin_name, labels_dirs)
            if not gt:
                print(f"  SKIP {bin_name}: no GT labels")
                continue

            resolve_thunks(functions, ext_by_func)

            print(f"  Predicting {bin_name} ({len(gt)} GT)...", end="", flush=True)
            preds, embs = predict_binary_with_embeddings(
                functions, ext_by_func, model, token_vocab, ext_vocab,
                sp_model, demo_cfg, device, beam_width=args.beam_width, use_amp=args.amp
            )

            matched = 0
            for i, pred_func in enumerate(preds):
                addr = pred_func['address']
                if not addr.startswith('0x'):
                    addr = '0x' + addr

                true_name = gt.get(addr)
                if not true_name:
                    try:
                        addr_minus4 = hex(int(addr, 16) - 4)
                        true_name = gt.get(addr_minus4)
                    except ValueError:
                        pass
                if not true_name:
                    continue

                entry = {
                    'pred_name': pred_func['predicted_name'],
                    'true_name': true_name,
                    'beam_score': pred_func['beam_score'],
                    'binary': bin_name,
                }
                all_preds.append(entry)
                all_embeddings.append(embs[i:i+1])
                all_pkg_labels.append(pkg)
                per_pkg_preds[pkg].append(entry)
                per_pkg_embeddings[pkg].append(embs[i:i+1])
                matched += 1

            print(f" {matched} matched")

    if not all_preds:
        print("  No cross-project predictions!")
        return

    all_demo_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"\n  Total cross-project: {len(all_preds)} functions, {len(set(all_pkg_labels))} packages")

    # --- Overall cross-project results ---
    # Decoder only
    xproj_decoder_names = [p['pred_name'] for p in all_preds]
    xproj_true_names = [p['true_name'] for p in all_preds]
    xproj_decoder = compute_metrics(xproj_decoder_names, xproj_true_names)
    print(f"\n  OVERALL Cross-Project:")
    print(f"    Decoder: EM={xproj_decoder['em']:.1%}, F1={xproj_decoder['f1']:.4f}, "
          f"NgSim={xproj_decoder['ngsim']:.4f}, EdSim={xproj_decoder['edsim']:.4f}")

    # k-NN k=1
    xproj_knn_names = batch_knn_lookup(all_demo_embeddings, train_normed, train_names, k=1)
    xproj_knn = compute_metrics(xproj_knn_names, xproj_true_names)
    print(f"    k-NN k=1: EM={xproj_knn['em']:.1%}, F1={xproj_knn['f1']:.4f}, "
          f"NgSim={xproj_knn['ngsim']:.4f}, EdSim={xproj_knn['edsim']:.4f}")

    # k-NN k=1 + P2 binary filter (batched per unique binary)
    xproj_pred_binaries = [p.get('binary', '') for p in all_preds]
    xproj_knn_p2_names = [''] * len(all_demo_embeddings)
    xproj_bins_unique = sorted(set(xproj_pred_binaries))
    for xbin in xproj_bins_unique:
        bin_indices = [i for i, b in enumerate(xproj_pred_binaries) if b == xbin]
        target_fp = all_fingerprints.get(xbin, set())
        bin_embs = all_demo_embeddings[bin_indices]
        names = batch_knn_lookup_with_p2(
            bin_embs, train_normed, train_names,
            train_binary_idx, unique_binaries, unique_binary_fps,
            target_fp, binary_sim_threshold=0.2, k=1
        )
        for j, idx in enumerate(bin_indices):
            xproj_knn_p2_names[idx] = names[j]
    xproj_knn_p2 = compute_metrics(xproj_knn_p2_names, xproj_true_names)
    print(f"    k-NN k=1 + P2: EM={xproj_knn_p2['em']:.1%}, F1={xproj_knn_p2['f1']:.4f}, "
          f"NgSim={xproj_knn_p2['ngsim']:.4f}, EdSim={xproj_knn_p2['edsim']:.4f}")

    # --- Per-package breakdown ---
    print(f"\n  Per-package cross-project results:")
    print(f"  {'Package':<12} {'N':>5} {'Dec EM':>7} {'Dec F1':>7} {'kNN EM':>7} {'kNN F1':>7} {'P2 EM':>7} {'P2 F1':>7}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    xproj_per_pkg = {}
    # Track cumulative position for per-package k-NN slicing
    pkg_start = 0
    for pkg in sorted(per_pkg_preds.keys()):
        pkg_preds = per_pkg_preds[pkg]
        n = len(pkg_preds)

        # Decoder metrics
        d_names = [p['pred_name'] for p in pkg_preds]
        t_names = [p['true_name'] for p in pkg_preds]
        dec_m = compute_metrics(d_names, t_names)

        # k-NN metrics — slice from all_demo_embeddings using pkg_labels
        pkg_indices = [j for j, lbl in enumerate(all_pkg_labels) if lbl == pkg]
        pkg_embs = all_demo_embeddings[pkg_indices]
        pkg_knn_names = batch_knn_lookup(pkg_embs, train_normed, train_names, k=1)
        knn_m = compute_metrics(pkg_knn_names, t_names)

        # k-NN + P2 per-package
        pkg_p2_names = [xproj_knn_p2_names[j] for j in pkg_indices]
        p2_m = compute_metrics(pkg_p2_names, t_names)

        xproj_per_pkg[pkg] = {'decoder': dec_m, 'knn_k1': knn_m, 'knn_p2': p2_m}

        print(f"  {pkg:<12} {n:>5} {dec_m['em']:>6.1%} {dec_m['f1']:>7.4f} "
              f"{knn_m['em']:>6.1%} {knn_m['f1']:>7.4f} "
              f"{p2_m['em']:>6.1%} {p2_m['f1']:>7.4f}")

    save_data['cross_project'] = {
        'decoder': xproj_decoder,
        'knn_k1': xproj_knn,
        'knn_p2': xproj_knn_p2,
        'per_package': xproj_per_pkg,
        'n_functions': len(all_preds),
        'n_packages': len(set(all_pkg_labels)),
    }

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Test Set ({test_decoder['n']} functions):")
    print(f"    Decoder: EM={test_decoder['em']:.1%}, F1={test_decoder['f1']:.4f}, "
          f"NgSim={test_decoder['ngsim']:.4f}, EdSim={test_decoder['edsim']:.4f}")
    print(f"    k-NN k=1: EM={test_knn['em']:.1%}, F1={test_knn['f1']:.4f}, "
          f"NgSim={test_knn['ngsim']:.4f}, EdSim={test_knn['edsim']:.4f}")
    print(f"    k-NN+P2:  EM={test_knn_p2['em']:.1%}, F1={test_knn_p2['f1']:.4f}, "
          f"NgSim={test_knn_p2['ngsim']:.4f}, EdSim={test_knn_p2['edsim']:.4f}")
    print(f"\n  Cross-Project ({xproj_decoder['n']} functions, {len(set(all_pkg_labels))} packages):")
    print(f"    Decoder: EM={xproj_decoder['em']:.1%}, F1={xproj_decoder['f1']:.4f}, "
          f"NgSim={xproj_decoder['ngsim']:.4f}, EdSim={xproj_decoder['edsim']:.4f}")
    print(f"    k-NN k=1: EM={xproj_knn['em']:.1%}, F1={xproj_knn['f1']:.4f}, "
          f"NgSim={xproj_knn['ngsim']:.4f}, EdSim={xproj_knn['edsim']:.4f}")
    print(f"    k-NN+P2:  EM={xproj_knn_p2['em']:.1%}, F1={xproj_knn_p2['f1']:.4f}, "
          f"NgSim={xproj_knn_p2['ngsim']:.4f}, EdSim={xproj_knn_p2['edsim']:.4f}")

    # --- Save ---
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {args.save}")


if __name__ == '__main__':
    main()
