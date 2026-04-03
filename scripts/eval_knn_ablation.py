#!/usr/bin/env python3
"""
k-NN Ablation Study: Systematically evaluate k-NN configurations for the paper.

Tests:
  1. k values: k=1, k=3, k=5, k=10 (majority vote for k>1)
  2. Strategies: forward hybrid, reverse hybrid, k-NN only
  3. Threshold sweep for best k on test set

Outputs a clean markdown summary table.

Usage:
  python3 scripts/eval_knn_ablation.py checkpoints/best_model_wulver.pt
  python3 scripts/eval_knn_ablation.py checkpoints/best_model_wulver.pt --config configs/optimized_large.yaml --amp
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
# Collate (same as eval_knn_hybrid.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Embedding extraction (reused from eval_knn_hybrid.py)
# ---------------------------------------------------------------------------
def extract_embeddings_and_predict(model, loader, device, sp_model, desc="",
                                   beam_width=5, use_amp=False):
    """Extract encoder embeddings z and beam search predictions+scores."""
    model.eval()
    all_embeddings = []
    all_predictions = []

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

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_predictions


# ---------------------------------------------------------------------------
# k-NN with majority vote
# ---------------------------------------------------------------------------
def build_knn_index(train_embeddings):
    """Pre-normalize training embeddings for fast cosine similarity."""
    norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    return train_embeddings / (norms + 1e-8)


def batch_knn_lookup(query_embeddings, train_normed, train_names, k=1,
                     return_similarities=False):
    """Batch k-NN lookup with majority vote for k>1.

    For k=1: returns the nearest neighbor name.
    For k>1: returns the most common name among the k nearest neighbors
             (ties broken by highest cosine similarity).
    """
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_normed = query_embeddings / (norms + 1e-8)

    chunk_size = 500
    results = []
    similarities = []

    for start in range(0, len(query_normed), chunk_size):
        end = min(start + chunk_size, len(query_normed))
        sims = query_normed[start:end] @ train_normed.T  # [chunk, N_train]

        if k == 1:
            best_idx = np.argmax(sims, axis=1)
            best_sims = np.max(sims, axis=1)
            for idx in best_idx:
                results.append(train_names[idx])
            similarities.extend(best_sims.tolist())
        else:
            # Get top-k indices (sorted descending by similarity)
            top_k_idx = np.argsort(sims, axis=1)[:, -k:][:, ::-1]
            for row_idx, row in enumerate(top_k_idx):
                # Majority vote: pick the most common name among top-k
                candidate_names = [train_names[idx] for idx in row]
                candidate_sims = [sims[row_idx, idx] for idx in row]

                # Count votes
                name_counts = Counter(candidate_names)
                # Find the max vote count
                max_votes = max(name_counts.values())
                # Among names with max votes, pick the one with highest similarity
                best_name = None
                best_sim = -999
                for name, sim in zip(candidate_names, candidate_sims):
                    if name_counts[name] == max_votes and sim > best_sim:
                        best_name = name
                        best_sim = sim

                results.append(best_name)
                # Return similarity of the top-1 neighbor (used for threshold decisions)
                similarities.append(candidate_sims[0])

    if return_similarities:
        return results, similarities
    return results


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------
def compute_metrics(final_names, true_names):
    """Compute EM, F1, NgSim, EdSim over lists of predictions and ground truths."""
    n = len(final_names)
    em = sum(1 for p, t in zip(final_names, true_names) if p == t)
    f1_sum = sum(compute_subtoken_f1(p, t) for p, t in zip(final_names, true_names))
    ngsim_sum = sum(compute_char_ngram_similarity(p, t) for p, t in zip(final_names, true_names))
    edsim_sum = sum(compute_edit_distance_similarity(p, t) for p, t in zip(final_names, true_names))
    return {
        'n': n,
        'em': em / n if n > 0 else 0,
        'f1': f1_sum / n if n > 0 else 0,
        'ngsim': ngsim_sum / n if n > 0 else 0,
        'edsim': edsim_sum / n if n > 0 else 0,
    }


def evaluate_forward_hybrid(preds, embeddings, train_normed, train_names,
                            threshold, k=1):
    """Forward hybrid: decoder default, k-NN fallback when beam score <= threshold."""
    knn_indices = [i for i, p in enumerate(preds) if p['beam_score'] <= threshold]

    knn_names = {}
    if knn_indices:
        knn_embs = embeddings[knn_indices]
        knn_results = batch_knn_lookup(knn_embs, train_normed, train_names, k=k)
        for idx, name in zip(knn_indices, knn_results):
            knn_names[idx] = name

    final_names = []
    true_names = []
    n_knn = 0
    for i, p in enumerate(preds):
        true_names.append(p['true_name'])
        if i in knn_names:
            final_names.append(knn_names[i])
            n_knn += 1
        else:
            final_names.append(p['pred_name'])

    metrics = compute_metrics(final_names, true_names)
    metrics['n_knn'] = n_knn
    metrics['n_decoder'] = len(preds) - n_knn
    return metrics


def evaluate_reverse_hybrid(preds, embeddings, train_normed, train_names,
                            threshold, k=1):
    """Reverse hybrid: k-NN default, decoder fallback when cosine sim < threshold."""
    knn_results, knn_sims = batch_knn_lookup(
        embeddings, train_normed, train_names, k=k, return_similarities=True
    )

    final_names = []
    true_names = []
    n_knn = 0
    for i, p in enumerate(preds):
        true_names.append(p['true_name'])
        if knn_sims[i] >= threshold:
            final_names.append(knn_results[i])
            n_knn += 1
        else:
            final_names.append(p['pred_name'])

    metrics = compute_metrics(final_names, true_names)
    metrics['n_knn'] = n_knn
    metrics['n_decoder'] = len(preds) - n_knn
    return metrics


def evaluate_knn_only(preds, embeddings, train_normed, train_names, k=1):
    """k-NN only: no decoder at all."""
    knn_results = batch_knn_lookup(embeddings, train_normed, train_names, k=k)

    true_names = [p['true_name'] for p in preds]
    metrics = compute_metrics(knn_results, true_names)
    metrics['n_knn'] = len(preds)
    metrics['n_decoder'] = 0
    return metrics


def evaluate_decoder_only(preds):
    """Decoder only baseline."""
    final_names = [p['pred_name'] for p in preds]
    true_names = [p['true_name'] for p in preds]
    metrics = compute_metrics(final_names, true_names)
    metrics['n_knn'] = 0
    metrics['n_decoder'] = len(preds)
    return metrics


# ---------------------------------------------------------------------------
# Demo support (same as eval_knn_hybrid.py)
# ---------------------------------------------------------------------------
DEMO_PACKAGES = [
    # Truly unseen cross-project packages (not in training)
    ("diffutils", "diff", [""]),
    ("diffutils", "cmp", [""]),
    ("diffutils", "sdiff", [""]),
    ("diffutils", "diff3", [""]),
    ("curl", "curl", ["O0", "O2"]),
    ("nginx", "nginx", ["O0", "O2"]),
    ("csplit2", "cflow", ["O0", "O2"]),
    ("datamash", "datamash", ["O0", "O2"]),
    ("rcs", "rcs", ["O0", "O2"]),
    ("cppi", "cppi", ["O0", "O2"]),
    ("hello", "hello", ["O0", "O2"]),
    ("tree", "tree", ["O0", "O2"]),
    ("dos2unix", "dos2unix", ["O0", "O2"]),
    ("dos2unix", "unix2dos", ["O0", "O2"]),
    ("bzip2", "bzip2", ["O0", "O2"]),
]


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
    """Run model inference on demo functions, returning predictions + embeddings."""
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
            if graph is None and name.startswith('sub_'):
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


def extract_demo_data(model, cfg, token_vocab, ext_vocab, sp, device, args):
    """Extract demo predictions and embeddings for all demo packages."""
    graphs_dirs = ["data/graphs", "demo/graphs"]
    labels_dirs = ["data/labels", "demo/labels"]
    ext_dirs = ["data/external_calls", "demo/external_calls"]

    all_preds = []
    all_embeddings = []
    all_pkg_labels = []

    for pkg, binary, opt_levels in DEMO_PACKAGES:
        for opt in opt_levels:
            bin_name = f"{pkg}_{binary}_{opt}" if opt else f"{pkg}_{binary}"

            functions = load_functions_from_graphs(bin_name, graphs_dirs)
            if not functions:
                alt_name = f"diffutils_{binary}"
                functions = load_functions_from_graphs(alt_name, graphs_dirs)
                if functions:
                    bin_name = alt_name
            if not functions:
                print(f"  SKIP {bin_name}: no graph files found")
                continue

            ext_by_func = load_external_calls(bin_name, ext_dirs)
            gt = load_ground_truth(bin_name, labels_dirs)
            if not gt:
                print(f"  SKIP {bin_name}: no ground truth labels")
                continue

            resolve_thunks(functions, ext_by_func)

            print(f"  Predicting {bin_name} ({len(gt)} GT)...", end="", flush=True)
            preds, embs = predict_binary_with_embeddings(
                functions, ext_by_func, model, token_vocab, ext_vocab,
                sp, cfg, device, beam_width=args.beam_width, use_amp=args.amp
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

                all_preds.append({
                    'pred_name': pred_func['predicted_name'],
                    'true_name': true_name,
                    'beam_score': pred_func['beam_score'],
                })
                all_embeddings.append(embs[i:i+1])
                all_pkg_labels.append(pkg)
                matched += 1

            print(f" {matched} matched")

    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        embeddings = np.array([])

    return all_preds, embeddings, all_pkg_labels


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------
def run_ablation(preds, embeddings, train_normed, train_names, split_name,
                 k_values=(1, 3, 5, 10),
                 forward_thresholds=(-0.02, -0.05, -0.10, -0.15, -0.20),
                 reverse_thresholds=(0.60, 0.70, 0.75, 0.80, 0.85, 0.90)):
    """Run the full ablation grid and return a list of result rows."""
    n = len(preds)
    rows = []

    # --- Decoder-only baseline ---
    baseline = evaluate_decoder_only(preds)
    rows.append({
        'config': 'Decoder only',
        'k': '-',
        'threshold': '-',
        **baseline,
    })
    print(f"  Decoder only: EM={baseline['em']:.4f}, F1={baseline['f1']:.4f}")

    # --- k-NN only (various k) ---
    print(f"\n  k-NN only ablation...")
    best_knn_k = 1
    best_knn_f1 = 0
    for k in k_values:
        m = evaluate_knn_only(preds, embeddings, train_normed, train_names, k=k)
        rows.append({
            'config': f'k-NN only (k={k})',
            'k': k,
            'threshold': '-',
            **m,
        })
        print(f"    k={k}: EM={m['em']:.4f}, F1={m['f1']:.4f}")
        if m['f1'] > best_knn_f1:
            best_knn_f1 = m['f1']
            best_knn_k = k

    # --- Forward hybrid (various k, best threshold from prior work: -0.02) ---
    print(f"\n  Forward hybrid ablation (decoder default, k-NN fallback)...")
    for k in k_values:
        for t in forward_thresholds:
            m = evaluate_forward_hybrid(preds, embeddings, train_normed, train_names,
                                        threshold=t, k=k)
            rows.append({
                'config': f'Forward (k={k}, t={t})',
                'k': k,
                'threshold': t,
                **m,
            })
            knn_pct = 100 * m['n_knn'] / n
            print(f"    k={k}, t={t}: EM={m['em']:.4f}, F1={m['f1']:.4f} "
                  f"(k-NN: {knn_pct:.1f}%)")

    # --- Reverse hybrid (various k, threshold on cosine sim) ---
    print(f"\n  Reverse hybrid ablation (k-NN default, decoder fallback)...")
    for k in k_values:
        for t in reverse_thresholds:
            m = evaluate_reverse_hybrid(preds, embeddings, train_normed, train_names,
                                        threshold=t, k=k)
            rows.append({
                'config': f'Reverse (k={k}, t={t})',
                'k': k,
                'threshold': t,
                **m,
            })
            knn_pct = 100 * m['n_knn'] / n
            print(f"    k={k}, t={t}: EM={m['em']:.4f}, F1={m['f1']:.4f} "
                  f"(k-NN: {knn_pct:.1f}%)")

    return rows


def print_summary_table(test_rows, demo_rows=None):
    """Print a clean markdown summary table."""
    print(f"\n{'='*100}")
    print("k-NN ABLATION SUMMARY")
    print(f"{'='*100}")

    # --- Test set table ---
    if test_rows:
        print(f"\n## Test Set ({test_rows[0]['n']} functions)\n")
        print(f"| {'Config':<30} | {'EM':>7} | {'F1':>7} | {'NgSim':>7} | {'EdSim':>7} | {'k-NN%':>6} |")
        print(f"|{'-'*32}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*8}|")

        best_f1 = max(r['f1'] for r in test_rows)
        for r in test_rows:
            knn_pct = 100 * r['n_knn'] / r['n'] if r['n'] > 0 else 0
            marker = " **" if r['f1'] == best_f1 else ""
            print(f"| {r['config']:<30} | {100*r['em']:>6.1f}% | {r['f1']:>7.4f} | "
                  f"{r['ngsim']:>7.4f} | {r['edsim']:>7.4f} | {knn_pct:>5.1f}% |{marker}")

    # --- Demo set table ---
    if demo_rows:
        print(f"\n## Demo Set ({demo_rows[0]['n']} functions)\n")
        print(f"| {'Config':<30} | {'EM':>7} | {'F1':>7} | {'NgSim':>7} | {'EdSim':>7} | {'k-NN%':>6} |")
        print(f"|{'-'*32}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*8}|")

        best_f1 = max(r['f1'] for r in demo_rows)
        for r in demo_rows:
            knn_pct = 100 * r['n_knn'] / r['n'] if r['n'] > 0 else 0
            marker = " **" if r['f1'] == best_f1 else ""
            print(f"| {r['config']:<30} | {100*r['em']:>6.1f}% | {r['f1']:>7.4f} | "
                  f"{r['ngsim']:>7.4f} | {r['edsim']:>7.4f} | {knn_pct:>5.1f}% |{marker}")


def main():
    parser = argparse.ArgumentParser(description='k-NN Ablation Study')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--amp', action='store_true', help='Use AMP for inference')
    parser.add_argument('--demo', action='store_true', help='Also run on demo set')
    parser.add_argument('--demo-only', action='store_true', help='Only run on demo set')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 3, 5, 10],
                        help='k values to test (default: 1 3 5 10)')
    parser.add_argument('--save', default='results/knn_ablation.json',
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

    # --- Step 1: Extract training embeddings ---
    print(f"\n{'='*80}")
    print(f"STEP 1: Extract training embeddings ({len(train_idx)} functions)")
    print(f"{'='*80}")
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
    print(f"  Train embeddings shape: {train_embeddings.shape}")

    print("  Building k-NN index (normalizing)...")
    train_normed = build_knn_index(train_embeddings)

    all_results = {'k_values': args.k_values}

    # --- Step 2: Test set ablation ---
    test_rows = None
    if not args.demo_only:
        print(f"\n{'='*80}")
        print(f"STEP 2: Test set ablation ({len(test_idx)} functions)")
        print(f"{'='*80}")
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
        print(f"  Test embeddings shape: {test_embeddings.shape}")

        test_rows = run_ablation(
            test_preds, test_embeddings, train_normed, train_names,
            split_name="test", k_values=args.k_values,
        )
        all_results['test'] = test_rows

    # --- Step 3: Demo set ablation ---
    demo_rows = None
    if args.demo or args.demo_only:
        print(f"\n{'='*80}")
        print(f"STEP 3: Demo set ablation")
        print(f"{'='*80}")
        demo_cfg = ckpt.get('config', cfg)
        demo_preds, demo_embeddings, demo_pkg_labels = extract_demo_data(
            model, demo_cfg, token_vocab, ext_vocab, sp_model, device, args
        )

        if demo_preds:
            print(f"  Demo predictions: {len(demo_preds)}")
            demo_rows = run_ablation(
                demo_preds, demo_embeddings, train_normed, train_names,
                split_name="demo", k_values=args.k_values,
            )
            all_results['demo'] = demo_rows

            # --- Per-package breakdown for best config ---
            # Find best demo config
            best_config = max(demo_rows, key=lambda r: r['f1'])
            print(f"\n  Best demo config: {best_config['config']} "
                  f"(F1={best_config['f1']:.4f}, EM={100*best_config['em']:.1f}%)")

    # --- Print summary ---
    print_summary_table(test_rows, demo_rows)

    # --- Save results ---
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    # Convert rows to JSON-serializable format
    save_data = {
        'checkpoint': args.checkpoint,
        'k_values': args.k_values,
        'beam_width': args.beam_width,
    }
    if test_rows:
        save_data['test'] = [{k: v for k, v in r.items()} for r in test_rows]
    if demo_rows:
        save_data['demo'] = [{k: v for k, v in r.items()} for r in demo_rows]

    with open(args.save, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {args.save}")


if __name__ == '__main__':
    main()
