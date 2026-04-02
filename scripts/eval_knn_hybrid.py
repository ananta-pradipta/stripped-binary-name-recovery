#!/usr/bin/env python3
"""
k-NN Hybrid Evaluation: Confidence-based fallback from decoder to k-NN retrieval.

Strategy B:
  - If beam score > threshold → use decoder prediction (model is confident)
  - If beam score <= threshold → use k-NN retrieval (model is uncertain)

This is inference-only — no model changes, no retraining.

Usage:
  python3 scripts/eval_knn_hybrid.py checkpoints/best_model_wulver.pt
  python3 scripts/eval_knn_hybrid.py checkpoints/best_model_wulver.pt --threshold -0.15
  python3 scripts/eval_knn_hybrid.py checkpoints/best_model_wulver.pt --sweep
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
import glob as glob_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import FunctionDataset, compute_block_features, compute_block_degrees, NUM_BLOCK_FEATURES
from src.evaluation.metrics import (
    compute_subtoken_f1,
    compute_exact_match,
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

            # --- Extract encoder embedding z ---
            with torch.amp.autocast('cuda', enabled=use_amp):
                block_embs = model.block_encoder(block_tokens, block_features=block_features)
                B, N, D = block_embs.shape
                x = block_embs.view(B * N, D)
                batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
                f, _, _ = model.graph_encoder(x, edge_index, batch_vec)

                # Fusion
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

            # Save embeddings
            all_embeddings.append(z.cpu().numpy())

            # --- Beam search predictions ---
            for i in range(B):
                tokens, score = model.decoder.generate(
                    z[i:i+1], sos_id, eos_id, beam_width
                )
                pred_name = sp_model.decode(tokens) if tokens else ""

                # Decode ground truth
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


def build_knn_index(train_embeddings):
    """Pre-normalize training embeddings for fast cosine similarity."""
    norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    return train_embeddings / (norms + 1e-8)


def batch_knn_lookup(query_embeddings, train_normed, train_names, k=1,
                     return_similarities=False):
    """Batch k-NN lookup using matrix multiplication. Returns list of nearest names
    and optionally their cosine similarities."""
    # Normalize queries
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_normed = query_embeddings / (norms + 1e-8)

    # Batch cosine similarity: [N_query, N_train]
    # Process in chunks to avoid OOM for large matrices
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
            top_k_idx = np.argsort(sims, axis=1)[:, -k:][:, ::-1]
            for row_idx, row in enumerate(top_k_idx):
                results.append(train_names[row[0]])  # Use top-1 for now
                similarities.append(sims[row_idx, row[0]])

    if return_similarities:
        return results, similarities
    return results


def evaluate_hybrid(test_preds, test_embeddings, train_normed, train_names,
                    threshold, k=1):
    """Evaluate with confidence-based k-NN fallback. Uses batch k-NN for speed."""
    results = {
        'decoder_used': 0,
        'knn_used': 0,
        'decoder_correct': 0,
        'knn_correct': 0,
        'decoder_f1_sum': 0.0,
        'knn_f1_sum': 0.0,
        'total_f1_sum': 0.0,
        'total_em': 0,
        'total_ngsim': 0.0,
        'total_edsim': 0.0,
        'details': [],
    }

    n = len(test_preds)

    # Split into decoder (confident) and knn (uncertain) sets
    knn_indices = []
    for i in range(n):
        if test_preds[i]['beam_score'] <= threshold:
            knn_indices.append(i)

    # Batch k-NN lookup for all uncertain functions at once
    knn_names = {}
    if knn_indices:
        knn_embeddings = test_embeddings[knn_indices]
        knn_results = batch_knn_lookup(knn_embeddings, train_normed, train_names, k=k)
        for idx, name in zip(knn_indices, knn_results):
            knn_names[idx] = name

    # Evaluate all functions
    for i in range(n):
        pred = test_preds[i]
        true_name = pred['true_name']
        decoder_name = pred['pred_name']
        beam_score = pred['beam_score']

        if i in knn_names:
            final_name = knn_names[i]
            source = 'knn'
            results['knn_used'] += 1
        else:
            final_name = decoder_name
            source = 'decoder'
            results['decoder_used'] += 1

        f1 = compute_subtoken_f1(final_name, true_name)
        em = compute_exact_match(final_name, true_name)
        ngsim = compute_char_ngram_similarity(final_name, true_name)
        edsim = compute_edit_distance_similarity(final_name, true_name)

        if source == 'decoder':
            results['decoder_f1_sum'] += f1
            if em:
                results['decoder_correct'] += 1
        else:
            results['knn_f1_sum'] += f1
            if em:
                results['knn_correct'] += 1

        results['total_f1_sum'] += f1
        if em:
            results['total_em'] += 1
        results['total_ngsim'] += ngsim
        results['total_edsim'] += edsim

        results['details'].append({
            'true_name': true_name,
            'decoder_name': decoder_name,
            'final_name': final_name,
            'beam_score': beam_score,
            'source': source,
            'f1': f1,
            'em': em,
        })

    return results


def evaluate_reverse_hybrid(test_preds, test_embeddings, train_normed, train_names,
                            threshold, k=1):
    """Reverse hybrid: k-NN default, decoder fallback when cosine similarity is low."""
    results = {
        'decoder_used': 0,
        'knn_used': 0,
        'decoder_correct': 0,
        'knn_correct': 0,
        'decoder_f1_sum': 0.0,
        'knn_f1_sum': 0.0,
        'total_f1_sum': 0.0,
        'total_em': 0,
        'total_ngsim': 0.0,
        'total_edsim': 0.0,
        'details': [],
    }

    n = len(test_preds)

    # Batch k-NN lookup for ALL functions (with similarities)
    knn_names, knn_sims = batch_knn_lookup(
        test_embeddings, train_normed, train_names, k=k, return_similarities=True
    )

    for i in range(n):
        pred = test_preds[i]
        true_name = pred['true_name']
        decoder_name = pred['pred_name']
        cos_sim = knn_sims[i]

        if cos_sim >= threshold:
            final_name = knn_names[i]
            source = 'knn'
            results['knn_used'] += 1
        else:
            final_name = decoder_name
            source = 'decoder'
            results['decoder_used'] += 1

        f1 = compute_subtoken_f1(final_name, true_name)
        em = compute_exact_match(final_name, true_name)
        ngsim = compute_char_ngram_similarity(final_name, true_name)
        edsim = compute_edit_distance_similarity(final_name, true_name)

        if source == 'decoder':
            results['decoder_f1_sum'] += f1
            if em:
                results['decoder_correct'] += 1
        else:
            results['knn_f1_sum'] += f1
            if em:
                results['knn_correct'] += 1

        results['total_f1_sum'] += f1
        if em:
            results['total_em'] += 1
        results['total_ngsim'] += ngsim
        results['total_edsim'] += edsim

        results['details'].append({
            'true_name': true_name,
            'decoder_name': decoder_name,
            'final_name': final_name,
            'cos_sim': cos_sim,
            'source': source,
            'f1': f1,
            'em': em,
        })

    return results


# --- Demo support ---
DEMO_PACKAGES = [
    ("diffutils", "diff", [""]),
    ("diffutils", "cmp", [""]),
    ("diffutils", "sdiff", [""]),
    ("diffutils", "diff3", [""]),
    ("datamash", "datamash", ["O0", "O2"]),
    ("direvent", "direvent", ["O0", "O2"]),
    ("csplit2", "cflow", ["O0", "O2"]),
    ("texinfo", "ginfo", ["O0", "O2"]),
    ("cppi", "cppi", ["O0", "O2"]),
    ("hello", "hello", ["O0", "O2"]),
    ("acct", "ac", ["O0", "O2"]),
    ("acct", "last", ["O0", "O2"]),
    ("acct", "lastcomm", ["O0", "O2"]),
    ("acct", "sa", ["O0", "O2"]),
    ("acct", "dump-utmp", ["O0", "O2"]),
    ("acct", "accton", ["O0", "O2"]),
    ("rush", "rush", ["O0", "O2"]),
    ("htop", "htop", ["O0", "O2"]),
    ("strace", "strace", ["O0", "O2"]),
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
                # Extract embedding
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

            # Beam search
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


def run_demo_knn_evaluation(model, cfg, token_vocab, ext_vocab, sp, device,
                             train_normed, train_names, args):
    """Run k-NN hybrid evaluation on demo set."""
    graphs_dirs = ["data/graphs", "demo/graphs"]
    labels_dirs = ["data/labels", "demo/labels"]
    ext_dirs = ["data/external_calls", "demo/external_calls"]

    per_pkg = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0,
                                    "ngsim_sum": 0, "edsim_sum": 0})
    per_pkg_baseline = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0})

    all_demo_preds = []
    all_demo_embeddings = []
    all_demo_true_names = []
    all_demo_pkg_labels = []

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

            # Match predictions to ground truth
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

                all_demo_preds.append({
                    'pred_name': pred_func['predicted_name'],
                    'true_name': true_name,
                    'beam_score': pred_func['beam_score'],
                })
                all_demo_embeddings.append(embs[i:i+1])
                all_demo_pkg_labels.append(pkg)
                matched += 1

                # Track baseline per pkg
                f1 = compute_subtoken_f1(pred_func['predicted_name'], true_name)
                em = 1 if pred_func['predicted_name'] == true_name else 0
                per_pkg_baseline[pkg]['correct'] += em
                per_pkg_baseline[pkg]['total'] += 1
                per_pkg_baseline[pkg]['f1_sum'] += f1

            print(f" {matched} matched")

    if not all_demo_preds:
        print("No demo predictions matched ground truth.")
        return

    demo_embeddings = np.concatenate(all_demo_embeddings, axis=0)
    n = len(all_demo_preds)

    baseline_f1 = sum(compute_subtoken_f1(p['pred_name'], p['true_name']) for p in all_demo_preds) / n
    baseline_em = sum(1 for p in all_demo_preds if p['pred_name'] == p['true_name']) / n
    print(f"\n  Demo predictions: {n}")
    print(f"  Baseline (decoder only): F1={baseline_f1:.4f}, EM={baseline_em:.4f}")

    scores = [p['beam_score'] for p in all_demo_preds]
    print(f"  Beam score distribution: min={min(scores):.3f}, median={np.median(scores):.3f}, "
          f"max={max(scores):.3f}, mean={np.mean(scores):.3f}")

    if args.sweep:
        if args.reverse:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            mode_label = "REVERSE"
            eval_fn = evaluate_reverse_hybrid
        else:
            thresholds = [-0.02, -0.05, -0.08, -0.10, -0.12, -0.15, -0.20, -0.25, -0.30, -0.40]
            mode_label = "FORWARD"
            eval_fn = evaluate_hybrid

        print(f"\n{'='*80}")
        print(f"THRESHOLD SWEEP ({mode_label}) — demo set ({n} functions)")
        print(f"{'='*80}")
        print(f"{'Threshold':>10} | {'Decoder%':>8} | {'k-NN%':>6} | {'F1':>6} | {'EM':>6} | "
              f"{'F1 delta':>8} | {'EM delta':>8}")
        print(f"{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

        best_f1 = baseline_f1
        best_threshold = None

        for t in thresholds:
            res = eval_fn(all_demo_preds, demo_embeddings, train_normed,
                          train_names, threshold=t, k=args.k)
            f1 = res['total_f1_sum'] / n
            em = res['total_em'] / n
            dec_pct = 100 * res['decoder_used'] / n
            knn_pct = 100 * res['knn_used'] / n
            f1_delta = f1 - baseline_f1
            em_delta = em - baseline_em

            marker = " ★" if f1 > best_f1 else ""
            print(f"{t:>10.3f} | {dec_pct:>7.1f}% | {knn_pct:>5.1f}% | {f1:>6.4f} | {em:>6.4f} | "
                  f"{f1_delta:>+8.4f} | {em_delta:>+8.4f}{marker}")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        print(f"\nBaseline: F1={baseline_f1:.4f}, EM={baseline_em:.4f}")
        if best_threshold is not None:
            print(f"Best threshold: {best_threshold} → F1={best_f1:.4f} (+{best_f1-baseline_f1:.4f})")
        else:
            print("No threshold improved over baseline.")

        # Per-package breakdown at best threshold
        best_t = best_threshold if best_threshold else (0.75 if args.reverse else -0.15)
        print(f"\n--- Per-package at threshold={best_t} ---")
        res = eval_fn(all_demo_preds, demo_embeddings, train_normed,
                      train_names, threshold=best_t, k=args.k)
        pkg_stats = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0})
        for i, detail in enumerate(res['details']):
            pkg = all_demo_pkg_labels[i]
            pkg_stats[pkg]['total'] += 1
            pkg_stats[pkg]['f1_sum'] += detail['f1']
            if detail['em']:
                pkg_stats[pkg]['correct'] += 1

        print(f"{'Package':>15} | {'Base EM':>8} | {'Hybrid EM':>9} | {'Delta':>7} | {'Base F1':>8} | {'Hybrid F1':>9}")
        print(f"{'-'*15}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}")
        for pkg in sorted(pkg_stats):
            ps = pkg_stats[pkg]
            pb = per_pkg_baseline[pkg]
            h_em = 100 * ps['correct'] / ps['total'] if ps['total'] > 0 else 0
            b_em = 100 * pb['correct'] / pb['total'] if pb['total'] > 0 else 0
            h_f1 = ps['f1_sum'] / ps['total'] if ps['total'] > 0 else 0
            b_f1 = pb['f1_sum'] / pb['total'] if pb['total'] > 0 else 0
            delta = h_em - b_em
            print(f"{pkg:>15} | {b_em:>7.1f}% | {h_em:>8.1f}% | {delta:>+6.1f}% | {b_f1:>8.4f} | {h_f1:>9.4f}")

    else:
        eval_fn = evaluate_reverse_hybrid if args.reverse else evaluate_hybrid
        mode_label = "reverse" if args.reverse else "forward"
        print(f"\n[Hybrid eval ({mode_label})] threshold={args.threshold}, k={args.k}")
        res = eval_fn(all_demo_preds, demo_embeddings, train_normed,
                      train_names, threshold=args.threshold, k=args.k)
        f1 = res['total_f1_sum'] / n
        em = res['total_em'] / n
        ngsim = res['total_ngsim'] / n
        edsim = res['total_edsim'] / n

        # Compute baseline ngsim/edsim
        baseline_ngsim = sum(compute_char_ngram_similarity(p['pred_name'], p['true_name']) for p in all_demo_preds) / n
        baseline_edsim = sum(compute_edit_distance_similarity(p['pred_name'], p['true_name']) for p in all_demo_preds) / n

        print(f"\n{'Metric':<12} | {'Baseline':>10} | {'Hybrid':>10} | {'Delta':>10}")
        print(f"{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        print(f"{'EM':<12} | {baseline_em:>10.4f} | {em:>10.4f} | {em-baseline_em:>+10.4f}")
        print(f"{'F1':<12} | {baseline_f1:>10.4f} | {f1:>10.4f} | {f1-baseline_f1:>+10.4f}")
        print(f"{'NgSim':<12} | {baseline_ngsim:>10.4f} | {ngsim:>10.4f} | {ngsim-baseline_ngsim:>+10.4f}")
        print(f"{'EdSim':<12} | {baseline_edsim:>10.4f} | {edsim:>10.4f} | {edsim-baseline_edsim:>+10.4f}")
        print(f"\nDecoder used: {res['decoder_used']} ({100*res['decoder_used']/n:.1f}%)")
        print(f"k-NN used:    {res['knn_used']} ({100*res['knn_used']/n:.1f}%)")

        # Per-package breakdown
        pkg_stats = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0,
                                          "ngsim_sum": 0, "edsim_sum": 0})
        for i, detail in enumerate(res['details']):
            pkg = all_demo_pkg_labels[i]
            pkg_stats[pkg]['total'] += 1
            pkg_stats[pkg]['f1_sum'] += detail['f1']
            pkg_stats[pkg]['ngsim_sum'] += compute_char_ngram_similarity(detail['final_name'], detail['true_name'])
            pkg_stats[pkg]['edsim_sum'] += compute_edit_distance_similarity(detail['final_name'], detail['true_name'])
            if detail['em']:
                pkg_stats[pkg]['correct'] += 1

        print(f"\n{'Package':>15} | {'EM':>7} | {'F1':>6} | {'NgSim':>6} | {'EdSim':>6}")
        print(f"{'-'*15}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
        for pkg in sorted(pkg_stats):
            ps = pkg_stats[pkg]
            t = ps['total']
            print(f"{pkg:>15} | {100*ps['correct']/t:>6.1f}% | {ps['f1_sum']/t:>6.4f} | "
                  f"{ps['ngsim_sum']/t:>6.4f} | {ps['edsim_sum']/t:>6.4f}")


def main():
    parser = argparse.ArgumentParser(description='k-NN Hybrid Evaluation')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--threshold', type=float, default=-0.15,
                        help='Beam score threshold (below = use k-NN)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep thresholds from -0.05 to -0.40')
    parser.add_argument('--k', type=int, default=1, help='Number of nearest neighbors')
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--splits', nargs='+', default=None,
                        help='Splits to evaluate (default: test unless --demo)')
    parser.add_argument('--demo', action='store_true',
                        help='Run on demo set (unseen packages)')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse hybrid: k-NN default, decoder fallback when cosine sim is low')
    args = parser.parse_args()

    # Default splits: test unless --demo is used alone
    if args.splits is None and not args.demo:
        args.splits = ['test']
    elif args.splits is None:
        args.splits = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load checkpoint
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

    train_idx, val_idx, test_idx = dataset.get_splits(
        cfg['data']['train_split'], cfg['data']['val_split']
    )
    split_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    # Build model
    ckpt_state = ckpt['model_state_dict']
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    if 'ext_encoder.embedding.weight' in ckpt_state:
        cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = ckpt_state['decoder.embedding.weight'].shape[0]

    model = FunctionNamer(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Val F1: {ckpt.get('val_f1', '?'):.4f}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Step 1: Extract training embeddings + names ---
    print(f"\n[Step 1] Extracting training embeddings ({len(train_idx)} functions)...")
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
    print(f"  Train embeddings: {train_embeddings.shape}")

    # Pre-normalize for fast cosine similarity
    print("  Building k-NN index (normalizing)...")
    train_normed = build_knn_index(train_embeddings)

    # --- Step 2: Extract test embeddings + predictions ---
    for split_name in args.splits:
        if split_name not in split_map:
            print(f"Unknown split: {split_name}")
            continue

        print(f"\n[Step 2] Extracting {split_name} embeddings + beam search predictions...")
        test_loader = DataLoader(
            Subset(dataset, split_map[split_name]),
            batch_size=cfg['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        test_embeddings, test_preds = extract_embeddings_and_predict(
            model, test_loader, device, sp_model,
            desc=f"{split_name} predict", beam_width=args.beam_width, use_amp=args.amp
        )
        print(f"  {split_name} embeddings: {test_embeddings.shape}")
        print(f"  {split_name} predictions: {len(test_preds)}")

        # Baseline: decoder-only metrics
        n = len(test_preds)
        baseline_f1 = sum(compute_subtoken_f1(p['pred_name'], p['true_name']) for p in test_preds) / n
        baseline_em = sum(1 for p in test_preds if compute_exact_match(p['pred_name'], p['true_name'])) / n
        print(f"\n  Baseline (decoder only): F1={baseline_f1:.4f}, EM={baseline_em:.4f}")

        # Score distribution
        scores = [p['beam_score'] for p in test_preds]
        print(f"  Beam score distribution: min={min(scores):.3f}, median={np.median(scores):.3f}, "
              f"max={max(scores):.3f}, mean={np.mean(scores):.3f}")

        # --- Step 3: Evaluate hybrid ---
        if args.sweep:
            if args.reverse:
                # Reverse: k-NN default, decoder fallback when cosine sim < threshold
                thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
                mode_label = "REVERSE"
                eval_fn = evaluate_reverse_hybrid
            else:
                thresholds = [-0.02, -0.05, -0.08, -0.10, -0.12, -0.15, -0.20, -0.25, -0.30, -0.40]
                mode_label = "FORWARD"
                eval_fn = evaluate_hybrid

            print(f"\n{'='*80}")
            print(f"THRESHOLD SWEEP ({mode_label}) — {split_name} set ({n} functions)")
            print(f"{'='*80}")
            print(f"{'Threshold':>10} | {'Decoder%':>8} | {'k-NN%':>6} | {'F1':>6} | {'EM':>6} | "
                  f"{'F1 delta':>8} | {'EM delta':>8}")
            print(f"{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

            best_f1 = baseline_f1
            best_threshold = None

            for t in thresholds:
                res = eval_fn(test_preds, test_embeddings, train_normed,
                              train_names, threshold=t, k=args.k)
                f1 = res['total_f1_sum'] / n
                em = res['total_em'] / n
                dec_pct = 100 * res['decoder_used'] / n
                knn_pct = 100 * res['knn_used'] / n
                f1_delta = f1 - baseline_f1
                em_delta = em - baseline_em

                marker = " ★" if f1 > best_f1 else ""
                print(f"{t:>10.3f} | {dec_pct:>7.1f}% | {knn_pct:>5.1f}% | {f1:>6.4f} | {em:>6.4f} | "
                      f"{f1_delta:>+8.4f} | {em_delta:>+8.4f}{marker}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

            print(f"\nBaseline: F1={baseline_f1:.4f}, EM={baseline_em:.4f}")
            if best_threshold is not None:
                print(f"Best threshold: {best_threshold} → F1={best_f1:.4f} (+{best_f1-baseline_f1:.4f})")
            else:
                print("No threshold improved over baseline.")

        else:
            # Single threshold evaluation
            eval_fn = evaluate_reverse_hybrid if args.reverse else evaluate_hybrid
            mode_label = "reverse" if args.reverse else "forward"
            print(f"\n[Step 3] Hybrid evaluation ({mode_label}, threshold={args.threshold}, k={args.k})...")
            res = eval_fn(test_preds, test_embeddings, train_normed,
                          train_names, threshold=args.threshold, k=args.k)

            f1 = res['total_f1_sum'] / n
            em = res['total_em'] / n
            ngsim = res['total_ngsim'] / n
            edsim = res['total_edsim'] / n

            print(f"\n{'='*70}")
            print(f"k-NN HYBRID RESULTS — {split_name}")
            print(f"{'='*70}")
            print(f"Threshold: {args.threshold}, k={args.k}")
            print(f"Decoder used: {res['decoder_used']} ({100*res['decoder_used']/n:.1f}%)")
            print(f"k-NN used:    {res['knn_used']} ({100*res['knn_used']/n:.1f}%)")
            print(f"")
            print(f"{'Metric':<12} | {'Baseline':>10} | {'Hybrid':>10} | {'Delta':>10}")
            print(f"{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
            print(f"{'F1':<12} | {baseline_f1:>10.4f} | {f1:>10.4f} | {f1-baseline_f1:>+10.4f}")
            print(f"{'EM':<12} | {baseline_em:>10.4f} | {em:>10.4f} | {em-baseline_em:>+10.4f}")
            print(f"{'NgSim':<12} | {'—':>10} | {ngsim:>10.4f} | {'—':>10}")
            print(f"{'EdSim':<12} | {'—':>10} | {edsim:>10.4f} | {'—':>10}")

            if res['decoder_used'] > 0:
                dec_f1 = res['decoder_f1_sum'] / res['decoder_used']
                dec_em = res['decoder_correct'] / res['decoder_used']
                print(f"\nDecoder bucket: F1={dec_f1:.4f}, EM={dec_em:.4f} ({res['decoder_used']} functions)")
            if res['knn_used'] > 0:
                knn_f1 = res['knn_f1_sum'] / res['knn_used']
                knn_em = res['knn_correct'] / res['knn_used']
                print(f"k-NN bucket:    F1={knn_f1:.4f}, EM={knn_em:.4f} ({res['knn_used']} functions)")

            # Save results
            save_data = {
                'config': {
                    'threshold': args.threshold,
                    'k': args.k,
                    'beam_width': args.beam_width,
                    'split': split_name,
                    'checkpoint': args.checkpoint,
                },
                'baseline': {'f1': baseline_f1, 'em': baseline_em},
                'hybrid': {'f1': f1, 'em': em, 'ngsim': ngsim, 'edsim': edsim},
                'decoder_used': res['decoder_used'],
                'knn_used': res['knn_used'],
            }

            os.makedirs('results', exist_ok=True)
            save_path = f'results/knn_hybrid_{split_name}.json'
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"\nSaved to {save_path}")

    # --- Demo evaluation ---
    if args.demo:
        print(f"\n{'='*80}")
        print(f"=== Demo set — k-NN hybrid evaluation ===")
        print(f"{'='*80}")
        # Use config from checkpoint for demo (has correct encoder settings)
        demo_cfg = ckpt.get('config', cfg)
        run_demo_knn_evaluation(model, demo_cfg, token_vocab, ext_vocab, sp_model, device,
                                 train_normed, train_names, args)


if __name__ == '__main__':
    main()
