#!/usr/bin/env python3
"""
Extract per-function decoder predictions for cross-project packages.

Lightweight script: loads checkpoint, runs decoder inference on cross-project
binaries, saves per-function predictions to JSON. No k-NN, no training
embedding extraction.

Usage:
  python3 scripts/extract_predictions.py checkpoints/best_model.pt \
      --config configs/optimized_large.yaml --amp \
      --save results/cross_project_predictions.json
"""
import argparse
import json
import os
import sys
import yaml
import torch
from tqdm import tqdm
from collections import defaultdict
import glob as glob_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import compute_block_features, compute_block_degrees, NUM_BLOCK_FEATURES


# Cross-project packages
DEMO_BINARIES = [
    ("pigz", "pigz", ["O0", "O2"]),
    ("rsync", "rsync", ["O0", "O2"]),
    ("dropbear", "dropbear", ["O0", "O2"]),
    ("dropbear", "dbclient", ["O0", "O2"]),
    ("dropbear", "dropbearkey", ["O0", "O2"]),
    ("socat", "socat", ["O0", "O2"]),
    ("tmux", "tmux", ["O0", "O2"]),
    ("tengine", "nginx", ["O0", "O2"]),
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


def predict_binary(functions, ext_by_func, model, token_vocab, ext_vocab,
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
    for func_name, func_data in tqdm(targets.items(), desc="Predict", leave=False):
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
            norm_score = score / max(len(pred_tokens), 1) if pred_tokens else -999.0

        results.append({
            'address': func_data['address'],
            'bap_name': func_name,
            'predicted_name': pred_name,
            'beam_score': float(norm_score),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Extract cross-project decoder predictions')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--amp', action='store_true', help='Use AMP for inference')
    parser.add_argument('--save', default='results/cross_project_predictions.json',
                        help='Path to save JSON predictions')
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

    # --- Build model ---
    ckpt_state = ckpt['model_state_dict']
    # Infer vocab sizes from checkpoint
    if token_vocab:
        cfg['block_encoder']['token_vocab_size'] = len(token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    if 'ext_encoder.embedding.weight' in ckpt_state:
        cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = ckpt_state['decoder.embedding.weight'].shape[0]

    model = FunctionNamer(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model: Epoch {ckpt.get('epoch', '?')}, Val F1: {ckpt.get('val_f1', '?'):.4f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Run cross-project inference ---
    demo_cfg = ckpt.get('config', cfg)
    graphs_dirs = ["data/graphs", "demo/graphs"]
    labels_dirs = ["data/labels", "demo/labels"]
    ext_dirs = ["data/external_calls", "demo/external_calls"]

    all_predictions = []
    summary = {}

    for pkg, binary, opt_levels in DEMO_BINARIES:
        for opt in opt_levels:
            bin_name = f"{pkg}_{binary}_{opt}"

            functions = load_functions_from_graphs(bin_name, graphs_dirs)
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
            preds = predict_binary(
                functions, ext_by_func, model, token_vocab, ext_vocab,
                sp_model, demo_cfg, device, beam_width=args.beam_width, use_amp=args.amp
            )

            matched = 0
            for pred_func in preds:
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

                all_predictions.append({
                    'package': pkg,
                    'binary': bin_name,
                    'address': addr,
                    'true_name': true_name,
                    'predicted_name': pred_func['predicted_name'],
                    'beam_score': pred_func['beam_score'],
                })
                matched += 1

            print(f" {matched} matched")
            summary[bin_name] = matched

    # --- Save ---
    print(f"\nTotal predictions: {len(all_predictions)}")
    n_correct = sum(1 for p in all_predictions if p['predicted_name'] == p['true_name'])
    print(f"Exact matches: {n_correct}/{len(all_predictions)} ({n_correct/max(len(all_predictions),1):.1%})")

    # Per-package summary
    pkg_counts = defaultdict(lambda: {'total': 0, 'correct': 0})
    for p in all_predictions:
        pkg_counts[p['package']]['total'] += 1
        if p['predicted_name'] == p['true_name']:
            pkg_counts[p['package']]['correct'] += 1

    print(f"\nPer-package:")
    for pkg in sorted(pkg_counts.keys()):
        c = pkg_counts[pkg]
        print(f"  {pkg:<12} {c['correct']}/{c['total']} ({c['correct']/max(c['total'],1):.1%})")

    output = {
        'checkpoint': args.checkpoint,
        'beam_width': args.beam_width,
        'n_predictions': len(all_predictions),
        'n_exact_match': n_correct,
        'predictions': all_predictions,
    }

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.save}")


if __name__ == '__main__':
    main()
