#!/usr/bin/env python3
"""
Wulver-compatible demo evaluation — NO BAP required.
Loads pre-processed graphs, labels, and external calls from disk,
runs model inference, and computes metrics.

Usage:
  python3 scripts/eval_demo_wulver.py checkpoints/best_model.pt
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import compute_block_features, compute_block_degrees, NUM_BLOCK_FEATURES
from src.evaluation.metrics import (
    compute_subtoken_f1,
    compute_char_ngram_similarity,
    compute_edit_distance_similarity,
)
from collections import defaultdict

# Demo packages: (package, binary_name, opt_levels)
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
    """Load all per-function graph JSONs for a binary."""
    functions = {}
    for gdir in graphs_dirs:
        pattern = os.path.join(gdir, f"{bin_name}_*.json")
        for path in glob.glob(pattern):
            with open(path) as f:
                data = json.load(f)
            fname = data.get('function_name', '')
            if fname:
                functions[fname] = data
    return functions


def load_external_calls(bin_name, ext_dirs):
    """Load external calls for a binary."""
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
    """Load ground truth labels (addr -> name) from labels JSON.

    Labels format: {"binary": "...", "num_functions": N, "functions": {"name": "0xADDR", ...}}
    We invert to addr -> name for matching.
    """
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
                        # Normalize address: strip leading zeros
                        if addr.startswith('0x'):
                            addr_norm = '0x' + addr[2:].lstrip('0')
                            if addr_norm == '0x':
                                addr_norm = '0x0'
                        else:
                            addr_norm = '0x' + addr.lstrip('0')
                            if addr_norm == '0x':
                                addr_norm = '0x0'
                        gt[addr_norm] = name
                        gt[addr] = name  # also keep full form
            return gt
    return {}


def resolve_thunks(functions, ext_by_func):
    """Resolve ENDBR64 thunks (O0 fix)."""
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
                   sp, cfg, device, beam_width=5):
    """Run model inference on all sub_XXXX functions."""
    sos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # Build address index for callee resolution
    func_by_addr = {}
    for fname, fdata in functions.items():
        addr = fdata.get('address', '')
        if addr:
            func_by_addr[addr] = fdata

    # Build reverse call graph for caller context
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

    def get_callee_tokens(func_data, max_callees=5, max_sig_tokens=10):
        callee_sigs = []
        for callee_name in func_data.get('internal_callees', [])[:max_callees]:
            callee_graph = functions.get(callee_name)
            if callee_graph is None and callee_name.startswith('sub_'):
                addr = '0x' + callee_name[4:]
                callee_graph = func_by_addr.get(addr)
            if callee_graph is None:
                callee_sigs.append([0] * max_sig_tokens)
                continue
            sig = []
            for block in callee_graph['blocks'][:3]:
                sig.extend(block['tokens'][:5])
                if len(sig) >= max_sig_tokens:
                    break
            sig_ids = [token_vocab.get(t, token_vocab.get('<UNK>', 1))
                       for t in sig[:max_sig_tokens]]
            sig_ids += [0] * (max_sig_tokens - len(sig_ids))
            callee_sigs.append(sig_ids)
        while len(callee_sigs) < max_callees:
            callee_sigs.append([0] * max_sig_tokens)
        return callee_sigs

    def get_caller_tokens(func_name, max_callers=5, max_sig_tokens=10):
        caller_sigs = []
        for caller_name in callers_of.get(func_name, [])[:max_callers]:
            caller_graph = functions.get(caller_name)
            if caller_graph is None and caller_name.startswith('sub_'):
                addr = '0x' + caller_name[4:]
                caller_graph = func_by_addr.get(addr)
            if caller_graph is None:
                caller_sigs.append([0] * max_sig_tokens)
                continue
            sig = []
            for block in caller_graph['blocks'][:3]:
                sig.extend(block['tokens'][:5])
                if len(sig) >= max_sig_tokens:
                    break
            sig_ids = [token_vocab.get(t, token_vocab.get('<UNK>', 1))
                       for t in sig[:max_sig_tokens]]
            sig_ids += [0] * (max_sig_tokens - len(sig_ids))
            caller_sigs.append(sig_ids)
        while len(caller_sigs) < max_callers:
            caller_sigs.append([0] * max_sig_tokens)
        return caller_sigs

    # Filter to sub_XXXX with >= 2 blocks
    targets = {name: data for name, data in functions.items()
               if name.startswith('sub_') and data['num_blocks'] >= 2}

    results = []
    for func_name, func_data in targets.items():
        blocks = func_data['blocks'][:max_blocks]

        # Token IDs
        block_token_ids = []
        for block in blocks:
            ids = [token_vocab.get(t, token_vocab.get('<UNK>', 1))
                   for t in block['tokens'][:max_tokens]]
            ids += [0] * (max_tokens - len(ids))
            block_token_ids.append(ids)

        num_blocks = len(block_token_ids)

        # Block features
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

        # External calls
        ext = ext_by_func.get(func_name, [])
        ext_ids = [ext_vocab.get(name, ext_vocab.get('<NO_EXT>', 0)) for name in ext]
        if not ext_ids:
            ext_ids = [0]

        # Tensors
        bt = torch.tensor([block_token_ids], dtype=torch.long, device=device)
        bf = torch.tensor([block_feats], dtype=torch.float32, device=device)
        ei_filtered = [[s, d] for s, d in edges_raw if s < num_blocks and d < num_blocks]
        if not ei_filtered:
            ei_filtered = [[0, 0]]
        ei = torch.tensor(ei_filtered, dtype=torch.long, device=device).t().contiguous()
        ec = torch.tensor([ext_ids], dtype=torch.long, device=device)

        ct = None
        if callee_enabled:
            callee_sigs = get_callee_tokens(func_data)
            ct = torch.tensor([callee_sigs], dtype=torch.long, device=device)

        crt = None
        if caller_enabled:
            caller_sigs = get_caller_tokens(func_name)
            crt = torch.tensor([caller_sigs], dtype=torch.long, device=device)

        with torch.no_grad():
            pred_results = model.predict(bt, ei, ec, sos_id, eos_id,
                                         beam_width=beam_width,
                                         block_features=bf,
                                         callee_tokens=ct,
                                         caller_tokens=crt)
        pred_tokens, score = pred_results[0]
        pred_name = sp.decode(pred_tokens).strip()

        results.append({
            'address': func_data['address'],
            'bap_name': func_name,
            'predicted_name': pred_name,
            'score': score / max(len(pred_tokens), 1) if pred_tokens else -999,
        })

    return results


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    outdir = "demo/results/expanded"
    os.makedirs(outdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories to search for preprocessed data
    graphs_dirs = ["data/graphs", "demo/graphs"]
    labels_dirs = ["data/labels", "demo/labels"]
    ext_dirs = ["data/external_calls", "demo/external_calls"]

    # Load checkpoint
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    token_vocab = ckpt['token_vocab']

    # Load name tokenizer
    votes_vocab_path = cfg['data'].get('votes_vocab_path')
    if votes_vocab_path and os.path.exists(votes_vocab_path):
        from src.preprocessing.build_votes import VotesTokenizer
        sp = VotesTokenizer(vocab_path=votes_vocab_path)
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=cfg['data']['bpe_model_path'])

    cfg['decoder']['bpe_vocab_size'] = sp.get_piece_size()

    # Load ext vocab
    if 'ext_vocab' in ckpt:
        ext_vocab = ckpt['ext_vocab']
        cfg['external_encoder']['vocab_size'] = len(ext_vocab)
    else:
        with open(cfg['data']['external_vocab_path']) as f:
            ext_data = json.load(f)
        ext_vocab = ext_data['vocabulary']
        cfg['external_encoder']['vocab_size'] = ext_data['vocab_size']

    cfg['block_encoder']['token_vocab_size'] = len(token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']

    # Build model
    model = FunctionNamer(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Output dir: {outdir}\n")

    # Evaluate each demo binary
    per_binary = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0, "ngsim_sum": 0, "edsim_sum": 0})
    per_opt = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0, "ngsim_sum": 0, "edsim_sum": 0})
    per_pkg = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0, "ngsim_sum": 0, "edsim_sum": 0})

    for pkg, binary, opt_levels in DEMO_PACKAGES:
        for opt in opt_levels:
            bin_name = f"{pkg}_{binary}_{opt}" if opt else f"{pkg}_{binary}"

            # Load preprocessed data
            functions = load_functions_from_graphs(bin_name, graphs_dirs)
            if not functions:
                # Try diffutils naming convention
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

            print(f"  Predicting {bin_name} ({len(gt)} GT, {len(functions)} graphs)...", end="", flush=True)

            # Resolve thunks
            resolve_thunks(functions, ext_by_func)

            # Run inference
            preds = predict_binary(functions, ext_by_func, model, token_vocab,
                                   ext_vocab, sp, cfg, device)

            # Save predictions
            pred_path = os.path.join(outdir, f"predictions_{bin_name}.json")
            with open(pred_path, 'w') as f:
                json.dump(preds, f, indent=2)

            # Match and compute metrics
            correct = 0
            total = 0
            f1_sum = 0
            ngsim_sum = 0
            edsim_sum = 0

            for pred_func in preds:
                addr = pred_func.get("address", "")
                pred_name = pred_func.get("predicted_name", "")

                if not addr.startswith("0x"):
                    addr = "0x" + addr

                true_name = gt.get(addr)
                if not true_name:
                    try:
                        addr_minus4 = hex(int(addr, 16) - 4)
                        true_name = gt.get(addr_minus4)
                    except ValueError:
                        pass
                if not true_name:
                    continue

                f1 = compute_subtoken_f1(pred_name, true_name)
                ngsim = compute_char_ngram_similarity(pred_name, true_name)
                edsim = compute_edit_distance_similarity(pred_name, true_name)
                total += 1
                f1_sum += f1
                ngsim_sum += ngsim
                edsim_sum += edsim
                if pred_name == true_name:
                    correct += 1

            opt_label = opt if opt else "O2"
            for d in [per_binary[bin_name], per_opt[opt_label], per_pkg[pkg]]:
                d["correct"] += correct
                d["total"] += total
                d["f1_sum"] += f1_sum
                d["ngsim_sum"] += ngsim_sum
                d["edsim_sum"] += edsim_sum

            em_pct = 100 * correct / total if total > 0 else 0
            f1_avg = f1_sum / total if total > 0 else 0
            ngsim_avg = ngsim_sum / total if total > 0 else 0
            edsim_avg = edsim_sum / total if total > 0 else 0
            print(f" {correct}/{total} EM ({em_pct:.1f}%), F1={f1_avg:.3f}, NgSim={ngsim_avg:.3f}, EdSim={edsim_avg:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"DEMO RESULTS (Wulver — no BAP)")
    print(f"{'='*70}")

    overall_c = sum(v["correct"] for v in per_binary.values())
    overall_t = sum(v["total"] for v in per_binary.values())
    overall_f1 = sum(v["f1_sum"] for v in per_binary.values()) / max(overall_t, 1)
    overall_ngsim = sum(v["ngsim_sum"] for v in per_binary.values()) / max(overall_t, 1)
    overall_edsim = sum(v["edsim_sum"] for v in per_binary.values()) / max(overall_t, 1)

    print(f"\nOverall: {overall_c}/{overall_t} EM ({100*overall_c/overall_t:.1f}%), F1={overall_f1:.4f}, NgSim={overall_ngsim:.4f}, EdSim={overall_edsim:.4f}")

    print(f"\nPer optimization level:")
    for opt in sorted(per_opt):
        v = per_opt[opt]
        em = 100 * v["correct"] / v["total"] if v["total"] > 0 else 0
        f1 = v["f1_sum"] / v["total"] if v["total"] > 0 else 0
        ngsim = v["ngsim_sum"] / v["total"] if v["total"] > 0 else 0
        edsim = v["edsim_sum"] / v["total"] if v["total"] > 0 else 0
        print(f"  {opt}: {v['correct']}/{v['total']} EM ({em:.1f}%), F1={f1:.4f}, NgSim={ngsim:.4f}, EdSim={edsim:.4f}")

    print(f"\nPer package:")
    for pkg in sorted(per_pkg):
        v = per_pkg[pkg]
        em = 100 * v["correct"] / v["total"] if v["total"] > 0 else 0
        f1 = v["f1_sum"] / v["total"] if v["total"] > 0 else 0
        ngsim = v["ngsim_sum"] / v["total"] if v["total"] > 0 else 0
        edsim = v["edsim_sum"] / v["total"] if v["total"] > 0 else 0
        print(f"  {pkg:15s}: {v['correct']:4d}/{v['total']:4d} EM ({em:5.1f}%), F1={f1:.4f}, NgSim={ngsim:.4f}, EdSim={edsim:.4f}")

    # Save
    save_data = {
        "overall": {"correct": overall_c, "total": overall_t,
                    "f1": overall_f1, "ngsim": overall_ngsim, "edsim": overall_edsim},
        "per_opt": {k: dict(v) for k, v in per_opt.items()},
        "per_pkg": {k: dict(v) for k, v in per_pkg.items()},
        "per_binary": {k: {"correct": v["correct"], "total": v["total"],
                           "f1": v["f1_sum"]/max(v["total"],1),
                           "ngsim": v["ngsim_sum"]/max(v["total"],1),
                           "edsim": v["edsim_sum"]/max(v["total"],1)} for k, v in per_binary.items()},
    }
    save_path = os.path.join(outdir, "expanded_eval.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
