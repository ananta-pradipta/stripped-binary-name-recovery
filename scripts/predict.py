"""
Predict function names for a stripped binary.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import subprocess

import torch
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import compute_block_features, compute_block_degrees, NUM_BLOCK_FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, help='Stripped binary path')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--output', default=None)
    parser.add_argument('--beam-width', type=int, default=5)
    args = parser.parse_args()

    print("═══ Function Name Prediction ═══")
    print(f"  Binary:     {args.binary}")
    print(f"  Checkpoint: {args.checkpoint}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device:     {device}")

    # Load checkpoint
    print("\n  Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    token_vocab = ckpt['token_vocab']

    # Load name tokenizer (votes or BPE)
    votes_vocab_path = cfg['data'].get('votes_vocab_path')
    if votes_vocab_path and os.path.exists(votes_vocab_path):
        from src.preprocessing.build_votes import VotesTokenizer
        sp = VotesTokenizer(vocab_path=votes_vocab_path)
    else:
        sp = spm.SentencePieceProcessor(model_file=cfg['data']['bpe_model_path'])
    sos_id = sp.bos_id()
    eos_id = sp.eos_id()
    cfg['decoder']['bpe_vocab_size'] = sp.get_piece_size()

    # Load ext vocab from checkpoint (immune to disk file changes)
    if 'ext_vocab' in ckpt:
        ext_vocab = ckpt['ext_vocab']
        cfg['external_encoder']['vocab_size'] = len(ext_vocab)
        print(f"  Ext vocab: {len(ext_vocab)} tokens (from checkpoint)")
    else:
        # Fallback to disk for old checkpoints without ext_vocab
        with open(cfg['data']['external_vocab_path']) as f:
            ext_data = json.load(f)
        ext_vocab = ext_data['vocabulary']
        cfg['external_encoder']['vocab_size'] = ext_data['vocab_size']
        print(f"  Ext vocab: {ext_data['vocab_size']} tokens (from disk — old checkpoint)")

    # Override token vocab
    cfg['block_encoder']['token_vocab_size'] = len(token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']

    model = FunctionNamer(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # Lift binary with BAP
    print("\n  Lifting with BAP...")
    bir_path = tempfile.mktemp(suffix='.bir')
    subprocess.run(['bap', args.binary, f'--dump=bir:{bir_path}'],
                   capture_output=True, timeout=300)
    print(f"  BAP-IR: {bir_path}")

    # Parse functions
    print("  Parsing functions...")
    from src.preprocessing.parse_bap import parse_bir_file
    functions = parse_bir_file(bir_path)
    print(f"  Found {len(functions)} functions")

    # Extract external calls via subprocess
    print("  Extracting external calls...")
    ext_tmp_dir = tempfile.mkdtemp()
    subprocess.run(
        ['python3', '-m', 'src.preprocessing.extract_external',
         '--bir', bir_path, '--binary-name', '_predict',
         '--output-dir', ext_tmp_dir,
         '--vocab-path', '/tmp/predict_ext_vocab.json'],
        capture_output=True,
    )
    ext_by_func = {}
    ext_file = os.path.join(ext_tmp_dir, '_predict_external.json')
    if os.path.exists(ext_file):
        with open(ext_file) as f:
            ext_data_parsed = json.load(f)
        for func in ext_data_parsed.get('functions', []):
            key = func.get('function_name', '')
            calls = [c['name'] for c in func.get('external_calls', [])]
            ext_by_func[key] = calls

    # Build callee lookup for inter-procedural context
    # Index all functions by address for callee resolution
    func_by_addr = {}
    for fname, fdata in functions.items():
        addr = fdata.get('address', '')
        if addr:
            func_by_addr[addr] = fdata

    def get_callee_tokens(func_data, max_callees=5, max_sig_tokens=10):
        """Build callee token signatures for a function."""
        callee_sigs = []
        for callee_name in func_data.get('internal_callees', [])[:max_callees]:
            # Look up callee in all functions
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

    # Build reverse call graph for caller context
    callers_of = {}  # func_name -> list of caller func_names
    for fname, fdata in functions.items():
        for callee_name in fdata.get('internal_callees', []):
            if callee_name not in callers_of:
                callers_of[callee_name] = []
            callers_of[callee_name].append(fname)

    def get_caller_tokens(func_name, max_callers=5, max_sig_tokens=10):
        """Build caller token signatures for a function."""
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

    callee_enabled = cfg.get('callee_encoder', {}).get('enabled', False)
    caller_enabled = cfg.get('caller_encoder', {}).get('enabled', False)

    # ── Thunk resolution (O0 ENDBR64 fix) ──
    # At O0 with PIE+CET, BAP lifts most functions as 1-token thunks:
    #   endbr64; jmp addr+4  →  CALL_INTERNAL to sub_(addr+4)
    # Resolve by replacing the thunk graph with its callee's graph.
    thunks_resolved = 0
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
                    # Inherit ext calls from callee
                    callee_ext = ext_by_func.get(callee_name, [])
                    if callee_ext:
                        ext_by_func[func_name] = callee_ext
                    thunks_resolved += 1
    if thunks_resolved > 0:
        print(f"  Thunk resolution: {thunks_resolved} O0 thunks resolved to real code")

    # Filter to sub_XXXX with >= 2 blocks
    targets = {name: data for name, data in functions.items()
               if name.startswith('sub_') and data['num_blocks'] >= 2}
    print(f"  Target functions (sub_XXXX with ≥2 blocks): {len(targets)}")
    if callee_enabled:
        print(f"  Callee context: ENABLED")

    # Predict
    print("\n  Predicting names...\n")
    max_blocks = cfg['data']['max_blocks_per_function']
    max_tokens = cfg['data']['max_tokens_per_block']

    results = []
    print(f"  {'Address':<12s} {'Predicted Name':<35s} {'Score':>8s} {'Blocks':>7s} {'ExtCalls':>9s}")
    print(f"  {'─'*12} {'─'*35} {'─'*8} {'─'*7} {'─'*9}")

    for func_name, func_data in sorted(targets.items(),
                                        key=lambda x: float('inf') if not x[1]['address'].startswith('0x')
                                        else int(x[1]['address'], 16)):
        blocks = func_data['blocks'][:max_blocks]

        # Token IDs
        block_token_ids = []
        for block in blocks:
            ids = [token_vocab.get(t, token_vocab.get('<UNK>', 1))
                   for t in block['tokens'][:max_tokens]]
            ids += [0] * (max_tokens - len(ids))
            block_token_ids.append(ids)

        num_blocks = len(block_token_ids)

        # Compute block statistics
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
        ei_raw = func_data['edges']
        ei_filtered = [[s, d] for s, d in ei_raw if s < num_blocks and d < num_blocks]
        if not ei_filtered:
            ei_filtered = [[0, 0]]
        ei = torch.tensor(ei_filtered, dtype=torch.long, device=device).t().contiguous()
        ec = torch.tensor([ext_ids], dtype=torch.long, device=device)

        # Callee context
        ct = None
        if callee_enabled:
            callee_sigs = get_callee_tokens(func_data)
            ct = torch.tensor([callee_sigs], dtype=torch.long, device=device)

        # Caller context
        crt = None
        if caller_enabled:
            caller_sigs = get_caller_tokens(func_name)
            crt = torch.tensor([caller_sigs], dtype=torch.long, device=device)

        # Predict
        pred_results = model.predict(bt, ei, ec, sos_id, eos_id,
                                     beam_width=args.beam_width,
                                     block_features=bf,
                                     callee_tokens=ct,
                                     caller_tokens=crt)
        pred_tokens, score = pred_results[0]
        pred_name = sp.decode(pred_tokens).strip()

        ext_summary = ', '.join(ext[:3])
        if len(ext) > 3:
            ext_summary += f"... +{len(ext)-3}"

        score_display = f"{score/max(len(pred_tokens),1):.3f}" if pred_tokens else "N/A"
        print(f"  {func_data['address']:<12s} {pred_name:<35s} {score_display:>8s} "
              f"{num_blocks:>7d} {len(ext):>5d} {ext_summary}")

        results.append({
            'address': func_data['address'],
            'bap_name': func_name,
            'predicted_name': pred_name,
            'score': score / max(len(pred_tokens), 1) if pred_tokens else -999,
            'num_blocks': num_blocks,
            'num_ext_calls': len(ext),
            'ext_calls': ext,
        })

    # Sort by score
    results.sort(key=lambda r: -r['score'] if r['score'] != -999 else float('inf'))

    # Save
    output_path = args.output or f"demo/results/predictions_{os.path.basename(args.binary).replace('_stripped','')}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  ═══════════════════════════════════════")
    print(f"  Total functions predicted: {len(results)}")
    print(f"  Avg ext calls per function: {sum(r['num_ext_calls'] for r in results)/max(len(results),1):.1f}")
    print(f"  Predictions saved to: {output_path}")
    print(f"  ═══════════════════════════════════════")

    # Cleanup
    if os.path.exists(bir_path):
        os.remove(bir_path)


if __name__ == '__main__':
    main()
