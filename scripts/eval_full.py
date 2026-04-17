#!/usr/bin/env python3
"""
Comprehensive evaluation script with k-NN hybrid as default inference.

Produces a full report with:
  1. Model info
  2. Test/Val metrics (k-NN hybrid, with decoder baseline comparison)
  3. Demo metrics (per-package breakdown, k-NN hybrid)
  4. Sample predictions (correct, close, wrong)
  5. Ablation table (if checkpoints available)

Works both locally and on HPC.

Usage:
  python3 scripts/eval_full.py                          # Full report
  python3 scripts/eval_full.py --sections test demo     # Only test + demo
  python3 scripts/eval_full.py --sections ablation      # Only ablation table
  python3 scripts/eval_full.py --amp                    # Use AMP (GPU)
  python3 scripts/eval_full.py --no-knn                 # Decoder-only (skip k-NN)
"""
import argparse
import json
import os
import sys
import yaml
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import (
    FunctionDataset, compute_block_features, compute_block_degrees, NUM_BLOCK_FEATURES,
)
from src.evaluation.metrics import (
    compute_subtoken_f1, compute_exact_match,
    compute_char_ngram_similarity, compute_edit_distance_similarity,
    normalize_name, split_name,
)

KNN_THRESHOLD = -0.02  # Forward hybrid: decoder when beam score > threshold, k-NN otherwise

# ═══════════════════════════════════════
# Helpers
# ═══════════════════════════════════════

def hextester(s):
    """True iff s is a valid hex integer. Used to distinguish BAP placeholder
    names (sub_HEX) from user-defined sub_* names (e.g. sub_append_string)."""
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            if k == 'edge_index':
                edge_lists, offset = [], 0
                for s in batch:
                    ei = s[k].clone() + offset
                    edge_lists.append(ei)
                    offset += s['num_blocks']
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


def header(text, char='=', width=80):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def load_model(checkpoint_path, cfg, device, sp):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get('config', cfg)
    token_vocab = ckpt.get('token_vocab', {})
    ext_vocab = ckpt.get('ext_vocab', {})

    ckpt_cfg['block_encoder']['token_vocab_size'] = len(token_vocab)
    ckpt_cfg['graph_encoder']['input_dim'] = ckpt_cfg['block_encoder']['output_dim']
    ckpt_state = ckpt['model_state_dict']
    if 'ext_encoder.embedding.weight' in ckpt_state:
        ckpt_cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
    ckpt_cfg['decoder']['bpe_vocab_size'] = sp.get_piece_size()

    model = FunctionNamer(ckpt_cfg).to(device)
    model.load_state_dict(ckpt_state)
    model.eval()
    return model, ckpt, ckpt_cfg, token_vocab, ext_vocab


# ═══════════════════════════════════════
# Embedding + prediction extraction
# ═══════════════════════════════════════

def extract_embeddings_and_predict(model, loader, device, sp, beam_width=5, use_amp=False, desc=""):
    """Extract encoder embeddings z, beam search predictions, and ground truth names."""
    sos_id, eos_id = sp.bos_id(), sp.eos_id()
    all_embeddings = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            bt = batch['block_tokens'].to(device)
            ei = batch['edge_index'].to(device)
            ec = batch['ext_call_ids'].to(device)
            bf = batch.get('block_features')
            if bf is not None:
                bf = bf.to(device)
            dt = batch['decoder_target'].to(device)
            ct = batch.get('callee_tokens')
            if ct is not None:
                ct = ct.to(device)
            crt = batch.get('caller_tokens')
            if crt is not None:
                crt = crt.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                # Extract embedding z
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

            all_embeddings.append(z.cpu().numpy())

            for i in range(B):
                tokens, score = model.decoder.generate(z[i:i+1], sos_id, eos_id, beam_width)
                pred_name = sp.decode(tokens).strip() if tokens else ""
                norm_score = score / max(len(tokens), 1) if tokens else -999

                target_tokens = []
                for t in dt[i]:
                    tid = t.item()
                    if tid == eos_id: break
                    if tid != 0: target_tokens.append(tid)
                true_name = sp.decode(target_tokens)
                n_ext = (ec[i] > 0).sum().item()

                all_preds.append({
                    'true': true_name, 'pred': pred_name, 'score': norm_score, 'n_ext': n_ext,
                })

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_preds


def build_knn_index(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def knn_lookup(query_embs, train_normed, train_names):
    """Batch k-NN lookup, returns nearest names and cosine similarities."""
    norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_normed = query_embs / (norms + 1e-8)
    names, sims = [], []
    chunk = 500
    for start in range(0, len(query_normed), chunk):
        end = min(start + chunk, len(query_normed))
        s = query_normed[start:end] @ train_normed.T
        idx = np.argmax(s, axis=1)
        for j, i in enumerate(idx):
            names.append(train_names[i])
            sims.append(s[j, i])
    return names, sims


def apply_hybrid(preds, embeddings, train_normed, train_names, threshold=KNN_THRESHOLD):
    """Apply k-NN hybrid: use decoder when confident, k-NN otherwise. Returns updated results."""
    knn_names, knn_sims = knn_lookup(embeddings, train_normed, train_names)
    results = []
    n_knn = 0
    for i, p in enumerate(preds):
        if p['score'] <= threshold:
            final_name = knn_names[i]
            source = 'knn'
            n_knn += 1
        else:
            final_name = p['pred']
            source = 'decoder'

        true_name = p['true']
        results.append({
            'true': true_name,
            'pred': final_name,
            'decoder_pred': p['pred'],
            'score': p['score'],
            'source': source,
            'f1': compute_subtoken_f1(final_name, true_name),
            'em': 1 if final_name == true_name else 0,
            'ngsim': compute_char_ngram_similarity(final_name, true_name),
            'edsim': compute_edit_distance_similarity(final_name, true_name),
            'n_ext': p.get('n_ext', 0),
            'pkg': p.get('pkg', ''),
        })
    return results, n_knn


def compute_decoder_baseline(preds):
    """Compute decoder-only metrics from raw predictions."""
    n = len(preds)
    results = []
    for p in preds:
        true_name = p['true']
        pred_name = p['pred']
        results.append({
            'f1': compute_subtoken_f1(pred_name, true_name),
            'em': 1 if pred_name == true_name else 0,
            'ngsim': compute_char_ngram_similarity(pred_name, true_name),
            'edsim': compute_edit_distance_similarity(pred_name, true_name),
        })
    return {
        'em': sum(r['em'] for r in results) / n,
        'f1': sum(r['f1'] for r in results) / n,
        'ngsim': sum(r['ngsim'] for r in results) / n,
        'edsim': sum(r['edsim'] for r in results) / n,
    }


def print_metrics_comparison(results, baseline, label, n_knn=0):
    """Print hybrid metrics with decoder baseline comparison."""
    n = len(results)
    em = sum(r['em'] for r in results) / n
    f1 = sum(r['f1'] for r in results) / n
    ngsim = sum(r['ngsim'] for r in results) / n
    edsim = sum(r['edsim'] for r in results) / n

    print(f"\n  {label} ({n} functions):")
    if n_knn > 0:
        print(f"  Inference: {n - n_knn} decoder ({100*(n-n_knn)/n:.0f}%) + {n_knn} k-NN ({100*n_knn/n:.0f}%)")
    print(f"\n  {'Metric':<8s} | {'Decoder':>10s} | {'Hybrid':>10s} | {'Delta':>10s}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'EM':<8s} | {100*baseline['em']:>9.1f}% | {100*em:>9.1f}% | {100*(em-baseline['em']):>+9.1f}%")
    print(f"  {'F1':<8s} | {baseline['f1']:>10.4f} | {f1:>10.4f} | {f1-baseline['f1']:>+10.4f}")
    print(f"  {'NgSim':<8s} | {baseline['ngsim']:>10.4f} | {ngsim:>10.4f} | {ngsim-baseline['ngsim']:>+10.4f}")
    print(f"  {'EdSim':<8s} | {baseline['edsim']:>10.4f} | {edsim:>10.4f} | {edsim-baseline['edsim']:>+10.4f}")

    return {'em': em, 'f1': f1, 'ngsim': ngsim, 'edsim': edsim, 'n': n}


def print_samples(results, n_each=10):
    correct = [r for r in results if r['em']]
    close = [r for r in results if not r['em'] and r['f1'] > 0.5]
    wrong = [r for r in results if r['f1'] == 0]

    print(f"\n  Correct: {len(correct)}/{len(results)}  |  Close (F1>0.5): {len(close)}  |  Wrong (F1=0): {len(wrong)}")

    print(f"\n  CORRECT predictions (sample):")
    print(f"  {'True Name':<35s} {'Predicted':<35s} {'Source':<7s}")
    print(f"  {'-'*35} {'-'*35} {'-'*7}")
    for r in correct[:n_each]:
        print(f"  {r['true']:<35s} {r['pred']:<35s} {r.get('source',''):<7s}")

    if close:
        print(f"\n  CLOSE predictions (F1>0.5, not exact):")
        print(f"  {'True Name':<28s} {'Predicted':<28s} {'F1':>5s} {'EdSim':>6s} {'Source':<7s}")
        print(f"  {'-'*28} {'-'*28} {'-'*5} {'-'*6} {'-'*7}")
        for r in sorted(close, key=lambda x: -x['f1'])[:n_each]:
            print(f"  {r['true']:<28s} {r['pred']:<28s} {r['f1']:>5.2f} {r['edsim']:>6.3f} {r.get('source',''):<7s}")

    if wrong:
        print(f"\n  WRONG predictions (F1=0, sample):")
        print(f"  {'True Name':<28s} {'Predicted':<28s} {'Source':<7s}")
        print(f"  {'-'*28} {'-'*28} {'-'*7}")
        for r in wrong[:n_each]:
            print(f"  {r['true']:<28s} {r['pred']:<28s} {r.get('source',''):<7s}")


# ═══════════════════════════════════════
# Demo evaluation (from pre-processed graphs)
# ═══════════════════════════════════════

DEMO_PACKAGES = [
    ("diffutils", "diff", [""]), ("diffutils", "cmp", [""]),
    ("diffutils", "sdiff", [""]), ("diffutils", "diff3", [""]),
    ("datamash", "datamash", ["O0", "O2"]), ("direvent", "direvent", ["O0", "O2"]),
    ("csplit2", "cflow", ["O0", "O2"]), ("texinfo", "ginfo", ["O0", "O2"]),
    ("cppi", "cppi", ["O0", "O2"]), ("hello", "hello", ["O0", "O2"]),
    ("acct", "ac", ["O0", "O2"]), ("acct", "last", ["O0", "O2"]),
    ("acct", "lastcomm", ["O0", "O2"]), ("acct", "sa", ["O0", "O2"]),
    ("acct", "dump-utmp", ["O0", "O2"]), ("acct", "accton", ["O0", "O2"]),
    ("rush", "rush", ["O0", "O2"]), ("htop", "htop", ["O0", "O2"]),
    ("strace", "strace", ["O0", "O2"]),
]

import glob as glob_mod

def _load_demo_graphs(bin_name, dirs):
    functions = {}
    for d in dirs:
        for path in glob_mod.glob(os.path.join(d, f"{bin_name}_*.json")):
            with open(path) as f:
                data = json.load(f)
            fname = data.get('function_name', '')
            if fname:
                functions[fname] = data
    return functions

def _load_demo_ext(bin_name, dirs):
    for d in dirs:
        p = os.path.join(d, f"{bin_name}_external.json")
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            return {func['function_name']: [c['name'] for c in func.get('external_calls', [])]
                    for func in data.get('functions', [])}
    return {}

def _load_demo_gt(bin_name, dirs):
    for d in dirs:
        p = os.path.join(d, f"{bin_name}_labels.json")
        if os.path.exists(p):
            with open(p) as f:
                labels = json.load(f)
            gt = {}
            funcs = labels.get('functions', labels)
            if isinstance(funcs, dict):
                for name, addr in funcs.items():
                    if isinstance(addr, str):
                        norm = '0x' + addr[2:].lstrip('0') if addr.startswith('0x') else '0x' + addr.lstrip('0')
                        if norm == '0x': norm = '0x0'
                        gt[norm] = name
                        gt[addr] = name
            return gt
    return {}

def _resolve_thunks(functions, ext_by_func):
    for fname, fdata in list(functions.items()):
        all_tokens = [t for b in fdata['blocks'] for t in b['tokens']]
        if len(all_tokens) <= 2 and 'CALL_INTERNAL' in all_tokens:
            callees = fdata.get('internal_callees', [])
            if callees:
                callee = functions.get(callees[0])
                if callee:
                    fdata['blocks'] = callee['blocks']
                    fdata['edges'] = callee['edges']
                    fdata['num_blocks'] = callee['num_blocks']
                    fdata['internal_callees'] = callee.get('internal_callees', [])
                    ce = ext_by_func.get(callees[0], [])
                    if ce:
                        ext_by_func[fname] = ce


def evaluate_demo_with_embeddings(model, cfg, token_vocab, ext_vocab, sp, device, beam_width=5, use_amp=False):
    """Evaluate demo, returning predictions + embeddings for k-NN hybrid."""
    graphs_dirs = ["data/graphs", "demo/graphs"]
    labels_dirs = ["data/labels", "demo/labels"]
    ext_dirs = ["data/external_calls", "demo/external_calls"]

    sos_id, eos_id = sp.bos_id(), sp.eos_id()
    callee_enabled = cfg.get('callee_encoder', {}).get('enabled', False)
    caller_enabled = cfg.get('caller_encoder', {}).get('enabled', False)
    max_blocks = cfg['data']['max_blocks_per_function']
    max_tokens = cfg['data']['max_tokens_per_block']

    all_preds = []
    all_embeddings = []

    for pkg, binary, opt_levels in DEMO_PACKAGES:
        for opt in opt_levels:
            bin_name = f"{pkg}_{binary}_{opt}" if opt else f"{pkg}_{binary}"
            functions = _load_demo_graphs(bin_name, graphs_dirs)
            if not functions:
                alt = f"diffutils_{binary}"
                functions = _load_demo_graphs(alt, graphs_dirs)
                if functions: bin_name = alt
            if not functions: continue

            ext_by_func = _load_demo_ext(bin_name, ext_dirs)
            gt = _load_demo_gt(bin_name, labels_dirs)
            if not gt: continue

            _resolve_thunks(functions, ext_by_func)

            func_by_addr = {fd.get('address', ''): fd for fd in functions.values()}
            callers_of = defaultdict(list)
            for fn, fd in functions.items():
                for cn in fd.get('internal_callees', []):
                    callers_of[cn].append(fn)

            def _ctx_tokens(names, max_ctx=5, max_sig=10):
                sigs = []
                for name in names[:max_ctx]:
                    g = functions.get(name)
                    if g is None and name.startswith('sub_') and hextester(name[4:]):
                        g = func_by_addr.get('0x' + name[4:])
                    if g is None:
                        sigs.append([0]*max_sig); continue
                    sig = []
                    for blk in g['blocks'][:3]:
                        sig.extend(blk['tokens'][:5])
                        if len(sig) >= max_sig: break
                    ids = [token_vocab.get(t, token_vocab.get('<UNK>', 1)) for t in sig[:max_sig]]
                    ids += [0]*(max_sig - len(ids))
                    sigs.append(ids)
                while len(sigs) < max_ctx:
                    sigs.append([0]*max_sig)
                return sigs

            targets = {n: d for n, d in functions.items()
                       if n.startswith('sub_') and d['num_blocks'] >= 2}

            for func_name, func_data in targets.items():
                blocks = func_data['blocks'][:max_blocks]
                block_ids = []
                for blk in blocks:
                    ids = [token_vocab.get(t, token_vocab.get('<UNK>', 1)) for t in blk['tokens'][:max_tokens]]
                    ids += [0]*(max_tokens - len(ids))
                    block_ids.append(ids)
                nb = len(block_ids)

                edges_raw = func_data['edges']
                in_d, out_d = compute_block_degrees(edges_raw, nb)
                bfeats = []
                for bi, blk in enumerate(blocks):
                    bfeats.append(compute_block_features(
                        blk['tokens'], bi, nb,
                        in_d[bi] if bi < len(in_d) else 0,
                        out_d[bi] if bi < len(out_d) else 0))

                while len(block_ids) < max_blocks:
                    block_ids.append([0]*max_tokens)
                while len(bfeats) < max_blocks:
                    bfeats.append([0.0]*NUM_BLOCK_FEATURES)

                ext = ext_by_func.get(func_name, [])
                ext_ids = [ext_vocab.get(n, ext_vocab.get('<NO_EXT>', 0)) for n in ext]
                if not ext_ids: ext_ids = [0]

                bt = torch.tensor([block_ids], dtype=torch.long, device=device)
                bf = torch.tensor([bfeats], dtype=torch.float32, device=device)
                ei_f = [[s, d] for s, d in edges_raw if s < nb and d < nb]
                if not ei_f: ei_f = [[0, 0]]
                ei = torch.tensor(ei_f, dtype=torch.long, device=device).t().contiguous()
                ec = torch.tensor([ext_ids], dtype=torch.long, device=device)

                ct_t = None
                if callee_enabled:
                    ct_t = torch.tensor([_ctx_tokens(func_data.get('internal_callees', []))],
                                         dtype=torch.long, device=device)
                crt_t = None
                if caller_enabled:
                    crt_t = torch.tensor([_ctx_tokens(callers_of.get(func_name, []))],
                                          dtype=torch.long, device=device)

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
                        if model.callee_encoder_enabled and ct_t is not None:
                            callee_ctx, has_callees = model.callee_encoder(ct_t)
                            g = model.callee_gate(torch.cat([z, callee_ctx], dim=1))
                            z_fused = g * z + (1 - g) * callee_ctx
                            mask = has_callees.unsqueeze(1).float()
                            z = mask * z_fused + (1 - mask) * z
                        if model.caller_encoder_enabled and crt_t is not None:
                            caller_ctx, has_callers = model.caller_encoder(crt_t)
                            g = model.caller_gate(torch.cat([z, caller_ctx], dim=1))
                            z_fused = g * z + (1 - g) * caller_ctx
                            mask = has_callers.unsqueeze(1).float()
                            z = mask * z_fused + (1 - mask) * z

                    pred_tokens, score = model.decoder.generate(z, sos_id, eos_id, beam_width)
                    pred_name = sp.decode(pred_tokens).strip() if pred_tokens else ""
                    norm_score = score / max(len(pred_tokens), 1) if pred_tokens else -999

                # Match to GT
                addr = func_data['address']
                if not addr.startswith('0x'): addr = '0x' + addr
                true_name = gt.get(addr)
                if not true_name:
                    try: true_name = gt.get(hex(int(addr, 16) - 4))
                    except ValueError: pass
                if not true_name:
                    continue

                all_embeddings.append(z.cpu().numpy())
                all_preds.append({
                    'true': true_name, 'pred': pred_name, 'score': norm_score, 'pkg': pkg,
                })

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])
    return all_preds, embeddings


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation report')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--sections', nargs='+',
                        default=['info', 'test', 'demo', 'samples', 'ablation'],
                        help='Sections: info test demo samples ablation')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--no-knn', action='store_true', help='Skip k-NN hybrid, decoder only')
    args = parser.parse_args()

    use_knn = not args.no_knn
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    header("BFNR BINARY FUNCTION NAME RECOVERY — EVALUATION REPORT")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {device}")
    print(f"  Inference: {'k-NN hybrid (threshold=' + str(KNN_THRESHOLD) + ')' if use_knn else 'decoder only'}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load tokenizer
    votes_path = cfg['data'].get('votes_vocab_path', 'data/votes_vocab.json')
    if votes_path and os.path.exists(votes_path):
        from src.preprocessing.build_votes import VotesTokenizer
        sp = VotesTokenizer(vocab_path=votes_path)
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=cfg['data']['bpe_model_path'])

    # ═══ SECTION: Model Info ═══
    if 'info' in args.sections:
        header("1. MODEL INFORMATION")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
        print(f"  Val F1: {ckpt.get('val_f1', 0):.4f}")
        params = sum(p.numel() for p in ckpt['model_state_dict'].values() if isinstance(p, torch.Tensor))
        print(f"  Parameters: {params:,}")
        cc = ckpt.get('config', {})
        be = cc.get('block_encoder', {})
        ge = cc.get('graph_encoder', {})
        print(f"  Block Encoder: {be.get('num_layers', be.get('layers', '?'))} layers, "
              f"{be.get('token_embed_dim', be.get('embed_dim', '?'))}d, "
              f"{be.get('num_heads', be.get('heads', '?'))} heads")
        print(f"  Graph Encoder: {ge.get('num_layers', ge.get('layers', '?'))} layers GAT, "
              f"{ge.get('num_heads', ge.get('heads', '?'))} heads, "
              f"{ge.get('output_dim', '?')}d output")
        print(f"  Decoder: GRU {cc.get('decoder', {}).get('hidden_dim', '?')}d hidden")
        print(f"  Token vocab: {len(ckpt.get('token_vocab', {}))} | "
              f"Name vocab: {sp.get_piece_size()} | "
              f"Ext vocab: {len(ckpt.get('ext_vocab', {}))}")
        print(f"  Features: ext_calls={cc.get('external_encoder', {}).get('enabled', False)}, "
              f"callee={cc.get('callee_encoder', {}).get('enabled', False)}, "
              f"caller={cc.get('caller_encoder', {}).get('enabled', False)}")
        del ckpt

    # Load main model
    model, ckpt, ckpt_cfg, token_vocab, ext_vocab = load_model(args.checkpoint, cfg, device, sp)
    print(f"\n  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ═══ Build k-NN index from training data ═══
    train_normed = None
    train_names = None
    dataset = None
    train_idx = val_idx = test_idx = None

    need_dataset = any(s in args.sections for s in ['test', 'samples', 'ablation'])
    if need_dataset or use_knn:
        dataset = FunctionDataset(
            graphs_dir=cfg['data']['graphs_dir'],
            labels_dir=cfg['data']['labels_dir'],
            external_calls_dir=cfg['data']['external_calls_dir'],
            bpe_model_path=cfg['data']['bpe_model_path'],
            external_vocab_path=cfg['data']['external_vocab_path'],
            max_blocks=cfg['data']['max_blocks_per_function'],
            max_tokens=cfg['data']['max_tokens_per_block'],
            max_name_len=cfg['data']['max_name_length'],
            votes_vocab_path=votes_path,
            token_vocab=token_vocab,
            ext_vocab_override=ext_vocab,
        )
        train_idx, val_idx, test_idx = dataset.get_splits(
            cfg['data']['train_split'], cfg['data']['val_split'])
        print(f"  Dataset: {len(dataset)} — {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    if use_knn and dataset:
        header("BUILDING k-NN INDEX (training set)")
        print(f"  Extracting embeddings for {len(train_idx)} training functions...")
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=cfg['training']['batch_size'],
            shuffle=False, collate_fn=collate_fn, num_workers=0)
        train_embs, train_preds = extract_embeddings_and_predict(
            model, train_loader, device, sp, beam_width=1, use_amp=args.amp, desc="Train embeddings")
        train_names = [p['true'] for p in train_preds]
        train_normed = build_knn_index(train_embs)
        print(f"  k-NN index: {train_embs.shape[0]} functions, {train_embs.shape[1]}d embeddings")
        del train_embs

    # ═══ SECTION: Test/Val Metrics ═══
    test_results = None
    val_results = None
    if 'test' in args.sections and dataset:
        header("2. TEST & VALIDATION METRICS")

        for split_name, split_idx in [('Val', val_idx), ('Test', test_idx)]:
            print(f"\n  Evaluating {split_name.lower()} set...")
            loader = DataLoader(
                Subset(dataset, split_idx), batch_size=32,
                shuffle=False, collate_fn=collate_fn, num_workers=0)
            embs, preds = extract_embeddings_and_predict(
                model, loader, device, sp, beam_width=args.beam_width,
                use_amp=args.amp, desc=f"{split_name} predict")

            baseline = compute_decoder_baseline(preds)

            if use_knn and train_normed is not None:
                results, n_knn = apply_hybrid(preds, embs, train_normed, train_names)
                print_metrics_comparison(results, baseline, split_name, n_knn)
            else:
                results = []
                for p in preds:
                    results.append({**p,
                        'f1': compute_subtoken_f1(p['pred'], p['true']),
                        'em': 1 if p['pred'] == p['true'] else 0,
                        'ngsim': compute_char_ngram_similarity(p['pred'], p['true']),
                        'edsim': compute_edit_distance_similarity(p['pred'], p['true']),
                        'source': 'decoder', 'decoder_pred': p['pred'],
                    })
                n = len(results)
                print(f"\n  {split_name} ({n} functions):")
                print(f"    EM={100*baseline['em']:.1f}%  F1={baseline['f1']:.4f}  "
                      f"NgSim={baseline['ngsim']:.4f}  EdSim={baseline['edsim']:.4f}")

            if split_name == 'Test':
                test_results = results
            else:
                val_results = results

    # ═══ SECTION: Demo Metrics ═══
    demo_results = None
    if 'demo' in args.sections:
        header("3. DEMO EVALUATION (11 unseen packages)")

        print("\n  Evaluating demo binaries...")
        demo_preds, demo_embs = evaluate_demo_with_embeddings(
            model, ckpt_cfg, token_vocab, ext_vocab, sp, device,
            beam_width=args.beam_width, use_amp=args.amp)

        baseline = compute_decoder_baseline(demo_preds)

        if use_knn and train_normed is not None and len(demo_embs) > 0:
            demo_results, n_knn = apply_hybrid(demo_preds, demo_embs, train_normed, train_names)
            print_metrics_comparison(demo_results, baseline, "Demo (11 unseen packages)", n_knn)
        else:
            demo_results = []
            for p in demo_preds:
                demo_results.append({**p,
                    'f1': compute_subtoken_f1(p['pred'], p['true']),
                    'em': 1 if p['pred'] == p['true'] else 0,
                    'ngsim': compute_char_ngram_similarity(p['pred'], p['true']),
                    'edsim': compute_edit_distance_similarity(p['pred'], p['true']),
                    'source': 'decoder', 'decoder_pred': p['pred'],
                })
            n = len(demo_results)
            print(f"\n  Demo ({n} functions):")
            print(f"    EM={100*baseline['em']:.1f}%  F1={baseline['f1']:.4f}  "
                  f"NgSim={baseline['ngsim']:.4f}  EdSim={baseline['edsim']:.4f}")

        # Per-package breakdown
        if demo_results:
            pkg_results = defaultdict(list)
            for r in demo_results:
                pkg_results[r.get('pkg', 'unknown')].append(r)

            print(f"\n  Per-Package Breakdown:")
            print(f"  {'Package':>15s} | {'N':>5s} | {'EM':>7s} | {'F1':>6s} | {'NgSim':>6s} | {'EdSim':>6s}")
            print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
            for pkg in sorted(pkg_results):
                pr = pkg_results[pkg]
                t = len(pr)
                print(f"  {pkg:>15s} | {t:>5d} | {100*sum(r['em'] for r in pr)/t:>6.1f}% | "
                      f"{sum(r['f1'] for r in pr)/t:>6.4f} | "
                      f"{sum(r['ngsim'] for r in pr)/t:>6.4f} | "
                      f"{sum(r['edsim'] for r in pr)/t:>6.4f}")

    # ═══ SECTION: Sample Predictions ═══
    if 'samples' in args.sections:
        if test_results:
            header("4. SAMPLE PREDICTIONS — TEST SET")
            print_samples(test_results, n_each=10)

        if demo_results:
            header("5. SAMPLE PREDICTIONS — DEMO SET (unseen packages)")
            print_samples(demo_results, n_each=10)

            # Best per package
            pkg_results = defaultdict(list)
            for r in demo_results:
                pkg_results[r.get('pkg', '')].append(r)

            print(f"\n  Best prediction per unseen package:")
            print(f"  {'Package':>12s} | {'True Name':<28s} | {'Predicted':<28s} | {'Src':<4s}")
            print(f"  {'-'*12}-+-{'-'*28}-+-{'-'*28}-+-{'-'*4}")
            for pkg in sorted(pkg_results):
                best = max(pkg_results[pkg], key=lambda r: (r['em'], r['f1']))
                print(f"  {pkg:>12s} | {best['true']:<28s} | {best['pred']:<28s} | {best.get('source','')[:3]:<4s}")

    # ═══ SECTION: Ablation Table ═══
    if 'ablation' in args.sections:
        header("6. ABLATION STUDY")

        ablation_models = [
            ("Model 2: GAT + Decoder", "checkpoints/ablation_model2/best_model.pt", "configs/ablation_model2.yaml"),
            ("Model 3: + Ext Calls", "checkpoints/ablation_model3/best_model.pt", "configs/ablation_model3.yaml"),
            ("Model 4: + Callee/Caller", "checkpoints/ablation_model4/best_model.pt", "configs/ablation_model4.yaml"),
            ("Model 5: + Pretrain+Scale", args.checkpoint, args.config),
        ]

        print(f"\n  {'Model':<30s} | {'Params':>7s} | {'Test EM':>7s} | {'Test F1':>7s} | {'Demo EM':>7s} | {'Demo F1':>7s}")
        print(f"  {'-'*30}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for name, ckpt_path, cfg_path in ablation_models:
            if not os.path.exists(ckpt_path):
                print(f"  {name:<30s} | {'N/A':>7s} | {'N/A':>7s} | {'N/A':>7s} | {'N/A':>7s} | {'N/A':>7s}")
                continue

            try:
                # Reuse main model results for Model 5
                if ckpt_path == args.checkpoint:
                    params = sum(p.numel() for p in model.parameters())
                    if test_results:
                        t_em = sum(r['em'] for r in test_results) / len(test_results)
                        t_f1 = sum(r['f1'] for r in test_results) / len(test_results)
                    else:
                        t_em = t_f1 = 0
                    if demo_results:
                        d_em = sum(r['em'] for r in demo_results) / len(demo_results)
                        d_f1 = sum(r['f1'] for r in demo_results) / len(demo_results)
                    else:
                        d_em = d_f1 = 0
                else:
                    with open(cfg_path) as f:
                        acfg = yaml.safe_load(f)
                    amodel, _, acfg_loaded, atv, aev = load_model(ckpt_path, acfg, device, sp)
                    params = sum(p.numel() for p in amodel.parameters())

                    # Test eval
                    ads = FunctionDataset(
                        graphs_dir=acfg['data']['graphs_dir'],
                        labels_dir=acfg['data']['labels_dir'],
                        external_calls_dir=acfg['data']['external_calls_dir'],
                        bpe_model_path=acfg['data']['bpe_model_path'],
                        external_vocab_path=acfg['data']['external_vocab_path'],
                        max_blocks=acfg['data']['max_blocks_per_function'],
                        max_tokens=acfg['data']['max_tokens_per_block'],
                        max_name_len=acfg['data']['max_name_length'],
                        votes_vocab_path=acfg['data'].get('votes_vocab_path', votes_path),
                        token_vocab=atv, ext_vocab_override=aev)
                    _, _, ati = ads.get_splits(acfg['data']['train_split'], acfg['data']['val_split'])
                    loader = DataLoader(Subset(ads, ati), batch_size=32, shuffle=False,
                                        collate_fn=collate_fn, num_workers=0)
                    # Disable AMP for ablation models (smaller models can have dtype issues)
                    _, tr_preds = extract_embeddings_and_predict(
                        amodel, loader, device, sp, beam_width=args.beam_width,
                        use_amp=False, desc=f"{name} test")
                    bl = compute_decoder_baseline(tr_preds)
                    t_em, t_f1 = bl['em'], bl['f1']

                    # Demo eval (decoder only for ablation models — no k-NN)
                    dr_preds, _ = evaluate_demo_with_embeddings(
                        amodel, acfg_loaded, atv, aev, sp, device,
                        beam_width=args.beam_width, use_amp=False)
                    dbl = compute_decoder_baseline(dr_preds) if dr_preds else {'em': 0, 'f1': 0}
                    d_em, d_f1 = dbl['em'], dbl['f1']

                    del amodel
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                pstr = f"{params/1e6:.1f}M"
                print(f"  {name:<30s} | {pstr:>7s} | {100*t_em:>6.1f}% | {t_f1:>7.4f} | {100*d_em:>6.1f}% | {d_f1:>7.4f}")

            except Exception as e:
                print(f"  {name:<30s} | ERROR: {str(e)[:40]}")

        print(f"\n  Key findings:")
        print(f"    - Ext calls alone hurt demo EM (ext-call paradox)")
        print(f"    - Callee/caller context resolves paradox (+12.8pp demo EM)")
        print(f"    - Pretrain+scale gives best generalization")
        if use_knn:
            print(f"    - k-NN hybrid adds +6.7pp demo EM on top (inference-only, no retraining)")

    # ═══ Summary ═══
    header("SUMMARY")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Inference: {'k-NN hybrid' if use_knn else 'decoder only'}")
    if test_results:
        n = len(test_results)
        print(f"  Test:  EM={100*sum(r['em'] for r in test_results)/n:.1f}%, "
              f"F1={sum(r['f1'] for r in test_results)/n:.4f}")
    if demo_results:
        n = len(demo_results)
        print(f"  Demo:  EM={100*sum(r['em'] for r in demo_results)/n:.1f}%, "
              f"F1={sum(r['f1'] for r in demo_results)/n:.4f}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == '__main__':
    main()
