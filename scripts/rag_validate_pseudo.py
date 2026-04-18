#!/usr/bin/env python3
"""
Validation experiment: treat a set of held-out training packages as pseudo-cross-project,
remove them from the k-NN index, then evaluate k-NN + decoder + Jaccard gate.

Tests whether τ_bin = 0.7 generalizes to unseen held-out packages without peeking at true xproj.

Also saves per-query features for learned-gate training.
"""
import argparse
import json
import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import FunctionDataset
from src.evaluation.metrics import compute_subtoken_f1
from src.preprocessing.build_votes import VotesTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from eval_cross_project import collate_fn, CROSS_PROJECT_PACKAGES


PSEUDO_XPROJ = {'libsodium', 'make', 'wget', 'enscript', 'nano', 'spell'}


def jaccard_similarity(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--binary-sim-threshold', type=float, default=0.5)
    parser.add_argument('--save', default='results/validate_pseudo.json')
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    token_vocab = ckpt.get('token_vocab')
    ext_vocab = ckpt.get('ext_vocab')

    votes_vocab_path = cfg['data'].get('votes_vocab_path', 'data/votes_vocab.json')
    sp = VotesTokenizer(vocab_path=votes_vocab_path)

    print("Loading dataset...")
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

    # Exclude true xproj + pseudo xproj from k-NN index
    xproj_prefixes = tuple(pkg + '_' for pkg in CROSS_PROJECT_PACKAGES)
    pseudo_prefixes = tuple(pkg + '_' for pkg in PSEUDO_XPROJ)
    clean_train_idx = []
    pseudo_xproj_idx = []
    for idx in train_idx:
        binary = dataset.samples[idx]['binary']
        if binary.startswith(xproj_prefixes):
            continue
        if binary.startswith(pseudo_prefixes):
            pseudo_xproj_idx.append(idx)
        else:
            clean_train_idx.append(idx)

    print(f"Clean train (k-NN index): {len(clean_train_idx)}")
    print(f"Pseudo-xproj queries:      {len(pseudo_xproj_idx)}")
    pseudo_pkg_counts = Counter()
    for idx in pseudo_xproj_idx:
        pseudo_pkg_counts[dataset.samples[idx]['binary'].split('_')[0]] += 1
    print(f"Pseudo-xproj by pkg:       {dict(pseudo_pkg_counts)}")

    # Fix cfg
    ckpt_state = ckpt['model_state_dict']
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    if 'ext_encoder.embedding.weight' in ckpt_state:
        cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = ckpt_state['decoder.embedding.weight'].shape[0]

    model = FunctionNamer(cfg).to(device)
    model.load_state_dict(ckpt_state)
    model.eval()

    from torch.utils.data import Subset, DataLoader

    def extract_embeddings(idxs, desc='Encode', gen_decoder=True):
        loader = DataLoader(Subset(dataset, idxs), batch_size=64, num_workers=0,
                            collate_fn=collate_fn)
        embs_out = []
        names_out = []
        binaries_out = []
        decoder_out = []
        num_blocks_out = []
        num_ext_out = []
        beam_scores_out = []
        sos_id = sp.bos_id()
        eos_id = sp.eos_id()
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                block_tokens = batch['block_tokens'].to(device)
                edge_index = batch['edge_index'].to(device)
                ext_call_ids = batch['ext_call_ids'].to(device)
                block_features = batch['block_features'].to(device) if 'block_features' in batch else None

                callee_tokens = batch['callee_tokens'].to(device) if 'callee_tokens' in batch else None
                caller_tokens = batch['caller_tokens'].to(device) if 'caller_tokens' in batch else None

                with torch.amp.autocast('cuda', enabled=args.amp):
                    B = block_tokens.size(0)
                    block_embs = model.block_encoder(block_tokens, block_features=block_features)
                    N = block_embs.size(1)
                    x = block_embs.view(B * N, block_embs.size(-1))
                    batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
                    f, _, _ = model.graph_encoder(x, edge_index, batch_vec)

                    z = f
                    if model.ext_encoder_enabled and model.fusion is not None:
                        c = model.ext_encoder(ext_call_ids)
                        has_ext = model._compute_has_ext_calls(ext_call_ids)
                        z, _ = model.fusion(z, c, has_ext_calls=has_ext)
                    if model.callee_encoder_enabled and callee_tokens is not None:
                        callee_ctx, has_callees = model.callee_encoder(callee_tokens.to(device))
                        g = model.callee_gate(torch.cat([z, callee_ctx], dim=1))
                        z = g * z + (1 - g) * callee_ctx * has_callees.unsqueeze(1).float() + \
                            z * (1 - has_callees.unsqueeze(1).float())
                    if model.caller_encoder_enabled and caller_tokens is not None:
                        caller_ctx, has_callers = model.caller_encoder(caller_tokens.to(device))
                        g = model.caller_gate(torch.cat([z, caller_ctx], dim=1))
                        z = g * z + (1 - g) * caller_ctx * has_callers.unsqueeze(1).float() + \
                            z * (1 - has_callers.unsqueeze(1).float())

                embs_out.append(z.cpu().numpy())
                # Decoder generation (optional — skip for train index)
                for i in range(z.size(0)):
                    if gen_decoder:
                        pred_tokens, score = model.decoder.generate(z[i:i+1], sos_id, eos_id, 5)
                        decoder_out.append(sp.decode(pred_tokens) if pred_tokens else '')
                        beam_scores_out.append(float(score) / max(len(pred_tokens), 1) if pred_tokens else -999)
                    else:
                        decoder_out.append('')
                        beam_scores_out.append(0.0)
                    names_out.append(batch['name'][i])
                    binaries_out.append(batch['binary'][i])
                    num_blocks_out.append(int(batch['num_blocks'][i]) if isinstance(batch['num_blocks'], list) else int(batch['num_blocks'][i].item()))
                    nonzero = (ext_call_ids[i] != 0).sum().item()
                    num_ext_out.append(nonzero)

        return (np.concatenate(embs_out, axis=0), names_out, binaries_out,
                decoder_out, beam_scores_out, num_blocks_out, num_ext_out)

    print("\nExtracting train embeddings (for k-NN index, NO decoder)...")
    tr_embs, tr_names, tr_binaries, _, _, _, _ = extract_embeddings(clean_train_idx, 'Train', gen_decoder=False)
    tr_normed = tr_embs / (np.linalg.norm(tr_embs, axis=1, keepdims=True) + 1e-8)

    print("\nExtracting pseudo-xproj embeddings + decoder preds...")
    ps_embs, ps_names, ps_binaries, ps_dec, ps_scores, ps_blocks, ps_ext = extract_embeddings(pseudo_xproj_idx, 'Pseudo', gen_decoder=True)
    ps_normed = ps_embs / (np.linalg.norm(ps_embs, axis=1, keepdims=True) + 1e-8)

    # Compute k-NN top-1 for each pseudo-xproj query
    print("\nRunning k-NN top-1...")
    sims_all = ps_normed @ tr_normed.T  # (N_pseudo, N_train)
    top1_idx = np.argmax(sims_all, axis=1)
    top1_sim = np.max(sims_all, axis=1)
    knn_names = [tr_names[i] for i in top1_idx]

    # Compute per-binary fingerprints + max Jaccard
    print("\nComputing binary fingerprints...")
    import glob
    all_fp = {}
    for ef in glob.glob('data/external_calls/*_external.json') + glob.glob('demo/external_calls/*_external.json'):
        binary = os.path.basename(ef).replace('_external.json', '')
        try:
            ed = json.load(open(ef))
            calls = set()
            for fn in ed.get('functions', []):
                for c in fn.get('external_calls', []):
                    calls.add(c['name'])
            all_fp[binary] = calls
        except Exception:
            pass
    # Max Jaccard per pseudo binary to any clean training binary
    unique_tr_bins = list(set(tr_binaries))
    tr_fps = [all_fp.get(b, set()) for b in unique_tr_bins]
    per_binary_jacc = {}
    for b in set(ps_binaries):
        tfp = all_fp.get(b, set())
        maxj = max((jaccard_similarity(tfp, f) for f in tr_fps), default=0.0)
        per_binary_jacc[b] = maxj

    # Per-query records
    records = []
    for i in range(len(pseudo_xproj_idx)):
        b = ps_binaries[i]
        pkg = b.split('_')[0]
        records.append({
            'pkg': pkg,
            'binary': b,
            'gt': ps_names[i],
            'knn': knn_names[i],
            'decoder': ps_dec[i],
            'sim': float(top1_sim[i]),
            'beam_score': ps_scores[i],
            'num_blocks': ps_blocks[i],
            'num_ext': ps_ext[i],
            'bin_jaccard_max': per_binary_jacc[b],
            'knn_f1': compute_subtoken_f1(knn_names[i], ps_names[i]),
            'dec_f1': compute_subtoken_f1(ps_dec[i], ps_names[i]),
        })

    # Save
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    json.dump({
        'pseudo_xproj_pkgs': sorted(PSEUDO_XPROJ),
        'n_queries': len(records),
        'records': records,
    }, open(args.save, 'w'), indent=2)

    # Quick analysis
    print(f"\nSaved {len(records)} records to {args.save}")
    print("\nPer-pkg summary:")
    by_pkg = defaultdict(list)
    for r in records:
        by_pkg[r['pkg']].append(r)
    for pkg in sorted(by_pkg.keys()):
        rs = by_pkg[pkg]
        n = len(rs)
        if n == 0: continue
        knn_f1 = sum(r['knn_f1'] for r in rs) / n
        dec_f1 = sum(r['dec_f1'] for r in rs) / n
        avg_jacc = sum(r['bin_jaccard_max'] for r in rs) / n
        print(f'  {pkg:<12} n={n:>4} knn_F1={knn_f1:.4f}  dec_F1={dec_f1:.4f}  max_Jacc={avg_jacc:.3f}')


if __name__ == '__main__':
    main()
