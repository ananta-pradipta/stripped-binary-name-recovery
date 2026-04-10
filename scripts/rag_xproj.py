#!/usr/bin/env python3
"""
RAG Hybrid for cross-project evaluation.

Retrieves examples from clean training set (cross-project packages excluded),
uses CodeGen to generate names via few-shot prompting, with threshold-based
hybrid fallback.
"""
import argparse
import json
import os
import sys
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import FunctionDataset

# Import from eval_cross_project
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from eval_cross_project import (
    collate_fn, extract_embeddings_and_predict, predict_binary_with_embeddings,
    load_functions_from_graphs, load_external_calls, load_ground_truth,
    resolve_thunks, DEMO_BINARIES, CROSS_PROJECT_PACKAGES
)
from rag_predict import (
    extract_features_from_desc, build_rag_prompt, batch_knn_lookup, compute_metrics
)
from src.preprocessing.build_text_dataset import build_text_prompt as build_desc_from_data


def extract_train_embeddings(model, dataset, clean_train_idx, device, sp_model, use_amp=False):
    """Extract train embeddings for k-NN index (excluding cross-project)."""
    loader = DataLoader(Subset(dataset, clean_train_idx), batch_size=32,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)

    embs, names = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Train embs", leave=False):
            bt = batch['block_tokens'].to(device)
            ei = batch['edge_index'].to(device)
            ec = batch['ext_call_ids'].to(device)
            bf = batch.get('block_features')
            if bf is not None: bf = bf.to(device)
            ct = batch.get('callee_tokens')
            if ct is not None: ct = ct.to(device)
            crt = batch.get('caller_tokens')
            if crt is not None: crt = crt.to(device)
            dt = batch['decoder_target']

            with torch.amp.autocast('cuda', enabled=use_amp):
                be = model.block_encoder(bt, block_features=bf)
                B, N, D = be.shape
                x = be.view(B*N, D)
                bv = torch.arange(B, device=x.device).repeat_interleave(N)
                f, _, _ = model.graph_encoder(x, ei, bv)
                z = f
                if model.ext_encoder_enabled and model.fusion is not None:
                    c = model.ext_encoder(ec)
                    he = model._compute_has_ext_calls(ec)
                    z, _ = model.fusion(z, c, has_ext_calls=he)
                if model.callee_encoder_enabled and ct is not None:
                    cc, hc = model.callee_encoder(ct)
                    g = model.callee_gate(torch.cat([z, cc], dim=1))
                    z_f = g*z + (1-g)*cc
                    mask = hc.unsqueeze(1).float()
                    z = mask * z_f + (1-mask) * z
                if model.caller_encoder_enabled and crt is not None:
                    cc2, hc2 = model.caller_encoder(crt)
                    g2 = model.caller_gate(torch.cat([z, cc2], dim=1))
                    z_f2 = g2*z + (1-g2)*cc2
                    mask2 = hc2.unsqueeze(1).float()
                    z = mask2 * z_f2 + (1-mask2) * z

            embs.append(z.cpu().numpy())

            eos_id = sp_model.eos_id()
            for i in range(B):
                toks = []
                for t in dt[i]:
                    tid = t.item()
                    if tid == eos_id: break
                    if tid != 0: toks.append(tid)
                names.append(sp_model.decode(toks))

    return np.concatenate(embs, axis=0), names


def main():
    parser = argparse.ArgumentParser(description='Cross-project RAG hybrid')
    parser.add_argument('checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--text-dataset', default='data/text_dataset_v2.json')
    parser.add_argument('--lm-model', default='Salesforce/codegen-350M-mono')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--samples-per-package', type=int, default=50,
                        help='Max samples per cross-project package (stratified)')
    parser.add_argument('--save', default='results/rag_xproj.json')
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

    # CRITICAL: exclude cross-project packages from k-NN index
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

    # Load text prompts (for k-NN example descriptions)
    print(f"Loading text prompts...")
    with open(args.text_dataset) as f:
        text_data = json.load(f)
    text_lookup = {}
    for entry in text_data:
        text_lookup[(entry['binary'], entry['bap_name'])] = entry['prompt']

    # Extract clean train embeddings
    print("Extracting clean training embeddings (k-NN index)...")
    train_embs, train_names = extract_train_embeddings(
        encoder, dataset, clean_train_idx, device, sp_model, use_amp=args.amp
    )
    print(f"Train: {len(train_embs)} embeddings")

    train_descs = []
    for idx in clean_train_idx:
        sample = dataset.samples[idx]
        key = (sample['binary'], sample['bap_name'])
        train_descs.append(text_lookup.get(key, ''))

    train_normed = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)

    # Load LM
    print(f"Loading LM: {args.lm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.lm_model).to(device)
    lm.eval()

    # Extract cross-project demo data (using same path as eval_cross_project.py)
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
                alt_name = f"diffutils_{binary}"
                functions = load_functions_from_graphs(alt_name, graphs_dirs)
                if functions:
                    bin_name = alt_name
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

            # Stratified sample: take up to args.samples_per_package per package
            # We'll accumulate across all binaries per pkg and subsample later

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

                # Build text description for this function via live extraction
                func_data = functions.get(pred_func['bap_name'])
                if func_data is None:
                    continue
                ext_calls = ext_by_func.get(pred_func['bap_name'], [])
                callees = func_data.get('internal_callees', [])
                # For cross-project, we don't know real callee names — use empty
                callee_real_names = []
                caller_real_names = []
                desc = build_desc_from_data(func_data, ext_calls,
                                            callee_real_names, caller_real_names)

                all_preds.append({
                    'package': pkg,
                    'binary': bin_name,
                    'address': addr,
                    'true_name': true_name,
                    'desc': desc,
                    'embedding': embs[i:i+1],
                })
                matched += 1
            print(f" {matched} matched")

    print(f"\nTotal: {len(all_preds)} cross-project functions")

    # Stratified sample: up to N per package
    by_pkg = defaultdict(list)
    for p in all_preds:
        by_pkg[p['package']].append(p)

    sampled = []
    for pkg, items in by_pkg.items():
        random.shuffle(items)
        sampled.extend(items[:args.samples_per_package])
    print(f"Stratified sample: {len(sampled)} functions ({len(by_pkg)} packages, ≤{args.samples_per_package}/pkg)")

    # Retrieve top-k neighbors for each sample
    print(f"\nRetrieving top-{args.k} neighbors...")
    val_embs = np.concatenate([p['embedding'] for p in sampled], axis=0)
    retrieved, sim_scores = batch_knn_lookup(val_embs, train_normed, train_names, train_descs, k=args.k)

    # Generate with LM
    print("Generating RAG predictions...")
    predictions = []
    knn_baseline = []
    top1_sims = []

    for i, (p, examples, scores) in enumerate(
        tqdm(list(zip(sampled, retrieved, sim_scores)), desc="RAG")
    ):
        knn_pred = examples[0][1] if examples else ""
        knn_baseline.append(knn_pred)
        top1_sims.append(scores[0] if scores else 0.0)

        prompt = build_rag_prompt(p['desc'], examples,
                                  similarity_scores=scores, max_examples=args.k)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                           max_length=900).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=args.amp):
                gen_ids = lm.generate(
                    **inputs, max_new_tokens=20, do_sample=False,
                    temperature=0.1, pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                )
        gen_text = tokenizer.decode(
            gen_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        gen_text = gen_text.split('\n')[0].split('(')[0].split(',')[0].split(' ')[0].strip()
        gen_text = gen_text.rstrip('.;:')
        predictions.append(gen_text)

    true_names = [p['true_name'] for p in sampled]

    # Metrics
    knn_m = compute_metrics(knn_baseline, true_names)
    rag_m = compute_metrics(predictions, true_names)

    print(f"\n=== Cross-Project Results ({len(sampled)} functions) ===")
    print(f"k-NN (top-1):  EM={knn_m['em']:.1%}, F1={knn_m['f1']:.4f}")
    print(f"RAG (LM):      EM={rag_m['em']:.1%}, F1={rag_m['f1']:.4f}")

    # Hybrid threshold sweep
    print(f"\n=== Hybrid Threshold Sweep ===")
    print(f"{'Threshold':>10} {'k-NN used':>10} {'LM used':>8} {'EM':>7} {'F1':>7}")
    best = None
    for t in [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        hybrid = []
        n_knn = 0
        for k_p, r_p, sim in zip(knn_baseline, predictions, top1_sims):
            if sim > t:
                hybrid.append(k_p)
                n_knn += 1
            else:
                hybrid.append(r_p)
        m = compute_metrics(hybrid, true_names)
        print(f"{t:>10.2f} {n_knn:>10} {len(hybrid)-n_knn:>8} {m['em']:>6.1%} {m['f1']:>7.4f}")
        if best is None or m['em'] > best['em']:
            best = {**m, 'threshold': t, 'n_knn': n_knn}

    print(f"\nBest hybrid: threshold={best['threshold']:.2f}, EM={best['em']:.1%}, F1={best['f1']:.4f}")

    # Per-package breakdown
    print(f"\n=== Per-Package Analysis ===")
    print(f"{'Package':<12} {'N':>4} {'k-NN':>7} {'RAG':>7} {'Hybrid':>8} {'AvgSim':>8}")
    pkg_stats = defaultdict(lambda: {'true': [], 'knn': [], 'rag': [], 'hybrid': [], 'sim': []})
    for p, k_p, r_p, sim in zip(sampled, knn_baseline, predictions, top1_sims):
        pkg = p['package']
        pkg_stats[pkg]['true'].append(p['true_name'])
        pkg_stats[pkg]['knn'].append(k_p)
        pkg_stats[pkg]['rag'].append(r_p)
        pkg_stats[pkg]['hybrid'].append(k_p if sim > best['threshold'] else r_p)
        pkg_stats[pkg]['sim'].append(sim)

    for pkg in sorted(pkg_stats.keys()):
        d = pkg_stats[pkg]
        n = len(d['true'])
        knn_em = sum(1 for p, t in zip(d['knn'], d['true']) if p == t) / n
        rag_em = sum(1 for p, t in zip(d['rag'], d['true']) if p == t) / n
        hyb_em = sum(1 for p, t in zip(d['hybrid'], d['true']) if p == t) / n
        avg_sim = sum(d['sim']) / n
        print(f"{pkg:<12} {n:>4} {knn_em:>6.1%} {rag_em:>6.1%} {hyb_em:>7.1%} {avg_sim:>8.3f}")

    # Sample predictions
    print(f"\n=== Sample predictions ===")
    for i in range(min(25, len(sampled))):
        match_knn = "✓" if knn_baseline[i] == true_names[i] else "✗"
        match_rag = "✓" if predictions[i] == true_names[i] else "✗"
        print(f"  [{match_knn}/{match_rag}] [{sampled[i]['package']}] sim={top1_sims[i]:.3f} true=\"{true_names[i]}\"")
        print(f"      k-NN: \"{knn_baseline[i]}\"")
        print(f"      RAG:  \"{predictions[i]}\"")

    # Save
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump({
            'n_eval': len(sampled),
            'knn': knn_m,
            'rag': rag_m,
            'best_hybrid': best,
            'predictions': [
                {'pkg': sampled[i]['package'], 'true': true_names[i],
                 'knn': knn_baseline[i], 'rag': predictions[i], 'sim': top1_sims[i]}
                for i in range(len(sampled))
            ],
        }, f, indent=2)
    print(f"\nSaved to {args.save}")


if __name__ == '__main__':
    main()
