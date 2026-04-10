#!/usr/bin/env python3
"""
Retrieval-Augmented Generation for Binary Function Name Prediction.

Pipeline:
1. Encoder produces embedding z for each function
2. k-NN retrieves top-K similar functions from training set
3. Build in-context prompt with retrieved examples
4. CodeGen generates function name given examples + target description

No training needed — pure inference using:
- Pre-trained encoder (our best_model.pt)
- Pre-trained CodeGen-350M (no fine-tuning)
"""
import argparse
import json
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict, Counter
import glob as glob_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import (
    FunctionDataset, compute_block_features, compute_block_degrees, NUM_BLOCK_FEATURES
)
from src.preprocessing.build_text_dataset import build_text_prompt
from src.evaluation.metrics import (
    compute_subtoken_f1,
    compute_exact_match,
    compute_char_ngram_similarity,
    compute_edit_distance_similarity,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def extract_embeddings_and_descriptions(model, dataset, indices, device, sp_model,
                                         batch_size=32, use_amp=False):
    """Extract encoder embeddings and text descriptions for a subset of functions."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    embeddings = []
    names = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract train embeddings", leave=False):
            bt = batch['block_tokens'].to(device)
            ei = batch['edge_index'].to(device)
            ec = batch['ext_call_ids'].to(device)
            bf = batch.get('block_features')
            if bf is not None:
                bf = bf.to(device)
            ct = batch.get('callee_tokens')
            if ct is not None:
                ct = ct.to(device)
            crt = batch.get('caller_tokens')
            if crt is not None:
                crt = crt.to(device)
            dt = batch['decoder_target'].to(device)

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
                    z_fused = g * z + (1 - g) * cc
                    mask = hc.unsqueeze(1).float()
                    z = mask * z_fused + (1 - mask) * z
                if model.caller_encoder_enabled and crt is not None:
                    cc2, hc2 = model.caller_encoder(crt)
                    g2 = model.caller_gate(torch.cat([z, cc2], dim=1))
                    z_fused2 = g2 * z + (1 - g2) * cc2
                    mask2 = hc2.unsqueeze(1).float()
                    z = mask2 * z_fused2 + (1 - mask2) * z

            embeddings.append(z.cpu().numpy())

            # Extract names from decoder_target
            eos_id = sp_model.eos_id()
            for i in range(B):
                tokens = []
                for t in dt[i]:
                    tid = t.item()
                    if tid == eos_id:
                        break
                    if tid != 0:
                        tokens.append(tid)
                name = sp_model.decode(tokens)
                names.append(name)

    return np.concatenate(embeddings, axis=0), names


def load_text_prompts(text_dataset_path):
    """Load pre-built text prompts for all training functions."""
    with open(text_dataset_path) as f:
        return json.load(f)


def extract_features_from_desc(desc):
    """Extract key features from a V2 description string."""
    lines = desc.split('\n')
    features = {}
    for line in lines:
        if 'Library calls:' in line:
            features['lib'] = line.split('Library calls:', 1)[1].strip().rstrip('* /')
        elif 'Internal calls:' in line:
            features['internal'] = line.split('Internal calls:', 1)[1].strip().rstrip('* /')
        elif 'Called by:' in line:
            features['callers'] = line.split('Called by:', 1)[1].strip().rstrip('* /')
        elif 'Size:' in line:
            features['size'] = line.split('Size:', 1)[1].strip().rstrip('* /')
    return features


def build_rag_prompt(target_description, retrieved_examples,
                     similarity_scores=None, max_examples=5):
    """Build RAG prompt framing examples as nearest neighbors.

    The k-NN retrieved functions are the MOST SIMILAR functions in our
    training database of 200K+ functions. We explicitly tell the LM this
    and ask it to predict a name consistent with these neighbors.
    """
    target_features = extract_features_from_desc(target_description)

    prompt = "// Binary function name recovery using nearest-neighbor analysis.\n"
    prompt += "// The following are the MOST SIMILAR functions (by code embedding)\n"
    prompt += "// to the target function in our database of 200,000 labeled functions.\n"
    prompt += "// Similar functions share naming patterns. Predict the target name.\n\n"

    for i, (desc, name) in enumerate(retrieved_examples[:max_examples], 1):
        feats = extract_features_from_desc(desc)
        sim_str = ""
        if similarity_scores is not None and i - 1 < len(similarity_scores):
            sim_str = f" (similarity: {similarity_scores[i-1]:.3f})"
        prompt += f"// Nearest neighbor #{i}{sim_str}:\n"
        prompt += f"//   library_calls: {feats.get('lib', '(none)')}\n"
        prompt += f"//   internal_calls: {feats.get('internal', '(none)')}\n"
        prompt += f"//   called_by: {feats.get('callers', '(none)')}\n"
        prompt += f"//   size: {feats.get('size', '(unknown)')}\n"
        prompt += f"//   name: {name}\n\n"

    prompt += "// Target function (predict name consistent with neighbors above):\n"
    prompt += f"//   library_calls: {target_features.get('lib', '(none)')}\n"
    prompt += f"//   internal_calls: {target_features.get('internal', '(none)')}\n"
    prompt += f"//   called_by: {target_features.get('callers', '(none)')}\n"
    prompt += f"//   size: {target_features.get('size', '(unknown)')}\n"
    prompt += "//   name:"

    return prompt


def batch_knn_lookup(query_embs, train_normed, train_names, train_descs, k=5):
    """Retrieve top-k most similar training functions with similarity scores."""
    query_normed = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)

    results = []
    scores = []
    for i in range(len(query_normed)):
        sims = query_normed[i] @ train_normed.T
        top_k_idx = np.argsort(sims)[-k:][::-1]
        examples = [(train_descs[idx], train_names[idx]) for idx in top_k_idx]
        results.append(examples)
        scores.append([float(sims[idx]) for idx in top_k_idx])
    return results, scores


def compute_metrics(preds, trues):
    n = len(preds)
    if n == 0:
        return {'n': 0, 'em': 0, 'f1': 0}
    em = sum(1 for p, t in zip(preds, trues) if p == t) / n
    f1 = sum(compute_subtoken_f1(p, t) for p, t in zip(preds, trues)) / n
    return {'n': n, 'em': em, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description='RAG-based function naming')
    parser.add_argument('checkpoint', help='Encoder checkpoint')
    parser.add_argument('--config', default='configs/optimized_large.yaml')
    parser.add_argument('--text-dataset', default='data/text_dataset.json')
    parser.add_argument('--lm-model', default='Salesforce/codegen-350M-mono')
    parser.add_argument('--k', type=int, default=5, help='Number of retrieved examples')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--max-eval', type=int, default=200,
                        help='Max samples to eval (for speed)')
    parser.add_argument('--eval-split', default='val', choices=['val', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--random-sample', action='store_true',
                        help='Random sample from split instead of first N')
    parser.add_argument('--stratified', action='store_true',
                        help='Stratified sample per binary for balanced evaluation')
    parser.add_argument('--save', default='results/rag_predictions.json')
    args = parser.parse_args()

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

    ckpt_state = ckpt['model_state_dict']
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    if 'ext_encoder.embedding.weight' in ckpt_state:
        cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = ckpt_state['decoder.embedding.weight'].shape[0]

    encoder = FunctionNamer(cfg).to(device)
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()
    print(f"Encoder loaded")

    # Load text dataset
    print(f"Loading text prompts: {args.text_dataset}")
    text_data = load_text_prompts(args.text_dataset)
    print(f"  {len(text_data)} text prompts")

    # Build text lookup: {(binary, bap_name): text_prompt}
    text_lookup = {}
    for entry in text_data:
        key = (entry['binary'], entry['bap_name'])
        text_lookup[key] = entry['prompt']

    # Get train embeddings and names
    print("\nExtracting training embeddings...")
    train_embs, train_names = extract_embeddings_and_descriptions(
        encoder, dataset, train_idx, device, sp_model, use_amp=args.amp
    )
    print(f"  Train: {len(train_embs)} embeddings, {len(set(train_names))} unique names")

    # Build train text prompts matching the train indices
    train_descs = []
    for idx in train_idx:
        sample = dataset.samples[idx]
        key = (sample['binary'], sample['bap_name'])
        desc = text_lookup.get(key, sample['binary'] + ' function')
        train_descs.append(desc)

    # Normalize for k-NN
    train_normed = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)

    # Load LM
    print(f"\nLoading LM: {args.lm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.lm_model).to(device)
    lm.eval()
    print(f"  LM loaded: {sum(p.numel() for p in lm.parameters()):,} params")

    # Evaluate on val or test
    split_idx = val_idx if args.eval_split == 'val' else test_idx
    if args.stratified:
        # Stratified sample: equal samples per binary
        from collections import defaultdict
        import random as _r
        _r.seed(42)
        by_binary = defaultdict(list)
        for idx in split_idx:
            by_binary[dataset.samples[idx]['binary']].append(idx)
        n_binaries = len(by_binary)
        per_binary = max(1, args.max_eval // n_binaries)
        split_idx = []
        for binary, idxs in sorted(by_binary.items()):
            sampled = _r.sample(idxs, min(per_binary, len(idxs)))
            split_idx.extend(sampled)
        print(f"Stratified sampling: {n_binaries} binaries, ~{per_binary}/binary → {len(split_idx)} total")
    elif args.random_sample:
        import random as _r
        _r.seed(42)
        split_idx = _r.sample(split_idx, min(args.max_eval, len(split_idx)))
    else:
        split_idx = split_idx[:args.max_eval]
    print(f"\nEvaluating on {len(split_idx)} {args.eval_split} samples...")
    eval_indices = split_idx

    val_embs, val_names = extract_embeddings_and_descriptions(
        encoder, dataset, eval_indices, device, sp_model, use_amp=args.amp
    )
    val_descs = []
    for idx in eval_indices:
        sample = dataset.samples[idx]
        key = (sample['binary'], sample['bap_name'])
        desc = text_lookup.get(key, sample['binary'] + ' function')
        val_descs.append(desc)

    # Retrieve top-k for each val sample
    print(f"Retrieving top-{args.k} examples per function...")
    retrieved, sim_scores = batch_knn_lookup(val_embs, train_normed, train_names, train_descs, k=args.k)

    # Generate with LM using RAG prompts
    print("Generating with RAG prompts...")
    predictions = []
    knn_baseline = []  # top-1 k-NN for comparison

    for i, (target_desc, true_name, examples, scores) in enumerate(
        tqdm(list(zip(val_descs, val_names, retrieved, sim_scores)), desc="RAG gen")
    ):
        # k-NN baseline: top-1 name
        knn_pred = examples[0][1] if examples else ""
        knn_baseline.append(knn_pred)

        # Build RAG prompt with similarity scores
        prompt = build_rag_prompt(target_desc, examples,
                                  similarity_scores=scores, max_examples=args.k)

        # Generate
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                           max_length=900).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=args.amp):
                gen_ids = lm.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                )

        gen_text = tokenizer.decode(
            gen_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        # Clean up
        gen_text = gen_text.split('\n')[0].split('(')[0].split(',')[0].split(' ')[0].strip()
        gen_text = gen_text.rstrip('.;:')

        predictions.append(gen_text)

    # Compute metrics
    rag_metrics = compute_metrics(predictions, val_names)
    knn_metrics = compute_metrics(knn_baseline, val_names)

    # Get top-1 similarity for each query (for hybrid threshold)
    top1_sims = [scores[0] if scores else 0.0 for scores in sim_scores]

    print(f"\n=== Results (on {len(val_names)} val functions) ===")
    print(f"k-NN (top-1):  EM={knn_metrics['em']:.1%}, F1={knn_metrics['f1']:.4f}")
    print(f"RAG (LM+k-NN): EM={rag_metrics['em']:.1%}, F1={rag_metrics['f1']:.4f}")

    # Hybrid: use k-NN when confident, LM otherwise
    print(f"\n=== Hybrid (k-NN if sim>threshold, else LM) ===")
    print(f"{'Threshold':>10} {'k-NN used':>10} {'LM used':>8} {'EM':>7} {'F1':>7}")
    print(f"{'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*7}")
    best_hybrid = None
    for threshold in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        hybrid_preds = []
        n_knn = 0
        for knn_p, rag_p, sim in zip(knn_baseline, predictions, top1_sims):
            if sim > threshold:
                hybrid_preds.append(knn_p)
                n_knn += 1
            else:
                hybrid_preds.append(rag_p)
        m = compute_metrics(hybrid_preds, val_names)
        n_lm = len(hybrid_preds) - n_knn
        print(f"{threshold:>10.2f} {n_knn:>10} {n_lm:>8} {m['em']:>6.1%} {m['f1']:>7.4f}")
        if best_hybrid is None or m['em'] > best_hybrid['em']:
            best_hybrid = {**m, 'threshold': threshold, 'n_knn': n_knn, 'n_lm': n_lm}

    print(f"\nBest hybrid: threshold={best_hybrid['threshold']:.2f}, "
          f"EM={best_hybrid['em']:.1%}, F1={best_hybrid['f1']:.4f}")
    print(f"  k-NN used: {best_hybrid['n_knn']}, LM used: {best_hybrid['n_lm']}")

    # Per-package analysis
    print(f"\n=== Per-package analysis ===")
    from collections import defaultdict as _dd
    pkg_data = _dd(lambda: {'trues': [], 'knns': [], 'rags': [], 'hybrids': [], 'sims': []})
    best_thresh = best_hybrid['threshold']
    for i, idx in enumerate(eval_indices):
        binary = dataset.samples[idx]['binary']
        pkg = binary.split('_')[0]
        pkg_data[pkg]['trues'].append(val_names[i])
        pkg_data[pkg]['knns'].append(knn_baseline[i])
        pkg_data[pkg]['rags'].append(predictions[i])
        sim = top1_sims[i]
        hybrid_pred = knn_baseline[i] if sim > best_thresh else predictions[i]
        pkg_data[pkg]['hybrids'].append(hybrid_pred)
        pkg_data[pkg]['sims'].append(sim)

    print(f"{'Package':<15} {'N':>4} {'k-NN EM':>8} {'RAG EM':>8} {'Hybrid EM':>10} {'Avg Sim':>8}")
    print(f"{'-'*15} {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for pkg in sorted(pkg_data.keys()):
        d = pkg_data[pkg]
        knn_em = sum(1 for p, t in zip(d['knns'], d['trues']) if p == t) / len(d['trues'])
        rag_em = sum(1 for p, t in zip(d['rags'], d['trues']) if p == t) / len(d['trues'])
        hyb_em = sum(1 for p, t in zip(d['hybrids'], d['trues']) if p == t) / len(d['trues'])
        avg_sim = sum(d['sims']) / len(d['sims'])
        print(f"{pkg:<15} {len(d['trues']):>4} {knn_em:>7.1%} {rag_em:>7.1%} {hyb_em:>9.1%} {avg_sim:>8.3f}")

    # Print sample predictions
    print(f"\n=== Sample predictions ===")
    for i in range(min(20, len(val_names))):
        match_rag = "✓" if predictions[i] == val_names[i] else "✗"
        match_knn = "✓" if knn_baseline[i] == val_names[i] else "✗"
        print(f"  [{match_knn}/{match_rag}] sim={top1_sims[i]:.3f} true=\"{val_names[i]}\"")
        print(f"      k-NN top-1: \"{knn_baseline[i]}\"")
        print(f"      RAG LM:     \"{predictions[i]}\"")

    # Save
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    with open(args.save, 'w') as f:
        json.dump({
            'k': args.k,
            'n_eval': len(val_names),
            'knn_metrics': knn_metrics,
            'rag_metrics': rag_metrics,
            'predictions': [
                {'true': t, 'knn_pred': k_, 'rag_pred': r}
                for t, k_, r in zip(val_names, knn_baseline, predictions)
            ],
        }, f, indent=2)
    print(f"\nSaved to {args.save}")


if __name__ == '__main__':
    main()
