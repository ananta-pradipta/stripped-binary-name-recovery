#!/bin/bash
# ============================================================
# Step 5: Full Evaluation & Analysis
# ============================================================
# Usage:
#   bash scripts/05_evaluate.sh          # evaluate all models
#   bash scripts/05_evaluate.sh 5        # evaluate Option B only
#   bash scripts/05_evaluate.sh optb     # same as above
#   bash scripts/05_evaluate.sh 2        # evaluate GCN only
#   bash scripts/05_evaluate.sh 3        # evaluate GAT only
#   bash scripts/05_evaluate.sh 4        # evaluate Option A only
#   bash scripts/05_evaluate.sh 1        # evaluate ExtraTrees only
# ============================================================
set -e
source ~/cs785-project/activate.sh
cd ~/cs785-project

VARIANT="${1:-all}"
case "$VARIANT" in
    optb|optB|b|B) VARIANT="5" ;;
    all|1|2|3|4|5) ;;
    *) echo "  ✗ Unknown variant: $VARIANT"; echo "  Usage: bash scripts/05_evaluate.sh [all|1|2|3|4|5|optb]"; exit 1 ;;
esac

echo "═══════════════════════════════════════════════"
echo " Step 5: Full Evaluation & Analysis"
echo "═══════════════════════════════════════════════"
echo "  Evaluating: ${VARIANT}"

# Pass VARIANT to Python via environment variable
export EVAL_VARIANT="$VARIANT"

python3 << 'EVALSCRIPT'
import json
import os
import sys
import re
import torch
import yaml
import numpy as np
import sentencepiece as spm
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.expanduser('~/cs785-project'))

from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import FunctionDataset
from src.evaluation.metrics import (
    compute_subtoken_f1, compute_subtoken_precision_recall,
    compute_exact_match, compute_char_ngram_similarity,
    compute_edit_distance_similarity, compute_all_metrics,
    normalize_name, split_name,
)

# Read variant from environment
VARIANT = os.environ.get('EVAL_VARIANT', 'all')

def collate_fn(batch):
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

with open('configs/optimized.yaml') as f:
    default_cfg = yaml.safe_load(f)

votes_vocab_path = default_cfg['data'].get('votes_vocab_path')
if votes_vocab_path and os.path.exists(votes_vocab_path):
    from src.preprocessing.build_votes import VotesTokenizer
    sp = VotesTokenizer(vocab_path=votes_vocab_path)
else:
    sp = spm.SentencePieceProcessor(model_file=default_cfg['data']['bpe_model_path'])
sos_id = sp.bos_id()
eos_id = sp.eos_id()

# Load only test split to avoid OOM with large models
import gc
sa = json.load(open('data/split_assignments.json'))
mi = json.load(open('data/match_index.json'))
test_bins = set(sa['test'])
test_entries = {k: v for k, v in mi.items() if v['binary'] in test_bins}
print(f"Test set: {len(test_entries)} functions from {len(test_bins)} binaries")

# Pre-load checkpoint token/ext vocab for correct ID mapping
_ckpt_preview = torch.load('checkpoints/best_model.pt', map_location='cpu')
_ckpt_token_vocab = _ckpt_preview.get('token_vocab')
_ckpt_ext_vocab = _ckpt_preview.get('ext_vocab')
del _ckpt_preview
gc.collect()

dataset = FunctionDataset(
    graphs_dir=default_cfg['data']['graphs_dir'],
    labels_dir=default_cfg['data']['labels_dir'],
    external_calls_dir=default_cfg['data']['external_calls_dir'],
    bpe_model_path=default_cfg['data']['bpe_model_path'],
    external_vocab_path=default_cfg['data']['external_vocab_path'],
    max_blocks=default_cfg['data']['max_blocks_per_function'],
    max_tokens=default_cfg['data']['max_tokens_per_block'],
    max_name_len=default_cfg['data']['max_name_length'],
    votes_vocab_path=votes_vocab_path,
    token_vocab=_ckpt_token_vocab,
    ext_vocab_override=_ckpt_ext_vocab,
    match_index_override=test_entries,
    enrich_callees=True,
)
dataset.training_mode = False

print(f"Test set: {len(dataset)} functions")

test_loader = DataLoader(
    dataset,
    batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0,
)
print(f"Test set: {len(dataset)} functions")


def evaluate_model(model, loader, sp_model):
    model.eval()
    results = []
    all_gates = []

    with torch.no_grad():
        for batch in loader:
            block_tokens = batch['block_tokens'].to(device)
            edge_index = batch['edge_index'].to(device)
            ext_call_ids = batch['ext_call_ids'].to(device)
            block_features = batch.get('block_features')
            if block_features is not None:
                block_features = block_features.to(device)
            decoder_input = batch['decoder_input'].to(device)
            decoder_target = batch['decoder_target'].to(device)
            true_names = batch['name']
            addresses = batch['address']

            callee_tok = batch.get('callee_tokens')
            if callee_tok is not None:
                callee_tok = callee_tok.to(device)
            caller_tok = batch.get('caller_tokens')
            if caller_tok is not None:
                caller_tok = caller_tok.to(device)
            model_out = model(
                block_tokens, edge_index, ext_call_ids,
                decoder_input, teacher_forcing_ratio=0.0,
                block_features=block_features,
                callee_tokens=callee_tok,
                caller_tokens=caller_tok,
            )
            logits = model_out[0]
            gate_values = model_out[1]

            if gate_values is not None:
                all_gates.append(gate_values.cpu())

            callee_tokens = batch.get('callee_tokens')
            if callee_tokens is not None:
                callee_tokens = callee_tokens.to(device)
            caller_tokens = batch.get('caller_tokens')
            if caller_tokens is not None:
                caller_tokens = caller_tokens.to(device)
            beam_results = model.predict(
                block_tokens, edge_index, ext_call_ids,
                sos_id=sos_id, eos_id=eos_id, beam_width=5,
                block_features=block_features,
                callee_tokens=callee_tokens,
                caller_tokens=caller_tokens,
            )

            B = len(true_names)
            for i in range(B):
                if i < len(beam_results):
                    pred_tokens, score = beam_results[i]
                    pred_name = sp_model.decode(pred_tokens).strip()
                else:
                    pred_name = ""
                    score = float('-inf')

                true_name = true_names[i]
                metrics = compute_all_metrics(pred_name, true_name)
                num_ext = (ext_call_ids[i] > 0).sum().item()

                results.append({
                    'address': addresses[i],
                    'true_name': true_name,
                    'predicted_name': pred_name,
                    'score': round(score / max(len(pred_tokens), 1), 4) if pred_tokens and score != float('-inf') else -999,
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'exact_match': metrics['exact_match'],
                    'char_ngram_sim': metrics['char_ngram_sim'],
                    'edit_sim': metrics['edit_sim'],
                    'num_ext_calls': num_ext,
                })

    gate_data = torch.cat(all_gates, dim=0) if all_gates else None
    return results, gate_data


def aggregate_metrics(results):
    n = len(results)
    if n == 0:
        return {}
    return {
        'num_functions': n,
        'precision': sum(r['precision'] for r in results) / n,
        'recall': sum(r['recall'] for r in results) / n,
        'f1': sum(r['f1'] for r in results) / n,
        'exact_match': sum(r['exact_match'] for r in results) / n,
        'char_ngram_sim': sum(r['char_ngram_sim'] for r in results) / n,
        'edit_sim': sum(r['edit_sim'] for r in results) / n,
    }


def load_and_evaluate(checkpoint_path, name):
    print(f"  Loading {name}...")
    if not os.path.exists(checkpoint_path):
        print(f"    ✗ Checkpoint not found: {checkpoint_path}")
        return None, None, None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    # CRITICAL: Use checkpoint vocab sizes (not dataset's, which may have extra tokens)
    ckpt_token_vocab = ckpt.get('token_vocab', dataset.token_vocab)
    dataset.token_vocab = ckpt_token_vocab  # Override dataset's expanded vocab
    cfg['block_encoder']['token_vocab_size'] = len(ckpt_token_vocab)
    if 'ext_vocab' in ckpt:
        dataset.ext_vocab = ckpt['ext_vocab']
        cfg['external_encoder']['vocab_size'] = len(ckpt['ext_vocab'])
    cfg['decoder']['bpe_vocab_size'] = sp.get_piece_size()
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']

    # Restore package vocab from checkpoint for binary context embedding
    if 'package_vocab' in ckpt and cfg.get('binary_context', {}).get('enabled', False):
        pkg_vocab = ckpt['package_vocab']
        cfg['binary_context']['num_packages'] = len(pkg_vocab)
        # Update dataset's package_vocab so __getitem__ uses the right IDs
        dataset.package_vocab = pkg_vocab

    try:
        model = FunctionNamer(cfg).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
    except RuntimeError as e:
        print(f"    ✗ Failed to load: {e}")
        return None, None, None

    # Clamp ext_call_ids to model's vocab size
    ext_vocab_size = cfg['external_encoder']['vocab_size']

    results, gate_data = evaluate_model(model, test_loader, sp)
    agg = aggregate_metrics(results)

    print(f"    P={agg['precision']:.4f}  R={agg['recall']:.4f}  "
          f"F1={agg['f1']:.4f}  EM={agg['exact_match']:.4f}  "
          f"NgSim={agg['char_ngram_sim']:.4f}  EdSim={agg['edit_sim']:.4f}")

    return agg, results, gate_data


print()
print("═══ Evaluating on Test Set ═══")
print()

all_results = {}
os.makedirs('results', exist_ok=True)

r5 = None
gate_data = None

# Variant 1: ExtraTrees
if VARIANT in ('all', '1'):
    et_path = 'results/baseline_extratrees.json'
    if os.path.exists(et_path):
        with open(et_path) as f:
            et = json.load(f)
        acc = et.get('accuracy', 0)
        all_results['DeBin (ExtraTrees)'] = {
            'precision': acc, 'recall': acc, 'f1': acc, 'exact_match': acc,
            'char_ngram_sim': acc, 'edit_sim': acc,
            'note': 'closed-vocab classification accuracy (not directly comparable to sub-token F1)',
        }
        print(f"  DeBin (ExtraTrees): accuracy={acc:.4f}")

# Variant 2: GCN
if VARIANT in ('all', '2'):
    m2, r2, _ = load_and_evaluate('checkpoints/variant2_gcn/best_model.pt', 'GCN + Mean Pool')
    if m2: all_results['GCN + Mean Pool'] = m2

# Variant 3: GAT
if VARIANT in ('all', '3'):
    m3, r3, _ = load_and_evaluate('checkpoints/variant3_gat/best_model.pt', 'GAT + Attn Pool')
    if m3: all_results['GAT + Attn Pool'] = m3

# Variant 4: Option A
if VARIANT in ('all', '4'):
    m4, r4, _ = load_and_evaluate('checkpoints/variant4_optA/best_model.pt', 'GAT + Option A')
    if m4: all_results['GAT + Option A'] = m4

# Variant 5: Option B
if VARIANT in ('all', '5'):
    m5, r5, gate_data = load_and_evaluate('checkpoints/best_model.pt', '★ GAT + Option B')
    if m5: all_results['★ GAT + Option B'] = m5

# ═════ ABLATION TABLE ═════
if all_results:
    print()
    print("═══════════════════════════════════════════════════════════════════════════════════")
    print("                              ABLATION TABLE")
    print("═══════════════════════════════════════════════════════════════════════════════════")
    print(f"  {'Model':<27s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'EM':>6s} {'NgSim':>6s} {'EdSim':>6s} {'Δ F1'}")
    print(f"  {'─'*27} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*10}")

    baseline_f1 = None
    for name, metrics in all_results.items():
        f1 = metrics['f1']
        if baseline_f1 is None:
            baseline_f1 = f1
            delta = " (base)"
        elif baseline_f1 > 0:
            pct = 100 * (f1 - baseline_f1) / baseline_f1
            delta = f" +{pct:.1f}%" if pct >= 0 else f" {pct:.1f}%"
        else:
            delta = " N/A"
        print(f"  {name:<27s} {metrics['precision']:>6.3f} {metrics['recall']:>6.3f} "
              f"{f1:>6.3f} {metrics['exact_match']:>6.3f} "
              f"{metrics['char_ngram_sim']:>6.3f} {metrics['edit_sim']:>6.3f}{delta}")

    print(f"  {'─'*83}")
    print()
    print("  Metrics explanation:")
    print("    Prec/Rec/F1  = Sub-token precision/recall/F1 (split name by _ and camelCase)")
    print("    EM           = Exact match rate (whole name correct)")
    print("    NgSim        = Character 3-gram similarity (Dice coefficient)")
    print("    EdSim        = Edit distance similarity (1 - normalized Levenshtein)")
    print()

    with open('results/ablation_table.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("  Saved: results/ablation_table.json")

# ═════ GATE ANALYSIS ═════
if gate_data is not None and r5 is not None:
    print()
    print("═══════════════════════════════════════════════════════════════════════════════════")
    print("                     GATE VALUE ANALYSIS (Option B)")
    print("═══════════════════════════════════════════════════════════════════════════════════")

    avg_gates = gate_data.mean(dim=1).numpy()
    groups = {'0 ext calls': [], '1-3 ext calls': [], '4+ ext calls': []}

    for i, func_result in enumerate(r5):
        if i >= len(avg_gates): break
        n_ext = func_result['num_ext_calls']
        g = avg_gates[i]
        if n_ext == 0: groups['0 ext calls'].append(g)
        elif n_ext <= 3: groups['1-3 ext calls'].append(g)
        else: groups['4+ ext calls'].append(g)

    print()
    print(f"  {'Group':<22s} {'Avg Gate':>10s} {'Std':>8s} {'Count':>7s} {'Interpretation'}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*7} {'─'*25}")

    gate_json = {'groups': {}}
    for group_name, values in groups.items():
        if values:
            avg = float(np.mean(values))
            std = float(np.std(values))
            interp = "trust code" if avg > 0.55 else ("balanced" if avg > 0.45 else "trust ext")
            print(f"  {group_name:<22s} {avg:>10.4f} {std:>8.4f} {len(values):>7d} {interp}")
            gate_json['groups'][group_name] = {'avg': avg, 'std': std, 'count': len(values)}

    overall_mean = float(np.mean(avg_gates))
    gate_json['overall_mean'] = overall_mean
    gate_json['overall_std'] = float(np.std(avg_gates))
    print(f"\n  Overall: mean={overall_mean:.4f}, std={gate_json['overall_std']:.4f}")

    paired = [(avg_gates[i], r5[i]) for i in range(min(len(avg_gates), len(r5)))]
    paired.sort(key=lambda x: x[0])

    print(f"\n  Top 5 — Gate TRUSTS EXTERNAL (g lowest):")
    for g, r in paired[:5]:
        print(f"    g={g:.3f}  true={r['true_name']:<30s} pred={r['predicted_name']:<25s} F1={r['f1']:.2f}  ext={r['num_ext_calls']}")

    print(f"\n  Top 5 — Gate TRUSTS CODE (g highest):")
    for g, r in paired[-5:]:
        print(f"    g={g:.3f}  true={r['true_name']:<30s} pred={r['predicted_name']:<25s} F1={r['f1']:.2f}  ext={r['num_ext_calls']}")

    with open('results/gate_analysis.json', 'w') as f:
        json.dump(gate_json, f, indent=2)
    print(f"\n  Saved: results/gate_analysis.json")

# ═════ ERROR ANALYSIS ═════
if r5:
    print()
    print("═══════════════════════════════════════════════════════════════════════════════════")
    print("                            ERROR ANALYSIS")
    print("═══════════════════════════════════════════════════════════════════════════════════")

    worst = sorted(r5, key=lambda r: r['f1'])[:15]
    print(f"\n  Worst 15 predictions:")
    print(f"  {'True Name':<30s} {'Predicted':<30s} {'F1':>5s} {'NgSim':>6s} {'EdSim':>6s} {'Ext':>4s}")
    print(f"  {'─'*30} {'─'*30} {'─'*5} {'─'*6} {'─'*6} {'─'*4}")
    for r in worst:
        print(f"  {r['true_name']:<30s} {r['predicted_name']:<30s} {r['f1']:>5.2f} "
              f"{r['char_ngram_sim']:>6.3f} {r['edit_sim']:>6.3f} {r['num_ext_calls']:>4d}")

    failures = [r for r in r5 if r['f1'] < 0.5]
    patterns = defaultdict(int)
    for r in failures:
        if r['num_ext_calls'] == 0: patterns['no_ext_calls'] += 1
        true_tokens = split_name(normalize_name(r['true_name']))
        if len(true_tokens) <= 1: patterns['short_name (1 token)'] += 1
        elif len(true_tokens) > 5: patterns['long_name (>5 tokens)'] += 1
        else: patterns['other'] += 1

    print(f"\n  Failure patterns (F1 < 0.5): {len(failures)} functions")
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"    {pattern:<25s}: {count:>4d} ({100*count/max(len(failures),1):.0f}%)")

    perfect = [r for r in r5 if r['exact_match']]
    print(f"\n  Perfect predictions (exact match): {len(perfect)}/{len(r5)} ({100*len(perfect)/max(len(r5),1):.1f}%)")
    for r in perfect[:10]:
        print(f"    {r['true_name']:<35s} (ext={r['num_ext_calls']})")
    if len(perfect) > 10:
        print(f"    ... and {len(perfect)-10} more")

    close = [r for r in r5 if not r['exact_match'] and r['edit_sim'] > 0.7]
    if close:
        print(f"\n  Close predictions (EdSim > 0.7 but not exact): {len(close)}")
        for r in close[:10]:
            print(f"    {r['true_name']:<30s} → {r['predicted_name']:<25s} EdSim={r['edit_sim']:.3f}  F1={r['f1']:.2f}")

    with open('results/predictions_test.json', 'w') as f:
        json.dump(r5, f, indent=2)
    print(f"\n  Saved: results/predictions_test.json")

print()
print("═══════════════════════════════════════════════════════════════════════════════════")
print("  Output files:")
print("    results/ablation_table.json     — all metrics for evaluated variants")
print("    results/gate_analysis.json      — Option B gate values by ext-call group")
print("    results/predictions_test.json   — per-function predictions + all metrics")
print("═══════════════════════════════════════════════════════════════════════════════════")
EVALSCRIPT

echo ""
echo "═══════════════════════════════════════════════"
echo " ✓ EVALUATION COMPLETE"
echo "═══════════════════════════════════════════════"
