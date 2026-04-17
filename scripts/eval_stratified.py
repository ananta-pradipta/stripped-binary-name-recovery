"""
Stratified Evaluation Script for BFNR Binary Function Name Recovery.

Loads the baseline checkpoint and reports metrics stratified by:
  - Easy:         functions with 1+ external calls
  - Hard-Seen:    functions with 0 ext calls, name appears in training set
  - Hard-Unseen:  functions with 0 ext calls, name NEVER seen in training

Also reports:
  - Per-binary breakdown
  - Duplicate function analysis (same name in multiple test binaries)
  - Deduped metrics (each unique (true_name, predicted_name) pair counted once)
  - Demo set (diffutils) analysis if available

Outputs: results/stratified_eval.json
"""
import json
import os
import sys
import gc
from collections import defaultdict, Counter

import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import FunctionDataset
from src.evaluation.metrics import (
    compute_subtoken_f1,
    compute_exact_match,
    compute_char_ngram_similarity,
    compute_edit_distance_similarity,
    compute_all_metrics,
    normalize_name,
    split_name,
)


# ── Configuration ──────────────────────────────────────────────────────────────

CHECKPOINT = 'checkpoints/baseline_v1_val0668.pt'
CONFIG_PATH = 'configs/optimized.yaml'
SPLIT_FILE  = 'data/split_assignments.json'
OUTPUT_PATH = 'results/stratified_eval.json'
DEMO_RESULTS_PATH = 'demo/results/comparison_diff.json'  # from 06_demo.sh

BEAM_WIDTH = 5
BATCH_SIZE = 16

# ── Helpers ────────────────────────────────────────────────────────────────────

def agg(items, key):
    if not items:
        return 0.0
    return sum(x[key] for x in items) / len(items)


def aggregate_metrics(records):
    if not records:
        return {'n': 0, 'f1': 0.0, 'em': 0.0, 'ngsim': 0.0, 'edsim': 0.0,
                'em_count': 0}
    n = len(records)
    return {
        'n': n,
        'f1': agg(records, 'f1'),
        'em': agg(records, 'exact_match'),
        'ngsim': agg(records, 'char_ngram_sim'),
        'edsim': agg(records, 'edit_sim'),
        'em_count': sum(1 for r in records if r['exact_match']),
    }


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
        elif k == 'num_blocks':
            result[k] = [s[k] for s in batch]
        else:
            result[k] = [s[k] for s in batch]
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load config ──
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # ── Load split assignments ──
    with open(SPLIT_FILE) as f:
        splits = json.load(f)
    train_bins = set(splits['train'])
    val_bins   = set(splits.get('val', []))
    test_bins  = set(splits['test'])

    # ── Load checkpoint ──
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    ckpt_cfg    = ckpt['config']
    token_vocab = ckpt['token_vocab']
    ext_vocab   = ckpt.get('ext_vocab', None)

    # ── Load name tokenizer ──
    votes_vocab_path = cfg['data'].get('votes_vocab_path')
    if votes_vocab_path and os.path.exists(votes_vocab_path):
        from src.preprocessing.build_votes import VotesTokenizer
        sp = VotesTokenizer(vocab_path=votes_vocab_path)
        print(f"Votes tokenizer: {sp.get_piece_size()} tokens")
    else:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=cfg['data']['bpe_model_path'])
        print(f"BPE tokenizer: {sp.get_piece_size()} tokens")
    sos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # ── Build dataset ──
    print("\nLoading dataset...")
    dataset = FunctionDataset(
        graphs_dir=cfg['data']['graphs_dir'],
        labels_dir=cfg['data']['labels_dir'],
        external_calls_dir=cfg['data']['external_calls_dir'],
        bpe_model_path=cfg['data']['bpe_model_path'],
        external_vocab_path=cfg['data']['external_vocab_path'],
        max_blocks=cfg['data']['max_blocks_per_function'],
        max_tokens=cfg['data']['max_tokens_per_block'],
        max_name_len=cfg['data']['max_name_length'],
        token_vocab=token_vocab,
        votes_vocab_path=votes_vocab_path,
        ext_vocab_override=ext_vocab,
    )

    # ── Apply splits ──
    train_idx, val_idx, test_idx = dataset.get_splits(
        cfg['data']['train_split'], cfg['data']['val_split'],
        split_file=SPLIT_FILE,
    )

    # ── Collect training names (for Hard-Seen classification) ──
    train_names = set()
    for i in train_idx:
        train_names.add(dataset.samples[i]['name'])
    val_names = set()
    for i in val_idx:
        val_names.add(dataset.samples[i]['name'])
    seen_names = train_names | val_names  # anything the model could have learned from

    print(f"\nTraining names (unique): {len(train_names)}")
    print(f"Val names (unique):      {len(val_names)}")
    print(f"Total seen names:        {len(seen_names)}")

    # ── Free memory: drop non-test graph data ──
    test_idx_set = set(test_idx)
    for i, sample in enumerate(dataset.samples):
        if i not in test_idx_set:
            sample['graph'] = None
    dataset._all_graphs.clear()
    dataset._graphs_by_addr.clear()
    dataset._callers.clear()
    gc.collect()
    print("Freed non-test graph memory")

    # ── Build model ──
    ckpt_cfg['block_encoder']['token_vocab_size'] = len(token_vocab)
    ckpt_cfg['graph_encoder']['input_dim'] = ckpt_cfg['block_encoder']['output_dim']
    if ext_vocab:
        ckpt_cfg['external_encoder']['vocab_size'] = len(ext_vocab)
    ckpt_cfg['decoder']['bpe_vocab_size'] = sp.get_piece_size()

    model = FunctionNamer(ckpt_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")

    # ── DataLoader ──
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"Test set: {len(test_idx)} functions across {len(test_bins)} binaries")

    # ── Run inference ──
    print(f"\nRunning inference (beam_width={BEAM_WIDTH})...")
    callee_enabled = ckpt_cfg.get('callee_encoder', {}).get('enabled', False)
    caller_enabled = ckpt_cfg.get('caller_encoder', {}).get('enabled', False)

    all_records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 20 == 0:
                processed = batch_idx * BATCH_SIZE
                print(f"  [{processed}/{len(test_idx)}]", end='\r', flush=True)

            block_tokens  = batch['block_tokens'].to(device)
            edge_index    = batch['edge_index'].to(device)
            ext_call_ids  = batch['ext_call_ids'].to(device)
            block_features = batch.get('block_features')
            if block_features is not None:
                block_features = block_features.to(device)

            callee_tok = batch.get('callee_tokens')
            if callee_tok is not None and callee_enabled:
                callee_tok = callee_tok.to(device)
            else:
                callee_tok = None

            caller_tok = batch.get('caller_tokens')
            if caller_tok is not None and caller_enabled:
                caller_tok = caller_tok.to(device)
            else:
                caller_tok = None

            beam_results = model.predict(
                block_tokens, edge_index, ext_call_ids,
                sos_id=sos_id, eos_id=eos_id,
                beam_width=BEAM_WIDTH,
                block_features=block_features,
                callee_tokens=callee_tok,
                caller_tokens=caller_tok,
            )

            true_names = batch['name']
            binaries   = batch['binary']
            addresses  = batch['address']
            B = len(true_names)

            for i in range(B):
                if i < len(beam_results):
                    pred_tokens, score = beam_results[i]
                    pred_name = sp.decode(pred_tokens).strip()
                    norm_score = score / max(len(pred_tokens), 1) if pred_tokens else -999.0
                else:
                    pred_name  = ""
                    norm_score = -999.0

                true_name = true_names[i]
                binary    = binaries[i]
                address   = addresses[i]
                num_ext   = int((ext_call_ids[i] > 0).sum().item())

                metrics = compute_all_metrics(pred_name, true_name)

                # Stratification
                if num_ext > 0:
                    stratum = 'easy'
                elif true_name in seen_names:
                    stratum = 'hard_seen'
                else:
                    stratum = 'hard_unseen'

                all_records.append({
                    'binary':        binary,
                    'address':       address,
                    'true_name':     true_name,
                    'predicted_name': pred_name,
                    'score':         round(norm_score, 4),
                    'num_ext_calls': num_ext,
                    'stratum':       stratum,
                    'f1':            metrics['f1'],
                    'exact_match':   metrics['exact_match'],
                    'char_ngram_sim': metrics['char_ngram_sim'],
                    'edit_sim':      metrics['edit_sim'],
                })

    print(f"\nInference complete. {len(all_records)} functions processed.")

    # ── Stratified metrics ──
    strata = defaultdict(list)
    for r in all_records:
        strata[r['stratum']].append(r)

    print("\n" + "=" * 70)
    print("STRATIFIED METRICS (baseline_v1_val0668.pt)")
    print("=" * 70)
    stratum_order = ['easy', 'hard_seen', 'hard_unseen']
    stratum_labels = {
        'easy':         'Easy         (1+ ext calls)',
        'hard_seen':    'Hard-Seen    (0 ext, name in train)',
        'hard_unseen':  'Hard-Unseen  (0 ext, name never seen)',
    }
    strat_aggs = {}
    for stratum in stratum_order:
        records = strata[stratum]
        a = aggregate_metrics(records)
        strat_aggs[stratum] = a
        label = stratum_labels[stratum]
        print(f"\n  {label}")
        print(f"    N={a['n']:>5d}  F1={a['f1']:.4f}  EM={a['em']:.4f} ({a['em_count']}/{a['n']})  "
              f"NgSim={a['ngsim']:.4f}  EdSim={a['edsim']:.4f}")

    overall = aggregate_metrics(all_records)
    print(f"\n  Overall")
    print(f"    N={overall['n']:>5d}  F1={overall['f1']:.4f}  EM={overall['em']:.4f} "
          f"({overall['em_count']}/{overall['n']})  "
          f"NgSim={overall['ngsim']:.4f}  EdSim={overall['edsim']:.4f}")

    # ── Mode collapse check ──
    pred_counter = Counter(r['predicted_name'] for r in all_records)
    top_name, top_count = pred_counter.most_common(1)[0]
    unique_preds = len(pred_counter)
    collapse_pct = 100.0 * top_count / len(all_records)
    print(f"\n  Prediction diversity: {unique_preds} unique / {len(all_records)} total")
    print(f"  Most common: '{top_name}' x{top_count} ({collapse_pct:.1f}%)")
    if collapse_pct > 10:
        print("  *** CRITICAL: MODE COLLAPSE DETECTED ***")
    else:
        print("  Collapse check: PASS (no mode collapse)")

    # ── Per-binary breakdown ──
    print("\n" + "=" * 70)
    print("PER-BINARY BREAKDOWN")
    print("=" * 70)
    by_binary = defaultdict(list)
    for r in all_records:
        by_binary[r['binary']].append(r)

    binary_rows = []
    for binary in sorted(by_binary.keys()):
        records = by_binary[binary]
        a = aggregate_metrics(records)
        pkg = binary.split('_')[0] if '_' in binary else binary
        binary_rows.append({
            'binary': binary,
            'package': pkg,
            **a,
        })

    print(f"\n  {'Binary':<30s} {'N':>5s} {'F1':>7s} {'EM':>7s} {'EM_cnt':>7s} {'Easy%':>7s}")
    print(f"  {'─'*30} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for row in binary_rows:
        records = by_binary[row['binary']]
        easy_n = sum(1 for r in records if r['stratum'] == 'easy')
        easy_pct = 100.0 * easy_n / max(len(records), 1)
        print(f"  {row['binary']:<30s} {row['n']:>5d} {row['f1']:>7.4f} {row['em']:>7.4f} "
              f"{row['em_count']:>7d} {easy_pct:>6.1f}%")

    # Binutils share
    binutils_records = [r for r in all_records if 'binutils' in r['binary']]
    binutils_pct = 100.0 * len(binutils_records) / len(all_records)
    print(f"\n  Binutils share: {len(binutils_records)}/{len(all_records)} = {binutils_pct:.1f}% of test set")

    # ── Duplicate function analysis ──
    print("\n" + "=" * 70)
    print("DUPLICATE FUNCTION ANALYSIS")
    print("=" * 70)

    # Functions with the same name appearing in multiple test binaries
    name_to_binaries = defaultdict(set)
    name_to_records  = defaultdict(list)
    for r in all_records:
        name_to_binaries[r['true_name']].add(r['binary'])
        name_to_records[r['true_name']].append(r)

    duplicated_names = {n: bins for n, bins in name_to_binaries.items() if len(bins) > 1}
    dup_records = [r for r in all_records if r['true_name'] in duplicated_names]
    unique_records = [r for r in all_records if r['true_name'] not in duplicated_names]

    print(f"\n  Names appearing in multiple test binaries: {len(duplicated_names)}")
    print(f"  Records with duplicated names: {len(dup_records)} "
          f"({100.0*len(dup_records)/len(all_records):.1f}% of test)")
    print(f"  Records with unique names:     {len(unique_records)}")

    # Top duplicated names
    print(f"\n  Top 15 most duplicated test function names:")
    print(f"  {'Name':<35s} {'# Binaries':>12s} {'# Records':>10s} {'Avg F1':>8s}")
    print(f"  {'─'*35} {'─'*12} {'─'*10} {'─'*8}")
    for name, bins in sorted(duplicated_names.items(),
                              key=lambda x: (-len(x[1]), x[0]))[:15]:
        recs = name_to_records[name]
        avg_f1 = sum(r['f1'] for r in recs) / len(recs)
        print(f"  {name:<35s} {len(bins):>12d} {len(recs):>10d} {avg_f1:>8.4f}")

    # ── Deduped metrics ──
    # For each unique (true_name, predicted_name) pair, count once
    print("\n" + "=" * 70)
    print("DEDUPED METRICS (each unique (true_name, pred_name) pair counted once)")
    print("=" * 70)

    seen_pairs = {}
    for r in all_records:
        pair = (r['true_name'], r['predicted_name'])
        if pair not in seen_pairs:
            seen_pairs[pair] = r

    deduped_records = list(seen_pairs.values())
    deduped_agg = aggregate_metrics(deduped_records)
    print(f"\n  Deduped records: {len(deduped_records)} (from {len(all_records)} total)")
    print(f"  F1={deduped_agg['f1']:.4f}  EM={deduped_agg['em']:.4f} "
          f"({deduped_agg['em_count']}/{deduped_agg['n']})  "
          f"NgSim={deduped_agg['ngsim']:.4f}  EdSim={deduped_agg['edsim']:.4f}")

    # Deduped per stratum
    deduped_strata = defaultdict(list)
    for r in deduped_records:
        deduped_strata[r['stratum']].append(r)
    for stratum in stratum_order:
        recs = deduped_strata[stratum]
        a = aggregate_metrics(recs)
        label = stratum_labels[stratum]
        print(f"\n  Deduped {label}")
        print(f"    N={a['n']:>5d}  F1={a['f1']:.4f}  EM={a['em']:.4f} ({a['em_count']}/{a['n']})")

    # ── Solvability analysis ──
    print("\n" + "=" * 70)
    print("TEST SET SOLVABILITY ANALYSIS")
    print("=" * 70)

    # Literally unsolvable: 0 ext calls AND name never seen in training
    unsolvable = strata['hard_unseen']
    solvable_hard = strata['hard_seen']
    easy = strata['easy']

    print(f"\n  Easy (1+ ext calls):              {len(easy):>5d}  ({100*len(easy)/len(all_records):.1f}%)")
    print(f"  Hard-Seen (0 ext, name known):    {len(solvable_hard):>5d}  ({100*len(solvable_hard)/len(all_records):.1f}%)")
    print(f"  Hard-Unseen (0 ext, name unseen): {len(unsolvable):>5d}  ({100*len(unsolvable)/len(all_records):.1f}%)")

    # What fraction of Hard-Seen are the model getting right?
    hard_seen_em = sum(1 for r in solvable_hard if r['exact_match'])
    print(f"\n  Hard-Seen EM: {hard_seen_em}/{len(solvable_hard)} = {100*hard_seen_em/max(len(solvable_hard),1):.1f}%")
    print(f"  Hard-Unseen EM: {sum(1 for r in unsolvable if r['exact_match'])}/{len(unsolvable)} "
          f"= {100*sum(1 for r in unsolvable if r['exact_match'])/max(len(unsolvable),1):.1f}%")

    # Binutils hard-unseen
    bins_hard_unseen = [r for r in unsolvable if 'binutils' in r['binary']]
    print(f"\n  Hard-Unseen in binutils: {len(bins_hard_unseen)}/{len(unsolvable)} "
          f"= {100*len(bins_hard_unseen)/max(len(unsolvable),1):.1f}%")

    # Test names that appear in training (upper bound for hard-seen)
    test_unique_names = set(r['true_name'] for r in all_records)
    test_names_in_train = test_unique_names & train_names
    test_names_unseen   = test_unique_names - seen_names
    print(f"\n  Unique test function names: {len(test_unique_names)}")
    print(f"  Names seen in training:     {len(test_names_in_train)} ({100*len(test_names_in_train)/len(test_unique_names):.1f}%)")
    print(f"  Names never seen:           {len(test_names_unseen)} ({100*len(test_names_unseen)/len(test_unique_names):.1f}%)")

    # Estimate theoretical ceiling
    # For Hard-Seen: if model memorized all names, max EM ≈ Hard-Seen count
    # For Hard-Unseen: theoretical max EM ≈ 0 (never seen)
    # For Easy: strong signal from ext calls
    theoretical_max_em = len(easy) + len(solvable_hard)
    theoretical_ceiling = theoretical_max_em / len(all_records)
    print(f"\n  Theoretical EM ceiling (if perfect on solvable): "
          f"{theoretical_max_em}/{len(all_records)} = {theoretical_ceiling:.4f} ({100*theoretical_ceiling:.1f}%)")
    print(f"  Current EM:                                      "
          f"{overall['em_count']}/{len(all_records)} = {overall['em']:.4f} ({100*overall['em']:.1f}%)")
    print(f"  Gap to ceiling: {theoretical_ceiling - overall['em']:.4f} ({100*(theoretical_ceiling - overall['em']):.1f}pp)")

    # ── Failure pattern breakdown ──
    print("\n" + "=" * 70)
    print("FAILURE PATTERN BREAKDOWN (F1 < 0.5)")
    print("=" * 70)
    failures = [r for r in all_records if r['f1'] < 0.5]
    fail_by_stratum = Counter(r['stratum'] for r in failures)
    print(f"\n  Total failures: {len(failures)}/{len(all_records)} "
          f"({100*len(failures)/len(all_records):.1f}%)")
    for stratum in stratum_order:
        n_fail = fail_by_stratum.get(stratum, 0)
        n_total = len(strata[stratum])
        print(f"  {stratum_labels[stratum]}: {n_fail}/{n_total} fail "
              f"({100*n_fail/max(n_total,1):.1f}%)")

    # Name-length breakdown for failures
    for stratum in stratum_order:
        recs = strata[stratum]
        f = [r for r in recs if r['f1'] < 0.5]
        if not f:
            continue
        short  = sum(1 for r in f if len(split_name(r['true_name'])) <= 1)
        long_  = sum(1 for r in f if len(split_name(r['true_name'])) > 5)
        mid    = len(f) - short - long_
        print(f"\n  {stratum_labels[stratum]} failures: {len(f)}")
        print(f"    Short name (<=1 token): {short} ({100*short/max(len(f),1):.0f}%)")
        print(f"    Mid   name (2-5 tokens): {mid} ({100*mid/max(len(f),1):.0f}%)")
        print(f"    Long  name (>5 tokens): {long_} ({100*long_/max(len(f),1):.0f}%)")

    # ── Demo set analysis ──
    demo_data = None
    if os.path.exists(DEMO_RESULTS_PATH):
        with open(DEMO_RESULTS_PATH) as f:
            demo_data = json.load(f)
        print("\n" + "=" * 70)
        print("DEMO SET (diffutils — completely unseen package)")
        print("=" * 70)
        # demo_data may be a list of functions or a dict with 'functions' key
        if isinstance(demo_data, list):
            funcs = demo_data
        else:
            funcs = demo_data.get('functions', [])
        # Filter to functions that have a true_name (i.e. have ground truth)
        funcs_with_gt = [fn for fn in funcs if fn.get('true_name') or fn.get('predicted')]
        # predictions_diff.json from 06_demo.sh has 'predicted', 'true_name', 'f1', 'em', 'ext' keys
        # comparison_diff.json may have different format — check both
        if funcs_with_gt:
            demo_em  = sum(1 for fn in funcs_with_gt if fn.get('em', False))
            demo_f1  = sum(fn.get('f1', 0) for fn in funcs_with_gt) / len(funcs_with_gt)
            demo_total = len(funcs_with_gt)
            print(f"\n  Functions with GT: {demo_total}")
            print(f"  EM: {demo_em}/{demo_total} = {100*demo_em/demo_total:.1f}%")
            print(f"  Avg F1: {demo_f1:.4f}")
            # By ext calls
            demo_easy = [fn for fn in funcs_with_gt if fn.get('ext', 0) > 0]
            demo_hard = [fn for fn in funcs_with_gt if fn.get('ext', 0) == 0]
            if demo_easy:
                easy_em = sum(1 for fn in demo_easy if fn.get('em'))
                print(f"  Easy (1+ ext): {len(demo_easy)}, EM={easy_em}/{len(demo_easy)} "
                      f"= {100*easy_em/len(demo_easy):.1f}%")
            if demo_hard:
                hard_em = sum(1 for fn in demo_hard if fn.get('em'))
                print(f"  Hard (0 ext):  {len(demo_hard)}, EM={hard_em}/{len(demo_hard)} "
                      f"= {100*hard_em/len(demo_hard):.1f}%")

    # ── Save results ──
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    out = {
        'checkpoint': CHECKPOINT,
        'beam_width': BEAM_WIDTH,
        'overall': overall,
        'mode_collapse': {
            'unique_predictions': unique_preds,
            'total_predictions': len(all_records),
            'top_prediction': top_name,
            'top_count': top_count,
            'top_pct': round(collapse_pct, 2),
            'collapsed': collapse_pct > 10,
        },
        'stratified': {
            'easy':        strat_aggs['easy'],
            'hard_seen':   strat_aggs['hard_seen'],
            'hard_unseen': strat_aggs['hard_unseen'],
        },
        'deduped': {
            'overall':     deduped_agg,
            'easy':        aggregate_metrics(deduped_strata['easy']),
            'hard_seen':   aggregate_metrics(deduped_strata['hard_seen']),
            'hard_unseen': aggregate_metrics(deduped_strata['hard_unseen']),
        },
        'per_binary': {
            row['binary']: {
                'package': row['package'],
                'n':        row['n'],
                'f1':       row['f1'],
                'em':       row['em'],
                'em_count': row['em_count'],
            }
            for row in binary_rows
        },
        'duplicate_analysis': {
            'duplicated_name_count': len(duplicated_names),
            'duplicate_record_count': len(dup_records),
            'duplicate_record_pct': round(100.0 * len(dup_records) / len(all_records), 2),
        },
        'solvability': {
            'unique_test_names':       len(test_unique_names),
            'names_in_training':       len(test_names_in_train),
            'names_never_seen':        len(test_names_unseen),
            'theoretical_max_em':      theoretical_max_em,
            'theoretical_ceiling':     round(theoretical_ceiling, 4),
            'actual_em':               overall['em_count'],
            'actual_em_rate':          round(overall['em'], 4),
            'gap_to_ceiling':          round(theoretical_ceiling - overall['em'], 4),
        },
        'binutils_share_pct':   round(binutils_pct, 2),
        'train_unique_names':   len(train_names),
    }

    os.makedirs('results', exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {OUTPUT_PATH}")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Checkpoint:      {CHECKPOINT}")
    print(f"  Test functions:  {len(all_records)}")
    print(f"  Test binaries:   {len(test_bins)}")
    print(f"  Binutils share:  {binutils_pct:.1f}%")
    print()
    print(f"  Overall   F1={overall['f1']:.4f}  EM={overall['em']:.4f}  "
          f"({overall['em_count']}/{overall['n']})")
    print(f"  Easy      F1={strat_aggs['easy']['f1']:.4f}  "
          f"EM={strat_aggs['easy']['em']:.4f}  ({strat_aggs['easy']['em_count']}/{strat_aggs['easy']['n']})")
    print(f"  Hard-Seen F1={strat_aggs['hard_seen']['f1']:.4f}  "
          f"EM={strat_aggs['hard_seen']['em']:.4f}  ({strat_aggs['hard_seen']['em_count']}/{strat_aggs['hard_seen']['n']})")
    print(f"  Hard-Unse F1={strat_aggs['hard_unseen']['f1']:.4f}  "
          f"EM={strat_aggs['hard_unseen']['em']:.4f}  ({strat_aggs['hard_unseen']['em_count']}/{strat_aggs['hard_unseen']['n']})")
    print()
    print(f"  Deduped   F1={deduped_agg['f1']:.4f}  EM={deduped_agg['em']:.4f}  "
          f"({deduped_agg['em_count']}/{deduped_agg['n']})")
    print()
    print(f"  Mode collapse: {'YES (collapsed)' if collapse_pct > 10 else 'NO'} "
          f"(top pred '{top_name}' = {collapse_pct:.1f}%)")
    print()
    print(f"  Theoretical EM ceiling: {100*theoretical_ceiling:.1f}%  |  Current: {100*overall['em']:.1f}%")
    print()
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
