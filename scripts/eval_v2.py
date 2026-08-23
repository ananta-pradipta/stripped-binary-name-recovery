#!/usr/bin/env python3
"""Dataset-v2 evaluation (split policy v3): one held-out TEST set, reported as
  micro F1/EM (n-weighted), per-package macro F1/EM, regime sub-rows (FT / NCT),
  and strata: scored / seen-body (dropped) / symbol-visible (dropped); inside scored:
  seen-name (name occurs in train) vs novel-name.

Usage:
  python3 scripts/eval_v2.py CKPT [--config configs/dualhead_v2_large.yaml] [--tiers test val]
                              [--decode greedy|beam] [--amp] [--save results/x.json]
Greedy is batched and matches the training-time validation metric; beam (k=5) is per-sample
(slower, ~+3-5% F1). Vocabularies come from the checkpoint (never rebuilt here).
"""
import argparse, json, os, sys, hashlib
from collections import defaultdict
import torch, yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.dataset_v2 import FunctionDatasetV2
from src.preprocessing.build_votes import VotesTokenizer
from src.training.collate import collate_fn
from src.evaluation.metrics import (compute_subtoken_f1, compute_char_ngram_similarity,
                                    compute_edit_distance_similarity)


def _to(batch, key, device):
    v = batch.get(key)
    return v.to(device) if v is not None else None


def decode_ids(ids, sos_id, eos_id):
    out = []
    for t in ids:
        t = int(t)
        if t == eos_id:
            break
        if t not in (sos_id, 0):
            out.append(t)
    return out


def run(model, dataset, idx, device, sp, decode, use_amp, batch_size, num_workers=0):
    sos_id, eos_id = sp.bos_id(), sp.eos_id()
    model.eval()
    preds = []
    loader = DataLoader(Subset(dataset, idx), batch_size=1 if decode == 'beam' else batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'eval[{decode}]', leave=False):
            bt, ei, ec = (batch['block_tokens'].to(device), batch['edge_index'].to(device),
                          batch['ext_call_ids'].to(device))
            kw = dict(block_features=_to(batch, 'block_features', device),
                      callee_tokens=_to(batch, 'callee_tokens', device),
                      caller_tokens=_to(batch, 'caller_tokens', device),
                      string_tokens=_to(batch, 'string_tokens', device),
                      binary_ext_ids=_to(batch, 'binary_ext_ids', device))
            if decode == 'beam':
                res = model.predict(bt, ei, ec, sos_id, eos_id, beam_width=5, **kw)
                preds.append(sp.decode(res[0][0]).strip())
            else:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits, *_ = model(bt, ei, ec, batch['decoder_input'].to(device),
                                       teacher_forcing_ratio=0.0, **kw)
                for row in logits.argmax(-1):
                    preds.append(sp.decode(decode_ids(row.tolist(), sos_id, eos_id)).strip())
    return preds


def agg(rows):
    n = len(rows)
    if n == 0:
        return {'n': 0}
    return {'n': n,
            'f1': sum(r['f1'] for r in rows) / n, 'em': sum(r['em'] for r in rows) / n,
            'ngsim': sum(r['ngsim'] for r in rows) / n, 'edsim': sum(r['edsim'] for r in rows) / n}


def macro(rows, key='pkg'):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    per = {k: agg(v) for k, v in sorted(g.items())}
    m = len(per)
    return {'n_groups': m, 'f1': sum(v['f1'] for v in per.values()) / m,
            'em': sum(v['em'] for v in per.values()) / m}, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--config', default=None, help='default: config stored in the checkpoint')
    ap.add_argument('--tiers', nargs='+', default=['test'])
    ap.add_argument('--decode', choices=['greedy', 'beam'], default='greedy')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--save', default=None)
    ap.add_argument('--dump-preds', default=None, help='TSV of binary, bap_name, true, pred, regime, stratum')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = yaml.safe_load(open(args.config)) if args.config else ckpt['config']
    assert cfg['data'].get('format') == 'v2', 'eval_v2.py is for dataset v2 checkpoints'
    split_file = cfg['data']['split_file']
    split = json.load(open(split_file))
    split_sha = hashlib.sha256(open(split_file, 'rb').read()).hexdigest()
    if ckpt.get('split_sha256') and ckpt['split_sha256'] != split_sha:
        print(f"WARNING: checkpoint split sha {ckpt['split_sha256'][:12]} != {split_file} sha {split_sha[:12]}")
    regime = split['meta'].get('test_regime', {})
    val_regime = {p: d['regime'] for p, d in split['meta']['tiers'].get('val_xproj', {}).items()}

    sp = VotesTokenizer(vocab_path=cfg['data']['votes_vocab_path'])
    dataset = FunctionDatasetV2(
        match_index_path=cfg['data']['match_index_path'], string_refs_dir=cfg['data'].get('string_refs_dir'),
        votes_vocab_path=cfg['data']['votes_vocab_path'],
        max_blocks=cfg['data']['max_blocks_per_function'], max_tokens=cfg['data']['max_tokens_per_block'],
        max_name_len=cfg['data']['max_name_length'], min_tokens=cfg['data'].get('min_tokens', 1),
        corpora=set(cfg['data']['corpora']) if cfg['data'].get('corpora') else None,
        token_vocab=ckpt['token_vocab'], ext_vocab=ckpt['ext_vocab'], vocab_binaries=set(split['train']),
        cache_path=cfg['data'].get('cache_path'))
    train_idx, val_idx, test_idx = dataset.get_splits(cfg['data']['train_split'], cfg['data']['val_split'],
                                                      split_file=split_file)
    if not val_idx and dataset.val_xproj_idx:
        val_idx = list(dataset.val_xproj_idx)
    raw = {'val': list(val_idx), 'test': list(test_idx)}
    train_idx, val_idx, test_idx = dataset.apply_split_policy(train_idx, val_idx, test_idx)
    scored = {'val': val_idx, 'test': test_idx}
    train_names = {dataset.samples[i]['name'] for i in train_idx}

    st = ckpt['model_state_dict']
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    if 'ext_encoder.embedding.weight' in st:
        cfg['external_encoder']['vocab_size'] = st['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = st['decoder.embedding.weight'].shape[0]
    model = FunctionNamer(cfg).to(device)
    model.load_state_dict(st)
    print(f"ckpt {args.checkpoint}: epoch {ckpt.get('epoch')} val_f1 {ckpt.get('val_f1')} "
          f"params {sum(p.numel() for p in model.parameters()):,}")

    report = {'checkpoint': args.checkpoint, 'decode': args.decode, 'split_sha256': split_sha,
              'policy': dataset.policy_stats, 'tiers': {}}
    dump = []
    for tier in args.tiers:
        reg = regime if tier == 'test' else val_regime
        idx = scored[tier]
        preds = run(model, dataset, idx, device, sp, args.decode, args.amp, args.batch_size, args.num_workers)
        rows = []
        for i, pred in zip(idx, preds):
            s = dataset.samples[i]
            true = s['name']
            pkg = s['binary'].split('_')[0]
            rows.append({'pkg': pkg, 'regime': reg.get(pkg, '?'), 'binary': s['binary'], 'bap_name': s['bap_name'],
                         'true': true, 'pred': pred, 'f1': compute_subtoken_f1(pred, true),
                         'em': 1.0 if pred == true else 0.0,
                         'ngsim': compute_char_ngram_similarity(pred, true),
                         'edsim': compute_edit_distance_similarity(pred, true),
                         'name_seen': true in train_names})
        mac, per_pkg = macro(rows)
        out = {'raw_fns': len(raw[tier]), 'scored': agg(rows), 'macro_pkg': mac, 'per_pkg': per_pkg,
               'by_regime': {}, 'by_name_stratum': {}}
        for r in ('FT', 'NCT'):
            sub = [x for x in rows if x['regime'] == r]
            m, _ = macro(sub) if sub else ({'n_groups': 0}, {})
            out['by_regime'][r] = {'micro': agg(sub), 'macro_pkg': m}
        for name, cond in (('seen_name', True), ('novel_name', False)):
            sub = [x for x in rows if x['name_seen'] == cond]
            out['by_name_stratum'][name] = agg(sub)
        dropped = dataset.policy_dropped[tier]
        out['dropped'] = {k: len(v) for k, v in dropped.items()}
        report['tiers'][tier] = out
        dump += [(tier, x) for x in rows]
        uniq = len({x['pred'] for x in rows}) / max(1, len(rows))
        print(f"\n=== {tier} ({args.decode}) === raw {len(raw[tier])} scored {len(rows)} "
              f"dropped {out['dropped']} pred-uniqueness {uniq:.3f}")
        print(f"  micro  F1 {out['scored']['f1']:.4f} EM {out['scored']['em']:.4f} "
              f"NgSim {out['scored']['ngsim']:.3f} EdSim {out['scored']['edsim']:.3f}")
        print(f"  macro  F1 {mac['f1']:.4f} EM {mac['em']:.4f} over {mac['n_groups']} pkgs")
        for r, d in out['by_regime'].items():
            if d['micro']['n']:
                print(f"  {r:<4} micro F1 {d['micro']['f1']:.4f} EM {d['micro']['em']:.4f} n={d['micro']['n']} | "
                      f"macro F1 {d['macro_pkg']['f1']:.4f} ({d['macro_pkg']['n_groups']} pkgs)")
        for k, d in out['by_name_stratum'].items():
            if d['n']:
                print(f"  {k:<10} F1 {d['f1']:.4f} EM {d['em']:.4f} n={d['n']}")
        print(f"  {'package':<12} {'reg':<4} {'n':>7} {'F1':>7} {'EM':>7}")
        for p, d in per_pkg.items():
            print(f"  {p:<12} {reg.get(p, '?'):<4} {d['n']:>7} {d['f1']:>7.4f} {d['em']:>7.4f}")
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        json.dump(report, open(args.save, 'w'), indent=1)
        print(f"saved {args.save}")
    if args.dump_preds:
        with open(args.dump_preds, 'w') as fh:
            fh.write('tier\tbinary\tbap_name\ttrue\tpred\tregime\tname_seen\tf1\n')
            for tier, x in dump:
                fh.write(f"{tier}\t{x['binary']}\t{x['bap_name']}\t{x['true']}\t{x['pred']}\t{x['regime']}\t{int(x['name_seen'])}\t{x['f1']:.3f}\n")
    print('EFFECT: eval_v2 done', {t: (d['scored']['n'], round(d['scored']['f1'], 4)) for t, d in report['tiers'].items()})


if __name__ == '__main__':
    main()
