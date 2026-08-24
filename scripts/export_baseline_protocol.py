#!/usr/bin/env python3
"""P4-ExternalBaselines: export the dataset-v2 protocol for external systems (SymGen, BLens).

Produces, from split_v2.json + the same record-level policy the P2 evaluation used:
  results/baseline_protocol_v2/{train,val,test}.jsonl
      one line per function: binary, stripped ELF path, entry address (hex),
      function name (label), package, regime (val/test), strata flags
  results/baseline_protocol_v2/PROTOCOL.md   (counts + rules, for the paper appendix)

External baselines must: train on train.jsonl functions only; report on the scored
val/test functions; the seen/novel-name stratum and FT/NCT regime come with each row.
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.dataset_v2 import FunctionDatasetV2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/dualhead_v2_large.yaml')
    ap.add_argument('--stripped-dir', default='data/stripped_v2')
    ap.add_argument('--out', default='results/baseline_protocol_v2')
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))['data']
    split = json.load(open(cfg['split_file']))
    regime = dict(split['meta'].get('test_regime', {}))
    for p, d in split['meta']['tiers'].get('val_xproj', {}).items():
        regime.setdefault(p, d['regime'])

    ds = FunctionDatasetV2(
        match_index_path=cfg['match_index_path'], string_refs_dir=None,
        votes_vocab_path=cfg['votes_vocab_path'],
        max_blocks=cfg['max_blocks_per_function'], max_tokens=cfg['max_tokens_per_block'],
        max_name_len=cfg['max_name_length'], min_tokens=cfg.get('min_tokens', 1),
        cache_path=cfg.get('cache_path'))
    train_idx, val_idx, test_idx = ds.get_splits(cfg['train_split'], cfg['val_split'],
                                                 split_file=cfg['split_file'])
    if not val_idx and ds.val_xproj_idx:
        val_idx = list(ds.val_xproj_idx)
    raw = {'val': list(val_idx), 'test': list(test_idx)}
    train_idx, val_idx, test_idx = ds.apply_split_policy(train_idx, val_idx, test_idx)
    train_names = {ds.samples[i]['name'] for i in train_idx}

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for tier, idx in (('train', train_idx), ('val', val_idx), ('test', test_idx)):
        path = os.path.join(args.out, tier + '.jsonl')
        n = 0
        with open(path, 'w') as fh:
            for i in idx:
                s = ds.samples[i]
                pkg = s['binary'].split('_')[0]
                row = {'binary': s['binary'],
                       'stripped_elf': os.path.join(args.stripped_dir, s['binary'] + '_stripped'),
                       'entry_addr': s['address'], 'name': s['name'], 'package': pkg}
                if tier != 'train':
                    row['regime'] = regime.get(pkg, '?')
                    row['name_seen_in_train'] = s['name'] in train_names
                fh.write(json.dumps(row) + '\n')
                n += 1
        counts[tier] = n
        print(f'{tier}: {n} -> {path}')
    pol = ds.policy_stats
    md = ['# Baseline protocol v2 (exported from split_v2.json sha ' +
          str(getattr(ds, 'split_sha256', ''))[:12] + ')', '',
          'Rules: package-disjoint tiers; train deduplicated to one sample per (token-hash, name); ',
          'val/test exclude functions whose token-hash occurs in train and exported (dynsym) symbols.', '',
          f"Counts: {json.dumps(counts)}", f"Policy: {json.dumps(pol)}", '',
          'External baselines MUST train only on train.jsonl and report on val/test.jsonl rows, ',
          'with regime (FT/NCT) and name_seen_in_train sub-rows.']
    open(os.path.join(args.out, 'PROTOCOL.md'), 'w').write('\n'.join(md) + '\n')
    print('EFFECT: export done', counts)


if __name__ == '__main__':
    main()
