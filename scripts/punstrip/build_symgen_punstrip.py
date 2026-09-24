#!/usr/bin/env python3
"""Build SymGen (CodeLlama-34B + LoRA) fine-tune / inference inputs from the Punstrip corpus.

Rows come from punstrip/data/<split>.jsonl (the exact row set every system is scored on); the decompiled text comes
from punstrip/decomp/json/<binary>.json (raw Ghidra export, NO module-context digest — SymGen gets plain decompilation,
as in its own pipeline). The Ghidra placeholder name is masked at EVERY occurrence (same convention as our inputs;
first-only masking leaked recursive self-calls). Output = raw ground-truth name (SymGen trains on raw names).

  --split train  -> symgen/train_input.json (+ train_metadata.json)           [fine-tune data]
  --split test   -> symgen/test_shards/shard_<k>.json + meta_<k>.json (8,000 rows each)   [inference]
Audits printed as EFFECT lines: rows, skipped, mask applied on 100%, GT-name-in-input rate (must match the row flag).
"""
import argparse, json, os, re
from collections import Counter

P = '/project/hz79/_shared/cs785/punstrip'
INSTR = ('Suppose you are an expert in software reverse engineering. Here is a piece of decompiled code, '
         'you should infer code semantics and tell me the original function name from the contents of the '
         'function to replace [MASK]. And you need to tell me your answer. Now the decompiled codes are as follows:')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--shard', type=int, default=8000, help='rows per inference shard (0 = single file)')
    a = ap.parse_args()
    out = f'{P}/symgen'; os.makedirs(out, exist_ok=True)

    cache = {}
    def decomp(binary):
        if binary not in cache:
            p = f'{P}/decomp/json/{binary}.json'
            cache[binary] = json.load(open(p)) if os.path.exists(p) else {}
            if len(cache) > 400: cache.pop(next(iter(cache)))   # bounded memory: rows are grouped by binary
        return cache[binary]

    inputs, meta = [], []
    skipped = Counter(); name_in = 0; flag_in = 0
    for line in open(f'{P}/data/{a.split}.jsonl'):
        r = json.loads(line)
        d = decomp(r['binary']).get(r['addr'])
        if not d or 'code' not in d or not d.get('ghidra_name'):
            skipped['no_decomp'] += 1; continue
        code = re.sub(r'\b' + re.escape(d['ghidra_name']) + r'\b', '[MASK]', d['code'])
        if '[MASK]' not in code:
            skipped['mask_not_applied'] += 1; continue
        if re.search(r'\b' + re.escape(d['ghidra_name']) + r'\b', code):
            skipped['ghidra_name_survives'] += 1; continue   # cannot happen after the sub; substring-in-longer-identifier is fine
        gt_visible = re.search(r'\b' + re.escape(r['name']) + r'\b', code) is not None
        name_in += gt_visible; flag_in += bool(r.get('name_in_input'))
        inputs.append({'instruction': INSTR, 'input': '\n\n' + code, 'output': r['name']})
        meta.append({'key': r['key'], 'binary': r['binary'], 'binpath': r['binpath'], 'package': r['package'],
                     'addr': r['addr'], 'gt_name': r['name'], 'name_seen': r.get('name_seen'),
                     'dynsym_visible': r.get('dynsym_visible'), 'gt_in_input': gt_visible})
    n = len(inputs)
    print(f'EFFECT: symgen {a.split}: {n} inputs, skipped {dict(skipped)}, mask applied 100% of kept, '
          f'GT-name-in-input {name_in/n:.1%} (row flag {flag_in/n:.1%}), pkgs {len({m["package"] for m in meta})}', flush=True)
    if a.shard and a.split != 'train':
        sd = f'{out}/{a.split}_shards'; os.makedirs(sd, exist_ok=True)
        for k in range(0, n, a.shard):
            json.dump(inputs[k:k + a.shard], open(f'{sd}/shard_{k // a.shard}.json', 'w'))
            json.dump(meta[k:k + a.shard], open(f'{sd}/meta_{k // a.shard}.json', 'w'))
        print(f'EFFECT: {a.split} shards: {(n + a.shard - 1) // a.shard} x <= {a.shard} rows in {sd}', flush=True)
    else:
        json.dump(inputs, open(f'{out}/{a.split}_input.json', 'w'))
        json.dump(meta, open(f'{out}/{a.split}_metadata.json', 'w'))
        print(f'EFFECT: wrote {out}/{a.split}_input.json ({os.path.getsize(f"{out}/{a.split}_input.json")/1e6:.0f} MB)', flush=True)


if __name__ == '__main__':
    main()
