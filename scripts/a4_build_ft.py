#!/usr/bin/env python3
"""A4 prep: build (masked Ghidra decompiled code -> name) fine-tune set for the
generation head from the v2 protocol train/val tiers. Masking identical to
symgen_v2/build_symgen_input.py (ghidra_name -> [MASK], first occurrence).
Output: results/a4_ft/{train,val}.jsonl + stats.json. Nothing is trained here."""
import json, os, sys
from collections import Counter, defaultdict
WS = '$WORKSPACE/dh2'
PROTO = f'{WS}/results/baseline_protocol_v2'
OUT = f'{WS}/results/a4_ft'
os.makedirs(OUT, exist_ok=True)

cache = {}
def decomp(b):
    if b not in cache:
        p = f'{WS}/symgen_v2/decomp/{b}.json'
        cache[b] = json.load(open(p)) if os.path.exists(p) else None
    return cache[b]

stats = {}
for tier in ['train', 'val']:
    n = kept = 0
    miss = Counter(); lens = []; pkgs = set(); bins_nodecomp = set()
    with open(f'{OUT}/{tier}.jsonl', 'w') as fh:
        for line in open(f'{PROTO}/{tier}.jsonl'):
            r = json.loads(line); n += 1
            d = decomp(r['binary'])
            if d is None:
                miss['no_decomp_file'] += 1; bins_nodecomp.add(r['binary']); continue
            e = d.get(r['entry_addr'])
            if not e or 'code' not in e:
                miss['addr_missing_or_failed'] += 1; continue
            code = e['code'].replace(e['ghidra_name'], '[MASK]', 1)
            if '[MASK]' not in code:
                miss['mask_not_applied'] += 1; continue
            row = {'key': f"{r['binary']}_{r['entry_addr']}", 'binary': r['binary'], 'package': r['package'],
                   'addr': r['entry_addr'], 'name': r['name'], 'code': code}
            if 'regime' in r: row['regime'] = r['regime']
            if 'name_seen_in_train' in r: row['name_seen'] = r['name_seen_in_train']
            fh.write(json.dumps(row) + '\n'); kept += 1
            lens.append(len(code)); pkgs.add(r['package'])
        cache.clear() if tier == 'train' else None
    lens.sort()
    def pct(p): return lens[min(len(lens)-1, int(p*len(lens)))] if lens else 0
    approx_tok = [l/3.5 for l in lens]  # decompiled C ~3.5 chars/token for code tokenizers
    st = {'rows_in': n, 'rows_out': kept, 'skipped': dict(miss), 'bins_without_decomp': sorted(bins_nodecomp),
          'packages': len(pkgs), 'code_chars_p50': pct(.5), 'code_chars_p90': pct(.9), 'code_chars_p99': pct(.99),
          'frac_le_512tok': sum(t <= 512 for t in approx_tok)/max(1,len(lens)),
          'frac_le_1024tok': sum(t <= 1024 for t in approx_tok)/max(1,len(lens)),
          'frac_le_2048tok': sum(t <= 2048 for t in approx_tok)/max(1,len(lens))}
    stats[tier] = st
    print(f'EFFECT: a4_ft {tier}: {kept}/{n} rows, skipped={dict(miss)}, pkgs={len(pkgs)}, '
          f'p50/p90/p99 chars={st["code_chars_p50"]}/{st["code_chars_p90"]}/{st["code_chars_p99"]}, '
          f'<=1024tok={st["frac_le_1024tok"]:.3f} <=2048tok={st["frac_le_2048tok"]:.3f}', flush=True)
json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
