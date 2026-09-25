#!/usr/bin/env python3
"""Extract the per-binary function layout used by the context digest (Wulver side, no GPU).
For every test-tier binary: the Ghidra function keys in address order, each key's Ghidra address,
its direct callee keys (resolved through ghidra_name addresses), the protocol test rows (scored
functions) and the label file's debug_elf path / elf_type. Output is small and is consumed by
tu_locality.py wherever the debug ELFs live (local machine for data/raw + cross_project, Wulver for
clang_train / ftdomains / raw_wulver)."""
import json, os, re, sys
from collections import defaultdict
WS = '/project/hz79/_shared/cs785/dh2'
RL = '/project/hz79/_shared/cs785/relift_ws'
REF_RE = re.compile(r'\b(?:FUN_|sub_)([0-9a-fA-F]{4,})')
GNAME_RE = re.compile(r'([0-9a-fA-F]+)$')
out_path = sys.argv[1] if len(sys.argv) > 1 else f'{WS}/results/ctx_layout/test_layout.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
split = json.load(open(f'{RL}/data/split_v2.json'))
proto = defaultdict(list)
for line in open(f'{WS}/results/baseline_protocol_v2/test.jsonl'):
    r = json.loads(line); proto[r['binary']].append({'addr': r['entry_addr'], 'name': r['name'], 'regime': r.get('regime'),
                                                     'seen': r.get('name_seen_in_train')})
out = {}; n_fn = 0; n_rows = 0; n_missing = 0
for b in split['test']:
    p = f'{WS}/symgen_v2/decomp/{b}.json'; lp = f'{RL}/data/labels_v2/{b}.json'
    if not os.path.exists(p) or not os.path.exists(lp): n_missing += 1; continue
    dec = json.load(open(p)); lab = json.load(open(lp))
    addrs = sorted(dec, key=lambda a: int(a, 16)); idx = {a: i for i, a in enumerate(addrs)}
    g2i = {}
    for a in addrs:
        m = GNAME_RE.search(dec[a].get('ghidra_name', '') or '')
        if m: g2i[int(m.group(1), 16)] = idx[a]
    callees = []
    for a in addrs:
        cs = set()
        for h in REF_RE.findall(dec[a].get('code', '')):
            j = g2i.get(int(h, 16))
            if j is not None and j != idx[a]: cs.add(j)
        callees.append(sorted(cs))
    out[b] = {'package': b.split('_')[0], 'debug_elf': lab.get('debug_elf'), 'elf_type': lab.get('elf_type'),
              'addrs': addrs, 'callees': callees, 'rows': proto.get(b, [])}
    n_fn += len(addrs); n_rows += len(proto.get(b, []))
json.dump(out, open(out_path, 'w'))
print(f'EFFECT: layout for {len(out)} test binaries ({n_missing} missing decomp/label), {n_fn} functions, {n_rows} protocol rows -> {out_path}', flush=True)
