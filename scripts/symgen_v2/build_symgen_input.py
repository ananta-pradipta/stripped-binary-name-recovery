#!/usr/bin/env python3
"""Assemble SymGen inference inputs from the v2 protocol + Ghidra decompilations.

--mode interim_c : stratified sample of FT test packages that did not exist in the
                   April corpus the existing LoRA was fine-tuned on (clean w.r.t. it).
--mode full      : every scored val/test function with a successful decompilation.
"""
import argparse, json, os, random
from collections import defaultdict

WS = '/project/hz79/_shared/cs785/dh2/symgen_v2'
PROTO = '/project/hz79/_shared/cs785/dh2/results/baseline_protocol_v2'
INSTR = ('Suppose you are an expert in software reverse engineering. Here is a piece of decompiled code, '
         'you should infer code semantics and tell me the original function name from the contents of the '
         'function to replace [MASK]. And you need to tell me your answer. Now the decompiled codes are as follows:')
# packages that did not exist in the April-2026 corpus (existing LoRA is clean on these)
INTERIM_C_PKGS = {'atop', 'bdb', 'byacc', 'entr', 'file', 'gdbm', 'icu', 'lsof', 'mawk', 'mksh', 'mutt',
                  'procps', 'pv', 'sbase', 'sysstat', 'tcsh', 'tdb',
                  'cvs', 'lighttpd', 'tinycc', 'jansson', 'lmdb', 'mbedtls', 'libsodium'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['interim_c', 'full'], default='interim_c')
    ap.add_argument('--cap-per-pkg', type=int, default=400)
    ap.add_argument('--tiers', nargs='+', default=['test'])
    args = ap.parse_args()
    rng = random.Random(42)

    decomp_cache = {}
    def decomp(binary):
        if binary not in decomp_cache:
            p = f'{WS}/decomp/{binary}.json'
            decomp_cache[binary] = json.load(open(p)) if os.path.exists(p) else {}
        return decomp_cache[binary]

    rows = []
    for tier in args.tiers:
        for line in open(f'{PROTO}/{tier}.jsonl'):
            r = json.loads(line)
            r['tier'] = tier
            rows.append(r)
    if args.mode == 'interim_c':
        rows = [r for r in rows if r['package'] in INTERIM_C_PKGS and r.get('regime') == 'FT']
        by_pkg = defaultdict(list)
        for r in rows:
            by_pkg[r['package']].append(r)
        rows = []
        for p, rs in sorted(by_pkg.items()):
            rng.shuffle(rs)
            rows += rs[:args.cap_per_pkg]
    inputs, meta, miss = [], [], 0
    for r in rows:
        d = decomp(r['binary']).get(r['entry_addr'])
        if not d or 'code' not in d:
            miss += 1
            continue
        code = d['code'].replace(d['ghidra_name'], '[MASK]', 1)
        if '[MASK]' not in code:
            miss += 1
            continue
        inputs.append({'instruction': INSTR, 'input': '\n\n' + code})
        meta.append({'key': f"{r['binary']}_{r['entry_addr']}", 'package': r['package'],
                     'binary': r['binary'], 'gt_name': r['name'], 'addr': r['entry_addr'],
                     'tier': r['tier'], 'regime': r.get('regime'), 'name_seen': r.get('name_seen_in_train')})
    tag = args.mode
    json.dump(inputs, open(f'{WS}/{tag}_input.json', 'w'))
    json.dump(meta, open(f'{WS}/{tag}_metadata.json', 'w'))
    pkgs = sorted({m['package'] for m in meta})
    print(f'EFFECT: {tag}: {len(inputs)} inputs ({miss} skipped, {len(pkgs)} pkgs: {",".join(pkgs)})')


if __name__ == '__main__':
    main()
