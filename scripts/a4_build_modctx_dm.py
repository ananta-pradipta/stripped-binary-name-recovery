#!/usr/bin/env python3
"""Idea #2 — demangled-target variant of the modctx A4 data (targets only; exact scorer canon).
Diagnosis (novel_head_analysis.json): A4 emits mangled-name fragments (epns/epkns/7board) on C++
test packages (icu 48K rows, F1 0.041). Investigation 2026-09-01: the decomp INPUT text of the icu
tools has NO mangled or demangled C++ identifiers at all (static + stripped, everything is FUN_xxx),
so an input-side rewrite is a no-op — the mangled junk is imitation learned from the ~4K TRAIN rows
whose GT targets are raw _Z symbols. This builder canonizes targets with EXACTLY the scorers' canon
(union_eval/score_symgen_full: c++filt demangle, strip args/templates, last-2 :: qualifiers joined
by _). Do NOT add extra character rewriting: a stricter canon (e.g. '.'->'_') changed 137K test
targets incl. foo.part.0-style GCC suffixes and diverges from what is scored (EM would be lost).
Reads results/a4_modctx/{tier}.jsonl (digest inputs unchanged), writes results/a4_modctx_dm/;
also writes the seen-flag under the key a4_predict.py reads (name_seen_in_train)."""
import argparse, json, os, re, subprocess
WS = '$WORKSPACE/dh2'
ap = argparse.ArgumentParser()
ap.add_argument('--src', default=f'{WS}/results/a4_modctx', help='row dir whose targets to canonize')
ap.add_argument('--out', default=f'{WS}/results/a4_modctx_dm')
args = ap.parse_args()
SRC, OUT = args.src, args.out
os.makedirs(OUT, exist_ok=True)

def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out

def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)

stats = {}
for tier in ['train', 'val', 'test']:
    rows = [json.loads(l) for l in open(f'{SRC}/{tier}.jsonl')]
    dem = demangle_many([r['name'] for r in rows])
    changed = 0
    with open(f'{OUT}/{tier}.jsonl', 'w') as fh:
        for r in rows:
            c = canon(r['name'], dem)
            if c != r['name']: changed += 1
            r['name'] = c
            if 'name_seen' in r: r['name_seen_in_train'] = r.pop('name_seen')
            fh.write(json.dumps(r) + '\n')
    stats[tier] = {'rows': len(rows), 'targets_canonized': changed}
    print(f'EFFECT: {os.path.basename(OUT)} {tier}: {len(rows)} rows, {changed} targets canonized', flush=True)
json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
