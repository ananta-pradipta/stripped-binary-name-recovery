#!/usr/bin/env python3
"""Idea #2 — demangling normalization for the modctx A4 head (targets AND inputs, one coherent change).
Diagnosis (novel_head_analysis.json): on icu (48K test rows, F1 0.041) A4 emits mangled-name
fragments (epns/epkns/7board). Root cause is two-sided: (a) C++ targets keep _Z mangling artifacts,
(b) the decomp INPUT text is full of mangled identifiers (C++ exports survive stripping via .dynsym),
so the model copy-learns mangled shapes. Targets-only is underpowered: only 2.2% of train rows are
mangled and val has zero, so this builder normalizes BOTH sides with the scorers' own canon
(demangle via c++filt, strip args/templates, keep last-2 :: qualifiers joined by _):
  targets: name -> canon(name)
  inputs:  mask ghidra_name FIRST, then rewrite every _Z identifier in the code text to its canon
           form; the ±10-neighbor evidence digest is rebuilt from the rewritten text (same recipe
           and format as a4_build_modctx.py: K=10, TOP=40, string literals + named calls).
Output: results/a4_modctx_dm/{train,val,test}.jsonl (overwrites the targets-only v1 rows)."""
import json, os, re, subprocess
from collections import Counter, defaultdict
WS = '/project/hz79/_shared/cs785/dh2'
PROTO = f'{WS}/results/baseline_protocol_v2'
OUT = f'{WS}/results/a4_modctx_dm'
os.makedirs(OUT, exist_ok=True)
K = 10; TOP = 40

STR_RE = re.compile(r'"((?:[^"\\\n]|\\.){3,200})"')
CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(')
MANGLE_RE = re.compile(r'\b_Z[A-Za-z0-9_]{3,}\b')
SPLIT_RE1 = re.compile(r'([a-z0-9])([A-Z])'); SPLIT_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
STOP = set('''if else while for return switch case break continue goto sizeof do
void int char long short float double unsigned signed bool true false null nullptr
uint ulong ushort byte undefined undefined1 undefined2 undefined4 undefined8 code
param local stack var unaff extraout in out ram fun sub ptr concat sext zext
memcpy memset strlen strcmp strcpy strncpy strncmp malloc calloc realloc free printf fprintf sprintf
the and for with not this that from func file line error warning failed invalid cannot could unable
halt baddata warning'''.split())

def toks(s):
    s = SPLIT_RE1.sub(r'\1_\2', s); s = SPLIT_RE2.sub(r'\1_\2', s)
    return [t for t in re.split(r'[^A-Za-z0-9]+', s.lower()) if 2 < len(t) < 25 and not t.isdigit() and t not in STOP]

def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out

def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]
    c = '_'.join(p for p in parts if p)
    return re.sub(r'[^A-Za-z0-9_]+', '_', c).strip('_') or n

def fn_evidence(code):
    ev = Counter()
    for m in STR_RE.findall(code): ev.update(toks(m))
    for c in CALL_RE.findall(code):
        if not c.startswith(('FUN_', 'sub_', '_')) and c.lower() not in STOP: ev.update(toks(c))
    return ev

def build_digests(codes):
    addrs = sorted(codes.keys(), key=lambda a: int(a, 16))
    ev = [fn_evidence(codes[a]) for a in addrs]
    dig = {}
    for i, a in enumerate(addrs):
        lo, hi = max(0, i - K), min(len(addrs), i + 1 + K)
        df = Counter(); tot = Counter()
        for j in range(lo, hi):
            if j == i: continue
            for t in ev[j]: df[t] += 1; tot[t] += ev[j][t]
        dig[a] = ' '.join(sorted(df, key=lambda t: (-df[t], -tot[t], t))[:TOP])
    return dig

stats = {}
for tier in ['train', 'val', 'test']:
    rows_by_bin = defaultdict(list); n = 0
    for line in open(f'{PROTO}/{tier}.jsonl'):
        r = json.loads(line); rows_by_bin[r['binary']].append(r); n += 1
    kept = 0; miss = Counter(); tgt_changed = 0; in_rewritten = 0
    with open(f'{OUT}/{tier}.jsonl', 'w') as fh:
        for b, rs in rows_by_bin.items():
            p = f'{WS}/symgen_v2/decomp/{b}.json'
            if not os.path.exists(p): miss['no_decomp_file'] += len(rs); continue
            dec = json.load(open(p))
            # mask first (per-function ghidra_name), then demangle-rewrite the whole text
            mangled = set()
            for e in dec.values(): mangled.update(MANGLE_RE.findall(e.get('code', '')))
            dem = demangle_many(mangled)
            def rewrite(code): return MANGLE_RE.sub(lambda m: canon(m.group(0), dem), code)
            masked = {}
            for a, e in dec.items():
                if 'code' not in e: continue
                masked[a] = rewrite(e['code'].replace(e['ghidra_name'], '[MASK]', 1))
            dig = build_digests(masked)
            dem_t = demangle_many([r['name'] for r in rs])
            for r in rs:
                a = r['entry_addr']
                code = masked.get(a)
                if code is None: miss['addr_missing_or_failed'] += 1; continue
                if '[MASK]' not in code: miss['mask_not_applied'] += 1; continue
                if MANGLE_RE.search(dec[a]['code']): in_rewritten += 1
                tgt = canon(r['name'], dem_t)
                if tgt != r['name']: tgt_changed += 1
                src = f"/* module context: {dig.get(a, '')} */\n{code}"
                row = {'key': f"{r['binary']}_{a}", 'binary': r['binary'], 'package': r['package'],
                       'addr': a, 'name': tgt, 'code': src}
                if 'regime' in r: row['regime'] = r['regime']
                if 'name_seen_in_train' in r: row['name_seen_in_train'] = r['name_seen_in_train']
                if 'in_dynsym' in r: row['in_dynsym'] = r['in_dynsym']
                fh.write(json.dumps(row) + '\n'); kept += 1
    stats[tier] = {'rows_in': n, 'rows_out': kept, 'skipped': dict(miss),
                   'targets_canonized': tgt_changed, 'inputs_with_mangled_ids': in_rewritten}
    print(f'EFFECT: a4_modctx_dm {tier}: {kept}/{n} rows, targets_canonized={tgt_changed}, '
          f'inputs_with_mangled_ids={in_rewritten}, skipped={dict(miss)}', flush=True)
json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
