#!/usr/bin/env python3
"""Idea #1 — module-context (TU-locality) input variant for the A4 generation head.
For every protocol row, prepend to the masked decomp code a compact digest of the evidence carried by
its +-K address-adjacent functions in the SAME stripped binary: identifier tokens mined from their
string literals and their named (import) calls, ranked by how many distinct neighbors contain them.
Motivation: zero_evidence_census.json — 65% of zero-evidence novel fns have >=1 GT token and 42% have
the naming-convention prefix within +-10 neighbors. Output: results/a4_modctx/{train,val,test}.jsonl
(same row schema as a4_baptext, trainable via a4_train_codet5p.py --data-dir, predictable via --rows)."""
import json, os, re, sys
from collections import Counter, defaultdict
WS = '$WORKSPACE/dh2'
PROTO = f'{WS}/results/baseline_protocol_v2'
OUT = f'{WS}/results/a4_modctx'
os.makedirs(OUT, exist_ok=True)
K = 10          # neighbors each side
TOP = 40        # digest tokens

STR_RE = re.compile(r'"((?:[^"\\\n]|\\.){3,200})"')
CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(')
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

def fn_evidence(code):
    """identifier tokens from one function's decomp text: string literals + named calls."""
    ev = Counter()
    for m in STR_RE.findall(code): ev.update(toks(m))
    for c in CALL_RE.findall(code):
        if not c.startswith(('FUN_', 'sub_', '_')) and c.lower() not in STOP: ev.update(toks(c))
    return ev

def build_digests(dec):
    addrs = sorted(dec.keys(), key=lambda a: int(a, 16))
    ev = [fn_evidence(dec[a].get('code', '')) for a in addrs]
    dig = {}
    for i, a in enumerate(addrs):
        lo, hi = max(0, i - K), min(len(addrs), i + 1 + K)
        df = Counter(); tot = Counter()
        for j in range(lo, hi):
            if j == i: continue
            for t in ev[j]: df[t] += 1; tot[t] += ev[j][t]
        ranked = sorted(df, key=lambda t: (-df[t], -tot[t], t))[:TOP]
        dig[a] = ' '.join(ranked)
    return dig

stats = {}
for tier in ['train', 'val', 'test']:
    rows_by_bin = defaultdict(list); n = 0
    for line in open(f'{PROTO}/{tier}.jsonl'):
        r = json.loads(line); rows_by_bin[r['binary']].append(r); n += 1
    kept = 0; miss = Counter(); empty_ctx = 0; lens = []
    with open(f'{OUT}/{tier}.jsonl', 'w') as fh:
        for b, rs in rows_by_bin.items():
            p = f'{WS}/symgen_v2/decomp/{b}.json'
            if not os.path.exists(p): miss['no_decomp_file'] += len(rs); continue
            dec = json.load(open(p))
            dig = build_digests(dec)
            for r in rs:
                e = dec.get(r['entry_addr'])
                if not e or 'code' not in e: miss['addr_missing_or_failed'] += 1; continue
                code = e['code'].replace(e['ghidra_name'], '[MASK]', 1)
                if '[MASK]' not in code: miss['mask_not_applied'] += 1; continue
                ctx = dig.get(r['entry_addr'], '')
                if not ctx: empty_ctx += 1
                src = f'/* module context: {ctx} */\n{code}'
                row = {'key': f"{r['binary']}_{r['entry_addr']}", 'binary': r['binary'], 'package': r['package'],
                       'addr': r['entry_addr'], 'name': r['name'], 'code': src}
                if 'regime' in r: row['regime'] = r['regime']
                if 'name_seen_in_train' in r: row['name_seen_in_train'] = r['name_seen_in_train']
                if 'in_dynsym' in r: row['in_dynsym'] = r['in_dynsym']
                fh.write(json.dumps(row) + '\n'); kept += 1; lens.append(len(src))
    lens.sort()
    def pct(q): return lens[min(len(lens)-1, int(q*len(lens)))] if lens else 0
    approx = [l/3.5 for l in lens]
    st = {'rows_in': n, 'rows_out': kept, 'skipped': dict(miss), 'empty_ctx': empty_ctx,
          'chars_p50': pct(.5), 'chars_p90': pct(.9),
          'frac_le_1024tok': round(sum(t <= 1024 for t in approx)/max(1, len(approx)), 3),
          'frac_le_1280tok': round(sum(t <= 1280 for t in approx)/max(1, len(approx)), 3),
          'frac_le_1536tok': round(sum(t <= 1536 for t in approx)/max(1, len(approx)), 3)}
    stats[tier] = st
    print(f'EFFECT: a4_modctx {tier}: {kept}/{n} rows (empty_ctx {empty_ctx}), skipped={dict(miss)}, '
          f'p50/p90 chars={st["chars_p50"]}/{st["chars_p90"]}, '
          f'<=1024tok={st["frac_le_1024tok"]} <=1280tok={st["frac_le_1280tok"]}', flush=True)
json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
# show 2 examples for the smoke log
for l, _ in zip(open(f'{OUT}/val.jsonl'), range(2)):
    r = json.loads(l); print('SAMPLE', r['name'], '|', r['code'][:400].replace('\n', ' '))
