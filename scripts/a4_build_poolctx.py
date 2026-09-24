#!/usr/bin/env python3
"""Idea #1b — pooled-evidence digest for the A4 generation head (extends a4_build_modctx.py).
zero_evidence_census.json: for zero-self-evidence novel fns, GT-token coverage is 65% in the ±10
address neighbors (harvested by modctx) but 36% in direct callees/callers and 94% anywhere in the
binary's decomp pool. This builder emits a 3-tier digest per function:
  module:   top-24 tokens from ±10 address neighbors (same ranking as modctx)
  calls:    top-16 tokens from direct callees+callers (FUN_/sub_ refs, both directions, cap 40 fns)
  binary:   top-16 tokens from the whole binary's evidence pool
Tiers 2/3 are ranked by local df × idf, where idf is computed over TRAIN-tier binaries only (one
token counted once per binary), so generic tokens (the census's caveat on the 94%) are down-weighted
and no test-corpus statistics shape the ranking. Output: results/a4_poolctx/{train,val,test}.jsonl,
same row schema as a4_modctx (train via a4_train_codet5p.py --data-dir, predict via --rows)."""
import json, math, os, re
from collections import Counter, defaultdict
WS = '/project/hz79/_shared/cs785/dh2'
PROTO = f'{WS}/results/baseline_protocol_v2'
OUT = f'{WS}/results/a4_poolctx'
os.makedirs(OUT, exist_ok=True)
K = 10
TOP_NEIGH, TOP_CALL, TOP_POOL = 24, 16, 16
CALL_FN_CAP = 40

STR_RE = re.compile(r'"((?:[^"\\\n]|\\.){3,200})"')
CALL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(')
REF_RE = re.compile(r'\b(?:FUN_|sub_)([0-9a-fA-F]{4,})')
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
    ev = Counter()
    for m in STR_RE.findall(code): ev.update(toks(m))
    for c in CALL_RE.findall(code):
        if not c.startswith(('FUN_', 'sub_', '_')) and c.lower() not in STOP: ev.update(toks(c))
    return ev

def load_rows(tier):
    by_bin = defaultdict(list); n = 0
    for line in open(f'{PROTO}/{tier}.jsonl'):
        r = json.loads(line); by_bin[r['binary']].append(r); n += 1
    return by_bin, n

# ---- pass 1: idf over train-tier binaries (one count per binary; evidence tokens only)
IDF_PATH = f'{OUT}/global_df.json'
if os.path.exists(IDF_PATH):
    gd = json.load(open(IDF_PATH)); GDF = Counter(gd['df']); NBIN = gd['n_binaries']
else:
    GDF = Counter(); NBIN = 0
    train_bins, _ = load_rows('train')
    for b in train_bins:
        p = f'{WS}/symgen_v2/decomp/{b}.json'
        if not os.path.exists(p): continue
        dec = json.load(open(p)); NBIN += 1
        seen = set()
        for e in dec.values(): seen.update(fn_evidence(e.get('code', '')))
        for t in seen: GDF[t] += 1
    json.dump({'n_binaries': NBIN, 'df': dict(GDF)}, open(IDF_PATH, 'w'))
print(f'EFFECT: idf over {NBIN} train binaries, {len(GDF)} tokens', flush=True)

def idf(t): return math.log((NBIN + 1) / (GDF.get(t, 0) + 1))

def build_digests(dec):
    addrs = sorted(dec.keys(), key=lambda a: int(a, 16))
    idx = {a: i for i, a in enumerate(addrs)}
    ev = [fn_evidence(dec[a].get('code', '')) for a in addrs]
    callers = defaultdict(set)
    callees = defaultdict(set)
    for a in addrs:
        for h in REF_RE.findall(dec[a].get('code', '')):
            c = '0x' + h.lower().lstrip('0x')
            if c in idx: callees[a].add(c); callers[c].add(a)
    pool_df = Counter()
    for e in ev:
        for t in e: pool_df[t] += 1
    dig = {}
    for i, a in enumerate(addrs):
        # tier 1: address neighbors (modctx ranking: neighbor-df, then total count)
        lo, hi = max(0, i - K), min(len(addrs), i + 1 + K)
        df = Counter(); tot = Counter()
        for j in range(lo, hi):
            if j == i: continue
            for t in ev[j]: df[t] += 1; tot[t] += ev[j][t]
        neigh = sorted(df, key=lambda t: (-df[t], -tot[t], t))[:TOP_NEIGH]
        used = set(neigh)
        # tier 2: direct callees + callers, df × idf
        cs = list(callees[a] | callers[a])[:CALL_FN_CAP]
        cdf = Counter()
        for c in cs:
            for t in ev[idx[c]]: cdf[t] += 1
        call = sorted((t for t in cdf if t not in used),
                      key=lambda t: (-cdf[t] * idf(t), t))[:TOP_CALL]
        used.update(call)
        # tier 3: whole-binary pool, local fn-df × idf
        pool = sorted((t for t in pool_df if t not in used),
                      key=lambda t: (-pool_df[t] * idf(t), t))[:TOP_POOL]
        dig[a] = f"module: {' '.join(neigh)} | calls: {' '.join(call)} | binary: {' '.join(pool)}"
    return dig

stats = {}
for tier in ['train', 'val', 'test']:
    rows_by_bin, n = load_rows(tier)
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
                if not re.search(r'[a-z]', ctx): empty_ctx += 1
                src = f'/* context {ctx} */\n{code}'
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
          'frac_le_1280tok': round(sum(t <= 1280 for t in approx)/max(1, len(approx)), 3),
          'frac_le_1536tok': round(sum(t <= 1536 for t in approx)/max(1, len(approx)), 3)}
    stats[tier] = st
    print(f'EFFECT: a4_poolctx {tier}: {kept}/{n} rows (empty_ctx {empty_ctx}), skipped={dict(miss)}, '
          f'p50/p90 chars={st["chars_p50"]}/{st["chars_p90"]}, <=1280tok={st["frac_le_1280tok"]}', flush=True)
json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
for l, _ in zip(open(f'{OUT}/val.jsonl'), range(2)):
    r = json.loads(l); print('SAMPLE', r['name'], '|', r['code'][:500].replace('\n', ' '))
