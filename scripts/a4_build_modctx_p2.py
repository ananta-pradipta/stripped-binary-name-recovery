#!/usr/bin/env python3
"""Idea #3 probe — two-pass name propagation, eval-only (no retraining).
Pass 1 = existing a4_codet5p220m_modctx_v1 predictions. This rebuilds the val/test modctx digests
with each ±10 neighbor's evidence counter augmented by the sub-tokens of that neighbor's pass-1
predicted name (weight 2 per token, contributes to neighbor-df like real evidence), keeping the
exact digest format the checkpoint was trained on (`/* module context: <40 tokens> */`, same
ranking). Pass 2 = a4_predict.py with the SAME checkpoint on these rows. If propagation helps at
all under this in-distribution probe, a propagation-aware retrain is justified.
Output: results/a4_modctx_p2/{val,test}.jsonl."""
import csv, json, os, re
from collections import Counter, defaultdict
WS = '/project/hz79/_shared/cs785/dh2'
PROTO = f'{WS}/results/baseline_protocol_v2'
PREDS = f'{WS}/results/a4_codet5p220m_modctx_v1/val_test_preds.tsv'
OUT = f'{WS}/results/a4_modctx_p2'
os.makedirs(OUT, exist_ok=True)
K = 10; TOP = 40; PRED_W = 2

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
    ev = Counter()
    for m in STR_RE.findall(code): ev.update(toks(m))
    for c in CALL_RE.findall(code):
        if not c.startswith(('FUN_', 'sub_', '_')) and c.lower() not in STOP: ev.update(toks(c))
    return ev

# pass-1 predictions keyed (tier, binary, addr)
p1 = {}
with open(PREDS) as fh:
    for x in csv.DictReader(fh, delimiter='\t'):
        p1[(x['tier'], x['binary'], x['entry_addr'])] = x['pred']
print(f'EFFECT: loaded {len(p1)} pass-1 predictions', flush=True)

def build_digests(dec, tier, binary):
    addrs = sorted(dec.keys(), key=lambda a: int(a, 16))
    ev = []
    n_aug = 0
    for a in addrs:
        e = fn_evidence(dec[a].get('code', ''))
        pn = p1.get((tier, binary, a))
        if pn:
            for t in toks(pn): e[t] += PRED_W
            n_aug += 1
        ev.append(e)
    dig = {}
    for i, a in enumerate(addrs):
        lo, hi = max(0, i - K), min(len(addrs), i + 1 + K)
        df = Counter(); tot = Counter()
        for j in range(lo, hi):
            if j == i: continue
            for t in ev[j]: df[t] += 1; tot[t] += ev[j][t]
        ranked = sorted(df, key=lambda t: (-df[t], -tot[t], t))[:TOP]
        dig[a] = ' '.join(ranked)
    return dig, n_aug

stats = {}
for tier in ['val', 'test']:
    rows_by_bin = defaultdict(list); n = 0
    for line in open(f'{PROTO}/{tier}.jsonl'):
        r = json.loads(line); rows_by_bin[r['binary']].append(r); n += 1
    kept = 0; miss = Counter(); aug_total = 0
    with open(f'{OUT}/{tier}.jsonl', 'w') as fh:
        for b, rs in rows_by_bin.items():
            p = f'{WS}/symgen_v2/decomp/{b}.json'
            if not os.path.exists(p): miss['no_decomp_file'] += len(rs); continue
            dec = json.load(open(p))
            dig, n_aug = build_digests(dec, tier, b); aug_total += n_aug
            for r in rs:
                e = dec.get(r['entry_addr'])
                if not e or 'code' not in e: miss['addr_missing_or_failed'] += 1; continue
                code = e['code'].replace(e['ghidra_name'], '[MASK]', 1)
                if '[MASK]' not in code: miss['mask_not_applied'] += 1; continue
                src = f"/* module context: {dig.get(r['entry_addr'], '')} */\n{code}"
                row = {'key': f"{r['binary']}_{r['entry_addr']}", 'binary': r['binary'], 'package': r['package'],
                       'addr': r['entry_addr'], 'name': r['name'], 'code': src}
                if 'regime' in r: row['regime'] = r['regime']
                if 'name_seen_in_train' in r: row['name_seen_in_train'] = r['name_seen_in_train']
                if 'in_dynsym' in r: row['in_dynsym'] = r['in_dynsym']
                fh.write(json.dumps(row) + '\n'); kept += 1
    stats[tier] = {'rows_in': n, 'rows_out': kept, 'skipped': dict(miss), 'fns_augmented': aug_total}
    print(f'EFFECT: a4_modctx_p2 {tier}: {kept}/{n} rows, fns_augmented={aug_total}, skipped={dict(miss)}', flush=True)
json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
