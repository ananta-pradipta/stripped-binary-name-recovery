#!/usr/bin/env python3
"""CPU diagnostics for the context-source study (advisor plan v2 §10, §12, §13). Runs on Wulver dh2.
  layout   : Delta_context = F1(address ctx) - F1(no ctx) per compiler x optimisation level (adopted head vs
             the no-context Ghidra head), test tier, joined on (binary, entry_addr).
  coverage : per context source, share of scored test functions whose digest contains >=1 / the first /
             all ground-truth sub-tokens (mechanistic explanation of the F1 ordering).
  boot     : package-level paired bootstrap (10k resamples over packages) of F1 differences between
             inference-swap arms on identical functions: address vs random, address vs callgraph, ...
"""
import argparse, csv, json, os, random, re, sys
from collections import defaultdict
WS = '/project/hz79/_shared/cs785/dh2'
SPLIT_RE1 = re.compile(r'([a-z0-9])([A-Z])'); SPLIT_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')


def subtoks(name):
    s = SPLIT_RE1.sub(r'\1_\2', name); s = SPLIT_RE2.sub(r'\1_\2', s)
    return [t for t in re.split(r'[^A-Za-z0-9]+', s.lower()) if t]


def read_preds(path, tier='test'):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['tier'] != tier: continue
            out[(r['binary'], r['entry_addr'])] = (float(r['f1_v2']), r['binary'].split('_')[0])
    return out


def build_of(b):
    comp = 'clang' if 'clang' in b else 'gcc'
    m = re.search(r'_O([0-3])(_|$)', b + '_')
    return comp, ('O' + m.group(1)) if m else '?'


def layout(args):
    A = read_preds(f'{WS}/results/a4_{args.addr_tag}/val_test_preds.tsv'); N = read_preds(args.noctx_file or f'{WS}/results/a4_{args.noctx_tag}/val_test_preds.tsv')
    keys = sorted(set(A) & set(N)); g = defaultdict(list)
    for k in keys: g[build_of(k[0])].append((A[k][0], N[k][0]))
    rows = []
    for (c, o), v in sorted(g.items()):
        a = sum(x for x, _ in v) / len(v); n = sum(y for _, y in v) / len(v)
        rows.append({'compiler': c, 'opt': o, 'n': len(v), 'no_ctx': round(n, 4), 'addr_ctx': round(a, 4), 'delta': round(a - n, 4)})
        print(f"LAYOUT {c:5s} {o} n={len(v):6d} no-ctx {n:.4f} addr-ctx {a:.4f} delta {a-n:+.4f}")
    print(f"EFFECT: layout joined {len(keys)} test functions ({len(A)} addr, {len(N)} no-ctx)")
    json.dump(rows, open(f'{WS}/results/ctx_layout/layout_delta.json', 'w'), indent=1)


def coverage(args):
    res = {}
    for m in args.modes.split(','):
        p = f'{WS}/results/a4_ctx_{m}_dm/test.jsonl'
        if not os.path.exists(p): print('missing', p); continue
        n = anyc = pre = allc = 0
        for line in open(p):
            r = json.loads(line); toks = subtoks(r['name'])
            if not toks: continue
            mm = re.match(r'/\* module context: (.*?) \*/', r['code'], re.S)
            dig = set((mm.group(1) if mm else '').split())
            n += 1; hit = [t in dig for t in toks]
            anyc += any(hit); pre += hit[0]; allc += all(hit)
        res[m] = {'n': n, 'any': round(anyc / n, 4), 'prefix': round(pre / n, 4), 'all': round(allc / n, 4)}
        print(f"COVERAGE {m:12s} n={n} any {anyc/n:.4f} prefix {pre/n:.4f} all {allc/n:.4f}")
    json.dump(res, open(f'{WS}/results/ctx_layout/coverage.json', 'w'), indent=1)
    print('EFFECT: coverage done')


def boot(args):
    arms = {m: read_preds(f'{WS}/results/a4_modctx_dm_sens_{m}/test_preds.tsv') for m in args.modes.split(',')}
    keys = set.intersection(*[set(v) for v in arms.values()])
    pkgs = sorted({arms['win10'][k][1] for k in keys})
    per = {m: defaultdict(list) for m in arms}
    for k in keys:
        for m in arms: per[m][arms[m][k][1]].append(arms[m][k][0])
    pk = {m: {p: sum(v) / len(v) for p, v in per[m].items()} for m in arms}
    fn = {m: sum(arms[m][k][0] for k in keys) / len(keys) for m in arms}
    rng = random.Random(0); out = {}
    for m in arms:
        if m == 'win10': continue
        d_fn = fn['win10'] - fn[m]; d_pk = sum(pk['win10'][p] - pk[m][p] for p in pkgs) / len(pkgs)
        bs = []
        for _ in range(10000):
            smp = [rng.choice(pkgs) for _ in pkgs]; bs.append(sum(pk['win10'][p] - pk[m][p] for p in smp) / len(smp))
        bs.sort(); lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]
        wins = sum(1 for p in pkgs if pk['win10'][p] > pk[m][p])
        out[m] = {'delta_fn': round(d_fn, 4), 'delta_pkg': round(d_pk, 4), 'ci95': [round(lo, 4), round(hi, 4)], 'pkg_wins': f'{wins}/{len(pkgs)}'}
        print(f"BOOT address - {m:12s}: fn {d_fn:+.4f} | pkg {d_pk:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}] | address wins {wins}/{len(pkgs)} packages")
    print(f"EFFECT: boot on {len(keys)} common functions, {len(pkgs)} packages")
    json.dump(out, open(f'{WS}/results/ctx_layout/bootstrap.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('what', choices=['layout', 'coverage', 'boot'])
    ap.add_argument('--modes', default='win10,win5,win20,callgraph,random,empty')
    ap.add_argument('--addr-tag', default='codet5p220m_modctx_dm_v1'); ap.add_argument('--noctx-tag', default='codet5p220m_v1'); ap.add_argument('--noctx-file', default=f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv')
    a = ap.parse_args(); os.makedirs(f'{WS}/results/ctx_layout', exist_ok=True)
    {'layout': layout, 'coverage': coverage, 'boot': boot}[a.what](a)
