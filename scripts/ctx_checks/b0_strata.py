#!/usr/bin/env python3
"""Refinement plan §10: stratified error analysis of the existing flat caller/callee head (B0) by context availability,
with the address head and the no-context head as analysis references only. Wulver, CPU, test tier.
Output: results/ctx_layout/b0_strata.json"""
import csv, json, re
from collections import defaultdict
WS = '/project/hz79/_shared/cs785/dh2'
CTX = re.compile(r'^/\* module context: (.*?) \*/', re.S)


def rd(p):
    out = {}
    for r in csv.DictReader(open(p), delimiter='\t'):
        if r['tier'] == 'test': out[f"{r['binary']}_{r['entry_addr']}"] = (float(r['f1_v2']), r['binary'].split('_')[0], 1.0 if r['true'] == r['pred'] else 0.0)
    return out


def main():
    lay = json.load(open(f'{WS}/results/ctx_layout/test_layout.json'))
    ncall = {}
    for b, L in lay.items():
        callers = defaultdict(set)
        for i, cs in enumerate(L['callees']):
            for j in cs: callers[j].add(i)
        for i, a in enumerate(L['addrs']): ncall[f'{b}_{a}'] = (len(callers[i]), len(L['callees'][i]))
    ntok = {}
    for line in open(f'{WS}/results/a4_ctx_callgraph_dm/test.jsonl'):
        r = json.loads(line); m = CTX.match(r['code']); ntok[r['key']] = len((m.group(1) if m else '').split())
    B0 = rd(f'{WS}/results/a4_codet5p220m_ctxcallgraph_dm_v1/val_test_preds.tsv')
    AD = rd(f'{WS}/results/a4_codet5p220m_modctx_dm_v1/val_test_preds.tsv')
    NO = rd(f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv')
    keys = [k for k in B0 if k in AD and k in NO and k in ncall]
    def strata(k):
        c, e = ncall[k]; n = c + e
        return ['empty digest' if ntok.get(k, 0) == 0 else 'non-empty digest',
                'neither' if n == 0 else 'caller only' if e == 0 else 'callee only' if c == 0 else 'both',
                '0 neighbours' if n == 0 else '1 neighbour' if n == 1 else '2-3 neighbours' if n <= 3 else '4+ neighbours']
    g = defaultdict(list)
    for k in keys:
        for st in strata(k): g[st].append(k)
    def pk(h, ks):
        d = defaultdict(list)
        for k in ks: d[h[k][1]].append(h[k][0])
        return sum(sum(v) / len(v) for v in d.values()) / len(d)
    out = {'n': len(keys), 'mean_callers': sum(ncall[k][0] for k in keys) / len(keys), 'mean_callees': sum(ncall[k][1] for k in keys) / len(keys),
           'mean_digest_tokens': sum(ntok.get(k, 0) for k in keys) / len(keys), 'strata': {}}
    print(f"n={len(keys)} mean callers {out['mean_callers']:.2f} mean callees {out['mean_callees']:.2f} mean digest tokens {out['mean_digest_tokens']:.1f}")
    for st in ['empty digest', 'non-empty digest', 'neither', 'caller only', 'callee only', 'both', '0 neighbours', '1 neighbour', '2-3 neighbours', '4+ neighbours']:
        ks = g[st]; n = len(ks)
        if not n: continue
        f = lambda h: sum(h[k][0] for k in ks) / n
        rec = {'n': n, 'share': round(n / len(keys), 4), 'B0_fn': round(f(B0), 4), 'B0_pkg': round(pk(B0, ks), 4), 'B0_em': round(sum(B0[k][2] for k in ks) / n, 4),
               'address_fn': round(f(AD), 4), 'none_fn': round(f(NO), 4)}
        out['strata'][st] = rec
        print(f"{st:18s} n={n:6d} ({rec['share']:6.1%}) | B0 fn {rec['B0_fn']} pkg {rec['B0_pkg']} EM {rec['B0_em']} | address fn {rec['address_fn']} | none fn {rec['none_fn']}")
    json.dump(out, open(f'{WS}/results/ctx_layout/b0_strata.json', 'w'), indent=1)
    print('EFFECT: b0_strata written')


if __name__ == '__main__':
    main()
