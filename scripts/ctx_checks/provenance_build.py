#!/usr/bin/env python3
"""Token provenance + same-unit / different-unit context sets (advisor request, 2026-09-25 #2). Wulver, CPU.

For every scored LineageBench test function whose binary has per-function source-file labels
(results/ctx_layout/files_map_test.json, from addr2line on the unstripped builds):
  * recompute the adopted ±10 digest with per-token contributors and label each of the 40 tokens by whether at
    least one contributing neighbour is in the SAME source file as the target;
  * record whether the ground-truth sub-tokens (and the first one, the "prefix") are in the digest and whether
    they came from a same-file neighbour;
  * build two alternative context sets at the same budget and recipe, both OUTSIDE the ±10 window:
      SU = up to 20 functions from the SAME source file,   DU = up to 20 functions from DIFFERENT source files;
    written as inference rows for the adopted head (results/a4_ctx_{su,du}_dm/test.jsonl).
Output: results/ctx_layout/provenance_test.json (one record per function).
"""
import argparse, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a4_build_ctxvar import fn_evidence, TOP  # same tokeniser / evidence / budget as every other variant
WS = '/project/hz79/_shared/cs785/dh2'
K = 10; N = 20; SEED = 20260925
SPLIT_RE1 = re.compile(r'([a-z0-9])([A-Z])'); SPLIT_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
CTX_RE = re.compile(r'^/\* module context: (.*?) \*/\n', re.S)


def subtoks(name):
    s = SPLIT_RE1.sub(r'\1_\2', name); s = SPLIT_RE2.sub(r'\1_\2', s)
    return [t for t in re.split(r'[^A-Za-z0-9]+', s.lower()) if t]


def rank_with_contrib(ev, src):
    df = Counter(); tot = Counter(); contrib = defaultdict(set)
    for j in src:
        for t in ev[j]: df[t] += 1; tot[t] += ev[j][t]; contrib[t].add(j)
    toks = sorted(df, key=lambda t: (-df[t], -tot[t], t))[:TOP]
    return toks, contrib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files-map', default=f'{WS}/results/ctx_layout/files_map_test.json')
    ap.add_argument('--out-root', default=f'{WS}/results')
    ap.add_argument('--prov-out', default=f'{WS}/results/ctx_layout/provenance_test.json')
    ap.add_argument('--min-rows', type=int, default=50000)
    args = ap.parse_args()
    files_map = json.load(open(args.files_map))
    layout = json.load(open(f'{WS}/results/ctx_layout/test_layout.json'))
    rows = {}
    for line in open(f'{WS}/results/a4_ctx_win10_dm/test.jsonl'):
        r = json.loads(line); rows[r['key']] = r
    os.makedirs(f'{args.out_root}/a4_ctx_su_dm', exist_ok=True); os.makedirs(f'{args.out_root}/a4_ctx_du_dm', exist_ok=True)
    fsu = open(f'{args.out_root}/a4_ctx_su_dm/test.jsonl', 'w'); fdu = open(f'{args.out_root}/a4_ctx_du_dm/test.jsonl', 'w')
    prov = {}; n_su_empty = n_du_empty = 0; n_out = 0
    for b, files in files_map.items():
        L = layout.get(b)
        if not L: continue
        p = f'{WS}/symgen_v2/decomp/{b}.json'
        if not os.path.exists(p): continue
        dec = json.load(open(p)); addrs = L['addrs']; n = len(addrs)
        assert len(files) == n, (b, len(files), n)
        ev = [fn_evidence(dec[a].get('code', '')) for a in addrs]
        idx = {a: i for i, a in enumerate(addrs)}
        by_file = defaultdict(list)
        for j, f in enumerate(files):
            if f is not None: by_file[f].append(j)
        for r in L['rows']:
            key = f"{b}_{r['addr']}"; row = rows.get(key)
            i = idx.get(r['addr']); f = files[i] if i is not None else None
            if row is None or i is None or f is None: continue
            win = [j for j in range(max(0, i - K), min(n, i + 1 + K)) if j != i]
            toks, contrib = rank_with_contrib(ev, win)
            same_tok = sum(1 for t in toks if any(files[j] == f for j in contrib[t]))
            gt = subtoks(row['name']); digest = set(toks)
            def src_of(t):
                if t not in digest: return 'absent'
                return 'same_file' if any(files[j] == f for j in contrib[t]) else 'other_file'
            gt_src = [src_of(t) for t in gt]
            same_fns = sum(1 for j in win if files[j] == f)
            su_pool = [j for j in by_file[f] if abs(j - i) > K]
            du_pool = [j for j in range(n) if files[j] is not None and files[j] != f and abs(j - i) > K]
            rng = random.Random(int(hashlib.sha1(f'{key}:{SEED}'.encode()).hexdigest()[:12], 16))
            su = rng.sample(su_pool, min(N, len(su_pool))); du = rng.sample(du_pool, min(N, len(du_pool)))
            su_toks, _ = rank_with_contrib(ev, su) if su else ([], None); du_toks, _ = rank_with_contrib(ev, du) if du else ([], None)
            m = CTX_RE.match(row['code']); assert m, key
            body = row['code'][m.end():]
            for fh, tk in ((fsu, su_toks), (fdu, du_toks)):
                rr = dict(row); rr['code'] = f"/* module context: {' '.join(tk)} */\n{body}"; fh.write(json.dumps(rr) + '\n')
            n_su_empty += (not su_toks); n_du_empty += (not du_toks); n_out += 1
            prov[key] = {'binary': b, 'package': row['package'], 'regime': row.get('regime'), 'seen': row.get('name_seen_in_train'),
                         'file': f, 'n_fns': n, 'n_same_file_in_binary': len(by_file[f]) - 1,
                         'win_same_file_fns': same_fns, 'win_n': len(win), 'digest_n': len(toks), 'digest_same_file_tokens': same_tok,
                         'gt_subtoks': gt, 'gt_sources': gt_src, 'prefix_source': gt_src[0] if gt_src else 'absent',
                         'su_n': len(su), 'su_pool': len(su_pool), 'su_digest_n': len(su_toks), 'du_n': len(du), 'du_pool': len(du_pool), 'du_digest_n': len(du_toks)}
    fsu.close(); fdu.close()
    json.dump(prov, open(args.prov_out, 'w'))
    tot = len(prov)
    print(f"EFFECT: provenance for {tot} functions in {len(files_map)} binaries; SU empty {n_su_empty} ({n_su_empty/max(1,tot):.1%}), DU empty {n_du_empty} ({n_du_empty/max(1,tot):.1%})")
    ms = sum(v['digest_same_file_tokens'] for v in prov.values()) / max(1, sum(v['digest_n'] for v in prov.values()))
    pre = Counter(v['prefix_source'] for v in prov.values())
    print(f"EFFECT: same-file share of digest TOKENS {ms:.3f}; prefix source: {dict(pre)}")
    if tot < args.min_rows: print('FATAL: too few provenance rows'); sys.exit(2)


if __name__ == '__main__':
    main()
