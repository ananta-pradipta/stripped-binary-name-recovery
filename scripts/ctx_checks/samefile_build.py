#!/usr/bin/env python3
"""Matched-population rows for the source-locality mechanism study (follow-up plan §10–§13). Wulver, CPU.

Population: LineageBench functions (train / val / test) whose binary has per-function source-file labels
(results/ctx_layout/files_map_<tier>.json) and for which BOTH conditions can be built outside the ±10 window:
  SFO  same-source-file-out : n functions from the target's source file with |rank difference| > 10
  DFF  different-file      : n functions from other source files (label known) with |rank difference| > 10
with n = min(20, |SFO pool|, |DFF pool|) >= 1, the SAME n for both conditions (fairness §12), seeded per function.
For the same functions we also emit the ±10 address condition (ADDR-M) and the no-context condition (NONE-M, masked
body only), so that all four heads are trained and scored on an identical population. Digest recipe, budget (40),
targets (canonical) and masking are those of the adopted head; rows are taken from results/a4_modctx_dm/<tier>.jsonl.
Output: results/a4_ctx_{sfo,dff,addrm,nonem}_dm/{train,val,test}.jsonl + results/ctx_layout/samefile_population.json
"""
import argparse, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a4_build_ctxvar import fn_evidence, rank
WS = '/project/hz79/_shared/cs785/dh2'
K = 10; NMAX = 20; SEED = 20260926
CTX_RE = re.compile(r'^/\* module context: (.*?) \*/\n', re.S)
CONDS = ('sfo', 'dff', 'addrm', 'nonem')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tiers', default='train,val,test'); ap.add_argument('--limit-bins', type=int, default=None)
    ap.add_argument('--out-root', default=f'{WS}/results'); ap.add_argument('--min-n', type=int, default=1)
    args = ap.parse_args()
    pop = {}
    for tier in args.tiers.split(','):
        files_map = json.load(open(f'{WS}/results/ctx_layout/files_map_{tier}.json'))
        layout = json.load(open(f'{WS}/results/ctx_layout/{tier}_layout.json'))
        rows = {}
        for line in open(f'{WS}/results/a4_modctx_dm/{tier}.jsonl'):
            r = json.loads(line); rows[r['key']] = r
        fh = {}
        for c in CONDS:
            os.makedirs(f'{args.out_root}/a4_ctx_{c}_dm', exist_ok=True); fh[c] = open(f'{args.out_root}/a4_ctx_{c}_dm/{tier}.jsonl', 'w')
        st = Counter(); nsizes = []
        bins = list(files_map)[:args.limit_bins] if args.limit_bins else list(files_map)
        for b in bins:
            files = files_map[b]; L = layout.get(b)
            if not L: st['no_layout'] += 1; continue
            p = f'{WS}/symgen_v2/decomp/{b}.json'
            if not os.path.exists(p): st['no_decomp'] += 1; continue
            dec = json.load(open(p)); addrs = L['addrs']; n = len(addrs)
            if len(files) != n: st['len_mismatch'] += 1; continue
            ev = [fn_evidence(dec[a].get('code', '')) for a in addrs]
            idx = {a: i for i, a in enumerate(addrs)}
            by_file = defaultdict(list)
            for j, f in enumerate(files):
                if f is not None: by_file[f].append(j)
            for r in L['rows']:
                key = f"{b}_{r['addr']}"; row = rows.get(key); i = idx.get(r['addr'])
                if row is None or i is None: st['row_missing'] += 1; continue
                f = files[i]
                if f is None: st['target_no_file'] += 1; continue
                sfo_pool = [j for j in by_file[f] if abs(j - i) > K]
                dff_pool = [j for j in range(n) if files[j] is not None and files[j] != f and abs(j - i) > K]
                nn = min(NMAX, len(sfo_pool), len(dff_pool))
                if nn < args.min_n: st['unmatched'] += 1; continue
                rng = random.Random(int(hashlib.sha1(f'{key}:{SEED}'.encode()).hexdigest()[:12], 16))
                sfo = rng.sample(sfo_pool, nn); dff = rng.sample(dff_pool, nn)
                win = [j for j in range(max(0, i - K), min(n, i + 1 + K)) if j != i]
                m = CTX_RE.match(row['code'])
                if not m: st['no_ctx_comment'] += 1; continue
                body = row['code'][m.end():]
                digests = {'sfo': rank(ev, sfo), 'dff': rank(ev, dff), 'addrm': rank(ev, win)}
                for c in CONDS:
                    rr = dict(row)
                    rr['code'] = body if c == 'nonem' else f"/* module context: {digests[c]} */\n{body}"
                    fh[c].write(json.dumps(rr) + '\n')
                st['rows'] += 1; nsizes.append(nn)
                st['sfo_empty'] += (not digests['sfo']); st['dff_empty'] += (not digests['dff'])
                pop[key] = {'tier': tier, 'binary': b, 'package': row['package'], 'n_ctx': nn, 'file': f,
                            'win_same_file': sum(1 for j in win if files[j] == f), 'win_n': len(win)}
        for c in CONDS: fh[c].close()
        ns = sorted(nsizes)
        print(f"EFFECT: samefile {tier}: rows {st['rows']} (binaries {len(bins)}), unmatched {st['unmatched']}, target_no_file {st['target_no_file']}, "
              f"row_missing {st['row_missing']}, n_ctx mean {sum(ns)/max(1,len(ns)):.2f} median {ns[len(ns)//2] if ns else 0} "
              f"(n=20: {sum(1 for x in ns if x==20)/max(1,len(ns)):.1%}), sfo_empty {st['sfo_empty']}, dff_empty {st['dff_empty']}", flush=True)
    json.dump(pop, open(f'{WS}/results/ctx_layout/samefile_population.json', 'w'))
    print(f'EFFECT: population {len(pop)} functions')


if __name__ == '__main__':
    main()
