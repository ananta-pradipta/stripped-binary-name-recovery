#!/usr/bin/env python3
"""Follow-up plan §5–§6, §8, §18: token-category provenance (same-file only / different-file only / both / unknown),
prefix-evidence coverage per context source (first GT sub-token present; any / all REMAINING GT sub-tokens present),
call-resolution statistics (unresolved FUN_ references, functions without any resolved relation) and extraction cost
(wall-clock per binary / per function) for the address window vs callers+callees. Wulver, CPU, test tier.
Output: results/ctx_layout/provenance_extra.json"""
import json, os, re, sys, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a4_build_ctxvar import fn_evidence, rank, REF_RE, GNAME_RE, TOP
WS = '/project/hz79/_shared/cs785/dh2'
K = 10
SPLIT_RE1 = re.compile(r'([a-z0-9])([A-Z])'); SPLIT_RE2 = re.compile(r'([A-Z]+)([A-Z][a-z])')
CTX_RE = re.compile(r'^/\* module context: (.*?) \*/', re.S)


def subtoks(name):
    s = SPLIT_RE1.sub(r'\1_\2', name); s = SPLIT_RE2.sub(r'\1_\2', s)
    return [t for t in re.split(r'[^A-Za-z0-9]+', s.lower()) if t]


def main():
    files_map = json.load(open(f'{WS}/results/ctx_layout/files_map_test.json'))
    layout = json.load(open(f'{WS}/results/ctx_layout/test_layout.json'))
    names = {}
    for line in open(f'{WS}/results/a4_ctx_win10_dm/test.jsonl'):
        r = json.loads(line); names[r['key']] = r['name']
    # ---- §5/§6 token categories on the labelled subset
    cat = Counter(); gt_cat = Counter(); n_fn = 0
    for b, files in files_map.items():
        L = layout.get(b); p = f'{WS}/symgen_v2/decomp/{b}.json'
        if not L or not os.path.exists(p): continue
        dec = json.load(open(p)); addrs = L['addrs']; n = len(addrs); idx = {a: i for i, a in enumerate(addrs)}
        ev = [fn_evidence(dec[a].get('code', '')) for a in addrs]
        for r in L['rows']:
            key = f"{b}_{r['addr']}"; i = idx.get(r['addr'])
            if i is None or key not in names or files[i] is None: continue
            f = files[i]; win = [j for j in range(max(0, i - K), min(n, i + 1 + K)) if j != i]
            df = Counter(); tot = Counter(); contrib = defaultdict(set)
            for j in win:
                for t in ev[j]: df[t] += 1; tot[t] += ev[j][t]; contrib[t].add(j)
            toks = sorted(df, key=lambda t: (-df[t], -tot[t], t))[:TOP]
            gt = set(subtoks(names[key])); n_fn += 1
            for t in toks:
                fs = {files[j] for j in contrib[t]}
                same = f in fs; diff = any(x is not None and x != f for x in fs); unk = all(x is None for x in fs)
                c = 'unknown' if unk else 'both' if (same and diff) else 'same_file_only' if same else 'different_file_only'
                cat[c] += 1
                if t in gt: gt_cat[c] += 1
    tot_t = sum(cat.values()); tot_g = sum(gt_cat.values())
    out = {'labelled_functions': n_fn, 'digest_tokens': {k: {'n': v, 'share': round(v / tot_t, 4)} for k, v in cat.items()},
           'gt_matching_tokens': {k: {'n': v, 'share': round(v / max(1, tot_g), 4)} for k, v in gt_cat.items()}}
    print('TOKENS', {k: round(v / tot_t, 3) for k, v in cat.items()}); print('GT-MATCHING TOKENS', {k: round(v / max(1, tot_g), 3) for k, v in gt_cat.items()})
    # ---- §8 prefix-evidence coverage per source (all scored test functions for win10/callgraph/random; labelled subset for su/du)
    cov = {}
    for m in ('win10', 'callgraph', 'random', 'su', 'du', 'addrcall'):
        p = f'{WS}/results/a4_ctx_{m}_dm/test.jsonl'
        if not os.path.exists(p): continue
        n = first = any_rem = all_rem = 0; n_multi = 0
        for line in open(p):
            r = json.loads(line); gt = subtoks(r['name'])
            if not gt: continue
            mm = CTX_RE.match(r['code']); dig = set((mm.group(1) if mm else '').split())
            n += 1; first += gt[0] in dig
            if len(gt) > 1:
                n_multi += 1; rem = gt[1:]; hits = [t in dig for t in rem]; any_rem += any(hits); all_rem += all(hits)
        cov[m] = {'n': n, 'first_present': round(first / n, 4), 'any_remaining_present': round(any_rem / max(1, n_multi), 4), 'all_remaining_present': round(all_rem / max(1, n_multi), 4), 'n_with_remaining': n_multi}
        print(f"COVERAGE {m:9s} n={n} first {cov[m]['first_present']} any-rem {cov[m]['any_remaining_present']} all-rem {cov[m]['all_remaining_present']}")
    out['prefix_evidence_coverage'] = cov
    # ---- §18 call resolution + extraction cost over all test binaries
    n_refs = n_res = 0; fn_total = fn_with_rel = fn_with_any_ref = 0; t_addr = t_call = 0.0; nb = 0
    for b, L in layout.items():
        p = f'{WS}/symgen_v2/decomp/{b}.json'
        if not os.path.exists(p): continue
        dec = json.load(open(p)); addrs = L['addrs']; n = len(addrs); idx = {a: i for i, a in enumerate(addrs)}
        t0 = time.perf_counter(); ev = [fn_evidence(dec[a].get('code', '')) for a in addrs]; t_ev = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in range(n): rank(ev, [j for j in range(max(0, i - K), min(n, i + 1 + K)) if j != i])
        t_addr += time.perf_counter() - t0 + t_ev
        t0 = time.perf_counter()
        g2i = {}
        for a in addrs:
            mm = GNAME_RE.search(dec[a].get('ghidra_name', '') or '')
            if mm: g2i[int(mm.group(1), 16)] = idx[a]
        callees = defaultdict(set); callers = defaultdict(set)
        for a in addrs:
            i = idx[a]; refs = REF_RE.findall(dec[a].get('code', ''))
            if refs: fn_with_any_ref += 1
            for h in refs:
                n_refs += 1; j = g2i.get(int(h, 16))
                if j is not None and j != i: callees[i].add(j); callers[j].add(i); n_res += 1
        for i in range(n): rank(ev, sorted(callees[i] | callers[i])[:40])
        t_call += time.perf_counter() - t0 + t_ev
        fn_total += n; fn_with_rel += sum(1 for i in range(n) if callees[i] or callers[i]); nb += 1
    out['call_resolution'] = {'binaries': nb, 'functions': fn_total, 'fun_references': n_refs, 'resolved_to_retained_function': n_res,
                              'resolved_share': round(n_res / max(1, n_refs), 4), 'functions_with_any_FUN_reference': round(fn_with_any_ref / fn_total, 4),
                              'functions_with_resolved_relation': round(fn_with_rel / fn_total, 4)}
    out['extraction_cost_seconds'] = {'address_window_total': round(t_addr, 2), 'callers_callees_total': round(t_call, 2),
                                      'address_per_function_ms': round(1000 * t_addr / fn_total, 3), 'callers_callees_per_function_ms': round(1000 * t_call / fn_total, 3),
                                      'note': 'CPU time to build the 40-token digest for every function of every test binary from the decompiled text (shared decompilation excluded); call resolution = FUN_ address lookup'}
    print('CALLS', out['call_resolution']); print('COST', out['extraction_cost_seconds'])
    json.dump(out, open(f'{WS}/results/ctx_layout/provenance_extra.json', 'w'), indent=1)
    print('EFFECT: provenance_extra written')


if __name__ == '__main__':
    main()
