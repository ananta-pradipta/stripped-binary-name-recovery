#!/usr/bin/env python3
"""Caller/callee representation variants (refinement plan, 2026-09-26). Wulver, CPU.
Baseline B0 = results/a4_ctx_callgraph_dm (flat 40-token digest over direct callers+callees). Same backbone, targets,
masking, budget; only the context TEXT changes. Direct relations resolved through ghidra_name addresses (as B0).

  b1_role   : same global ranking as B0 (40 tokens), tokens grouped by the role of their contributors:
              "callees: ... callers: ... both: ..."
  b2_bal    : up to 20 tokens ranked over callees only + up to 20 over callers only; unused capacity stays unused
  b3_sig    : per-neighbour compact signature (strings + named calls + normalised code tokens + constant classes),
              ranked by training-corpus IDF, k = max(3, 40 // n_neighbours) tokens per neighbour, total <= 40,
              "callee: ... | caller: ..."
  b4_typed  : as b3 but each neighbour's tokens grouped "api: ... str: ... code: ..."
Markers are plain lower-case words (cheap for the CodeT5+ tokenizer) and do not count toward the 40 evidence tokens.
IDF over per-function evidence in the TRAIN tier decompilations is computed once (results/ctx_layout/cg_idf.json).
Output: results/a4_ctx_<mode>_dm/{train,val,test}.jsonl (+ stats.json), rows identical to B0 except 'code'.
"""
import argparse, json, math, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a4_build_ctxvar import fn_evidence, toks, REF_RE, GNAME_RE, STOP, TOP, STR_RE, CALL_RE, demangle_many, canon
WS = '/project/hz79/_shared/cs785/dh2'
PROTO = f'{WS}/results/baseline_protocol_v2'
CAP = 40
KW = {'if', 'else', 'while', 'for', 'return', 'switch', 'case', 'goto', 'break', 'continue', 'do'}
IDENT_RE = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')
NUM_RE = re.compile(r'\b(0x[0-9a-fA-F]+|\d+)\b')
GHIDRA_AUTO = re.compile(r'^(FUN_|DAT_|LAB_|PTR_|thunk_|switchD_|caseD_|joined_|local_|param_|in_|unaff_|extraout_|[a-z]{1,2}Var\d+|[a-z]{1,2}Stack\w*|uStack\w*|iStack\w*|pcVar\d+|puVar\d+|undefined\d*|code|_DAT)')


def const_class(v):
    if v == 0: return 'zero'
    if v == 1: return 'one'
    if v & (v - 1) == 0: return 'pow2'
    if v < 256: return 'small'
    if v > 0xFFFF: return 'large'
    return 'const'


def code_tokens(code):
    """normalised behavioural tokens of one decompiled function (no names, no addresses)."""
    ev = Counter()
    for kw in KW:
        n = len(re.findall(r'\b' + kw + r'\b', code))
        if n: ev[kw] += n
    for ident in IDENT_RE.findall(code):
        if GHIDRA_AUTO.match(ident) or ident in KW or ident.lower() in STOP: continue
        for t in toks(ident): ev[t] += 1
    for m in NUM_RE.findall(code):
        try: v = int(m, 16) if m.startswith('0x') else int(m)
        except ValueError: continue
        if v > 0xFFFF and len(m) > 6: continue  # addresses
        ev[const_class(v)] += 1
    return ev


def typed_evidence(code):
    api = Counter(); strs = Counter()
    for m in STR_RE.findall(code): strs.update(toks(m))
    for c in CALL_RE.findall(code):
        if not c.startswith(('FUN_', 'sub_', '_')) and c.lower() not in STOP: api.update(toks(c))
    cod = code_tokens(code)
    for t in list(cod):
        if t in api or t in strs or re.fullmatch(r'x[0-9a-f]{2}', t): del cod[t]
    for t in list(strs):
        if re.fullmatch(r'x[0-9a-f]{2}', t): del strs[t]
    return api, strs, cod


def rank_df(ev_list, top):
    df = Counter(); tot = Counter()
    for ev in ev_list:
        for t in ev: df[t] += 1; tot[t] += ev[t]
    return sorted(df, key=lambda t: (-df[t], -tot[t], t))[:top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='b1_role,b2_bal,b3_sig,b4_typed'); ap.add_argument('--tiers', default='train,val,test')
    ap.add_argument('--limit-bins', type=int, default=None); ap.add_argument('--out-root', default=f'{WS}/results')
    args = ap.parse_args(); modes = args.modes.split(','); tiers = args.tiers.split(',')
    # ---- IDF over train-tier functions (typed evidence + code tokens), cached
    idf_path = f'{WS}/results/ctx_layout/cg_idf.json'
    if os.path.exists(idf_path):
        d = json.load(open(idf_path)); DF = Counter(d['df']); NF = d['n']
    else:
        DF = Counter(); NF = 0; bins = set()
        for line in open(f'{PROTO}/train.jsonl'): bins.add(json.loads(line)['binary'])
        for b in sorted(bins):
            p = f'{WS}/symgen_v2/decomp/{b}.json'
            if not os.path.exists(p): continue
            for e in json.load(open(p)).values():
                api, strs, code = typed_evidence(e.get('code', '')); NF += 1
                for t in set(api) | set(strs) | set(code): DF[t] += 1
        json.dump({'n': NF, 'df': dict(DF)}, open(idf_path, 'w'))
    def idf(t): return math.log((NF + 1) / (DF.get(t, 0) + 1))
    print(f'EFFECT: idf over {NF} train functions, {len(DF)} tokens', flush=True)
    fhs = {}; stats = {m: {} for m in modes}
    for m in modes:
        os.makedirs(f'{args.out_root}/a4_ctx_{m}_dm', exist_ok=True)
        for t in tiers: fhs[(m, t)] = open(f'{args.out_root}/a4_ctx_{m}_dm/{t}.jsonl', 'w')
    for tier in tiers:
        rows_by_bin = defaultdict(list); n_in = 0
        for line in open(f'{PROTO}/{tier}.jsonl'):
            r = json.loads(line); rows_by_bin[r['binary']].append(r); n_in += 1
        bins = list(rows_by_bin)[:args.limit_bins] if args.limit_bins else list(rows_by_bin)
        acc = {m: Counter() for m in modes}; tokcount = {m: 0 for m in modes}
        dem = demangle_many([r['name'] for b in bins for r in rows_by_bin[b]])
        for b in bins:
            p = f'{WS}/symgen_v2/decomp/{b}.json'
            if not os.path.exists(p):
                for m in modes: acc[m]['no_decomp'] += len(rows_by_bin[b])
                continue
            dec = json.load(open(p)); addrs = sorted(dec, key=lambda a: int(a, 16)); idx = {a: i for i, a in enumerate(addrs)}
            codes = [dec[a].get('code', '') for a in addrs]
            g2i = {}
            for a in addrs:
                mm = GNAME_RE.search(dec[a].get('ghidra_name', '') or '')
                if mm: g2i[int(mm.group(1), 16)] = idx[a]
            callees = defaultdict(set); callers = defaultdict(set)
            for a in addrs:
                i = idx[a]
                for h in REF_RE.findall(codes[i]):
                    j = g2i.get(int(h, 16))
                    if j is not None and j != i: callees[i].add(j); callers[j].add(i)
            ev_cache = {}
            def ev(j):
                if j not in ev_cache: ev_cache[j] = fn_evidence(codes[j])
                return ev_cache[j]
            typed_cache = {}
            def typed(j):
                if j not in typed_cache: typed_cache[j] = typed_evidence(codes[j])
                return typed_cache[j]
            for r in rows_by_bin[b]:
                a = r['entry_addr']; e = dec.get(a)
                if not e or 'code' not in e:
                    for m in modes: acc[m]['addr_missing'] += 1
                    continue
                code = e['code'].replace(e['ghidra_name'], '[MASK]', 1)
                if '[MASK]' not in code:
                    for m in modes: acc[m]['mask_not_applied'] += 1
                    continue
                i = idx[a]; ce = sorted(callees[i])[:40]; cr = sorted(callers[i])[:40]
                cname = canon(r['name'], dem)
                for m in modes:
                    if m == 'b1_role':
                        allev = [ev(j) for j in ce + cr]; top = rank_df(allev, TOP)
                        has_ce = {t for j in ce for t in ev(j)}; has_cr = {t for j in cr for t in ev(j)}
                        seg = {'callees': [t for t in top if t in has_ce and t not in has_cr], 'callers': [t for t in top if t in has_cr and t not in has_ce], 'both': [t for t in top if t in has_ce and t in has_cr]}
                        ctx = ' '.join(f"{k}: {' '.join(v)}" for k, v in seg.items() if v); ntok = len(top)
                    elif m == 'b2_bal':
                        tce = rank_df([ev(j) for j in ce], 20); tcr = rank_df([ev(j) for j in cr], 20)
                        parts = []
                        if tce: parts.append('callees: ' + ' '.join(tce))
                        if tcr: parts.append('callers: ' + ' '.join(tcr))
                        ctx = ' '.join(parts); ntok = len(tce) + len(tcr)
                    else:
                        nb = [(j, 'callee') for j in ce] + [(j, 'caller') for j in cr]
                        if not nb: ctx = ''; ntok = 0
                        else:
                            k = max(3, CAP // len(nb)); parts = []; ntok = 0; budget = CAP
                            for j, role in nb:
                                if budget <= 0: break
                                api, strs, cod = typed(j)
                                if m == 'b3_sig':
                                    allt = Counter(); allt.update(api); allt.update(strs); allt.update(cod)
                                    sel = sorted(allt, key=lambda t: (-idf(t) * min(allt[t], 3), t))[:min(k, budget)]
                                    if sel: parts.append(f"{role}: {' '.join(sel)}"); ntok += len(sel); budget -= len(sel)
                                else:
                                    kk = min(k, budget); segs = []; used = 0
                                    for lab, cnt in (('api', api), ('str', strs), ('code', cod)):
                                        q = max(1, kk // 3) if cnt else 0
                                        sel = sorted(cnt, key=lambda t: (-idf(t) * min(cnt[t], 3), t))[:q]
                                        if sel: segs.append(f"{lab}: {' '.join(sel)}"); used += len(sel)
                                    if segs: parts.append(f"{role}: " + ' '.join(segs)); ntok += used; budget -= used
                            ctx = ' | '.join(parts)
                    text = f'/* call context: {ctx} */\n{code}' if ctx else f'/* call context: */\n{code}'
                    row = {'key': f"{b}_{a}", 'binary': b, 'package': r['package'], 'addr': a, 'name': cname, 'code': text}
                    for kf in ('regime', 'name_seen_in_train', 'in_dynsym'):
                        if kf in r: row[kf] = r[kf]
                    fhs[(m, tier)].write(json.dumps(row) + '\n'); acc[m]['rows'] += 1; acc[m]['empty'] += (ntok == 0); tokcount[m] += ntok
                    acc[m]['neither' if not ce and not cr else 'callee_only' if not cr else 'caller_only' if not ce else 'both'] += 1
        for m in modes:
            st = dict(acc[m]); st['mean_tokens'] = round(tokcount[m] / max(1, acc[m]['rows']), 2); stats[m][tier] = st
            print(f"EFFECT: {m} {tier}: rows {st.get('rows',0)}/{n_in}, empty {st.get('empty',0)} ({st.get('empty',0)/max(1,st.get('rows',1)):.1%}), mean tokens {st['mean_tokens']}, "
                  f"neither {st.get('neither',0)} caller_only {st.get('caller_only',0)} callee_only {st.get('callee_only',0)} both {st.get('both',0)}", flush=True)
    for fh in fhs.values(): fh.close()
    for m in modes: json.dump(stats[m], open(f'{args.out_root}/a4_ctx_{m}_dm/stats.json', 'w'), indent=1)
    for m in modes:
        for l, _ in zip(open(f'{args.out_root}/a4_ctx_{m}_dm/{tiers[-1]}.jsonl'), range(1)):
            r = json.loads(l); print(f'SAMPLE {m}', r['code'][:260].replace('\n', ' '))


if __name__ == '__main__':
    main()
