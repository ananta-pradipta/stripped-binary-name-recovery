#!/usr/bin/env python3
"""Deep analysis of the generation heads on NOVEL-NAME test rows: our A4 (CodeT5+ 220m on Ghidra text)
vs SymGen-34B (LoRA on our tier), with R for reference. Questions: (1) where does each win (per package,
per name/function scale); (2) are they complementary or redundant (oracle, exclusive-win counts);
(3) error character (precision vs recall, prediction length, genericity, top predictions);
(4) evidence vs prior: is the GT name recoverable from the input text both models see (masked decomp) —
for rows where it is not, only pretraining prior can produce it (contamination-sensitivity cut);
(5) dump qualitative examples (SymGen-only exact hits, A4-only hits, both-fail-high-evidence)."""
import json, os, re, sys, glob, random, collections, subprocess
sys.path.insert(0, '$WORKSPACE/dh2')
from src.evaluation.metrics import compute_subtoken_f1, split_name
WS = '$WORKSPACE/dh2'

def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out
def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)
def sg_clean(p):
    p = p.replace('</s>', '').strip(); p = re.sub(r'^The predicted function name is\s*', '', p).strip()
    return p.split()[0] if p else ''

# ---- join (same as score_symgen_full) ----
sg, meta_by_key = {}, {}
for k in range(34):
    meta = json.load(open(f'{WS}/symgen_v2/fulltest_shards/meta_{k}.json'))
    preds = json.load(open(f'{WS}/symgen_v2/results_fulltest/shard_{k}/predicted_function_name.json'))
    for m, p in zip(meta, preds):
        if p['ground_truth'] != m['gt_name']: continue
        key = ('test', m['binary'], int(m['addr'], 16))
        sg[key] = sg_clean(p['predicted_name']); meta_by_key[key] = m
feat = {}; hdr = None
for l in open(f'{WS}/results/router_c1soft_l10_interim_features.tsv'):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r))
    if d['bap'].startswith('sub_'):
        a = int(d['bap'][4:], 16)
        for da in (0, 4, -4): feat.setdefault((d['tier'], d['binary'], a + da), d)
a4 = {}
for l in open(f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv'):
    r = l.rstrip('\n').split('\t')
    if r[0] == 'test': a4[(r[0], r[1], int(r[2], 16))] = r[4]
rows = []
for key, p in sg.items():
    m = meta_by_key[key]
    if m.get('name_seen'): continue          # novel-name rows only
    d = feat.get(key); a = a4.get(key)
    if d is None or a is None: continue
    rows.append({'key': key, 'pkg': m['package'], 'binary': m['binary'], 'addr': m['addr'],
                 'true': m['gt_name'], 'regime': m['regime'], 'SG': p, 'A4': a, 'R': d['r_pred']})
print(f'novel rows joined: {len(rows)}')
dem = demangle_many([r['true'] for r in rows] + [r[h] for r in rows for h in ('SG', 'A4', 'R')])
for r in rows:
    r['ct'] = canon(r['true'], dem); r['tt'] = split_name(r['ct'])
    for h in ('SG', 'A4', 'R'):
        cp = canon(r[h], dem) if r[h] else ''
        r['c' + h] = cp; r['f_' + h] = compute_subtoken_f1(cp, r['ct']) if r[h] else 0.0
        pt, tt = split_name(cp), r['tt']
        inter = sum((collections.Counter(pt) & collections.Counter(tt)).values())
        r['p_' + h] = inter / len(pt) if pt else 0.0
        r['r_' + h] = inter / len(tt) if tt else 0.0
        r['len_' + h] = len(pt)

rep = {'n': len(rows)}
# 1. head-to-head per package
pkg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for h in ('SG', 'A4', 'R'): pkg[r['pkg']][h].append(r['f_' + h])
tab = []
for p, d in pkg.items():
    tab.append({'pkg': p, 'n': len(d['SG']), 'SG': sum(d['SG'])/len(d['SG']), 'A4': sum(d['A4'])/len(d['A4']), 'R': sum(d['R'])/len(d['R'])})
tab.sort(key=lambda x: -x['n'])
rep['per_pkg'] = [{**t, 'SG': round(t['SG'], 3), 'A4': round(t['A4'], 3), 'R': round(t['R'], 3)} for t in tab]
rep['pkg_wins'] = {'A4>SG': sum(1 for t in tab if t['A4'] > t['SG'] + 0.005), 'SG>A4': sum(1 for t in tab if t['SG'] > t['A4'] + 0.005), 'tie': sum(1 for t in tab if abs(t['SG'] - t['A4']) <= 0.005)}
# 2. complementarity
em = lambda r, h: 1 if r['f_' + h] == 1.0 else 0
both = sum(1 for r in rows if em(r, 'SG') and em(r, 'A4'))
sg_only = sum(1 for r in rows if em(r, 'SG') and not em(r, 'A4'))
a4_only = sum(1 for r in rows if em(r, 'A4') and not em(r, 'SG'))
rep['em_overlap'] = {'both': both, 'SG_only': sg_only, 'A4_only': a4_only}
rep['oracle_A4_SG'] = round(sum(max(r['f_A4'], r['f_SG']) for r in rows) / len(rows), 4)
rep['oracle_A4_SG_R'] = round(sum(max(r['f_A4'], r['f_SG'], r['f_R']) for r in rows) / len(rows), 4)
rep['mean'] = {h: round(sum(r['f_' + h] for r in rows) / len(rows), 4) for h in ('SG', 'A4', 'R')}
# F1-level complementarity (not just EM)
rep['f1_wins'] = {'SG>A4': sum(1 for r in rows if r['f_SG'] > r['f_A4'] + 1e-9), 'A4>SG': sum(1 for r in rows if r['f_A4'] > r['f_SG'] + 1e-9), 'tie': sum(1 for r in rows if abs(r['f_SG'] - r['f_A4']) < 1e-9)}
# 3. error character
for h in ('SG', 'A4', 'R'):
    rep['char_' + h] = {'precision': round(sum(r['p_' + h] for r in rows) / len(rows), 4),
                        'recall': round(sum(r['r_' + h] for r in rows) / len(rows), 4),
                        'pred_len': round(sum(r['len_' + h] for r in rows) / len(rows), 2),
                        'uniq_preds': len({r['c' + h] for r in rows}),
                        'empty': sum(1 for r in rows if not r['c' + h])}
rep['gt_len'] = round(sum(len(r['tt']) for r in rows) / len(rows), 2)
for h in ('SG', 'A4'):
    toks = collections.Counter(t for r in rows for t in split_name(r['c' + h]))
    rep['top_tokens_' + h] = toks.most_common(20)
# 4. evidence-vs-prior on a sample: GT tokens present in the masked decomp both models read
random.seed(0); sample = random.sample(rows, min(6000, len(rows)))
by_bin = collections.defaultdict(list)
for r in sample: by_bin[r['binary']].append(r)
ev = []
for b, rs in by_bin.items():
    p = f'{WS}/symgen_v2/decomp/{b}.json'
    if not os.path.exists(p): continue
    dec = json.load(open(p))
    for r in rs:
        e = dec.get(r['addr'])
        if not e or 'code' not in e: continue
        code = e['code'].replace(e['ghidra_name'], ' ', 1).lower()
        cov = sum(1 for t in set(r['tt']) if t in code) / max(1, len(set(r['tt'])))
        ev.append((r, cov))
rep['evidence_sample_n'] = len(ev)
for lo, hi, tag in ((0.999, 2, 'full_evidence'), (0.5, 0.999, 'partial'), (0.001, 0.5, 'weak'), (-1, 0.001, 'zero_evidence')):
    sub = [r for r, c in ev if lo < c <= hi]
    if sub:
        rep['by_evidence_' + tag] = {'n': len(sub), 'share': round(len(sub)/len(ev), 3),
                                     **{h: round(sum(r['f_' + h] for r in sub)/len(sub), 4) for h in ('SG', 'A4', 'R')}}
# 5. qualitative dumps
ex = {'SG_only_EM': [], 'A4_only_EM': [], 'SG_big_win': [], 'A4_big_win': []}
for r in rows:
    if em(r, 'SG') and not em(r, 'A4') and len(ex['SG_only_EM']) < 40: ex['SG_only_EM'].append((r['pkg'], r['true'], r['cSG'], r['cA4']))
    if em(r, 'A4') and not em(r, 'SG') and len(ex['A4_only_EM']) < 40: ex['A4_only_EM'].append((r['pkg'], r['true'], r['cA4'], r['cSG']))
    if r['f_SG'] - r['f_A4'] > 0.6 and len(ex['SG_big_win']) < 25: ex['SG_big_win'].append((r['pkg'], r['true'], r['cSG'], r['cA4']))
    if r['f_A4'] - r['f_SG'] > 0.6 and len(ex['A4_big_win']) < 25: ex['A4_big_win'].append((r['pkg'], r['true'], r['cA4'], r['cSG']))
rep['examples'] = ex
json.dump(rep, open(f'{WS}/results/novel_head_analysis.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k not in ('per_pkg', 'examples')}, indent=1))
print('\nper-pkg (top 15 by n): pkg n SG A4 R')
for t in rep['per_pkg'][:15]: print(f"  {t['pkg']:<14} {t['n']:>7} {t['SG']:.3f} {t['A4']:.3f} {t['R']:.3f}")
print('\nEFFECT: novel_head_analysis ->', f'{WS}/results/novel_head_analysis.json')
