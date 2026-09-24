#!/usr/bin/env python3
"""Score EXTRA systems (e.g. SymGen-34B LoRA retrained on Punstrip-train, BLens retrained by us) on the Punstrip/BLens
cross-project test split next to ours and the published baselines, on ONE common key set, with both scorers:
 (a) BLens's scorer (blens_scorer.score_preset full/strict; raw predictions -> their canonicaliser -> 1024-label vocab),
 (b) our sub-token F1 metric v2 (raw names for ours/extra; canonical space for published baselines = approximate).
Run with the blensnlp python (needs enchant/nltk for NLP.py). Env: OUT (dir with system_preds.tsv from punstrip_system.py).

  --extra NAME=path.tsv       TSV with header 'key\\tpred' (key = '<binary>_<addr>' as in punstrip/data/*.jsonl); repeatable
  --symgen-dir NAME=DIR       build the extra from SymGen shards: DIR/results_test/shard_k/predicted_function_name.json aligned
                              with punstrip/symgen/test_shards/meta_k.json (cleaning rule = dh2/scripts/matched_baselines.sg_clean)
  --blens-log NAME=LOGFILE    build the extra from a BLens LORD-inference-logs-test-*.txt (target:/output: pairs in the order of
                              punstrip/blens/data/xflBlensXProjectData[2]); output is ALREADY canonical -> flagged canonical
  --self-test                 add 'selftest: generation head' from our own predA column (must reproduce the main table row)
Writes OUT/score_report_extra_<tag>.json and prints EFFECT tables."""
import argparse, csv, json, os, re, sys, collections, pickle, signal
sys.path.insert(0, '/project/hz79/_shared/cs785/punstrip/scripts'); sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
sys.path.insert(0, '/project/hz79/_shared/cs785/baselines/blens')
import blens_scorer as B
from src.evaluation.metrics import compute_subtoken_f1
from NLP import NLP

P = '/project/hz79/_shared/cs785/punstrip'
OUT = os.environ['OUT']
VOCAB = set(json.load(open(f'{P}/manifest/blens_label_vocab_1024.json')))

ap = argparse.ArgumentParser()
ap.add_argument('--extra', action='append', default=[])
ap.add_argument('--symgen-dir', action='append', default=[])
ap.add_argument('--blens-log', action='append', default=[])
ap.add_argument('--self-test', action='store_true')
ap.add_argument('--tag', default='extra')
a = ap.parse_args()

# ---- ours + published csv, keyed (binPath, vaddr)
sysrows = [x for x in csv.DictReader(open(f'{OUT}/system_preds.tsv'), delimiter='\t') if x['tier'] == 'test']
ours = {(x['binpath'], int(x['vaddr'])): x for x in sysrows}
bykey = {f"{x['binary']}_{x['addr']}": (x['binpath'], int(x['vaddr'])) for x in sysrows}
csvrows = B.load_csv(); csvmap = {(r['binPath'], r['vaddr']): r for r in csvrows}
base_keys = [k for k in ours if k in csvmap]

# ---- extra systems -> {name: (dict key->pred, is_canonical)}
def sg_clean(p):
    p = p.replace('</s>', '').strip(); p = re.sub(r'^The predicted function name is\s*', '', p).strip()
    return p.split()[0] if p else ''
extra = {}
for spec in a.extra:
    name, path = spec.split('=', 1); d = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        if r['key'] in bykey: d[bykey[r['key']]] = r['pred']
    extra[name] = (d, False)
for spec in a.symgen_dir:
    name, dr = spec.split('=', 1); d = {}; n_shards = 0
    for k in range(64):
        mp, pp = f'{P}/symgen/test_shards/meta_{k}.json', f'{dr}/results_test/shard_{k}/predicted_function_name.json'
        if not os.path.exists(mp): break
        if not os.path.exists(pp): print(f'WARN {name}: shard {k} predictions missing', flush=True); continue
        meta, preds = json.load(open(mp)), json.load(open(pp))
        assert len(meta) == len(preds), (k, len(meta), len(preds))
        for m, p in zip(meta, preds):
            assert p['ground_truth'] == m['gt_name'], (k, m['key'], p['ground_truth'], m['gt_name'])   # alignment check
            if m['key'] in bykey: d[bykey[m['key']]] = sg_clean(p['predicted_name'])
        n_shards += 1
    print(f'EFFECT: {name}: {n_shards} shards, {len(d)} predictions joined', flush=True)
    extra[name] = (d, False)
for spec in a.blens_log:
    name, lf = spec.split('=', 1)
    recs = pickle.load(open(f'{P}/blens/data/xflBlensXProjectData', 'rb'))[2]
    lines = [l.rstrip('\n') for l in open(lf)]
    tg = [l[len('target:'):].strip() for l in lines if l.startswith('target:')]
    op = [l[len('output:'):].strip() for l in lines if l.startswith('output:')]
    assert len(tg) == len(op) == len(recs), (len(tg), len(op), len(recs))
    pref = f'{P}/rebuild/bins'; d = {}; agree = 0
    for r, t, o in zip(recs, tg, op):
        k = (r[0].replace(pref, '', 1), int(r[1]))
        if k in csvmap: agree += (t.replace(' ', '_') == csvmap[k]['groundtruth']); d[k] = o.replace(' ', '_')
    print(f'EFFECT: {name}: {len(d)} rows joined; target agreement with published GT {agree/max(1,len(d)):.4f}', flush=True)
    extra[name] = (d, True)
if a.self_test:
    extra['selftest: generation head'] = ({k: ours[k]['predA'] for k in base_keys}, False)

keys = [k for k in base_keys if all(k in d for d, _ in extra.values())]
print(f'EFFECT: keys: ours∩csv {len(base_keys)} | common with extras {len(keys)} ({len(keys)/len(base_keys):.1%})', flush=True)

# ---- canonicaliser with the digit-run guard (BLens recursive_split is exponential in digit-run length)
DIGITRUN = re.compile(r'\d{5,}')
class _CanonTimeout(BaseException): pass
def _alarm(*x): raise _CanonTimeout()
signal.signal(signal.SIGALRM, _alarm)
nlp = NLP(); cache = {}; timeouts = []
def canon(name):
    if name not in cache:
        raw = DIGITRUN.sub('_', name.replace('::', '_')) if name else ''
        signal.alarm(60)
        try: c = nlp.tristan_canonical_name(raw) if raw else ''
        except _CanonTimeout: c = ''; timeouts.append(name)
        except Exception: c = ''
        finally: signal.alarm(0)
        cache[name] = '_'.join(x for x in c.split('_') if x in VOCAB)
    return cache[name]
def to_label_space(pred, is_canon):
    return '_'.join(x for x in pred.split('_') if x in VOCAB) if is_canon else canon(pred)

def recs(getpred, is_canon):
    return [(k[0], k[1], csvmap[k]['name'], csvmap[k]['groundtruth'], to_label_space(getpred(k), is_canon)) for k in keys]
meta = {k: (k[0].split('/')[2], ours[k]['name_seen'] == '1', ours[k]['dynsym_visible'] == '1') for k in keys}
def our_metric(pairs):
    f = [compute_subtoken_f1(p, t) if p else 0.0 for p, t, *_ in pairs]
    pkg = collections.defaultdict(list)
    for (p, t, pk, seen, dyn), v in zip(pairs, f): pkg[pk].append(v)
    m = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {'micro': m(f), 'macro_pkg': m([m(v) for v in pkg.values()]),
            'seen': m([v for (p, t, pk, s, d), v in zip(pairs, f) if s]), 'novel': m([v for (p, t, pk, s, d), v in zip(pairs, f) if not s]),
            'excl_dynsym': m([v for (p, t, pk, s, d), v in zip(pairs, f) if not d]), 'n': len(f)}

systems = [('ours: system (router)', lambda k: ours[k]['pred'], False), ('ours: generation head', lambda k: ours[k]['predA'], False),
           ('ours: retrieval head', lambda k: ours[k]['predR'], False)]
systems += [(name, (lambda d: (lambda k: d[k]))(d), is_c) for name, (d, is_c) in extra.items()]
systems += [(f'{c} (published csv)', (lambda c: (lambda k: csvmap[k][c] if csvmap[k][c] != '<Not in the dataset>' else ''))(c), True)
            for c in ['BLens', 'XFL', 'SymLM', 'AsmDepictor']]
report = {'n_keys': len(keys)}
print(f'\n=== (a) BLens scorer, common key set n={len(keys)} ===\n{"system":40s} {"full F1":>8s} {"strict F1":>10s} {"n_strict":>8s}')
dup, ex = B.load_strict()
for label, gp, is_c in systems:
    R = recs(gp, is_c); full = B.score_preset(R, 'full'); st = B.score_preset(R, 'strict', dup, ex)
    report[label] = {'blens_full': full, 'blens_strict': st}
    print(f'{label:40s} {full["micro_f1"]:8.3f} {st["micro_f1"]:10.3f} {st["n"]:8d}', flush=True)
print(f'\n=== (b) our metric v2, same keys (ours/extra on raw names; published baselines + canonical extras in canonical space ≈) ===')
print(f'{"system":40s} {"micro":>7s} {"macro":>7s} {"seen":>7s} {"novel":>7s} {"excl-dyn":>9s}')
for label, gp, is_c in systems:
    pairs = [(gp(k), csvmap[k]['groundtruth'] if is_c else ours[k]['true'], *meta[k]) for k in keys]
    r = our_metric(pairs); report[label]['ours_v2' + ('_canonspace' if is_c else '')] = r
    print(f'{label:40s} {r["micro"]:7.4f} {r["macro_pkg"]:7.4f} {r["seen"]:7.4f} {r["novel"]:7.4f} {r["excl_dynsym"]:9.4f}', flush=True)
json.dump(report, open(f'{OUT}/score_report_extra_{a.tag}.json', 'w'), indent=1)
print(f'EFFECT: score_report_extra_{a.tag}.json written; canon timeouts {len(timeouts)}', flush=True)
