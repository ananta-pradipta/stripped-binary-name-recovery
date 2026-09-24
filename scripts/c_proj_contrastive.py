#!/usr/bin/env python3
"""Name-supervised contrastive PROJECTION head for retrieval (user-approved 2026-09-03).
Motivation: selection-ceiling census — best-possible-copy F1 over train names is 0.505 on novel
rows where top-1 cosine currently gets 0.034 (gap 0.47). The generation-shaped embedding clusters
code clones; this trains a small MLP projection ON TOP OF THE FROZEN pooled embeddings so that
functions with token-overlapping NAMES embed nearby (soft supervised-contrastive, weights = name
token F1). Backbone untouched -> single-backbone story intact, seen-name lookup protected (raw
space still available; final system may route between spaces, but here we evaluate projected-only
vs raw-only first). Val-gated. Outputs preds tsv in a4_predict format + system numbers with the
adopted MLP router and dm generation head, unified canon scoring throughout."""
import csv, json, re, subprocess, sys, collections
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, '$WORKSPACE/dh2')
from src.evaluation.metrics import compute_subtoken_f1, split_name
WS = '$WORKSPACE/dh2'
SRC = f'{WS}/results/c_lmemb_knn'
OUT = f'{WS}/results/c_proj_contrastive'
import os; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)
dev = 'cuda'

def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out
def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)

# ---- load frozen pooled embeddings + rows
def rows_of(tier): return [json.loads(l) for l in open(f'{WS}/results/a4_modctx/{tier}.jsonl')]
tr_rows, va_rows, te_rows = rows_of('train'), rows_of('val'), rows_of('test')
Etr = torch.load(f'{SRC}/train_emb.pt', weights_only=True).float()
Eva = torch.load(f'{SRC}/val_emb.pt', weights_only=True).float()
Ete = torch.load(f'{SRC}/test_emb.pt', weights_only=True).float()
assert Etr.shape[0] == len(tr_rows) and Eva.shape[0] == len(va_rows) and Ete.shape[0] == len(te_rows)

dem = demangle_many([r['name'] for r in tr_rows + va_rows + te_rows])
def toks_of(name): return sorted(set(t.lower() for t in split_name(canon(name, dem))))
vocab = {}
def tok_ids(name):
    ids = []
    for t in toks_of(name):
        if t not in vocab: vocab[t] = len(vocab)
        ids.append(vocab[t])
    return ids
tr_tok = [tok_ids(r['name']) for r in tr_rows]
V = len(vocab); print(f'EFFECT: name-token vocab {V}', flush=True)

# ---- projection model + soft SupCon training on frozen embeddings
proj = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 256)).to(dev)
opt = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=1e-4)
Etr_g = Etr.to(dev)
N = Etr_g.shape[0]; BS = 1024; TAU = 0.07; STEPS = 3000
def name_f1_matrix(idx):
    B = torch.zeros(len(idx), V, device=dev)
    for j, i in enumerate(idx):
        ids = tr_tok[i]
        if ids: B[j, ids] = 1.0
    inter = B @ B.T; sz = B.sum(1)
    f1 = 2 * inter / (sz[:, None] + sz[None, :]).clamp(min=1)
    f1.fill_diagonal_(0)
    return f1
def val_probe(space_tr, space_q, q_rows, sub=4000):
    qs = space_q[:sub].to(dev); sims = qs @ space_tr.T
    top = sims.argmax(1).tolist()
    f1s = []
    for j, t in enumerate(top):
        ct = canon(q_rows[j]['name'], dem); pn = canon(tr_rows[t]['name'], dem)
        f1s.append(compute_subtoken_f1(pn, ct))
    return sum(f1s)/len(f1s)
raw_val = val_probe(torch.nn.functional.normalize(Etr_g, dim=-1), torch.nn.functional.normalize(Eva, dim=-1), va_rows)
print(f'EFFECT: RAW-space val probe (4K) micro {raw_val:.4f}', flush=True)
best = (-1, None)
for step in range(1, STEPS+1):
    idx = np.random.choice(N, BS, replace=False)
    W = name_f1_matrix(idx)
    Z = torch.nn.functional.normalize(proj(Etr_g[idx]), dim=-1)
    logits = (Z @ Z.T) / TAU
    logits.fill_diagonal_(-1e4)
    rowmass = W.sum(1)
    keep = rowmass > 0
    if keep.sum() == 0: continue
    logp = torch.log_softmax(logits[keep], dim=1)
    target = W[keep] / rowmass[keep][:, None]
    loss = -(target * logp).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        with torch.no_grad():
            Ptr = torch.nn.functional.normalize(proj(Etr_g), dim=-1)
            Pva = torch.nn.functional.normalize(proj(Eva.to(dev)), dim=-1)
            v = val_probe(Ptr, Pva, va_rows)
        print(f'  step {step} loss {loss.item():.4f} val-probe {v:.4f}', flush=True)
        if v > best[0]: best = (v, {k: p.detach().clone() for k, p in proj.state_dict().items()})
proj.load_state_dict(best[1]); print(f'EFFECT: best val-probe {best[0]:.4f} (raw {raw_val:.4f})', flush=True)
torch.save(proj.state_dict(), f'{OUT}/proj.pt')

# ---- full kNN in projected space -> preds tsv (a4_predict format)
with torch.no_grad():
    Ptr = torch.nn.functional.normalize(proj(Etr_g), dim=-1)
    names_tr = [r['name'] for r in tr_rows]
    with open(f'{OUT}/preds.tsv', 'w') as fh:
        fh.write('tier\tbinary\tentry_addr\ttrue\tpred\tregime\tname_seen\tf1_raw\tf1_v2\tconf\n')
        for tier, E, rs in (('val', Eva, va_rows), ('test', Ete, te_rows)):
            for i in range(0, len(rs), 4096):
                Q = torch.nn.functional.normalize(proj(E[i:i+4096].to(dev)), dim=-1)
                v2, ix = (Q @ Ptr.T).topk(2, dim=1)
                for j in range(Q.shape[0]):
                    r = rs[i+j]; pn = names_tr[ix[j,0]]
                    seen = bool(r.get('name_seen_in_train', r.get('name_seen')))
                    fh.write(f"{tier}\t{r['binary']}\t{r['addr']}\t{r['name']}\t{pn}\t{r.get('regime','?')}\t{int(seen)}\t0\t0\t{v2[j,0].item():.4f}\n")
                    fh.write('') if True else None
print('EFFECT: projected preds written', flush=True)
# margins file for router
marg = {}
with torch.no_grad():
    for tier, E, rs in (('val', Eva, va_rows), ('test', Ete, te_rows)):
        for i in range(0, len(rs), 4096):
            Q = torch.nn.functional.normalize(proj(E[i:i+4096].to(dev)), dim=-1)
            v2, _ = (Q @ Ptr.T).topk(2, dim=1)
            for j in range(Q.shape[0]):
                r = rs[i+j]; marg[(tier, r['binary'], r['addr'])] = (float(v2[j,0]), float(v2[j,0]-v2[j,1]))
# ---- unified scoring + MLP-routed system vs dm generation head
def load_preds(path):
    d = {}
    with open(path) as fh:
        for x in csv.DictReader(fh, delimiter='\t'):
            if x['tier'] in ('val','test'): d[(x['tier'], x['binary'], x['entry_addr'])] = (x['pred'], float(x['conf']))
    return d
proto = {}
for tier in ('val','test'):
    for l in open(f'{WS}/results/baseline_protocol_v2/{tier}.jsonl'):
        r = json.loads(l); proto[(tier, r['binary'], r['entry_addr'])] = (r['name'], r.get('regime','?'), bool(r.get('name_seen_in_train')))
Rp = load_preds(f'{OUT}/preds.tsv'); A = load_preds(f'{WS}/results/a4_codet5p220m_modctx_dm_v1/val_test_preds.tsv')
keys = sorted(set(Rp) & set(A) & set(proto) & set(marg))
dem2 = demangle_many([proto[k][0] for k in keys] + [Rp[k][0] for k in keys] + [A[k][0] for k in keys])
rows = []
for k in keys:
    ct = canon(proto[k][0], dem2)
    fR = compute_subtoken_f1(canon(Rp[k][0], dem2), ct) if Rp[k][0] else 0.0
    fA = compute_subtoken_f1(canon(A[k][0], dem2), ct) if A[k][0] else 0.0
    s, m = marg[k]
    rows.append({'tier': k[0], 'pkg': k[1].split('_')[0], 'regime': proto[k][1], 'seen': proto[k][2],
                 'fR': fR, 'fA': fA, 'emR': canon(Rp[k][0], dem2)==ct, 'X': [s, m, A[k][1], s-A[k][1]]})
val = [r for r in rows if r['tier']=='val']; test = [r for r in rows if r['tier']=='test']
def strat(rs, f, e=None):
    o = {}
    for s, sub in (('all', rs), ('seen', [r for r in rs if r['seen']]), ('novel', [r for r in rs if not r['seen']]),
                   ('FT', [r for r in rs if r['regime']=='FT']), ('NCT', [r for r in rs if r['regime']=='NCT'])):
        if sub: o[s] = round(sum(r[f] for r in sub)/len(sub), 4)
    return o
print('EFFECT: PROJECTED retrieval head val', strat(val, 'fR'), flush=True)
print('EFFECT: PROJECTED retrieval head test', strat(test, 'fR'), flush=True)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
Xv = np.array([r['X'] for r in val]); Xt = np.array([r['X'] for r in test])
yv = np.array([1 if r['fR'] >= r['fA'] else 0 for r in val])
wv = np.abs(np.array([r['fR']-r['fA'] for r in val])) + 1e-3
sc = StandardScaler().fit(Xv)
idx2 = np.random.default_rng(42).choice(len(val), size=min(len(val)*3, 60000), p=wv/wv.sum())
mlp = MLPClassifier((64,32), max_iter=800, early_stopping=True, random_state=42).fit(sc.transform(Xv[idx2]), yv[idx2])
for label, rs, X in (('val', val, Xv), ('test', test, Xt)):
    useR = mlp.predict(sc.transform(X)).astype(bool)
    for r, u in zip(rs, useR): r['fU'] = r['fR'] if u else r['fA']
    o = round(sum(max(r['fR'], r['fA']) for r in rs)/len(rs), 4)
    pkg = collections.defaultdict(list)
    for r in rs: pkg[r['pkg']].append(r['fU'])
    mac = round(sum(sum(v)/len(v) for v in pkg.values())/len(pkg), 4)
    print(f"EFFECT: SYSTEM(proj-R + dm) {label} routed micro {round(sum(r['fU'] for r in rs)/len(rs),4)} macro {mac} R_rate {float(np.mean(useR)):.3f} oracle {o}", flush=True)
print('EFFECT: done', flush=True)
