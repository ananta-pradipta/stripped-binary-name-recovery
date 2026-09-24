"""Category (seen/novel-known/novel-OOV) recomputation with the CANONICAL-name seen flag; per-category F1/EM for R, G, routed, SymGen, BLens."""
import csv, json, re, subprocess, sys, collections, os, pickle
import numpy as np, torch
sys.path.insert(0, "/project/hz79/_shared/cs785/dh2")
from src.evaluation.metrics import split_name
from collections import Counter
WS = "/project/hz79/_shared/cs785/dh2"; OUT = f"{WS}/results/c_lmemb_knn"
BLENS_LOG = sys.argv[1]
def prf(pred, true):
    pt, tt = split_name(pred) if pred else [], split_name(true)
    if not pt and not tt: return 1.0, 1.0, 1.0
    if not pt or not tt: return 0.0, 0.0, 0.0
    pc, tc = Counter(pt), Counter(tt); tp = sum((pc & tc).values())
    p = tp / sum(pc.values()); r = tp / sum(tc.values()); f = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f
def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith("_Z")}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(["c++filt"], input="\n".join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out
def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r"\(.*\)$", "", n); n = re.sub(r"<[^<>]*>", "", n); n = re.sub(r"<[^<>]*>", "", n)
    parts = n.split("::")[-2:] if "::" in n else [n]; return "_".join(p for p in parts if p)
def sg_clean(p):
    p = p.replace('</s>', '').strip(); p = re.sub(r'^The predicted function name is\s*', '', p).strip()
    return p.split()[0] if p else ''
proto = {}
for tier in ("val","test"):
    for l in open(f"{WS}/results/baseline_protocol_v2/{tier}.jsonl"):
        r = json.loads(l); proto[(tier, r["binary"], r["entry_addr"])] = (r["name"], r.get("regime","?"), bool(r.get("name_seen_in_train")), r.get("package") or r["binary"].split("_")[0])
train_names_raw = [json.loads(l)["name"] for l in open(f"{WS}/results/baseline_protocol_v2/train.jsonl")]
def load_preds(path):
    d = {}
    with open(path) as fh:
        for x in csv.DictReader(fh, delimiter="\t"):
            if x["tier"] in ("val","test"): d[(x["tier"], x["binary"], x["entry_addr"])] = (x["pred"], float(x["conf"]))
    return d
R = load_preds(f"{OUT}/preds.tsv"); A = load_preds(f"{WS}/results/a4_codet5p220m_modctx_dm_v1/val_test_preds.tsv")
dev = "cuda" if torch.cuda.is_available() else "cpu"
Etr = torch.load(f"{OUT}/train_emb.pt", weights_only=True).to(dev).float()
margins = {}
for tier in ("val","test"):
    rows_m = [json.loads(l) for l in open(f"{WS}/results/a4_modctx/{tier}.jsonl")]
    E = torch.load(f"{OUT}/{tier}_emb.pt", weights_only=True).to(dev).float()
    for i in range(0, E.shape[0], 2048):
        v, _ = (E[i:i+2048] @ Etr.T).topk(2, dim=1)
        for j in range(v.shape[0]):
            r = rows_m[i+j]; margins[(tier, r["binary"], r["addr"])] = float(v[j,0]-v[j,1])
keys = sorted(set(R) & set(A) & set(proto) & set(margins))
sg = {}
for k in range(34):
    meta = json.load(open(f"{WS}/symgen_v2/fulltest_shards/meta_{k}.json")); preds = json.load(open(f"{WS}/symgen_v2/results_fulltest/shard_{k}/predicted_function_name.json"))
    for m, p in zip(meta, preds):
        if p['ground_truth'] != m['gt_name']: continue
        sg[('test', m['binary'], int(m['addr'], 16))] = sg_clean(p['predicted_name'])
blens = {}
pairs = []; lines = open(BLENS_LOG, errors='ignore').read().splitlines()
for i, l in enumerate(lines):
    if l.startswith('target: ') and i + 1 < len(lines) and lines[i+1].startswith('output: '):
        pairs.append((l[8:].strip(), lines[i+1][8:].strip()))
test = pickle.load(open(f"{WS}/blens_ours_v2/xflBlensXProjectData", 'rb'))[2]
for (t, o), e in zip(pairs, test):
    b = os.path.basename(e[0]); b = b[:-9] if b.endswith('_stripped') else b
    blens[('test', b, int(e[1]))] = o
dem = demangle_many([proto[k][0] for k in keys] + [R[k][0] for k in keys] + [A[k][0] for k in keys] + list(sg.values()) + train_names_raw)
train_canon = set(canon(n, dem) for n in train_names_raw); train_raw = set(train_names_raw)
train_tokens = set(t.lower() for n in train_canon for t in split_name(n))
rows = []
for k in keys:
    ct = canon(proto[k][0], dem)
    pR = canon(R[k][0], dem) if R[k][0] else ""; pA = canon(A[k][0], dem) if A[k][0] else ""
    tt = [t.lower() for t in split_name(ct)]
    cat = "seen" if ct in train_canon else ("novel_known" if tt and all(t in train_tokens for t in tt) else "novel_oov")
    cat_raw = "seen" if proto[k][2] else ("novel_known" if tt and all(t in train_tokens for t in tt) else "novel_oov")
    e = {"tier": k[0], "pkg": proto[k][3], "regime": proto[k][1], "cat": cat, "seen_flag": proto[k][2], "raw_in_train": proto[k][0] in train_raw, "canon_in_train": ct in train_canon,
         "cat_raw": cat_raw, "X": [R[k][1], margins[k], A[k][1], R[k][1]-A[k][1]]}
    e["f_R"] = prf(pR, ct)[2]; e["em_R"] = (pR == ct); e["f_G"] = prf(pA, ct)[2]; e["em_G"] = (pA == ct)
    ik = ('test', k[1], int(k[2], 16)) if k[0] == 'test' else None
    if ik and ik in sg:
        cs = canon(sg[ik], dem) if sg[ik] else ""; e["f_SymGen"] = prf(cs, ct)[2]; e["em_SymGen"] = (cs == ct)
    if ik and ik in blens:
        cb = blens[ik].replace(" ", "_") if blens[ik] else ""; e["f_BLens"] = prf(cb, ct)[2]; e["em_BLens"] = (cb == ct)
    rows.append(e)
val = [r for r in rows if r["tier"]=="val"]; test_rows = [r for r in rows if r["tier"]=="test"]
Xv = np.array([r["X"] for r in val]); Xt = np.array([r["X"] for r in test_rows])
yv = np.array([1 if r["f_R"] >= r["f_G"] else 0 for r in val]); wv = np.abs(np.array([r["f_R"]-r["f_G"] for r in val])) + 1e-3
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(Xv)
idx = np.random.default_rng(42).choice(len(val), size=min(len(val)*3, 60000), p=wv/wv.sum())
mlp = MLPClassifier((64,32), max_iter=800, early_stopping=True, random_state=42).fit(sc.transform(Xv[idx]), yv[idx])
useR = mlp.predict(sc.transform(Xt)).astype(bool)
for r, u in zip(test_rows, useR): r["f_routed"] = r["f_R"] if u else r["f_G"]; r["em_routed"] = r["em_R"] if u else r["em_G"]
import collections as _c
out={}
for key in ("cat","cat_raw"):
    out[key]={}
    for cat in ("seen","novel_known","novel_oov"):
        rs=[r for r in test_rows if r[key]==cat]; row={"n":len(rs),"pct":100*len(rs)/len(test_rows)}
        for h in ("R","G","routed","SymGen","BLens"):
            hs=[r for r in rs if ("f_"+h) in r]
            if hs: row[h]={"f1":float(np.mean([r["f_"+h] for r in hs])),"em":float(np.mean([r["em_"+h] for r in hs])),"n":len(hs)}
        out[key][cat]=row
        print("%s %-12s n=%6d (%.1f%%) "%(key,cat,len(rs),row["pct"])+" ".join("%s F1 %.3f EM %.1f%%"%(h,row[h]["f1"],100*row[h]["em"]) for h in ("R","G","routed","SymGen","BLens") if h in row), flush=True)
moved=sum(1 for r in test_rows if r["cat"]!=r["cat_raw"]); print("rows recategorised:",moved)
print("retrieval EM on canonical-novel rows:", sum(1 for r in test_rows if r["cat"]!="seen" and r["em_R"]))
json.dump(out, open(f"{OUT}/canon_categories.json","w"), indent=1)
print("EFFECT rc=0 wrote canon_categories.json")
