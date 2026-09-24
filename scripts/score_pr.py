"""Precision / recall / F1 / EM (function-level means and package-level means) for the ADOPTED single-backbone system
(R, G=A, routed via MLP router as in per_pkg_single_backbone.py), SymGen-34B (fulltest shards) and BLens (LORD log),
on the joined test keys. Usage: python3 score_pr.py <blens_lord_log> -> results/c_lmemb_knn/score_pr.json"""
import csv, json, re, subprocess, sys, collections, os, pickle
import numpy as np, torch
sys.path.insert(0, "/project/hz79/_shared/cs785/dh2")
from src.evaluation.metrics import split_name
from collections import Counter
WS = "/project/hz79/_shared/cs785/dh2"; OUT = f"{WS}/results/c_lmemb_knn"
BLENS_LOG = sys.argv[1] if len(sys.argv) > 1 else None
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
print("our keys", len(keys), flush=True)
# SymGen
sg = {}
for k in range(34):
    meta = json.load(open(f"{WS}/symgen_v2/fulltest_shards/meta_{k}.json")); preds = json.load(open(f"{WS}/symgen_v2/results_fulltest/shard_{k}/predicted_function_name.json"))
    assert len(preds) == len(meta), (k, len(preds), len(meta))
    for m, p in zip(meta, preds):
        if p['ground_truth'] != m['gt_name']: continue
        sg[('test', m['binary'], int(m['addr'], 16))] = sg_clean(p['predicted_name'])
print("symgen keys", len(sg), flush=True)
# BLens
blens = {}
if BLENS_LOG:
    pairs = []; lines = open(BLENS_LOG, errors='ignore').read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith('target: ') and i + 1 < len(lines) and lines[i+1].startswith('output: '):
            pairs.append((l[8:].strip(), lines[i+1][8:].strip()))
    test = pickle.load(open(f"{WS}/blens_ours_v2/xflBlensXProjectData", 'rb'))[2]
    assert len(pairs) == len(test), (len(pairs), len(test))
    for (t, o), e in zip(pairs, test):
        b = os.path.basename(e[0]); b = b[:-9] if b.endswith('_stripped') else b
        blens[('test', b, int(e[1]))] = o
print("blens keys", len(blens), flush=True)
dem = demangle_many([proto[k][0] for k in keys] + [R[k][0] for k in keys] + [A[k][0] for k in keys] + list(sg.values()))
rows = []
for k in keys:
    ct = canon(proto[k][0], dem)
    pR = canon(R[k][0], dem) if R[k][0] else ""; pA = canon(A[k][0], dem) if A[k][0] else ""
    e = {"tier": k[0], "pkg": proto[k][3], "regime": proto[k][1], "X": [R[k][1], margins[k], A[k][1], R[k][1]-A[k][1]]}
    for h, p in (("R", pR), ("G", pA)):
        e["p_"+h], e["r_"+h], e["f_"+h] = prf(p, ct); e["em_"+h] = (p == ct)
    ik = ('test', k[1], int(k[2], 16)) if k[0] == 'test' else None
    e["has_sg"] = ik in sg if ik else False; e["has_bl"] = ik in blens if ik else False
    if e["has_sg"]:
        p = canon(sg[ik], dem) if sg[ik] else ""; e["p_SymGen"], e["r_SymGen"], e["f_SymGen"] = prf(p, ct); e["em_SymGen"] = (p == ct)
    if e["has_bl"]:
        p = blens[ik].replace(" ", "_") if blens[ik] else ""; e["p_BLens"], e["r_BLens"], e["f_BLens"] = prf(p, ct); e["em_BLens"] = (p == ct)
    rows.append(e)
val = [r for r in rows if r["tier"]=="val"]; test = [r for r in rows if r["tier"]=="test"]
Xv = np.array([r["X"] for r in val]); Xt = np.array([r["X"] for r in test])
yv = np.array([1 if r["f_R"] >= r["f_G"] else 0 for r in val]); wv = np.abs(np.array([r["f_R"]-r["f_G"] for r in val])) + 1e-3
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(Xv)
idx = np.random.default_rng(42).choice(len(val), size=min(len(val)*3, 60000), p=wv/wv.sum())
mlp = MLPClassifier((64,32), max_iter=800, early_stopping=True, random_state=42).fit(sc.transform(Xv[idx]), yv[idx])
useR = mlp.predict(sc.transform(Xt)).astype(bool)
for r, u in zip(test, useR):
    h = "R" if u else "G"
    for m in ("p", "r", "f", "em"): r[m+"_routed"] = r[m+"_"+h]
def agg(rs, h):
    rs = [r for r in rs if ("f_"+h) in r]
    if not rs: return None
    pk = collections.defaultdict(list)
    for r in rs: pk[r["pkg"]].append(r)
    fn = {m: float(np.mean([r[m+"_"+h] for r in rs])) for m in ("p","r","f","em")}
    pkg = {m: float(np.mean([np.mean([r[m+"_"+h] for r in g]) for g in pk.values()])) for m in ("p","r","f","em")}
    return {"n": len(rs), "pkgs": len(pk), "fn": fn, "pkg": pkg}
common = [r for r in test if r["has_sg"] and r["has_bl"]]
out = {"n_test": len(test), "n_common": len(common), "blens_log": BLENS_LOG}
for name, rs in (("all_ours", test), ("common", common), ("common_FT", [r for r in common if r["regime"]=="FT"]), ("common_NCT", [r for r in common if r["regime"]=="NCT"])):
    out[name] = {h: agg(rs, h) for h in ("R","G","routed","SymGen","BLens")}
    print("===", name, "n=%d" % len(rs))
    for h, v in out[name].items():
        if v: print("  %-7s fn P %.4f R %.4f F1 %.4f EM %.4f | pkg P %.4f R %.4f F1 %.4f EM %.4f (%d fns, %d pkgs)" % (h, v["fn"]["p"], v["fn"]["r"], v["fn"]["f"], v["fn"]["em"], v["pkg"]["p"], v["pkg"]["r"], v["pkg"]["f"], v["pkg"]["em"], v["n"], v["pkgs"]), flush=True)
json.dump(out, open(f"{OUT}/score_pr.json", "w"), indent=1)
print("EFFECT rc=0 wrote", f"{OUT}/score_pr.json")
