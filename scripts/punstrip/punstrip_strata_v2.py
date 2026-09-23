"""Punstrip test strata (seen / novel-known / novel-OOV vs Punstrip-train names) for our heads, the published baselines,
and the SymGen-34B LoRA (Punstrip-train) predictions; our sub-token F1 + case-sensitive canonical EM on the 22,926 joined keys."""
import csv, json, re, sys, collections
sys.path.insert(0, "/project/hz79/_shared/cs785/dh2")
from src.evaluation.metrics import compute_subtoken_f1, split_name
P="/project/hz79/_shared/cs785/punstrip"
def canon(n):
    n=re.sub(r"\(.*\)$","",n); n=re.sub(r"<[^<>]*>","",n); parts=n.split("::")[-2:] if "::" in n else [n]; return "_".join(p for p in parts if p)
try:
    sys.path.insert(0, "/project/hz79/_shared/cs785/dh2/scripts"); from matched_baselines import sg_clean
    print("sg_clean imported")
except Exception as ex:
    print("sg_clean fallback:", ex)
    def sg_clean(p):
        p=p.replace('</s>','').strip(); p=re.sub(r'^The predicted function name is\s*','',p).strip(); return p
train_tokens=set(); train_names=set()
for l in open(f"{P}/data/train.jsonl"):
    d=json.loads(l); n=canon(d["name"]); train_names.add(n); train_tokens.update(t.lower() for t in split_name(n))
print("train names",len(train_names),"tokens",len(train_tokens))
csvmap={}
for r in csv.DictReader(open("/project/hz79/_shared/cs785/baselines/blens/evaluation/cross-project.csv")):
    csvmap[(r["binPath"],int(r["vaddr"]))]=r
rows=[r for r in csv.DictReader(open(f"{P}/results/system/system_preds.tsv"),delimiter="\t") if r["tier"]=="test"]
# SymGen: shards aligned with meta; key '<binary>_<addr>' -> (binpath, vaddr) via our rows
bykey={f"{x['binary']}_{x['addr']}":(x['binpath'],int(x['vaddr'])) for x in rows}
sg={}
for k in range(3):
    meta=json.load(open(f"{P}/symgen/test_shards/meta_{k}.json")); preds=json.load(open(f"{P}/symgen/results_test/shard_{k}/predicted_function_name.json"))
    assert len(meta)==len(preds), (k,len(meta),len(preds))
    for m,p in zip(meta,preds):
        if m["key"] in bykey: sg[bykey[m["key"]]]=sg_clean(p["predicted_name"])
print("symgen joined",len(sg))
BASE=["BLens","XFL","SymLM","AsmDepictor"]
out=[]
for r in rows:
    k=(r["binpath"],int(r["vaddr"]))
    if k not in csvmap: continue
    t=canon(r["true"]); toks=[x.lower() for x in split_name(t)]
    cat="seen" if t in train_names else ("novel_known" if toks and all(x in train_tokens for x in toks) else "novel_oov")
    e={"cat":cat,"pkg":r["binpath"].split("/")[2]}
    for h,p in (("R",r["predR"]),("G",r["predA"]),("routed",r["pred"]),("SymGen",sg.get(k,""))):
        cp=canon(p) if p else ""; e["f_"+h]=compute_subtoken_f1(cp,t) if cp else 0.0; e["em_"+h]=(cp==t)
    c=csvmap[k]; gt=c["groundtruth"]
    for b in BASE:
        p=c[b]; ok=p not in ("","<Not in the dataset>")
        e["f_"+b]=compute_subtoken_f1(p.replace(" ","_"),gt.replace(" ","_")) if ok else 0.0; e["em_"+b]=ok and p==gt
    out.append(e)
print("joined",len(out),"symgen missing among joined",sum(1 for r in rows if (r["binpath"],int(r["vaddr"])) in csvmap and (r["binpath"],int(r["vaddr"])) not in sg))
def agg(rs,h):
    n=len(rs); return sum(x["f_"+h] for x in rs)/n, sum(x["em_"+h] for x in rs)/n
rep={}
for cat in ("seen","novel_known","novel_oov","all"):
    rs=out if cat=="all" else [x for x in out if x["cat"]==cat]; print("=== %s n=%d (%.1f%%)"%(cat,len(rs),100*len(rs)/len(out))); rep[cat]={"n":len(rs)}
    for h in ("R","G","routed","SymGen")+tuple(BASE):
        f,em=agg(rs,h); print("   %-12s F1 %.3f EM %.1f%%"%(h,f,100*em)); rep[cat][h]={"f1":f,"em":em}
json.dump(rep,open(f"{P}/results/system/punstrip_strata_v2.json","w"),indent=1)
print("EFFECT rc=0 wrote results/system/punstrip_strata_v2.json")
