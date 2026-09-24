import json, os, sys
WS="/project/hz79/_shared/cs785/dh2/blens_ours_v2"; K=int(sys.argv[1]) if len(sys.argv)>1 else 24
bins=[l.strip() for l in open(f"{WS}/bins.txt") if l.strip()]
def nfn(p):
    try: return len(json.load(open(f"{WS}/clap_jsons/{os.path.basename(p)}.clap.json")))
    except Exception: return 0
load=sorted(((nfn(b),b) for b in bins), reverse=True)
chunks=[[] for _ in range(K)]; tot=[0]*K
for n,b in load:                      # LPT: heaviest first into the lightest chunk
    i=min(range(K), key=lambda j: tot[j]); chunks[i].append(b); tot[i]+=n
os.makedirs(f"{WS}/pt_chunks", exist_ok=True)
for i,c in enumerate(chunks):
    open(f"{WS}/pt_chunks/chunk_{i:02d}.txt","w").write("\n".join(c)+"\n")
print("chunks",K,"bins",sum(map(len,chunks)),"fns",sum(tot),"max load",max(tot),"min load",min(tot))
for i in range(K): print(i, len(chunks[i]), tot[i], os.path.basename(chunks[i][0]))
