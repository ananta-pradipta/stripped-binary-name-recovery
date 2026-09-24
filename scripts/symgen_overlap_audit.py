import json, re, os
E = "/course/2026/spring/cs/785/ACCOUNT/USER/cs785/baselines/SymGen/zenodo/extracted/x86_64"
roles = json.load(open("$WORKSPACE/dh2/data/split_v2.json"))["meta"]["roles"]
base = {}
for pkg, role in roles.items():
    k = re.sub(r"\d+$", "", pkg.lower())
    base.setdefault(k, set()).add(role)
def norm(p):
    p = p.lower().replace("openssl-openssl", "openssl")
    return re.sub(r"[-_]\d[\d.]*[a-z]?$", "", p)
sg = sorted(os.listdir(E + "/O0"))
tiers = {}
for p in sg:
    n = norm(p)
    hit = {k: sorted(base[k]) for k in base if k == n or (len(n) > 4 and (k.startswith(n) or n.startswith(k)))}
    tag = "new" if not hit else "OVERLAP " + json.dumps(hit)
    print(f"{p:<28} -> {n:<14} {tag}")
    for k, rs in hit.items():
        for r in rs: tiers.setdefault(r, set()).add(p)
print("SUMMARY:", {r: sorted(v) for r, v in tiers.items()})
print("new projects:", [p for p in sg if not any(k == norm(p) or (len(norm(p)) > 4 and (k.startswith(norm(p)) or norm(p).startswith(k))) for k in base)])
