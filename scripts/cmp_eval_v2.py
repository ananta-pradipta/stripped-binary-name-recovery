import json, sys
tags = sys.argv[1:]
for tag in tags:
    r = json.load(open(f"/project/hz79/_shared/cs785/dh2/results/{tag}_eval_greedy.json"))
    for t, d in r["tiers"].items():
        s = d["scored"]; ft = d["by_regime"]["FT"]; nct = d["by_regime"]["NCT"]; ns = d["by_name_stratum"]
        print(f"{tag:<4} {t:<4} micro {s['f1']:.4f} EM {s['em']:.4f} macro {d['macro_pkg']['f1']:.4f} | FT {ft['micro'].get('f1',0):.4f} (macro {ft['macro_pkg'].get('f1',0):.4f}) NCT {nct['micro'].get('f1',0):.4f} (macro {nct['macro_pkg'].get('f1',0):.4f}) | seen {ns['seen_name']['f1']:.4f} novel {ns['novel_name']['f1']:.4f} n={s['n']}")
if len(tags) == 2:
    a = json.load(open(f"/project/hz79/_shared/cs785/dh2/results/{tags[0]}_eval_greedy.json"))["tiers"]["test"]["per_pkg"]
    b = json.load(open(f"/project/hz79/_shared/cs785/dh2/results/{tags[1]}_eval_greedy.json"))["tiers"]["test"]["per_pkg"]
    diffs = sorted(((b[p]["f1"] - a[p]["f1"], p, a[p]["f1"], b[p]["f1"], a[p]["n"]) for p in a if p in b))
    print("largest drops (pkg, A1a, A3+, n):", [(p, round(x, 3), round(y, 3), n) for d, p, x, y, n in diffs[:8]])
    print("largest gains:", [(p, round(x, 3), round(y, 3), n) for d, p, x, y, n in diffs[-5:]])
    print("pkgs better/worse:", sum(1 for d, *_ in diffs if d > 0), sum(1 for d, *_ in diffs if d < 0))
