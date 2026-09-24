#!/usr/bin/env python3
"""BLens fair retrain — data prep on OUR frozen v2 protocol (train/val/test tiers).
Writes <OUT>/xflBlensXProjectData_ours_v2 = [train_rows, val_rows, test_rows] in BLens's row format
  [stripped_elf_path, addr(int), name, name, addr(int), binary_index, function_index]
plus labels/<bin>.json and bins.txt for the Ghidra/CLAP/PalmTree embedding chain (blens_v2/*.sbatch pattern).
Embeddings must then be generated for every binary listed in bins.txt (train+val+test)."""
import json, os, pickle
PROTO = '$WORKSPACE/dh2/results/baseline_protocol_v2'
LBL = '$WORKSPACE/relift_ws/data/labels_v2'
STRIP = '$WORKSPACE/relift_ws/data/stripped_v2'
OUT = '$WORKSPACE/dh2/blens_ours_v2'
os.makedirs(OUT + '/labels', exist_ok=True)
tiers = {t: [json.loads(l) for l in open(f'{PROTO}/{t}.jsonl')] for t in ('train', 'val', 'test')}
bins = sorted({r['binary'] for rows in tiers.values() for r in rows})
bidx = {b: i for i, b in enumerate(bins)}
parts = []; fid = 0
for t in ('train', 'val', 'test'):
    rows = []
    for r in tiers[t]:
        a = int(r['entry_addr'], 16)
        rows.append([f"{STRIP}/{r['binary']}", a, r['name'], r['name'], a, bidx[r['binary']], fid]); fid += 1
    parts.append(rows)
os.makedirs(OUT + '/embedding', exist_ok=True)
pickle.dump(parts, open(f'{OUT}/xflBlensXProjectData', 'wb'))   # RunExp.py hardcodes this filename under -data-dir; embeddings go to <OUT>/embedding/{clap,palmtree}
for b in bins:
    d = json.load(open(f'{LBL}/{b}.json'))
    json.dump({'functions': {n: v['addr'] for n, v in d['functions'].items()}}, open(f'{OUT}/labels/{b}.json', 'w'))
open(f'{OUT}/bins.txt', 'w').write('\n'.join(f'{STRIP}/{b}' for b in bins) + '\n')
print(f'EFFECT: blens ours_v2 prep: train {len(parts[0])} val {len(parts[1])} test {len(parts[2])} fns; {len(bins)} binaries -> {OUT}')
