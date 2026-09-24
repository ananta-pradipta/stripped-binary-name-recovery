#!/usr/bin/env python3
"""BLens v2 interim eval prep: labels conversion + bin list + nlpData pkl for the
same 7,532-function clean-FT sample used by SymGen interim-C."""
import json, os, pickle

WS = '$WORKSPACE/dh2/blens_v2'
META = '$WORKSPACE/dh2/symgen_v2/interim_c_metadata.json'
LBL = '$WORKSPACE/relift_ws/data/labels_v2'
STRIP = '$WORKSPACE/relift_ws/data/stripped_v2'
os.makedirs(WS + '/labels', exist_ok=True)
os.makedirs(WS + '/clap_jsons', exist_ok=True)

meta = json.load(open(META))
bins = sorted({m['binary'] for m in meta})
rows = []
for fid, m in enumerate(meta):
    rows.append([f"{STRIP}/{m['binary']}", int(m['addr'], 16), m['gt_name'], '',
                 int(m['addr'], 16), bins.index(m['binary']), fid])
pickle.dump([[], [], rows], open(f'{WS}/nlpData_v2ftc.pkl', 'wb'))
for b in bins:
    d = json.load(open(f'{LBL}/{b}.json'))
    fns = {name: v['addr'] for name, v in d['functions'].items()}
    json.dump({'functions': fns}, open(f'{WS}/labels/{b}.json', 'w'))
open(f'{WS}/bins.txt', 'w').write('\n'.join(f'{STRIP}/{b}' for b in bins) + '\n')
print('EFFECT: prep', len(rows), 'fns', len(bins), 'bins')
