"""CLAP encode for the v2 clean-FT sample; keys = (full binPath, vaddr) as builder.loadData expects."""
import json, pickle, os
from transformers import AutoTokenizer, AutoModel
import torch, numpy as np

os.environ.setdefault("HF_HOME", "$WORKSPACE/baselines/hf_cache")
WS = '$WORKSPACE/dh2/blens_ours_v2'
NLP_PATH = f'{WS}/xflBlensXProjectData'
CLAP_DIR = f'{WS}/clap_jsons'
OUT = f'{WS}/embedding/clap'
BATCH = 16
PIE_OFFSET = 0x100000

tok = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)
model = AutoModel.from_pretrained("hustcw/clap-asm", trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

nlp = pickle.load(open(NLP_PATH, 'rb'))
test = nlp[0] + nlp[1] + nlp[2]   # ours_v2: embed train+val+test (retrain needs all tiers)
by_bin = {}
for e in test:
    by_bin.setdefault(e[0], []).append(e[1])
embeddings = {}
for bin_path, vaddrs in sorted(by_bin.items()):
    cj = os.path.join(CLAP_DIR, os.path.basename(bin_path) + '.clap.json')
    if not os.path.exists(cj):
        print('MISSING', bin_path); continue
    by_entry = {e[0]: e[1] for e in json.load(open(cj))}
    direct = sum(1 for v in vaddrs if v in by_entry)
    shifted = sum(1 for v in vaddrs if v + PIE_OFFSET in by_entry)
    use_shift = shifted > direct
    vin = [(v, v + PIE_OFFSET if use_shift else v) for v in vaddrs]
    vin = [(v, p) for v, p in vin if p in by_entry]
    print(os.path.basename(bin_path), len(vaddrs), 'direct', direct, 'shifted', shifted)
    for i in range(0, len(vin), BATCH):
        bt = vin[i:i + BATCH]
        funcs = [by_entry[p] for _, p in bt]
        with torch.no_grad():
            enc = tok(funcs, padding=True, return_tensors='pt').to(device)
            out = model(**enc)
        embs = out.cpu().numpy() if hasattr(out, 'cpu') else out.last_hidden_state.cpu().numpy()
        for (v, _), e in zip(bt, embs):
            embeddings[(bin_path, v)] = torch.tensor(e, dtype=torch.float)
print('EFFECT: clap embeddings', len(embeddings), '/', len(test))
pickle.dump(embeddings, open(OUT, 'wb'))
