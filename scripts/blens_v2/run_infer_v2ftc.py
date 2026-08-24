"""BLens inference on the v2 clean-FT sample (mirror of run_combo_fixed.py)."""
import os, sys, pickle
sys.path.insert(0, '/project/hz79/_shared/cs785/baselines/blens')
from builder import loadNLPData, loadData
from inferenceCOMBO import inferenceCOMBO
from inferenceLORD import inferenceLORD

WS = '/project/hz79/_shared/cs785/dh2/blens_v2'
DATA_DIR = '/project/hz79/_shared/cs785/baselines/blens_user_env/blens_data'
EXP_DIR = os.path.join(DATA_DIR, 'xp', 'ours-cp')
MODE = os.environ.get('MODE', 'lord')
EPOCH = int(os.environ.get('EPOCH', '59'))

nlpData = pickle.load(open(f'{WS}/nlpData_v2ftc.pkl', 'rb'))
with open(os.path.join(DATA_DIR, 'tokenizer', 'Tokenizer-Debin-1024-Projects'), 'rb') as f:
    tokenizer = pickle.load(f)
with open(os.path.join(EXP_DIR, 'params'), 'rb') as f:
    params = pickle.load(f)
listOfFEmbeddings = [('clap', True, pickle.load(open(f'{WS}/clap_v2ftc.pkl', 'rb'))),
                     ('palmtree', True, pickle.load(open(f'{WS}/palmtree_v2ftc.pkl', 'rb')))]
test = nlpData[2]
print(f'test size: {len(test)} mode: {MODE} epoch: {EPOCH}')
testData = loadData(test, tokenizer, params, listOfFEmbeddings)
import torch
for k in ('clap', 'palmtree'):
    if k in testData:
        t = testData[k]
        try:
            print(k, 'mean|v|', t.abs().mean().item())
        except Exception:
            pass
if MODE == 'combo':
    inferenceCOMBO(EXP_DIR, params, tokenizer, testData, bias=0.0)
else:
    inferenceLORD(EXP_DIR, params, tokenizer, testData, specialCode=f'v2ftc-{EPOCH}', bias=0.0, epoch=EPOCH)
print('EFFECT: infer done')
