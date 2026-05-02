"""Inference with CORRECT test-split embeddings (clap_test + palmtree_test)."""
import os, sys, pickle, json
sys.path.insert(0, '<shared-project-root>/baselines/blens')

from builder import loadNLPData, loadData
from inferenceCOMBO import inferenceCOMBO
from inferenceLORD import inferenceLORD

DATA_DIR = '<shared-project-root>/baselines/blens_user_env/blens_data'
EXP_DIR = os.path.join(DATA_DIR, 'xp', 'ours-cp')
MODE = os.environ.get('MODE', 'combo')
EPOCH = int(os.environ.get('EPOCH', '59'))

nlpData = loadNLPData(os.path.join(DATA_DIR, 'xflBlensXProjectData'))
with open(os.path.join(DATA_DIR, 'tokenizer', 'Tokenizer-Debin-1024-Projects'), 'rb') as f:
    tokenizer = pickle.load(f)
with open(os.path.join(EXP_DIR, 'params'), 'rb') as f:
    params = pickle.load(f)

# Load TEST-split embeddings (the per-split file with full-path keys, 100% coverage)
listOfFEmbeddings = []
with open(os.path.join(DATA_DIR, 'embedding', 'clap_test'), 'rb') as f:
    listOfFEmbeddings.append(('clap', True, pickle.load(f)))
with open(os.path.join(DATA_DIR, 'embedding', 'palmtree_test'), 'rb') as f:
    listOfFEmbeddings.append(('palmtree', True, pickle.load(f)))

test = nlpData[2]
print(f'test size: {len(test)}  mode: {MODE}  epoch: {EPOCH}')
testData = loadData(test, tokenizer, params, listOfFEmbeddings)
print('testData loaded.')

# Verify test data has non-zero clap embeddings
import torch
zero_count = sum(1 for d in testData['clap'] if torch.all(d == 0).item()) if 'clap' in testData else 'n/a'
# testData is a dict of stacked tensors, check non-zero magnitude instead
if 'clap' in testData:
    mag = testData['clap'].abs().mean().item()
    n_zero = (testData['clap'].abs().sum(dim=-1) < 1e-6).sum().item()
    print(f'CLAP embeddings: mean|v|={mag:.4f}, n_zero_rows={n_zero}/{testData["clap"].shape[0]}')
if 'palmtree' in testData:
    mag = testData['palmtree'].abs().mean().item()
    print(f'PalmTree embeddings: mean|v|={mag:.4f}, shape={testData["palmtree"].shape}')

if MODE == 'combo':
    inferenceCOMBO(EXP_DIR, params, tokenizer, testData, bias=0.0)
else:  # lord
    inferenceLORD(EXP_DIR, params, tokenizer, testData, specialCode=f'test-fixed-{EPOCH}', bias=0.0, epoch=EPOCH)
print('DONE')
