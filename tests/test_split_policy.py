"""Unit test for FunctionDatasetV2.apply_split_policy on synthetic samples."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.dataset_v2 import FunctionDatasetV2

def mk(samples):
    ds = FunctionDatasetV2.__new__(FunctionDatasetV2)
    ds.samples = samples
    return ds

def test_policy():
    S = [
        {'tok_hash': 'h1', 'name': 'a', 'in_dynsym': False},  # 0 train
        {'tok_hash': 'h1', 'name': 'a', 'in_dynsym': False},  # 1 train dup -> dropped
        {'tok_hash': 'h1', 'name': 'b', 'in_dynsym': False},  # 2 train same hash diff name -> kept
        {'tok_hash': 'h1', 'name': 'a', 'in_dynsym': False},  # 3 eval body-in-train -> dropped
        {'tok_hash': 'h2', 'name': 'c', 'in_dynsym': True},   # 4 eval exported -> dropped
        {'tok_hash': 'h3', 'name': 'd', 'in_dynsym': False},  # 5 eval scored
        {'tok_hash': 'h4', 'name': 'e', 'in_dynsym': False},  # 6 test scored
    ]
    ds = mk(S)
    tr, va, te = ds.apply_split_policy([0, 1, 2], [3, 4, 5], [6], quiet=True)
    assert tr == [0, 2], tr
    assert va == [5], va
    assert te == [6], te
    assert ds.policy_stats['val'] == {'raw': 3, 'drop_body_in_train': 1, 'drop_in_dynsym': 1, 'scored': 1}
    assert ds.policy_dropped['val']['body_in_train'] == [3]
    assert ds.policy_dropped['val']['in_dynsym'] == [4]
    # scored eval NEVER contains a train tok_hash
    train_hashes = {S[i]['tok_hash'] for i in tr}
    assert all(S[i]['tok_hash'] not in train_hashes for i in va + te)

if __name__ == '__main__':
    test_policy(); print('PASS test_policy')
