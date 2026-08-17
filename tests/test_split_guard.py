"""B8 regression test: the loader must refuse to silently default unassigned
binaries to train, must reject overlapping split lists, and must expose the
split-file hash for checkpoint provenance."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.build_dataset import FunctionDataset


def _ds(binaries):
    ds = FunctionDataset.__new__(FunctionDataset)
    ds.samples = [{'binary': b, 'name': f'f{i}'} for i, b in enumerate(binaries)]
    return ds


def _write(d, obj):
    p = os.path.join(d, 'split.json'); json.dump(obj, open(p, 'w')); return p


def test_unassigned_binary_raises():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, {'train': ['a_x_O0'], 'val': ['b_x_O0'], 'test': ['c_x_O0']})
        ds = _ds(['a_x_O0', 'b_x_O0', 'c_x_O0', 'dash_dash_O2'])
        try:
            ds.get_splits(split_file=p)
        except RuntimeError as e:
            assert 'SPLIT ASSIGNMENT GUARD' in str(e) and 'dash_dash_O2' in str(e)
        else:
            raise AssertionError('unassigned binary silently accepted (B8 regression)')


def test_bypass_env_defaults_to_train():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, {'train': ['a_x_O0'], 'val': ['b_x_O0'], 'test': ['c_x_O0']})
        ds = _ds(['a_x_O0', 'b_x_O0', 'c_x_O0', 'dash_dash_O2'])
        os.environ['ALLOW_UNASSIGNED_TO_TRAIN'] = '1'
        try:
            tr, va, te = ds.get_splits(split_file=p)
        finally:
            del os.environ['ALLOW_UNASSIGNED_TO_TRAIN']
        assert len(tr) == 2 and len(va) == 1 and len(te) == 1


def test_overlap_raises_and_heldout_dropped_and_hash_exposed():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, {'train': ['a_x_O0', 'b_x_O0'], 'val': ['b_x_O0'], 'test': ['c_x_O0']})
        ds = _ds(['a_x_O0', 'b_x_O0', 'c_x_O0'])
        try:
            ds.get_splits(split_file=p)
        except RuntimeError as e:
            assert 'SPLIT FILE INVALID' in str(e)
        else:
            raise AssertionError('overlapping split lists accepted')
        p = _write(d, {'train': ['a_x_O0'], 'val_indist': ['b_x_O0'], 'val_xproj': ['e_x_O0'],
                       'test': ['c_x_O0'], 'xproject': ['dash_dash_O2']})
        ds = _ds(['a_x_O0', 'b_x_O0', 'c_x_O0', 'dash_dash_O2', 'e_x_O0'])
        tr, va, te = ds.get_splits(split_file=p)
        assert (len(tr), len(va), len(te)) == (1, 1, 1)
        assert ds.val_xproj_idx == [4]
        assert ds.split_schema == 'v2' and len(ds.split_sha256) == 64
        # xproject binary must be in no split
        assert all(ds.samples[i]['binary'] != 'dash_dash_O2' for i in tr + va + te)


if __name__ == '__main__':
    for fn in [test_unassigned_binary_raises, test_bypass_env_defaults_to_train,
               test_overlap_raises_and_heldout_dropped_and_hash_exposed]:
        fn(); print('PASS', fn.__name__)
