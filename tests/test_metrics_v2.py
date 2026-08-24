"""Metric v2 unit tests: camelCase split, digits, F1 symmetry, EM-alignment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import split_name, compute_subtoken_f1

def test_camel_split():
    assert split_name('selectExpander') == ['select', 'expander']
    assert split_name('select_expander') == ['select', 'expander']
    assert split_name('fts3MIBufferAlloc') == ['fts3', 'mi', 'buffer', 'alloc']
    assert split_name('HTMLParser') == ['html', 'parser']
    assert split_name('ngx_http_core_set_aio') == ['ngx', 'http', 'core', 'set', 'aio']

def test_case_equivalence_scores_one():
    assert compute_subtoken_f1('select_expander', 'selectExpander') == 1.0
    assert compute_subtoken_f1('restart_model', 'RestartModel') == 1.0

def test_partial_and_zero():
    assert 0 < compute_subtoken_f1('buffer_alloc', 'fts3MIBufferAlloc') < 1
    assert compute_subtoken_f1('os_close', '__db_stat_print') == 0.0

def test_bpe_artifacts_still_normalized():
    assert compute_subtoken_f1('close _ stdout', 'close_stdout') == 1.0

if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v(); print('PASS', k)
