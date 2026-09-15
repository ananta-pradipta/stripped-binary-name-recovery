"""Patch punstrip_score.py: guard BLens canonicaliser against exponential recursive_split on long digit runs."""
import sys
p = sys.argv[1]
s = open(p).read()
if 'DIGITRUN' in s:
    print('already patched'); sys.exit(0)
i = s.index('def canon(name):')
j = s.index('    return cache[name]', i) + len('    return cache[name]')
new = '''import re, signal
DIGITRUN = re.compile(r"\\d{5,}")   # BLens NLP.recursive_split is exponential in digit-run length (job 1286044 hung >1h on a
                                    # 14-digit run); digit tokens can never be in the 1024-label vocab, so stripping runs of
                                    # >=5 digits cannot change any score (vocab audited: no token contains a 5+ digit run)
class _CanonTimeout(BaseException): pass   # BaseException so NLP.py's own except-Exception blocks cannot swallow it
def _alarm(*a): raise _CanonTimeout()
signal.signal(signal.SIGALRM, _alarm)
CANON_TIMEOUTS = []
def canon(name):
    """BLens label space: their canonicaliser, then restrict to their 1024-label vocabulary (validated: reproduces their
    groundtruth column on 98% of keys). Long digit runs are pre-stripped and a 60 s hard timeout guards the recursion."""
    if name not in cache:
        raw = DIGITRUN.sub("_", name.replace("::", "_")) if name else ""
        signal.alarm(60)
        try: c = nlp.tristan_canonical_name(raw) if raw else ""
        except _CanonTimeout: c = ""; CANON_TIMEOUTS.append(name); print(f"WARN canon timeout: {name!r}", flush=True)
        except Exception: c = ""
        finally: signal.alarm(0)
        cache[name] = "_".join(x for x in c.split("_") if x in VOCAB)
    return cache[name]'''
s = s[:i] + new + s[j:]
old_tail = "print('EFFECT: score_report.json written', flush=True)"
if old_tail in s:
    s = s.replace(old_tail, "print(f'EFFECT: score_report.json written; canon timeouts {len(CANON_TIMEOUTS)}', flush=True)")
open(p, 'w').write(s)
print('patched', p)
