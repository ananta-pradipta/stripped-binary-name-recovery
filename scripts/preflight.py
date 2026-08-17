#!/usr/bin/env python3
"""Preflight assertions for CS785 experiment jobs.

WHY THIS EXISTS
---------------
Every expensive mistake in this project has the same shape: the job runs, exits
0, prints plausible numbers — and measures the wrong thing. Crashes are cheap;
these are not. Recorded instances (see results/experiment_log.md):

  2026-08-04  pretrain ran 2-4x slow          num_workers hardcoded 0, no error
  2026-08-04  clang match index 1,390/141,932 wrong join key, "succeeded" at 1%
  2026-08-04  AMP dtype crash 2h in           untested eval path
  2026-08-05  FT-fix eval used OLD graphs     patched a file the driver didn't use
  2026-08-03  "new domains" were training pkgs novelty checked against one source
  2026-08-02  prefix eval timed out           cheap arms swept before the vital one

The common cause is asserting on CONFIGURATION ("I passed --graphs-fixed") rather
than on EFFECT ("N functions actually came from graphs_fixed"). Configuration can
be right while the effect is absent. Assert on effect.

USAGE (put at the top of every job, after staging, before the expensive part):

    from scripts.preflight import Preflight
    pf = Preflight("corrected FT eval")
    pf.checkpoint("checkpoints/best_model_paper_clean.pt", expect_vocab=7004)
    pf.split("data/split_assignments.json", expect_train=821, forbid_in_train=["clang"])
    pf.count("graphs from graphs_fixed", n_from_fixed, minimum=200)
    pf.differs_from("results/baseline.json", this_run_will_change=True)
    pf.done()          # raises SystemExit(1) with a summary if anything failed

Every check PRINTS its evidence, so the job log itself is the audit trail.

SBATCH USAGE — CRITICAL
-----------------------
A failing preflight only aborts the job if the shell propagates the exit code.
On 2026-08-05 a preflight printed "FATAL: ftdomains2/graphs empty" and the job
RAN ANYWAY for 22 GPU-minutes, because the sbatch lacked error handling. Always:

    #!/bin/bash -l
    set -euo pipefail          # <-- REQUIRED. Without it, sys.exit(1) is ignored.
    ...
    python3 - <<'PY'
    ...preflight assertions...
    PY

Verify after writing any new sbatch:  grep -c 'set -euo pipefail' job.sbatch
"""
import json
import os
import sys


class PreflightError(AssertionError):
    pass


class Preflight:
    def __init__(self, name):
        self.name = name
        self.failures = []
        self.checks = 0
        print("=" * 78)
        print(f"PREFLIGHT: {name}")
        print("=" * 78)

    # ---- internals ---------------------------------------------------------
    def _ok(self, label, evidence):
        self.checks += 1
        print(f"  [ok]   {label}: {evidence}")

    def _fail(self, label, evidence):
        self.checks += 1
        self.failures.append(label)
        print(f"  [FAIL] {label}: {evidence}")

    # ---- checks ------------------------------------------------------------
    def checkpoint(self, path, expect_vocab=None, expect_params=None):
        """Assert the checkpoint is the one intended (not a stale/leaky twin)."""
        if not os.path.exists(path):
            return self._fail("checkpoint", f"missing: {path}")
        try:
            import torch
            c = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:                                   # noqa: BLE001
            return self._fail("checkpoint", f"unreadable: {e}")
        params = sum(v.numel() for v in c["model_state_dict"].values())
        vocab = len(c.get("votes_vocab", {})) or c.get("config", {}).get(
            "decoder", {}).get("bpe_vocab_size")
        ev = (f"{os.path.basename(path)} params={params:,} vocab={vocab} "
              f"val_f1={c.get('val_f1')} epoch={c.get('epoch')}")
        if expect_vocab and vocab and vocab != expect_vocab:
            return self._fail("checkpoint", f"{ev} — expected vocab {expect_vocab}")
        if expect_params and abs(params - expect_params) / expect_params > 0.02:
            return self._fail("checkpoint", f"{ev} — expected ~{expect_params:,} params")
        self._ok("checkpoint", ev)

    def split(self, path, expect_train=None, forbid_in_train=(), require_buckets=()):
        """Assert the split staged is the split intended, and is leakage-clean."""
        if not os.path.exists(path):
            return self._fail("split", f"missing: {path}")
        sp = json.load(open(path))
        sizes = {k: len(v) for k, v in sp.items() if isinstance(v, list)}
        ev = " ".join(f"{k}={v}" for k, v in sorted(sizes.items()))
        for b in require_buckets:
            if not sp.get(b):
                return self._fail("split", f"{ev} — bucket '{b}' empty/missing")
        for pat in forbid_in_train:
            bad = [x for x in sp.get("train", []) if pat in x]
            if bad:
                return self._fail(
                    "split", f"{ev} — {len(bad)} train entries match forbidden "
                             f"'{pat}' e.g. {bad[:3]}")
        if expect_train is not None and sizes.get("train") != expect_train:
            return self._fail("split", f"{ev} — expected train={expect_train}")
        self._ok("split", ev)

    def count(self, label, value, minimum=None, maximum=None, expect=None):
        """Assert a magnitude is in the plausible range (catches 1%-yield bugs)."""
        ev = f"{value:,}" if isinstance(value, int) else str(value)
        if expect is not None and value != expect:
            return self._fail(label, f"{ev} — expected {expect:,}")
        if minimum is not None and value < minimum:
            return self._fail(label, f"{ev} — below minimum {minimum:,}")
        if maximum is not None and value > maximum:
            return self._fail(label, f"{ev} — above maximum {maximum:,}")
        self._ok(label, ev)

    def effect(self, label, observed, expected_desc):
        """Assert the INTENDED CHANGE is observable, not merely configured.

        This is the check that would have caught the FT-fix bug: the config was
        right, the file existed, and the driver still read the old graphs.
        """
        if observed:
            self._ok(f"effect: {label}", expected_desc)
        else:
            self._fail(f"effect: {label}",
                       f"NOT OBSERVED — expected {expected_desc}. The change is "
                       f"configured but not in force; the job would measure the "
                       f"OLD behaviour and look successful.")

    def differs_from(self, reference_json, key_path, tolerance=1e-6):
        """Warn if this run is about to reproduce a reference result exactly.

        Byte-identical agreement with a baseline you intended to change is the
        loudest possible signal that the change did not apply.
        """
        if not os.path.exists(reference_json):
            return self._ok("differs_from", f"no reference at {reference_json} (skipped)")
        self._ok("differs_from",
                 f"reference {os.path.basename(reference_json)} recorded; compare "
                 f"'{key_path}' after the run — identical to {tolerance} means the "
                 f"change did not apply")

    def dataloader(self, loader, expect_workers=None):
        """Assert the loader is configured for throughput (num_workers=0 bug)."""
        nw = getattr(loader, "num_workers", None)
        ev = f"num_workers={nw} batch_size={getattr(loader, 'batch_size', '?')}"
        if expect_workers is not None and nw != expect_workers:
            return self._fail("dataloader", f"{ev} — expected {expect_workers} workers")
        if nw == 0:
            return self._fail("dataloader", f"{ev} — serial loading starves the GPU")
        self._ok("dataloader", ev)

    def done(self):
        print("-" * 78)
        if self.failures:
            print(f"PREFLIGHT FAILED: {len(self.failures)}/{self.checks} checks — "
                  f"{', '.join(self.failures)}")
            print("Aborting before the expensive part. Fix and resubmit.")
            print("=" * 78)
            sys.exit(1)
        print(f"PREFLIGHT PASSED: {self.checks}/{self.checks} checks")
        print("=" * 78)
