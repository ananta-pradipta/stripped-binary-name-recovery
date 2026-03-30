#!/usr/bin/env python3
"""
Expanded demo evaluation: run predict.py on diverse unseen packages,
compare against ground truth, report per-package and per-optimization metrics.

Uses predict.py for proper callee/caller context handling.
"""
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import compute_subtoken_f1, compute_char_ngram_similarity, compute_edit_distance_similarity


# Unseen demo packages: (package, binary_name, opt_levels)
# Excludes groff (C++ — different naming convention)
DEMO_PACKAGES = [
    # Diffutils (completely unseen, the original demo package)
    ("diffutils", "diff", [""]),
    ("diffutils", "cmp", [""]),
    ("diffutils", "sdiff", [""]),
    ("diffutils", "diff3", [""]),
    # Data processing (medium, C)
    ("datamash", "datamash", ["O0", "O2"]),
    # File system monitoring (medium, C)
    ("direvent", "direvent", ["O0", "O2"]),
    # Source code analysis (medium, C)
    ("csplit2", "cflow", ["O0", "O2"]),
    # Info viewer (medium, C)
    ("texinfo", "ginfo", ["O0", "O2"]),
    # Source preprocessor (small, C)
    ("cppi", "cppi", ["O0", "O2"]),
    # Hello (tiny, C)
    ("hello", "hello", ["O0", "O2"]),
    # GNU Accounting (system admin, gnulib)
    ("acct", "ac", ["O0", "O2"]),
    ("acct", "last", ["O0", "O2"]),
    ("acct", "lastcomm", ["O0", "O2"]),
    ("acct", "sa", ["O0", "O2"]),
    ("acct", "dump-utmp", ["O0", "O2"]),
    ("acct", "accton", ["O0", "O2"]),
    # Rush (security/sysadmin, gnulib)
    ("rush", "rush", ["O0", "O2"]),
    # Dico (DICT protocol, gnulib)
    ("dico", "dico", ["O0", "O2"]),
    # htop (non-gnulib control)
    ("htop", "htop", ["O0", "O2"]),
    # strace (non-gnulib control)
    ("strace", "strace", ["O0", "O2"]),
]


def get_ground_truth(raw_path, labels_path=None):
    """Extract function names from labels JSON or debug binary using nm.

    Prefers labels JSON (consistent with Wulver eval). Falls back to nm.
    Deduplicates by name to avoid counting the same function at multiple addresses.
    """
    gt = {}

    # Try labels JSON first (consistent with Wulver eval)
    if labels_path and os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = json.load(f)
        funcs = labels.get('functions', {})
        for name, addr in funcs.items():
            if isinstance(addr, str):
                addr_norm = '0x' + addr[2:].lstrip('0') if addr.startswith('0x') else '0x' + addr.lstrip('0')
                if addr_norm == '0x':
                    addr_norm = '0x0'
                gt[addr_norm] = name
                gt[addr] = name  # keep full form too
        return gt

    # Fallback to nm, but deduplicate by name
    result = subprocess.run(
        ["nm", "--defined-only", raw_path],
        capture_output=True, text=True
    )
    seen_names = set()
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 3 and parts[1] in ("T", "t"):
            name = parts[2]
            if name in seen_names:
                continue  # skip duplicate names at different addresses
            seen_names.add(name)
            addr = "0x" + parts[0].lstrip("0")
            if addr == "0x":
                addr = "0x0"
            gt[addr] = name
    return gt


def run_predict(stripped_path, checkpoint, output_path):
    """Run predict.py on a stripped binary."""
    result = subprocess.run(
        ["python3", "scripts/predict.py",
         "--binary", stripped_path,
         "--checkpoint", checkpoint,
         "--output", output_path,
         "--beam-width", "5"],
        capture_output=True, text=True, timeout=1800,
    )
    return result.returncode == 0


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
    outdir = "demo/results/expanded"
    os.makedirs(outdir, exist_ok=True)

    print(f"Checkpoint: {checkpoint}")
    print(f"Output dir: {outdir}")
    print()

    per_binary = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0, "ngsim_sum": 0, "edsim_sum": 0})
    per_opt = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0, "ngsim_sum": 0, "edsim_sum": 0})
    per_pkg = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0, "ngsim_sum": 0, "edsim_sum": 0})

    for pkg, binary, opt_levels in DEMO_PACKAGES:
        for opt in opt_levels:
            if opt:
                bin_name = f"{pkg}_{binary}_{opt}"
                raw_path = f"data/raw/{bin_name}_sym"
                stripped_path = f"data/stripped/{bin_name}_stripped"
            else:
                bin_name = f"{pkg}_{binary}"
                raw_path = f"data/raw/{bin_name}_sym"
                stripped_path = f"data/stripped/{bin_name}_stripped"
                # diffutils might be in demo/stripped
                if not os.path.exists(stripped_path):
                    stripped_path = f"demo/stripped/{bin_name}_stripped"
                if not os.path.exists(raw_path):
                    raw_path = f"demo/raw/{bin_name}_sym"

            if not os.path.exists(raw_path):
                print(f"  SKIP {bin_name}: raw binary not found at {raw_path}")
                continue
            if not os.path.exists(stripped_path):
                print(f"  SKIP {bin_name}: stripped binary not found")
                continue

            # Get ground truth (prefer labels JSON for consistency with Wulver eval)
            labels_path = f"data/labels/{bin_name}_labels.json"
            if not os.path.exists(labels_path):
                labels_path = f"demo/labels/{bin_name}_labels.json"
            if not os.path.exists(labels_path):
                labels_path = None
            gt = get_ground_truth(raw_path, labels_path=labels_path)
            if not gt:
                print(f"  SKIP {bin_name}: no ground truth")
                continue

            # Run predict.py
            pred_path = os.path.join(outdir, f"predictions_{bin_name}.json")
            print(f"  Predicting {bin_name} ({len(gt)} GT functions)...", end="", flush=True)

            if not run_predict(stripped_path, checkpoint, pred_path):
                print(" FAILED")
                continue

            if not os.path.exists(pred_path):
                print(" no output")
                continue

            # Load predictions
            with open(pred_path) as f:
                preds = json.load(f)

            # Match predictions to ground truth by address
            correct = 0
            total = 0
            f1_sum = 0
            ngsim_sum = 0
            edsim_sum = 0

            for pred_func in preds:
                addr = pred_func.get("address", "")
                pred_name = pred_func.get("predicted_name", pred_func.get("name", ""))

                # Normalize address format
                if not addr.startswith("0x"):
                    addr = "0x" + addr

                true_name = gt.get(addr)
                # O0 thunk fix: if no match, try addr-4 (thunk jumps to addr+4)
                if not true_name:
                    try:
                        addr_minus4 = hex(int(addr, 16) - 4)
                        true_name = gt.get(addr_minus4)
                    except ValueError:
                        pass
                if not true_name:
                    continue

                f1 = compute_subtoken_f1(pred_name, true_name)
                ngsim = compute_char_ngram_similarity(pred_name, true_name)
                edsim = compute_edit_distance_similarity(pred_name, true_name)
                total += 1
                f1_sum += f1
                ngsim_sum += ngsim
                edsim_sum += edsim
                if pred_name == true_name:
                    correct += 1

            opt_label = opt if opt else "O2"
            per_binary[bin_name]["correct"] += correct
            per_binary[bin_name]["total"] += total
            per_binary[bin_name]["f1_sum"] += f1_sum
            per_binary[bin_name]["ngsim_sum"] += ngsim_sum
            per_binary[bin_name]["edsim_sum"] += edsim_sum
            per_opt[opt_label]["correct"] += correct
            per_opt[opt_label]["total"] += total
            per_opt[opt_label]["f1_sum"] += f1_sum
            per_opt[opt_label]["ngsim_sum"] += ngsim_sum
            per_opt[opt_label]["edsim_sum"] += edsim_sum
            per_pkg[pkg]["correct"] += correct
            per_pkg[pkg]["total"] += total
            per_pkg[pkg]["f1_sum"] += f1_sum
            per_pkg[pkg]["ngsim_sum"] += ngsim_sum
            per_pkg[pkg]["edsim_sum"] += edsim_sum

            em_pct = 100 * correct / total if total > 0 else 0
            f1_avg = f1_sum / total if total > 0 else 0
            ngsim_avg = ngsim_sum / total if total > 0 else 0
            edsim_avg = edsim_sum / total if total > 0 else 0
            print(f" {correct}/{total} EM ({em_pct:.1f}%), F1={f1_avg:.3f}, NgSim={ngsim_avg:.3f}, EdSim={edsim_avg:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"EXPANDED DEMO RESULTS (with callee/caller context)")
    print(f"{'='*70}")

    overall_c = sum(v["correct"] for v in per_binary.values())
    overall_t = sum(v["total"] for v in per_binary.values())
    overall_f1 = sum(v["f1_sum"] for v in per_binary.values()) / max(overall_t, 1)
    overall_ngsim = sum(v["ngsim_sum"] for v in per_binary.values()) / max(overall_t, 1)
    overall_edsim = sum(v["edsim_sum"] for v in per_binary.values()) / max(overall_t, 1)

    print(f"\nOverall: {overall_c}/{overall_t} EM ({100*overall_c/overall_t:.1f}%), F1={overall_f1:.4f}, NgSim={overall_ngsim:.4f}, EdSim={overall_edsim:.4f}")

    print(f"\nPer optimization level:")
    for opt in sorted(per_opt):
        v = per_opt[opt]
        em = 100 * v["correct"] / v["total"] if v["total"] > 0 else 0
        f1 = v["f1_sum"] / v["total"] if v["total"] > 0 else 0
        ngsim = v["ngsim_sum"] / v["total"] if v["total"] > 0 else 0
        edsim = v["edsim_sum"] / v["total"] if v["total"] > 0 else 0
        print(f"  {opt}: {v['correct']}/{v['total']} EM ({em:.1f}%), F1={f1:.4f}, NgSim={ngsim:.4f}, EdSim={edsim:.4f}")

    print(f"\nPer package:")
    for pkg in sorted(per_pkg):
        v = per_pkg[pkg]
        em = 100 * v["correct"] / v["total"] if v["total"] > 0 else 0
        f1 = v["f1_sum"] / v["total"] if v["total"] > 0 else 0
        ngsim = v["ngsim_sum"] / v["total"] if v["total"] > 0 else 0
        edsim = v["edsim_sum"] / v["total"] if v["total"] > 0 else 0
        print(f"  {pkg:15s}: {v['correct']:4d}/{v['total']:4d} EM ({em:5.1f}%), F1={f1:.4f}, NgSim={ngsim:.4f}, EdSim={edsim:.4f}")

    print(f"\nPer binary:")
    for bn in sorted(per_binary):
        v = per_binary[bn]
        em = 100 * v["correct"] / v["total"] if v["total"] > 0 else 0
        f1 = v["f1_sum"] / v["total"] if v["total"] > 0 else 0
        ngsim = v["ngsim_sum"] / v["total"] if v["total"] > 0 else 0
        edsim = v["edsim_sum"] / v["total"] if v["total"] > 0 else 0
        opt_tag = "[O0]" if "_O0" in bn else "[O2]"
        print(f"  {opt_tag} {bn:35s}: {v['correct']:4d}/{v['total']:4d} EM ({em:5.1f}%), F1={f1:.4f}, NgSim={ngsim:.4f}, EdSim={edsim:.4f}")

    # Save
    save_data = {
        "overall": {"correct": overall_c, "total": overall_t,
                    "f1": overall_f1, "ngsim": overall_ngsim, "edsim": overall_edsim},
        "per_opt": {k: dict(v) for k, v in per_opt.items()},
        "per_pkg": {k: dict(v) for k, v in per_pkg.items()},
        "per_binary": {k: {"correct": v["correct"], "total": v["total"],
                           "f1": v["f1_sum"]/max(v["total"],1),
                           "ngsim": v["ngsim_sum"]/max(v["total"],1),
                           "edsim": v["edsim_sum"]/max(v["total"],1)} for k, v in per_binary.items()},
    }
    save_path = os.path.join(outdir, "expanded_eval.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
