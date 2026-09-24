#!/usr/bin/env bash
# Resumable full re-lift chain for a 12 GB WSL host. Safe to re-run: relift_v2.py skips done ids.
cd "$(dirname "$0")/.." || exit 1
source activate.sh >/dev/null 2>&1
export OCAMLRUNPARAM='s=4M,i=32M,o=80'
LOG=results/phase1/relift_driver.log
{
echo "=== driver start $(date -Is) mem=$(free -g | awk '/Mem/{print $2}')G"
echo "=== [1] main <2MB, 3 workers";            python3 scripts/relift_v2.py --sources main --max-mb 2 --workers 3 --no-assert
echo "=== [2] main 2-10MB ascending, 2 workers"; python3 scripts/relift_v2.py --sources main --min-mb 2 --max-mb 10 --ascending --workers 2 --no-assert
echo "=== [3] main >=10MB ascending, 1 worker";  python3 scripts/relift_v2.py --sources main --min-mb 10 --ascending --workers 1 --no-assert
echo "=== [4] ftdomains/ftdomains2, 3 workers";  python3 scripts/relift_v2.py --sources ftdomains ftdomains2 --ascending --workers 3 --no-assert
echo "=== [5] clang, 3 workers";                 python3 scripts/relift_v2.py --sources clang --ascending --workers 3 --no-assert
echo "=== [6] final effect check";               python3 scripts/relift_v2.py --sources main ftdomains ftdomains2 clang --workers 1
echo "=== driver end $(date -Is)"
} >> "$LOG" 2>&1
