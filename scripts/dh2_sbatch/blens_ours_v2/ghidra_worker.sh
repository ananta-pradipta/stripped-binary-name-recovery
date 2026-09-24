#!/bin/bash
# $1 = full path to stripped binary
set -e
WS=$WORKSPACE/dh2/blens_ours_v2
BNAME=$(basename "$1")
OUT="$WS/clap_jsons/$BNAME.clap.json"
[ -s "$OUT" ] && { echo "SKIP $BNAME"; exit 0; }
PROJ=${TMPDIR:-/tmp}/USER_bl_$BNAME
mkdir -p "$PROJ"
$WORKSPACE/baselines/ghidra_11.3.1_PUBLIC/support/analyzeHeadless \
  "$PROJ" tmpProj -import "$1" \
  -scriptPath $WORKSPACE/baselines/blens_user_env \
  -preScript CreateFunctionsFromLabels.py "$WS/labels/$BNAME.json" \
  -postScript ExportClapAssembly.py "$OUT" \
  -deleteProject -max-cpu 2 >/dev/null 2>&1 || true
rm -rf "$PROJ"
[ -s "$OUT" ] && echo "OK $BNAME" || echo "FAIL $BNAME"
