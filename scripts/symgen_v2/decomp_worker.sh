#!/bin/bash
# $1 = binary id. Decompile the protocol addresses of one stripped_v2 ELF.
WS=$WORKSPACE/dh2/symgen_v2
TOOLS=$WORKSPACE/tools
JH=$TOOLS/jdk-21.0.12+8
STRIP=$WORKSPACE/relift_ws/data/stripped_v2/$1
OUT=$WS/decomp/$1.json
GP=${TMPDIR:-/tmp}/USER_sg_$1
[ -s "$OUT" ] && exit 0
[ -f "$STRIP" ] || { echo "$1 MISSING_STRIPPED"; exit 0; }
BASE=$(cat $WS/bases/$1 2>/dev/null || echo 0x0)
for attempt in 1 2 3; do
  rm -rf "$GP"; mkdir -p "$GP"
  JAVA_HOME=$JH PATH=$JH/bin:$PATH timeout 2400 $TOOLS/ghidra_11.2.1_PUBLIC/support/analyzeHeadless \
    "$GP" p -import "$STRIP" -scriptPath $WS \
    -postScript GhidraExportDecompMasked.py $WS/addrs/$1.addrs "$OUT" "$BASE" \
    -deleteProject -max-cpu 2 > $WS/logs/$1.log 2>&1
  if grep -q EXPORT_DONE $WS/logs/$1.log; then grep -h EXPORT_DONE $WS/logs/$1.log | sed "s/^/$1 attempt=$attempt /"; break; fi
  echo "$1 attempt=$attempt FAILED"
done
rm -rf "$GP"
