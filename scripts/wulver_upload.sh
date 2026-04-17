#!/bin/bash
# Upload preprocessed data and code to Slurm HPC
# Uses SSH multiplexing — authenticate ONCE, all transfers reuse the connection.
#
# Usage:
#   1. First, set up SSH config (one-time):
#        mkdir -p ~/.ssh/sockets
#        Add to ~/.ssh/config:
#          Host wulver
#              HostName <hpc-host>
#              User <hpc-user>
#              ControlMaster auto
#              ControlPath ~/.ssh/sockets/%r@%h-%p
#              ControlPersist 2h
#
#   2. Open a connection (authenticates with Duo once):
#        ssh <hpc-host>
#        # Keep this terminal open!
#
#   3. In another terminal, run:
#        bash scripts/wulver_upload.sh
#
# The script uses tar for large directories (graphs: 241K files, 2.2GB)
# to avoid slow per-file scp transfers.

set -e

WULVER="<hpc-host>"  # Uses SSH config alias (multiplexed connection)
REMOTE_BASE="<hpc-user-dir>"
REMOTE_DIR="${REMOTE_BASE}/bfnr"
LOCAL_DIR="$HOME/bfnr-project"
TMP_DIR="/tmp/wulver_upload"

# Check SSH connection is alive
echo "=== Checking SSH connection ==="
if ! ssh -O check ${WULVER} 2>/dev/null; then
    echo "ERROR: No active SSH connection to HPC."
    echo "Open a terminal and run:  ssh <hpc-host>"
    echo "Then re-run this script in another terminal."
    exit 1
fi
echo "SSH multiplexed connection active."
echo ""

echo "=== Uploading to Slurm HPC ==="
echo "Remote: ${REMOTE_DIR}"
echo ""

# Create remote directory structure
echo "[1/8] Creating remote directories..."
ssh ${WULVER} "mkdir -p ${REMOTE_DIR}/{data/bpe_model,src,configs,checkpoints,scripts,demo,results}"

# --- Large directories: tar + upload + extract ---
mkdir -p ${TMP_DIR}

echo "[2/8] Packing and uploading data/graphs (2.2GB, 241K files)..."
if ssh ${WULVER} "[ -d ${REMOTE_DIR}/data/graphs ] && [ \$(ls ${REMOTE_DIR}/data/graphs/ 2>/dev/null | head -5 | wc -l) -gt 0 ]" 2>/dev/null; then
    # Check file count on remote
    REMOTE_GRAPHS=$(ssh ${WULVER} "find ${REMOTE_DIR}/data/graphs -type f 2>/dev/null | wc -l")
    LOCAL_GRAPHS=$(find ${LOCAL_DIR}/data/graphs -type f | wc -l)
    echo "  Remote has ${REMOTE_GRAPHS} files, local has ${LOCAL_GRAPHS} files."
    if [ "${REMOTE_GRAPHS}" -eq "${LOCAL_GRAPHS}" ]; then
        echo "  SKIP — graphs already fully uploaded."
    else
        echo "  Mismatch — re-uploading graphs..."
        tar czf ${TMP_DIR}/graphs.tar.gz -C ${LOCAL_DIR}/data graphs
        scp ${TMP_DIR}/graphs.tar.gz ${WULVER}:${REMOTE_DIR}/data/
        ssh ${WULVER} "cd ${REMOTE_DIR}/data && rm -rf graphs && tar xzf graphs.tar.gz && rm graphs.tar.gz"
        echo "  Done."
    fi
else
    echo "  Compressing..."
    tar czf ${TMP_DIR}/graphs.tar.gz -C ${LOCAL_DIR}/data graphs
    echo "  Uploading (~500MB compressed)..."
    scp ${TMP_DIR}/graphs.tar.gz ${WULVER}:${REMOTE_DIR}/data/
    echo "  Extracting on HPC..."
    ssh ${WULVER} "cd ${REMOTE_DIR}/data && tar xzf graphs.tar.gz && rm graphs.tar.gz"
    echo "  Done."
fi

echo "[3/8] Packing and uploading data/labels (27MB, 450 files)..."
tar czf ${TMP_DIR}/labels.tar.gz -C ${LOCAL_DIR}/data labels
scp ${TMP_DIR}/labels.tar.gz ${WULVER}:${REMOTE_DIR}/data/
ssh ${WULVER} "cd ${REMOTE_DIR}/data && rm -rf labels && tar xzf labels.tar.gz && rm labels.tar.gz"

echo "[4/8] Packing and uploading data/external_calls (59MB, 415 files)..."
tar czf ${TMP_DIR}/external_calls.tar.gz -C ${LOCAL_DIR}/data external_calls
scp ${TMP_DIR}/external_calls.tar.gz ${WULVER}:${REMOTE_DIR}/data/
ssh ${WULVER} "cd ${REMOTE_DIR}/data && rm -rf external_calls && tar xzf external_calls.tar.gz && rm external_calls.tar.gz"

echo "[5/8] Packing and uploading data/string_refs (13MB, 334 files)..."
tar czf ${TMP_DIR}/string_refs.tar.gz -C ${LOCAL_DIR}/data string_refs
scp ${TMP_DIR}/string_refs.tar.gz ${WULVER}:${REMOTE_DIR}/data/
ssh ${WULVER} "cd ${REMOTE_DIR}/data && rm -rf string_refs && tar xzf string_refs.tar.gz && rm string_refs.tar.gz"

# --- Small files: direct scp ---
echo "[6/8] Uploading index files, vocabs, BPE model..."
scp ${LOCAL_DIR}/data/match_index.json ${WULVER}:${REMOTE_DIR}/data/
scp ${LOCAL_DIR}/data/split_assignments.json ${WULVER}:${REMOTE_DIR}/data/
scp ${LOCAL_DIR}/data/votes_vocab.json ${WULVER}:${REMOTE_DIR}/data/
scp ${LOCAL_DIR}/data/build_manifest.json ${WULVER}:${REMOTE_DIR}/data/ 2>/dev/null || true
scp -r ${LOCAL_DIR}/data/bpe_model ${WULVER}:${REMOTE_DIR}/data/

echo "[7/8] Uploading source code, configs, scripts, demo, checkpoints..."
scp -r ${LOCAL_DIR}/src ${WULVER}:${REMOTE_DIR}/
scp -r ${LOCAL_DIR}/configs ${WULVER}:${REMOTE_DIR}/
scp -r ${LOCAL_DIR}/scripts ${WULVER}:${REMOTE_DIR}/
scp ${LOCAL_DIR}/checkpoints/pretrained_encoder.pt ${WULVER}:${REMOTE_DIR}/checkpoints/

# Demo data
tar czf ${TMP_DIR}/demo.tar.gz -C ${LOCAL_DIR} demo
scp ${TMP_DIR}/demo.tar.gz ${WULVER}:${REMOTE_DIR}/
ssh ${WULVER} "cd ${REMOTE_DIR} && rm -rf demo && tar xzf demo.tar.gz && rm demo.tar.gz"

# --- Verification ---
echo "[8/8] Verifying upload..."
echo ""
ssh ${WULVER} "
echo '=== Remote file counts ==='
echo \"graphs:         \$(find ${REMOTE_DIR}/data/graphs -type f 2>/dev/null | wc -l) files\"
echo \"labels:         \$(find ${REMOTE_DIR}/data/labels -type f 2>/dev/null | wc -l) files\"
echo \"external_calls: \$(find ${REMOTE_DIR}/data/external_calls -type f 2>/dev/null | wc -l) files\"
echo \"string_refs:    \$(find ${REMOTE_DIR}/data/string_refs -type f 2>/dev/null | wc -l) files\"
echo \"\"
echo '=== Key files ==='
for f in match_index.json split_assignments.json votes_vocab.json; do
    if [ -f ${REMOTE_DIR}/data/\$f ]; then
        echo \"  ✓ data/\$f (\$(du -h ${REMOTE_DIR}/data/\$f | cut -f1))\"
    else
        echo \"  ✗ data/\$f MISSING\"
    fi
done
echo \"\"
if [ -f ${REMOTE_DIR}/checkpoints/pretrained_encoder.pt ]; then
    echo \"  ✓ checkpoints/pretrained_encoder.pt (\$(du -h ${REMOTE_DIR}/checkpoints/pretrained_encoder.pt | cut -f1))\"
else
    echo \"  ✗ checkpoints/pretrained_encoder.pt MISSING\"
fi
echo \"\"
echo \"src/: \$(find ${REMOTE_DIR}/src -type f 2>/dev/null | wc -l) files\"
echo \"configs/: \$(find ${REMOTE_DIR}/configs -type f 2>/dev/null | wc -l) files\"
echo \"scripts/: \$(find ${REMOTE_DIR}/scripts -type f 2>/dev/null | wc -l) files\"
echo \"demo/: \$(find ${REMOTE_DIR}/demo -type f 2>/dev/null | wc -l) files\"
"

# Cleanup temp files
rm -rf ${TMP_DIR}

echo ""
echo "=== Upload complete! ==="
echo ""
echo "Expected local file counts for comparison:"
echo "  graphs:         $(find ${LOCAL_DIR}/data/graphs -type f | wc -l) files"
echo "  labels:         $(find ${LOCAL_DIR}/data/labels -type f | wc -l) files"
echo "  external_calls: $(find ${LOCAL_DIR}/data/external_calls -type f | wc -l) files"
echo "  string_refs:    $(find ${LOCAL_DIR}/data/string_refs -type f | wc -l) files"
echo ""
echo "Next steps on HPC:"
echo "  # In the SSH terminal that's already open:"
echo "  bash bfnr/scripts/wulver_setup.sh     # first time only"
echo "  cd ${REMOTE_DIR}"
echo "  sbatch scripts/wulver_train.sbatch"
echo "  squeue -u <hpc-user>"
