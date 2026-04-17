#!/bin/bash
# ============================================================
# Phase 2 expansion: Fix label mismatches + new packages
# ============================================================
# 1. Busybox: compile locally, save debug, extract labels, BAP
# 2. GNU Chess: compile from scratch
# 3. wdiff: compile from scratch
# 4. wget: compile O1/O2/O3
# 5. zstd: compile O1/O3
# 6. nettle: compile from scratch (all opts)
# ============================================================
source ~/bfnr-project/activate.sh
cd ~/bfnr-project
set +e

BUILD_DIR="$HOME/bfnr-project/build_tmp"
DATA_RAW="data/raw"
DATA_STRIPPED="data/stripped"
DATA_DEBUG="data/debug"
DATA_BIR="data/bir"
DATA_GRAPHS="data/graphs"
DATA_LABELS="data/labels"
DATA_EXT="data/external_calls"

mkdir -p "$DATA_RAW" "$DATA_STRIPPED" "$DATA_DEBUG" "$DATA_BIR" "$DATA_GRAPHS" "$DATA_LABELS" "$DATA_EXT"

COMPILED=0

# ══ 1. Busybox — recompile to get debug binaries + correct labels ══
echo "═══════════════════════════════════════════════"
echo " 1. Busybox (O0 O1 O2 O3)"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR/busybox-1.36.1"
for opt in O0 O1 O2 O3; do
    name="busybox_busybox_${opt}"
    if [ -f "$HOME/bfnr-project/$DATA_RAW/${name}_sym" ]; then
        echo "  [$opt] already has debug binary, skipping compile"
    else
        echo "  [$opt] compiling..."
        make clean 2>/dev/null || true
        make -j$(nproc) CFLAGS="-g -${opt}" LDFLAGS="-g" 2>/dev/null && {
            if [ -f busybox_unstripped ]; then
                cp busybox_unstripped "$HOME/bfnr-project/$DATA_RAW/${name}_sym"
                cp busybox_unstripped "$HOME/bfnr-project/$DATA_DEBUG/${name}"
                strip -s busybox_unstripped -o "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped"
                echo "  [$opt] compiled"
                COMPILED=$((COMPILED + 1))
            elif [ -f busybox ]; then
                cp busybox "$HOME/bfnr-project/$DATA_RAW/${name}_sym"
                cp busybox "$HOME/bfnr-project/$DATA_DEBUG/${name}"
                strip -s busybox -o "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped"
                echo "  [$opt] compiled (from busybox)"
                COMPILED=$((COMPILED + 1))
            fi
        } || echo "  [$opt] FAILED"
    fi
    # Extract labels from debug binary
    raw_bin="$HOME/bfnr-project/$DATA_RAW/${name}_sym"
    labels_file="$HOME/bfnr-project/$DATA_LABELS/${name}_labels.json"
    if [ -f "$raw_bin" ] && [ ! -f "$labels_file" ]; then
        echo -n "  [$opt] extracting labels... "
        nm --defined-only "$raw_bin" 2>/dev/null | awk '$2 ~ /[tT]/ {printf "{\"0x%s\": \"%s\"}\n", $1, $3}' | python3 -c "
import sys, json
binary = '$name'
labels = {}
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        labels.update(d)
    except: pass
output = {'binary': binary, 'num_functions': len(labels), 'functions': labels, 'addr_to_name': labels, 'name_to_addr': {v:k for k,v in labels.items()}}
with open('$labels_file', 'w') as f:
    json.dump(output, f, indent=2)
print(f'{len(labels)} labels')
"
    fi
done
cd ~/bfnr-project

# ══ 2. GNU Chess ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 2. GNU Chess (O0 O1 O2 O3)"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
GNUCHESS_TAR="gnuchess-6.2.9.tar.gz"
GNUCHESS_DIR="gnuchess-6.2.9"
if [ ! -f "$GNUCHESS_TAR" ]; then
    wget -q "https://ftp.gnu.org/gnu/chess/gnuchess-6.2.9.tar.gz" -O "$GNUCHESS_TAR"
fi
[ ! -d "$GNUCHESS_DIR" ] && tar xf "$GNUCHESS_TAR"
for opt in O0 O1 O2 O3; do
    name="gnuchess_gnuchess_${opt}"
    if [ -f "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped" ]; then
        echo "  [$opt] already compiled"
        continue
    fi
    cd "$BUILD_DIR/$GNUCHESS_DIR"
    make distclean 2>/dev/null || make clean 2>/dev/null || true
    ./configure CFLAGS="-g -${opt}" CXXFLAGS="-g -${opt}" --quiet --without-readline 2>/dev/null && \
    make -j$(nproc) --quiet 2>/dev/null && {
        bin_path="src/gnuchess"
        actual="$bin_path"
        [ -f "$bin_path" ] && ! file "$bin_path" | grep -q ELF && actual="src/.libs/gnuchess"
        [ -f "$actual" ] || actual="$bin_path"
        if [ -f "$actual" ] && file "$actual" | grep -q ELF; then
            cp "$actual" "$HOME/bfnr-project/$DATA_RAW/${name}_sym"
            cp "$actual" "$HOME/bfnr-project/$DATA_DEBUG/${name}"
            strip -s "$actual" -o "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped"
            echo "  [$opt] compiled"
            COMPILED=$((COMPILED + 1))
        else
            echo "  [$opt] binary not found"
        fi
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/bfnr-project

# ══ 3. wdiff ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 3. wdiff (O0 O1 O2 O3)"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
WDIFF_TAR="wdiff-1.2.2.tar.gz"
WDIFF_DIR="wdiff-1.2.2"
if [ ! -f "$WDIFF_TAR" ]; then
    wget -q "https://ftp.gnu.org/gnu/wdiff/wdiff-1.2.2.tar.gz" -O "$WDIFF_TAR"
fi
[ ! -d "$WDIFF_DIR" ] && tar xf "$WDIFF_TAR"
for opt in O0 O1 O2 O3; do
    name="wdiff_wdiff_${opt}"
    if [ -f "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped" ]; then
        echo "  [$opt] already compiled"
        continue
    fi
    cd "$BUILD_DIR/$WDIFF_DIR"
    make distclean 2>/dev/null || make clean 2>/dev/null || true
    ./configure CFLAGS="-g -${opt}" --quiet 2>/dev/null && \
    make -j$(nproc) --quiet 2>/dev/null && {
        bin="src/wdiff"
        actual="$bin"
        [ -f "$bin" ] && ! file "$bin" | grep -q ELF && actual="src/.libs/wdiff"
        [ -f "$actual" ] || actual="$bin"
        if [ -f "$actual" ] && file "$actual" | grep -q ELF; then
            cp "$actual" "$HOME/bfnr-project/$DATA_RAW/${name}_sym"
            cp "$actual" "$HOME/bfnr-project/$DATA_DEBUG/${name}"
            strip -s "$actual" -o "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped"
            echo "  [$opt] compiled"
            COMPILED=$((COMPILED + 1))
        fi
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/bfnr-project

# ══ 4. wget O1/O2/O3 ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 4. wget (O1 O2 O3)"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
WGET_TAR="wget-1.21.4.tar.gz"
WGET_DIR="wget-1.21.4"
if [ ! -f "$WGET_TAR" ]; then
    wget -q "https://ftp.gnu.org/gnu/wget/wget-1.21.4.tar.gz" -O "$WGET_TAR"
fi
[ ! -d "$WGET_DIR" ] && tar xf "$WGET_TAR"
for opt in O0 O1 O2 O3; do
    name="wget_wget_${opt}"
    if [ -f "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped" ]; then
        echo "  [$opt] already compiled"
        continue
    fi
    cd "$BUILD_DIR/$WGET_DIR"
    make distclean 2>/dev/null || make clean 2>/dev/null || true
    ./configure CFLAGS="-g -${opt}" --quiet --without-ssl 2>/dev/null && \
    make -j$(nproc) --quiet 2>/dev/null && {
        bin="src/wget"
        actual="$bin"
        [ -f "$bin" ] && ! file "$bin" | grep -q ELF && actual="src/.libs/wget"
        [ -f "$actual" ] || actual="$bin"
        if [ -f "$actual" ] && file "$actual" | grep -q ELF; then
            cp "$actual" "$HOME/bfnr-project/$DATA_RAW/${name}_sym"
            cp "$actual" "$HOME/bfnr-project/$DATA_DEBUG/${name}"
            strip -s "$actual" -o "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped"
            echo "  [$opt] compiled"
            COMPILED=$((COMPILED + 1))
        fi
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/bfnr-project

# ══ 5. zstd O1/O3 ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 5. zstd (O0 O1 O2 O3) [Makefile]"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
ZSTD_TAR="zstd-1.5.6.tar.gz"
ZSTD_DIR="zstd-1.5.6"
if [ ! -f "$ZSTD_TAR" ]; then
    wget -q "https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz" -O "$ZSTD_TAR"
fi
[ ! -d "$ZSTD_DIR" ] && tar xf "$ZSTD_TAR"
for opt in O0 O1 O2 O3; do
    name="zstd_zstd_${opt}"
    if [ -f "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped" ] && [ -f "$HOME/bfnr-project/$DATA_RAW/${name}_sym" ]; then
        echo "  [$opt] already compiled"
        continue
    fi
    cd "$BUILD_DIR/$ZSTD_DIR"
    make clean 2>/dev/null || true
    make -j$(nproc) CFLAGS="-g -${opt}" LDFLAGS="-g" 2>/dev/null && {
        if [ -f programs/zstd ]; then
            cp programs/zstd "$HOME/bfnr-project/$DATA_RAW/${name}_sym"
            cp programs/zstd "$HOME/bfnr-project/$DATA_DEBUG/${name}"
            strip -s programs/zstd -o "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped"
            echo "  [$opt] compiled"
            COMPILED=$((COMPILED + 1))
        fi
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/bfnr-project

# ══ 6. nettle ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 6. nettle (O0 O1 O2 O3)"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
NETTLE_TAR="nettle-3.10.tar.gz"
NETTLE_DIR="nettle-3.10"
if [ ! -f "$NETTLE_TAR" ]; then
    wget -q "https://ftp.gnu.org/gnu/nettle/nettle-3.10.tar.gz" -O "$NETTLE_TAR"
fi
[ ! -d "$NETTLE_DIR" ] && tar xf "$NETTLE_TAR"
for opt in O0 O1 O2 O3; do
    for bin_name in nettle-hash sexp-conv; do
        name="nettle_${bin_name}_${opt}"
        if [ -f "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped" ]; then
            echo "  [$opt] $bin_name already compiled"
            continue
        fi
    done
    cd "$BUILD_DIR/$NETTLE_DIR"
    make distclean 2>/dev/null || make clean 2>/dev/null || true
    ./configure CFLAGS="-g -${opt}" --quiet --disable-shared 2>/dev/null && \
    make -j$(nproc) --quiet 2>/dev/null && {
        for bin_name in nettle-hash sexp-conv; do
            bin="$bin_name"
            actual="$bin"
            [ -f "$bin" ] && ! file "$bin" | grep -q ELF && actual=".libs/$bin_name"
            [ -f "$actual" ] || actual="$bin"
            name="nettle_${bin_name}_${opt}"
            if [ -f "$actual" ] && file "$actual" | grep -q ELF; then
                cp "$actual" "$HOME/bfnr-project/$DATA_RAW/${name}_sym"
                cp "$actual" "$HOME/bfnr-project/$DATA_DEBUG/${name}"
                strip -s "$actual" -o "$HOME/bfnr-project/$DATA_STRIPPED/${name}_stripped"
                echo "  [$opt] $bin_name"
                COMPILED=$((COMPILED + 1))
            fi
        done
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/bfnr-project

echo ""
echo "═══════════════════════════════════════════════"
echo " Phase 2 compilation done. Compiled: $COMPILED"
echo "═══════════════════════════════════════════════"
echo ""
echo "Next: run BAP pipeline"
echo "  bash scripts/expand_bap_pipeline.sh"
