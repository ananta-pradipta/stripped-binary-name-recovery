#!/bin/bash
# ============================================================
# Dataset Expansion: BAP-only, fill O1/O3 gaps + new packages
# ============================================================
# Tier 1: Recompile existing packages at missing O1/O3
# Tier 2: Compile new packages (gdb, ed, parted, etc.)
# Tier 3: More binaries from existing packages
#
# Usage: bash scripts/expand_bap_only.sh [tier1|tier2|tier3|all]
# ============================================================
set -eo pipefail
source ~/bfnr-project/activate.sh
cd ~/bfnr-project
# Don't exit on individual package failures
set +e

BUILD_DIR="$HOME/bfnr-project/build_tmp"
DATA_RAW="$HOME/bfnr-project/data/raw"
DATA_STRIPPED="$HOME/bfnr-project/data/stripped"
DATA_DEBUG="$HOME/bfnr-project/data/debug"
DATA_BIR="$HOME/bfnr-project/data/bir"
DATA_GRAPHS="$HOME/bfnr-project/data/graphs"
DATA_LABELS="$HOME/bfnr-project/data/labels"
DATA_EXT="$HOME/bfnr-project/data/external_calls"

mkdir -p "$BUILD_DIR" "$DATA_RAW" "$DATA_STRIPPED" "$DATA_DEBUG" "$DATA_BIR" "$DATA_GRAPHS" "$DATA_LABELS" "$DATA_EXT"

TIER="${1:-all}"
COMPILED=0
FAILED=0

# ── Helper: compile one package at specific opt levels ──
compile_at_opts() {
    local pkg_name="$1"
    local url="$2"
    local tarball="$3"
    local src_dir="$4"
    shift 4
    local opts=()
    local bins=()
    local reading_bins=false

    # Parse args: opts come first (O0/O1/O2/O3), then -- separator, then binary paths
    for arg in "$@"; do
        if [ "$arg" = "--" ]; then
            reading_bins=true
            continue
        fi
        if $reading_bins; then
            bins+=("$arg")
        else
            opts+=("$arg")
        fi
    done

    echo ""
    echo "── $pkg_name (${opts[*]}) ──"
    cd "$BUILD_DIR"

    # Download
    if [ ! -f "$tarball" ]; then
        echo "  Downloading..."
        wget -q "$url" -O "$tarball" 2>&1 || {
            echo "  ⚠ Download failed. Skipping."
            FAILED=$((FAILED + 1))
            cd ~/bfnr-project
            return 1
        }
    fi

    # Extract
    if [ ! -d "$src_dir" ]; then
        tar xf "$tarball" 2>/dev/null || {
            echo "  ⚠ Extract failed. Skipping."
            FAILED=$((FAILED + 1))
            cd ~/bfnr-project
            return 1
        }
    fi

    for opt in "${opts[@]}"; do
        # Check if all binaries already exist for this opt
        local all_exist=true
        for bin_path in "${bins[@]}"; do
            local bin_name=$(basename "$bin_path")
            local name="${pkg_name}_${bin_name}_${opt}"
            if [ ! -f "$DATA_STRIPPED/${name}_stripped" ]; then
                all_exist=false
                break
            fi
        done
        if $all_exist; then
            echo "  [$opt] already compiled, skipping"
            continue
        fi

        cd "$BUILD_DIR/$src_dir"
        make distclean 2>/dev/null || make clean 2>/dev/null || true

        ./configure CFLAGS="-g -${opt}" --quiet 2>/dev/null || {
            echo "  [$opt] configure failed"
            FAILED=$((FAILED + 1))
            cd "$BUILD_DIR"
            continue
        }

        make -j$(nproc) --quiet 2>/dev/null || {
            echo "  [$opt] compile failed"
            FAILED=$((FAILED + 1))
            cd "$BUILD_DIR"
            continue
        }

        local count=0
        for bin_path in "${bins[@]}"; do
            local actual_bin="$bin_path"
            # For autotools: real binary is in .libs/ (wrapper is a shell script)
            if [ -f "$bin_path" ] && ! file "$bin_path" 2>/dev/null | grep -q "ELF"; then
                local libs_path="$(dirname "$bin_path")/.libs/$(basename "$bin_path")"
                [ -f "$libs_path" ] && actual_bin="$libs_path"
            fi
            if [ ! -f "$actual_bin" ]; then
                local libs_path="$(dirname "$bin_path")/.libs/$(basename "$bin_path")"
                [ -f "$libs_path" ] && actual_bin="$libs_path"
            fi
            if [ -f "$actual_bin" ] && file "$actual_bin" 2>/dev/null | grep -q "ELF"; then
                local bin_name=$(basename "$bin_path")
                local name="${pkg_name}_${bin_name}_${opt}"
                cp "$actual_bin" "$DATA_RAW/${name}_sym"
                cp "$actual_bin" "$DATA_DEBUG/${name}"
                strip -s "$actual_bin" -o "$DATA_STRIPPED/${name}_stripped"
                count=$((count + 1))
                COMPILED=$((COMPILED + 1))
            else
                echo "    ⚠ $(basename "$bin_path") not found or not ELF"
            fi
        done
        echo "  [$opt] $count binaries compiled"
        cd "$BUILD_DIR"
    done

    cd ~/bfnr-project
}

# ══════════════════════════════════════════════
# TIER 1: Fill O1/O3 gaps for existing packages
# ══════════════════════════════════════════════
if [ "$TIER" = "tier1" ] || [ "$TIER" = "all" ]; then
    echo "═══════════════════════════════════════════════"
    echo " TIER 1: Filling O1/O3 gaps"
    echo "═══════════════════════════════════════════════"

    compile_at_opts libxml2 \
        "https://download.gnome.org/sources/libxml2/2.12/libxml2-2.12.6.tar.xz" \
        "libxml2-2.12.6.tar.xz" "libxml2-2.12.6" \
        O1 O3 -- xmllint

    compile_at_opts libarchive \
        "https://www.libarchive.org/downloads/libarchive-3.7.2.tar.xz" \
        "libarchive-3.7.2.tar.xz" "libarchive-3.7.2" \
        O1 O3 -- bsdtar bsdcpio bsdcat

    compile_at_opts libpng \
        "https://download.sourceforge.net/libpng/libpng-1.6.42.tar.xz" \
        "libpng-1.6.42.tar.xz" "libpng-1.6.42" \
        O1 O3 -- pngtest

    compile_at_opts pcre2 \
        "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.42/pcre2-10.42.tar.gz" \
        "pcre2-10.42.tar.gz" "pcre2-10.42" \
        O1 O3 -- pcre2grep pcre2test

    compile_at_opts libyaml \
        "https://github.com/yaml/libyaml/releases/download/0.2.5/yaml-0.2.5.tar.gz" \
        "yaml-0.2.5.tar.gz" "yaml-0.2.5" \
        O1 O3 -- tests/run-emitter tests/run-parser

    compile_at_opts expat \
        "https://github.com/libexpat/libexpat/releases/download/R_2_6_2/expat-2.6.2.tar.xz" \
        "expat-2.6.2.tar.xz" "expat-2.6.2" \
        O1 O3 -- xmlwf/xmlwf

    # bzip2, dos2unix, tree use plain Makefiles (not autotools)
    echo ""
    echo "── bzip2 (O1 O3) [Makefile] ──"
    cd "$BUILD_DIR"
    if [ ! -f "bzip2-1.0.8.tar.gz" ]; then
        wget -q "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz" -O "bzip2-1.0.8.tar.gz"
    fi
    [ ! -d "bzip2-1.0.8" ] && tar xf "bzip2-1.0.8.tar.gz"
    for opt in O1 O3; do
        name="bzip2_bzip2_${opt}"
        if [ -f "$DATA_STRIPPED/${name}_stripped" ]; then
            echo "  [$opt] already compiled, skipping"
            continue
        fi
        cd "$BUILD_DIR/bzip2-1.0.8"
        make clean 2>/dev/null || true
        make -j$(nproc) CFLAGS="-g -${opt} -Wall" bzip2 2>/dev/null && {
            cp bzip2 "$DATA_RAW/${name}_sym"
            cp bzip2 "$DATA_DEBUG/${name}"
            strip -s bzip2 -o "$DATA_STRIPPED/${name}_stripped"
            echo "  [$opt] bzip2"
            COMPILED=$((COMPILED + 1))
        } || echo "  [$opt] FAILED"
        cd "$BUILD_DIR"
    done

    echo ""
    echo "── dos2unix (O1 O3) [Makefile] ──"
    cd "$BUILD_DIR"
    if [ ! -f "dos2unix-7.5.2.tar.gz" ]; then
        wget -q "https://waterlan.home.xs4all.nl/dos2unix/dos2unix-7.5.2.tar.gz" -O "dos2unix-7.5.2.tar.gz"
    fi
    [ ! -d "dos2unix-7.5.2" ] && tar xf "dos2unix-7.5.2.tar.gz"
    for opt in O1 O3; do
        cd "$BUILD_DIR/dos2unix-7.5.2"
        make clean 2>/dev/null || true
        make -j$(nproc) CFLAGS="-g -${opt}" 2>/dev/null && {
            for bin in dos2unix unix2dos; do
                name="dos2unix_${bin}_${opt}"
                if [ -f "$bin" ] && [ ! -f "$DATA_STRIPPED/${name}_stripped" ]; then
                    cp "$bin" "$DATA_RAW/${name}_sym"
                    cp "$bin" "$DATA_DEBUG/${name}"
                    strip -s "$bin" -o "$DATA_STRIPPED/${name}_stripped"
                    echo "  [$opt] $bin"
                    COMPILED=$((COMPILED + 1))
                fi
            done
        } || echo "  [$opt] FAILED"
        cd "$BUILD_DIR"
    done

    echo ""
    echo "── tree (O1 O3) [Makefile] ──"
    cd "$BUILD_DIR"
    TREE_TAR="tree-2.1.3.tar.gz"
    TREE_DIR="tree-2.1.3"
    if [ ! -f "$TREE_TAR" ]; then
        wget -q "https://gitlab.com/OldManProgrammer/unix-tree/-/archive/2.1.3/unix-tree-2.1.3.tar.gz" -O "$TREE_TAR" || \
        wget -q "http://mama.indstate.edu/users/ice/tree/src/tree-2.1.3.tgz" -O "$TREE_TAR" || true
    fi
    [ ! -d "$TREE_DIR" ] && tar xf "$TREE_TAR" 2>/dev/null
    # tree source dir might be unix-tree-2.1.3
    [ ! -d "$TREE_DIR" ] && [ -d "unix-tree-2.1.3" ] && TREE_DIR="unix-tree-2.1.3"
    for opt in O1 O3; do
        name="tree_tree_${opt}"
        if [ -f "$DATA_STRIPPED/${name}_stripped" ]; then
            echo "  [$opt] already compiled, skipping"
            continue
        fi
        cd "$BUILD_DIR/$TREE_DIR"
        make clean 2>/dev/null || true
        make -j$(nproc) CFLAGS="-g -${opt}" 2>/dev/null && {
            cp tree "$DATA_RAW/${name}_sym"
            cp tree "$DATA_DEBUG/${name}"
            strip -s tree -o "$DATA_STRIPPED/${name}_stripped"
            echo "  [$opt] tree"
            COMPILED=$((COMPILED + 1))
        } || echo "  [$opt] FAILED"
        cd "$BUILD_DIR"
    done

    compile_at_opts rcs \
        "https://ftp.gnu.org/gnu/rcs/rcs-5.10.1.tar.lz" \
        "rcs-5.10.1.tar.lz" "rcs-5.10.1" \
        O1 O3 -- src/rcs

    # libsodium needs special configure
    echo ""
    echo "── libsodium (O1 O3) ──"
    cd "$BUILD_DIR"
    LSODIUM_TAR="libsodium-1.0.19-RELEASE.tar.gz"
    LSODIUM_DIR="libsodium-1.0.19-RELEASE"
    if [ ! -f "$LSODIUM_TAR" ]; then
        wget -q "https://download.libsodium.org/libsodium/releases/libsodium-1.0.19-RELEASE.tar.gz" -O "$LSODIUM_TAR"
    fi
    if [ ! -d "$LSODIUM_DIR" ]; then
        tar xf "$LSODIUM_TAR"
    fi
    for opt in O1 O3; do
        cd "$BUILD_DIR/$LSODIUM_DIR"
        make distclean 2>/dev/null || true
        ./configure CFLAGS="-g -${opt}" --quiet 2>/dev/null && \
        make -j$(nproc) --quiet 2>/dev/null && \
        make check -j$(nproc) --quiet 2>/dev/null || true
        # Find test binaries
        for test_bin in test/default/.libs/aead_aegis128l test/default/.libs/hash test/default/.libs/keygen test/default/.libs/pwhash_scrypt test/default/.libs/scalarmult_ed25519; do
            if [ -f "$test_bin" ]; then
                bin_name=$(basename "$test_bin")
                name="libsodium_${bin_name}_${opt}"
                cp "$test_bin" "$DATA_RAW/${name}_sym"
                cp "$test_bin" "$DATA_DEBUG/${name}"
                strip -s "$test_bin" -o "$DATA_STRIPPED/${name}_stripped"
                echo "  [$opt] $bin_name"
                COMPILED=$((COMPILED + 1))
            fi
        done
        cd "$BUILD_DIR"
    done
    cd ~/bfnr-project

    echo ""
    echo "Tier 1 done. Compiled: $COMPILED, Failed: $FAILED"
fi

# ══════════════════════════════════════════════
# TIER 2: New packages
# ══════════════════════════════════════════════
if [ "$TIER" = "tier2" ] || [ "$TIER" = "all" ]; then
    echo ""
    echo "═══════════════════════════════════════════════"
    echo " TIER 2: New packages"
    echo "═══════════════════════════════════════════════"

    compile_at_opts ed \
        "https://ftp.gnu.org/gnu/ed/ed-1.20.1.tar.lz" \
        "ed-1.20.1.tar.lz" "ed-1.20.1" \
        O0 O1 O2 O3 -- ed

    compile_at_opts sharutils \
        "https://ftp.gnu.org/gnu/sharutils/sharutils-4.15.2.tar.xz" \
        "sharutils-4.15.2.tar.xz" "sharutils-4.15.2" \
        O0 O1 O2 O3 -- src/shar src/unshar src/uuencode src/uudecode

    compile_at_opts diction \
        "https://ftp.gnu.org/gnu/diction/diction-1.14.tar.gz" \
        "diction-1.14.tar.gz" "diction-1.14" \
        O0 O1 O2 O3 -- diction style

    compile_at_opts gcal \
        "https://ftp.gnu.org/gnu/gcal/gcal-4.1.tar.xz" \
        "gcal-4.1.tar.xz" "gcal-4.1" \
        O0 O1 O2 O3 -- src/gcal

    compile_at_opts idutils \
        "https://ftp.gnu.org/gnu/idutils/idutils-4.6.tar.xz" \
        "idutils-4.6.tar.xz" "idutils-4.6" \
        O0 O1 O2 O3 -- src/mkid src/lid src/fid src/fnid src/xtokid

    compile_at_opts parted \
        "https://ftp.gnu.org/gnu/parted/parted-3.6.tar.xz" \
        "parted-3.6.tar.xz" "parted-3.6" \
        O0 O1 O2 O3 -- parted/parted

    # NOTE: recutils is CROSS-PROJECT — do NOT add to training
    # NOTE: curl is already in training (6,134 fns)
    # NOTE: tengine, angie, nginx118 are CROSS-PROJECT — do NOT add

    compile_at_opts gdb \
        "https://ftp.gnu.org/gnu/gdb/gdb-14.2.tar.xz" \
        "gdb-14.2.tar.xz" "gdb-14.2" \
        O0 O2 -- gdb/gdb

    echo ""
    echo "Tier 2 done. Total compiled: $COMPILED, Failed: $FAILED"
fi

# ══════════════════════════════════════════════
# TIER 3: More binaries from existing packages
# ══════════════════════════════════════════════
if [ "$TIER" = "tier3" ] || [ "$TIER" = "all" ]; then
    echo ""
    echo "═══════════════════════════════════════════════"
    echo " TIER 3: More binaries from existing packages"
    echo "═══════════════════════════════════════════════"

    # More coreutils binaries (we already have coreutils2 with some)
    compile_at_opts coreutils3 \
        "https://ftp.gnu.org/gnu/coreutils/coreutils-9.4.tar.xz" \
        "coreutils-9.4.tar.xz" "coreutils-9.4" \
        O0 O1 O2 O3 -- \
        src/dirname src/env src/expr src/id src/link src/ln \
        src/mkdir src/stat src/touch src/yes src/tee src/split \
        src/fold src/expand src/comm src/nproc src/seq

    # More binutils binaries
    compile_at_opts binutils2 \
        "https://ftp.gnu.org/gnu/binutils/binutils-2.42.tar.xz" \
        "binutils-2.42.tar.xz" "binutils-2.42" \
        O0 O1 O2 O3 -- \
        binutils/ar binutils/ranlib binutils/strip-new binutils/objcopy

    # More inetutils binaries
    compile_at_opts inetutils2 \
        "https://ftp.gnu.org/gnu/inetutils/inetutils-2.5.tar.xz" \
        "inetutils-2.5.tar.xz" "inetutils-2.5" \
        O0 O1 O2 O3 -- \
        src/hostname src/logger src/whois src/dnsdomainname src/ifconfig

    echo ""
    echo "Tier 3 done. Total compiled: $COMPILED, Failed: $FAILED"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo " EXPANSION COMPLETE"
echo " Total compiled: $COMPILED binaries"
echo " Failed: $FAILED"
echo "═══════════════════════════════════════════════"
echo ""
echo "Next: Run BAP pipeline on new binaries"
echo "  bash scripts/expand_bap_pipeline.sh"
