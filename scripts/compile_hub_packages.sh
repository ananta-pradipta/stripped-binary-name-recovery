#!/bin/bash
# ============================================================
# Compile universal "hub" packages for training data expansion
# Static builds to expose all internal library functions
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

source configs/packages.conf 2>/dev/null || true

BUILD_DIR="/tmp/hub_builds"
INSTALL_DIR="/tmp/hub_install"
PROJECT_ROOT="$(pwd)"
STRIPPED_DIR="$PROJECT_ROOT/data/stripped"
DEBUG_DIR="$PROJECT_ROOT/data/debug"
OPT_LEVELS=("O0" "O2")

mkdir -p "$BUILD_DIR" "$INSTALL_DIR" "$STRIPPED_DIR" "$DEBUG_DIR"

log() { echo "=== [$1] $2 ==="; }

compile_package() {
    local name="$1"
    local url="$2"
    local configure_cmd="$3"
    local binary_paths="$4"  # colon-separated relative paths to binaries

    log "$name" "Starting"

    local src_dir="$BUILD_DIR/$name"
    local archive="$BUILD_DIR/$(basename "$url")"

    # Download (re-download if zero-byte)
    if [ ! -s "$archive" ]; then
        log "$name" "Downloading"
        rm -f "$archive"
        wget -q -O "$archive" "$url" || { log "$name" "Download FAILED"; return 1; }
    fi

    for opt in "${OPT_LEVELS[@]}"; do
        local build_dir="$BUILD_DIR/${name}_${opt}"
        log "$name" "Building $opt"

        # Extract fresh
        rm -rf "$build_dir"
        mkdir -p "$build_dir"
        tar xf "$archive" -C "$build_dir" --strip-components=1

        cd "$build_dir"

        # Set compiler flags
        export CFLAGS="-${opt} -g -static"
        export LDFLAGS="-static"

        # Configure
        eval "$configure_cmd"

        # Build
        make -j$(nproc) 2>&1 | tail -5 || {
            # Try without static if static fails
            log "$name" "Static build failed for $opt, trying with -g only"
            export CFLAGS="-${opt} -g"
            export LDFLAGS=""
            make clean 2>/dev/null || true
            eval "$configure_cmd"
            make -j$(nproc) 2>&1 | tail -5
        }

        # Copy binaries
        IFS=':' read -ra BINS <<< "$binary_paths"
        for bin_rel in "${BINS[@]}"; do
            local bin_path="$build_dir/$bin_rel"
            if [ -f "$bin_path" ]; then
                local bin_name=$(basename "$bin_rel")
                local debug_name="${name}_${bin_name}_${opt}"
                local stripped_name="${name}_${bin_name}_${opt}"

                cp "$bin_path" "$DEBUG_DIR/${debug_name}"
                cp "$bin_path" "$STRIPPED_DIR/${stripped_name}"
                strip "$STRIPPED_DIR/${stripped_name}"

                local func_count=$(nm "$DEBUG_DIR/${debug_name}" -g --defined-only 2>/dev/null | grep -c ' T \| t ' || echo 0)
                log "$name" "$bin_name $opt: $func_count functions"
            else
                log "$name" "WARNING: $bin_rel not found"
            fi
        done

        cd "$OLDPWD"
    done

    log "$name" "Done"
}

# ============================================================
# Hub Package Definitions
# ============================================================

echo "╔══════════════════════════════════════════════════╗"
echo "║  Compiling Universal Hub Packages                ║"
echo "╚══════════════════════════════════════════════════╝"

# 1. zlib (6,230 reverse deps)
compile_package "zlib" \
    "https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz" \
    "./configure --static" \
    "minigzip:example"

# 2. xz-utils / liblzma (1,420 reverse deps)
compile_package "xz" \
    "https://github.com/tukaani-project/xz/releases/download/v5.4.5/xz-5.4.5.tar.gz" \
    "./configure --disable-shared --enable-static" \
    "src/xz/xz:src/xzdec/xzdec:src/lzmainfo/lzmainfo"

# 3. zstd (1,150 reverse deps) — uses Makefile, not autotools
log "zstd" "Starting"
ZSTD_URL="https://github.com/facebook/zstd/releases/download/v1.5.5/zstd-1.5.5.tar.gz"
ZSTD_ARCHIVE="$BUILD_DIR/zstd-1.5.5.tar.gz"
[ ! -s "$ZSTD_ARCHIVE" ] && rm -f "$ZSTD_ARCHIVE" && wget -q -O "$ZSTD_ARCHIVE" "$ZSTD_URL"
for opt in "${OPT_LEVELS[@]}"; do
    build_dir="$BUILD_DIR/zstd_${opt}"
    rm -rf "$build_dir" && mkdir -p "$build_dir"
    tar xf "$ZSTD_ARCHIVE" -C "$build_dir" --strip-components=1
    cd "$build_dir"
    log "zstd" "Building $opt"
    make -j$(nproc) CFLAGS="-${opt} -g" 2>&1 | tail -3 || true
    if [ -f "programs/zstd" ]; then
        cp "programs/zstd" "$DEBUG_DIR/zstd_zstd_${opt}"
        cp "programs/zstd" "$STRIPPED_DIR/zstd_zstd_${opt}"
        strip "$STRIPPED_DIR/zstd_zstd_${opt}"
        func_count=$(nm "$DEBUG_DIR/zstd_zstd_${opt}" -g --defined-only 2>/dev/null | grep -c ' [Tt] ' || echo 0)
        log "zstd" "zstd $opt: $func_count functions"
    fi
    cd "$OLDPWD"
done
log "zstd" "Done"

# 4. libxml2 (844 reverse deps)
compile_package "libxml2" \
    "https://download.gnome.org/sources/libxml2/2.12/libxml2-2.12.4.tar.xz" \
    "./configure --disable-shared --enable-static --without-python --without-icu" \
    "xmllint:.libs/xmllint"

# 5. libpng (574 reverse deps)
compile_package "libpng" \
    "https://download.sourceforge.net/libpng/libpng-1.6.40.tar.gz" \
    "./configure --disable-shared --enable-static" \
    "pngtest:.libs/pngtest"

# 6. libexpat (273 reverse deps)
compile_package "expat" \
    "https://github.com/libexpat/libexpat/releases/download/R_2_6_0/expat-2.6.0.tar.gz" \
    "./configure --disable-shared --enable-static" \
    "xmlwf/.libs/xmlwf"

# 7. pcre2 (92 reverse deps, but widely vendored)
compile_package "pcre2" \
    "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.42/pcre2-10.42.tar.gz" \
    "./configure --disable-shared --enable-static" \
    "pcre2grep/.libs/pcre2grep:pcre2test/.libs/pcre2test"

# 8. jansson (115 reverse deps)
compile_package "jansson" \
    "https://github.com/akheron/jansson/releases/download/v2.14/jansson-2.14.tar.gz" \
    "./configure --disable-shared --enable-static" \
    "src/.libs/libjansson.a"

# 9. lz4 (138 reverse deps) — uses Makefile
log "lz4" "Starting"
LZ4_URL="https://github.com/lz4/lz4/releases/download/v1.9.4/lz4-1.9.4.tar.gz"
LZ4_ARCHIVE="$BUILD_DIR/lz4-1.9.4.tar.gz"
[ ! -s "$LZ4_ARCHIVE" ] && rm -f "$LZ4_ARCHIVE" && wget -q -O "$LZ4_ARCHIVE" "$LZ4_URL"
for opt in "${OPT_LEVELS[@]}"; do
    build_dir="$BUILD_DIR/lz4_${opt}"
    rm -rf "$build_dir" && mkdir -p "$build_dir"
    tar xf "$LZ4_ARCHIVE" -C "$build_dir" --strip-components=1
    cd "$build_dir"
    log "lz4" "Building $opt"
    make -j$(nproc) CFLAGS="-${opt} -g" 2>&1 | tail -3 || true
    if [ -f "programs/lz4" ]; then
        cp "programs/lz4" "$DEBUG_DIR/lz4_lz4_${opt}"
        cp "programs/lz4" "$STRIPPED_DIR/lz4_lz4_${opt}"
        strip "$STRIPPED_DIR/lz4_lz4_${opt}"
        func_count=$(nm "$DEBUG_DIR/lz4_lz4_${opt}" -g --defined-only 2>/dev/null | grep -c ' [Tt] ' || echo 0)
        log "lz4" "lz4 $opt: $func_count functions"
    fi
    cd "$OLDPWD"
done
log "lz4" "Done"

# 10. libyaml (73 reverse deps)
compile_package "libyaml" \
    "https://github.com/yaml/libyaml/releases/download/0.2.5/yaml-0.2.5.tar.gz" \
    "./configure --disable-shared --enable-static" \
    "tests/run-emitter:tests/run-parser"

# 11. libarchive (99 reverse deps)
compile_package "libarchive" \
    "https://github.com/libarchive/libarchive/releases/download/v3.7.2/libarchive-3.7.2.tar.gz" \
    "./configure --disable-shared --enable-static --without-openssl --without-xml2" \
    "bsdtar:bsdcat:bsdcpio"

# 12. libsodium (72 reverse deps)
compile_package "libsodium" \
    "https://download.libsodium.org/libsodium/releases/libsodium-1.0.19.tar.gz" \
    "./configure --disable-shared --enable-static" \
    "test/default/.libs/aead_aes256gcm:test/default/.libs/auth"

# ============================================================
# OpenSSL — special handling (not autotools)
# ============================================================
log "openssl" "Starting"
OPENSSL_URL="https://github.com/openssl/openssl/releases/download/openssl-3.2.1/openssl-3.2.1.tar.gz"
OPENSSL_ARCHIVE="$BUILD_DIR/openssl-3.2.1.tar.gz"

if [ ! -s "$OPENSSL_ARCHIVE" ]; then
    log "openssl" "Downloading"
    rm -f "$OPENSSL_ARCHIVE"
    wget -q -O "$OPENSSL_ARCHIVE" "$OPENSSL_URL"
fi

for opt in "${OPT_LEVELS[@]}"; do
    build_dir="$BUILD_DIR/openssl_${opt}"
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    tar xf "$OPENSSL_ARCHIVE" -C "$build_dir" --strip-components=1
    cd "$build_dir"

    log "openssl" "Building $opt"
    ./config no-shared "-${opt}" -g 2>&1 | tail -3
    make -j$(nproc) 2>&1 | tail -5

    # OpenSSL produces the openssl CLI tool
    for bin in apps/openssl; do
        if [ -f "$bin" ]; then
            bin_name=$(basename "$bin")
            cp "$bin" "$DEBUG_DIR/openssl_${bin_name}_${opt}"
            cp "$bin" "$STRIPPED_DIR/openssl_${bin_name}_${opt}"
            strip "$STRIPPED_DIR/openssl_${bin_name}_${opt}"
            func_count=$(nm "$DEBUG_DIR/openssl_${bin_name}_${opt}" -g --defined-only 2>/dev/null | grep -c ' T \| t ' || echo 0)
            log "openssl" "$bin_name $opt: $func_count functions"
        fi
    done
    cd "$OLDPWD"
done
log "openssl" "Done"

# ============================================================
# Summary
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Compilation Complete                             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Debug binaries:    $(ls $DEBUG_DIR/*_O0 $DEBUG_DIR/*_O2 2>/dev/null | wc -l)"
echo "Stripped binaries: $(ls $STRIPPED_DIR/*_O0 $STRIPPED_DIR/*_O2 2>/dev/null | wc -l)"
echo ""
echo "Next steps:"
echo "  1. Run BAP preprocessing: bash scripts/03_preprocess.sh"
echo "  2. Update match_index.json"
echo "  3. Retrain model with expanded dataset"
