#!/bin/bash
# ============================================================
# Compile cross-project evaluation packages
# These packages USE the hub libraries we trained on
# Static builds to include library functions in the binary
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="/tmp/xproj_builds"
INSTALL_DIR="/tmp/hub_install"  # Where hub libs were installed
STRIPPED_DIR="data/stripped"
DEBUG_DIR="data/debug"
OPT_LEVELS=("O0" "O2")

mkdir -p "$BUILD_DIR" "$STRIPPED_DIR" "$DEBUG_DIR"

log() { echo "=== [$1] $2 ==="; }

compile_package() {
    local name="$1"
    local url="$2"
    local configure_cmd="$3"
    local binary_paths="$4"

    log "$name" "Starting"

    local archive="$BUILD_DIR/$(basename "$url")"

    if [ ! -f "$archive" ]; then
        log "$name" "Downloading"
        wget -q -O "$archive" "$url" || { log "$name" "Download failed"; return 1; }
    fi

    for opt in "${OPT_LEVELS[@]}"; do
        local build_dir="$BUILD_DIR/${name}_${opt}"
        log "$name" "Building $opt"

        rm -rf "$build_dir"
        mkdir -p "$build_dir"
        tar xf "$archive" -C "$build_dir" --strip-components=1 2>/dev/null || {
            # Some archives have different structure
            tar xf "$archive" -C "$build_dir" 2>/dev/null
        }

        cd "$build_dir"

        export CFLAGS="-${opt} -g"
        export LDFLAGS="-static"
        export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:$INSTALL_DIR/lib64/pkgconfig"
        export CPPFLAGS="-I$INSTALL_DIR/include"
        export LIBS="-L$INSTALL_DIR/lib -L$INSTALL_DIR/lib64"

        # Configure and build
        eval "$configure_cmd" 2>&1 | tail -3 || {
            log "$name" "Configure failed for $opt, trying without static"
            export LDFLAGS=""
            eval "$configure_cmd" 2>&1 | tail -3
        }

        make -j$(nproc) 2>&1 | tail -5 || {
            log "$name" "Build failed for $opt"
            cd "$OLDPWD"
            continue
        }

        # Copy binaries
        IFS=':' read -ra BINS <<< "$binary_paths"
        for bin_rel in "${BINS[@]}"; do
            # Handle glob patterns
            for bin_path in $build_dir/$bin_rel; do
                if [ -f "$bin_path" ]; then
                    local bin_name=$(basename "$bin_path")
                    local out_name="${name}_${bin_name}_${opt}"

                    cp "$bin_path" "$DEBUG_DIR/${out_name}"
                    cp "$bin_path" "$STRIPPED_DIR/${out_name}"
                    strip "$STRIPPED_DIR/${out_name}"

                    local func_count=$(nm "$DEBUG_DIR/${out_name}" -g --defined-only 2>/dev/null | grep -c ' [Tt] ' || echo 0)
                    log "$name" "$bin_name $opt: $func_count functions"
                fi
            done
        done

        cd "$OLDPWD"
    done

    log "$name" "Done"
}

echo "╔══════════════════════════════════════════════════╗"
echo "║  Compiling Cross-Project Evaluation Packages      ║"
echo "╚══════════════════════════════════════════════════╝"

# 1. pigz — parallel gzip (tiny, pure zlib test)
compile_package "pigz" \
    "https://zlib.net/pigz/pigz-2.8.tar.gz" \
    "true"  \
    "pigz"
# pigz uses simple Makefile, need special handling
# Actually override:
log "pigz" "Special build"
for opt in "${OPT_LEVELS[@]}"; do
    build_dir="$BUILD_DIR/pigz_${opt}"
    if [ -d "$build_dir" ]; then
        cd "$build_dir"
        make clean 2>/dev/null || true
        make CFLAGS="-${opt} -g" -j$(nproc) 2>&1 | tail -3 || true
        for bin in pigz; do
            if [ -f "$bin" ]; then
                cp "$bin" "$DEBUG_DIR/pigz_${bin}_${opt}"
                cp "$bin" "$STRIPPED_DIR/pigz_${bin}_${opt}"
                strip "$STRIPPED_DIR/pigz_${bin}_${opt}"
                func_count=$(nm "$DEBUG_DIR/pigz_${bin}_${opt}" -g --defined-only 2>/dev/null | grep -c ' [Tt] ' || echo 0)
                log "pigz" "$bin $opt: $func_count functions"
            fi
        done
        cd "$OLDPWD"
    fi
done

# 2. rsync (zlib + openssl)
compile_package "rsync" \
    "https://download.samba.org/pub/rsync/src/rsync-3.2.7.tar.gz" \
    "./configure --disable-xxhash --disable-zstd --disable-lz4 --disable-openssl" \
    "rsync"

# 3. tmux (ncurses + libevent)
compile_package "tmux" \
    "https://github.com/tmux/tmux/releases/download/3.3a/tmux-3.3a.tar.gz" \
    "./configure" \
    "tmux"

# 4. socat (openssl + readline)
compile_package "socat" \
    "http://www.dest-unreach.org/socat/download/socat-1.8.0.0.tar.gz" \
    "./configure --disable-openssl" \
    "socat"

# 5. dropbear (lightweight SSH, own crypto + zlib)
compile_package "dropbear" \
    "https://matt.ucc.asn.au/dropbear/releases/dropbear-2024.85.tar.bz2" \
    "./configure --disable-zlib" \
    "dropbear:dbclient:dropbearkey"

# 6. fossil (VCS, zlib + openssl)
compile_package "fossil" \
    "https://fossil-scm.org/home/tarball/version-2.23/fossil-2.23.tar.gz" \
    "./configure" \
    "fossil"

# 7. xmlstarlet (libxml2 heavy user)
compile_package "xmlstarlet" \
    "https://downloads.sourceforge.net/xmlstar/xmlstarlet-1.6.1.tar.gz" \
    "./configure --with-libxml-prefix=/usr" \
    "xml"

# 8. stunnel (TLS proxy, openssl heavy)
compile_package "stunnel" \
    "https://www.stunnel.org/downloads/stunnel-5.72.tar.gz" \
    "./configure" \
    "src/stunnel"

# 9. lighttpd (web server)
compile_package "lighttpd" \
    "https://download.lighttpd.net/lighttpd/releases-1.4.x/lighttpd-1.4.74.tar.gz" \
    "./configure --without-openssl --without-pcre2 --without-zlib" \
    "src/lighttpd"

# 10. w3m (text browser)
compile_package "w3m" \
    "https://github.com/tats/w3m/releases/download/v0.5.3+git20230129/w3m-0.5.3+git20230129.tar.gz" \
    "./configure --disable-image" \
    "w3m"

# ============================================================
# Keep existing cross-project packages that aren't moving to training
# ============================================================
echo ""
echo "Note: nginx, rcs, tree, dos2unix remain as cross-project packages"
echo "(they don't heavily use hub libraries, testing true generalization)"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Cross-Project Compilation Complete               ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Debug binaries:    $(ls $DEBUG_DIR/ 2>/dev/null | wc -l) total"
echo "Stripped binaries: $(ls $STRIPPED_DIR/ 2>/dev/null | wc -l) total"
