#!/bin/bash
# Compile 6 truly-unseen packages at O0/O1/O2/O3 for cross-project sanity eval.
# lighttpd, fossil, yash, tinycc, zsh, cvs (mercurial is python, substituted cvs)
set -e

PROJECT_ROOT=/mmfs1/project/hz79/_shared/cs785
DEBUG_OUT=$PROJECT_ROOT/data/cross_project/debug_unseen
STRIPPED_OUT=$PROJECT_ROOT/data/cross_project/stripped_unseen
SRC_DIR=/tmp/unseen_xproj_sources
mkdir -p "$DEBUG_OUT" "$STRIPPED_OUT" "$SRC_DIR"

log() { echo "=== [$1] $2 ==="; }

download_extract() {
    local name="$1"; local url="$2"
    local dir="$SRC_DIR/$name"
    if [ -d "$dir" ]; then return 0; fi
    local ar="$SRC_DIR/$(basename "$url")"
    [ -f "$ar" ] || wget -q -O "$ar" "$url"
    mkdir -p "$dir"
    tar xf "$ar" -C "$dir" --strip-components=1 2>/dev/null || tar xf "$ar" -C "$dir"
}

# Build one package at 4 opt levels, produce binary_name_O{0,1,2,3}{,_stripped}
build_opt() {
    local pkg="$1"; local binname="$2"; local srcdir="$3"; local makecmd="${4:-make}"
    local configure_script="${5:-./configure}"
    for opt in O0 O1 O2 O3; do
        local bdir="$srcdir/build_$opt"
        if [ -f "$DEBUG_OUT/${pkg}_${binname}_${opt}" ]; then
            log "$pkg-$opt" "exists, skipping"
            continue
        fi
        log "$pkg-$opt" "building"
        mkdir -p "$bdir"
        (cd "$bdir" && CFLAGS="-g -$opt" CXXFLAGS="-g -$opt" $configure_script 2>&1 | tail -5 && $makecmd -j8 2>&1 | tail -5) || { log "$pkg-$opt" "build FAILED"; continue; }
        local src_bin="$bdir/src/$binname"
        [ -f "$src_bin" ] || src_bin="$bdir/$binname"
        [ -f "$src_bin" ] || { log "$pkg-$opt" "no binary at $bdir/src/$binname"; continue; }
        cp "$src_bin" "$DEBUG_OUT/${pkg}_${binname}_${opt}"
        strip -s "$src_bin" -o "$STRIPPED_OUT/${pkg}_${binname}_${opt}_stripped"
    done
}

# 1. lighttpd
download_extract "lighttpd" "https://download.lighttpd.net/lighttpd/releases-1.4.x/lighttpd-1.4.76.tar.xz"
build_opt "lighttpd" "lighttpd" "$SRC_DIR/lighttpd" "make" "./configure --without-zlib --without-bzip2 --disable-shared"

# 2. yash
download_extract "yash" "https://osdn.net/frs/g_redir.php?m=iijnet&f=yash%2F80985%2Fyash-2.57.tar.xz"
build_opt "yash" "yash" "$SRC_DIR/yash" "make"

# 3. tinycc
download_extract "tinycc" "https://download.savannah.gnu.org/releases/tinycc/tcc-0.9.27.tar.bz2"
build_opt "tinycc" "tcc" "$SRC_DIR/tinycc" "make"

# 4. zsh
download_extract "zsh" "https://sourceforge.net/projects/zsh/files/zsh/5.9/zsh-5.9.tar.xz/download"
build_opt "zsh" "zsh" "$SRC_DIR/zsh" "make" "./configure --disable-shared"

# 5. fossil
download_extract "fossil" "https://fossil-scm.org/home/tarball/version-2.25/fossil-src-2.25.tar.gz"
build_opt "fossil" "fossil" "$SRC_DIR/fossil" "make"

# 6. cvs (replacing mercurial which is Python)
download_extract "cvs" "https://ftp.gnu.org/non-gnu/cvs/source/feature/1.12.13/cvs-1.12.13.tar.gz"
build_opt "cvs" "cvs" "$SRC_DIR/cvs" "make"

log "DONE" "all builds"
ls "$STRIPPED_OUT/" | head -30
