#!/bin/bash
# Compile 6 truly-unseen packages at O0/O1/O2/O3 for cross-project sanity eval.
# lighttpd, fossil, yash, tinycc, zsh, cvs (mercurial is python, substituted cvs)
# NOTE: no `set -e` — each build_opt call handles its own failure with `continue`,
# and set -e combined with tar/wget subshells caused silent aborts on yash/OSDN redirects.

PROJECT_ROOT=/mmfs1/project/hz79/_shared/cs785
DEBUG_OUT=$PROJECT_ROOT/data/cross_project/debug_unseen
STRIPPED_OUT=$PROJECT_ROOT/data/cross_project/stripped_unseen
SRC_DIR=/tmp/unseen_xproj_sources
mkdir -p "$DEBUG_OUT" "$STRIPPED_OUT" "$SRC_DIR"

log() { echo "=== [$1] $2 ==="; }

download_extract() {
    local name="$1"; local url="$2"
    local dir="$SRC_DIR/$name"
    # Consider it usable only if it has some content
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        log "$name" "already extracted, skipping download"
        return 0
    fi
    local ar="$SRC_DIR/${name}_source"
    log "$name" "downloading from $url"
    if ! wget --tries=3 --timeout=60 -L -O "$ar" "$url" 2>&1 | tail -5; then
        log "$name" "download FAILED"
        return 1
    fi
    if [ ! -s "$ar" ]; then
        log "$name" "downloaded file is empty"
        return 1
    fi
    mkdir -p "$dir"
    log "$name" "extracting"
    if ! tar xf "$ar" -C "$dir" --strip-components=1 2>/dev/null; then
        tar xf "$ar" -C "$dir" 2>&1 | tail -3 || { log "$name" "extract FAILED"; return 1; }
    fi
    if [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        log "$name" "extract produced empty dir"
        return 1
    fi
    return 0
}

# Build one package at 4 opt levels by running configure + make in $srcdir
# (in-tree build; make distclean between opt levels resets state).
# $5 is the full configure command line relative to srcdir, default "./configure".
# $6 optional list of candidate relative paths for the built binary.
build_opt() {
    local pkg="$1"; local binname="$2"; local srcdir="$3"; local makecmd="${4:-make}"
    local configure_cmd="${5:-./configure}"
    local bin_paths="${6:-src/$binname $binname Src/$binname}"
    for opt in O0 O1 O2 O3; do
        if [ -f "$DEBUG_OUT/${pkg}_${binname}_${opt}" ]; then
            log "$pkg-$opt" "exists, skipping"
            continue
        fi
        log "$pkg-$opt" "building in $srcdir"
        (
            cd "$srcdir"
            # Reset between opt levels
            make distclean > /dev/null 2>&1 || make clean > /dev/null 2>&1 || true
            # Generate ./configure if it's missing but autogen is present
            if [[ "$configure_cmd" == ./configure* ]] && [ ! -f configure ]; then
                if [ -x autogen.sh ]; then
                    echo "[autogen.sh]"
                    ./autogen.sh 2>&1 | tail -8 || true
                elif [ -f configure.ac ] || [ -f configure.in ]; then
                    echo "[autoreconf]"
                    autoreconf -fi 2>&1 | tail -8 || true
                fi
            fi
            CFLAGS="-g -$opt" CXXFLAGS="-g -$opt" $configure_cmd 2>&1 | tail -8
            $makecmd -j8 2>&1 | tail -8
        ) || { log "$pkg-$opt" "build FAILED"; continue; }
        local src_bin=""
        for cand in $bin_paths; do
            if [ -f "$srcdir/$cand" ]; then src_bin="$srcdir/$cand"; break; fi
        done
        [ -n "$src_bin" ] || { log "$pkg-$opt" "no binary found (searched: $bin_paths)"; continue; }
        cp "$src_bin" "$DEBUG_OUT/${pkg}_${binname}_${opt}"
        strip -s "$src_bin" -o "$STRIPPED_OUT/${pkg}_${binname}_${opt}_stripped"
        log "$pkg-$opt" "OK ($src_bin)"
    done
}

# Wrap each package in a function so per-package failures don't kill the loop
pkg_lighttpd() {
    download_extract "lighttpd" "https://download.lighttpd.net/lighttpd/releases-1.4.x/lighttpd-1.4.76.tar.xz" || return
    build_opt "lighttpd" "lighttpd" "$SRC_DIR/lighttpd" "make" \
        "./configure --without-zlib --without-bzip2 --without-pcre --disable-shared" \
        "src/lighttpd"
}

pkg_yash() {
    # OSDN redirector was silently erroring on wulver wget; try SF mirror + a couple of fallbacks
    download_extract "yash" "https://master.dl.sourceforge.net/project/yash/yash/2.57/yash-2.57.tar.xz?viasf=1" \
        || download_extract "yash" "https://mirror.yash.sh/pub/yash-2.57.tar.xz" \
        || return
    build_opt "yash" "yash" "$SRC_DIR/yash" "make" "./configure" "yash"
}

pkg_tinycc() {
    download_extract "tinycc" "https://download.savannah.gnu.org/releases/tinycc/tcc-0.9.27.tar.bz2" || return
    build_opt "tinycc" "tcc" "$SRC_DIR/tinycc" "make" "./configure" "tcc"
}

pkg_zsh() {
    download_extract "zsh" "https://sourceforge.net/projects/zsh/files/zsh/5.9/zsh-5.9.tar.xz/download" || return
    build_opt "zsh" "zsh" "$SRC_DIR/zsh" "make" "./configure --disable-shared" "Src/zsh"
}

pkg_fossil() {
    download_extract "fossil" "https://fossil-scm.org/home/tarball/version-2.25/fossil-src-2.25.tar.gz" || return
    # Drop --static: Wulver compute node lacks libcrypto.a / libc static libs
    build_opt "fossil" "fossil" "$SRC_DIR/fossil" "make" "./configure --with-openssl=none" "fossil"
}

pkg_cvs() {
    download_extract "cvs" "https://ftp.gnu.org/non-gnu/cvs/source/feature/1.12.13/cvs-1.12.13.tar.gz" || return
    build_opt "cvs" "cvs" "$SRC_DIR/cvs" "make" "./configure" "src/cvs"
}

for fn in pkg_lighttpd pkg_yash pkg_tinycc pkg_zsh pkg_fossil pkg_cvs; do
    log "DRIVER" "starting $fn"
    $fn || log "DRIVER" "$fn failed, continuing"
done

log "DONE" "all builds"
ls "$STRIPPED_OUT/" | head -60
