#!/bin/bash
# ============================================================
# Compile additional small/medium packages locally for BAP
# These are packages that BAP can handle (<5MB binary)
# ============================================================
source ~/cs785-project/activate.sh
cd ~/cs785-project

BUILD_DIR="$HOME/cs785-project/build_tmp"
DATA_RAW="data/raw"
DATA_STRIPPED="data/stripped"

mkdir -p "$BUILD_DIR" "$DATA_RAW" "$DATA_STRIPPED"

compile_gnu() {
    local pkg=$1 url=$2 tarball=$3 srcdir=$4 bins=$5

    echo "=== $pkg ==="
    cd "$BUILD_DIR"

    # Download
    if [ ! -f "$tarball" ]; then
        echo "  Downloading..."
        wget -q "$url" -O "$tarball" 2>/dev/null || { echo "  DOWNLOAD FAILED"; return; }
    fi

    # Extract
    if [ ! -d "$srcdir" ]; then
        tar xf "$tarball" 2>/dev/null || { echo "  EXTRACT FAILED"; return; }
    fi

    for opt in O0 O1 O2 O3; do
        existing=$(ls "$HOME/cs785-project/$DATA_STRIPPED/${pkg}_"*"_${opt}_stripped" 2>/dev/null | wc -l)
        if [ "$existing" -gt 0 ]; then
            echo "  SKIP $opt (exists)"
            continue
        fi

        cd "$BUILD_DIR/$srcdir"
        make clean >/dev/null 2>&1 || true

        CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" ./configure --quiet >/dev/null 2>&1
        make -j$(nproc) >/dev/null 2>&1

        for bin_path in $bins; do
            bin_name=$(basename "$bin_path")
            src_bin="$BUILD_DIR/$srcdir/$bin_path"
            # Try .libs for libtool
            [ ! -f "$src_bin" ] && src_bin="$BUILD_DIR/$srcdir/$(dirname $bin_path)/.libs/$bin_name"
            if [ -f "$src_bin" ] && file "$src_bin" | grep -q "ELF"; then
                cp "$src_bin" "$HOME/cs785-project/$DATA_RAW/${pkg}_${bin_name}_${opt}"
                cp "$src_bin" "$HOME/cs785-project/$DATA_STRIPPED/${pkg}_${bin_name}_${opt}_stripped"
                strip "$HOME/cs785-project/$DATA_STRIPPED/${pkg}_${bin_name}_${opt}_stripped"
                echo "  OK: ${pkg}_${bin_name}_${opt}"
            fi
        done
    done
    cd ~/cs785-project
}

# Well-known GNU packages, easy to compile, diverse functionality
compile_gnu "gdbm" "https://ftp.gnu.org/gnu/gdbm/gdbm-1.23.tar.gz" "gdbm-1.23.tar.gz" "gdbm-1.23" "src/gdbmtool"
compile_gnu "wdiff" "https://ftp.gnu.org/gnu/wdiff/wdiff-1.2.2.tar.gz" "wdiff-1.2.2.tar.gz" "wdiff-1.2.2" "src/wdiff"
compile_gnu "plotutils" "https://ftp.gnu.org/gnu/plotutils/plotutils-2.6.tar.gz" "plotutils-2.6.tar.gz" "plotutils-2.6" "graph/graph pic2plot/pic2plot plot/plot tek2plot/tek2plot plotfont/plotfont"
compile_gnu "gsl" "https://ftp.gnu.org/gnu/gsl/gsl-2.7.1.tar.gz" "gsl-2.7.1.tar.gz" "gsl-2.7.1" "gsl-config"
compile_gnu "nettle" "https://ftp.gnu.org/gnu/nettle/nettle-3.9.1.tar.gz" "nettle-3.9.1.tar.gz" "nettle-3.9.1" "tools/nettle-hash tools/sexp-conv"
compile_gnu "libtasn1" "https://ftp.gnu.org/gnu/libtasn1/libtasn1-4.19.0.tar.gz" "libtasn1-4.19.0.tar.gz" "libtasn1-4.19.0" "src/asn1Parser src/asn1Decoding src/asn1Coding"
compile_gnu "libidn2" "https://ftp.gnu.org/gnu/libidn/libidn2-2.3.7.tar.gz" "libidn2-2.3.7.tar.gz" "libidn2-2.3.7" "src/idn2"
compile_gnu "libmicrohttpd" "https://ftp.gnu.org/gnu/libmicrohttpd/libmicrohttpd-0.9.77.tar.gz" "libmicrohttpd-0.9.77.tar.gz" "libmicrohttpd-0.9.77" "src/testcurl/test_get"
compile_gnu "gnuchess" "https://ftp.gnu.org/gnu/chess/gnuchess-6.2.9.tar.gz" "gnuchess-6.2.9.tar.gz" "gnuchess-6.2.9" "src/gnuchess"
compile_gnu "mailutils" "https://ftp.gnu.org/gnu/mailutils/mailutils-3.16.tar.gz" "mailutils-3.16.tar.gz" "mailutils-3.16" "mail/mail readmsg/readmsg movemail/movemail"

echo ""
echo "═══════════════════════════════════════════════"
echo " Local compilation done. Run BAP preprocessing next."
echo "═══════════════════════════════════════════════"
