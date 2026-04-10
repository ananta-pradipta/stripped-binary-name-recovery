#!/bin/bash
# ============================================================
# Batch download, compile, and preprocess packages on Wulver
# Uses Wulver's BAP (installed at ~/.opam/4.14.2/bin/bap)
# Submits as SLURM jobs for parallel processing
# ============================================================

PROJ=/course/2026/spring/cs/785/hz79/adp232/cs785
BUILD=$PROJ/build_tmp
DATA_RAW=$PROJ/data/raw
DATA_STRIPPED=$PROJ/data/stripped
DATA_BIR=$PROJ/data/bir
DATA_GRAPHS=$PROJ/data/graphs
DATA_LABELS=$PROJ/data/labels
DATA_EXT=$PROJ/data/external_calls

# Setup BAP environment
setup_bap() {
    module load bright easybuild GCCcore/13.3.0 GMP/6.3.0 2>/dev/null
    export PATH=$HOME/.opam/4.14.2/bin:$HOME/.local/bin:$PATH
    eval $(opam env --switch=4.14.2 2>/dev/null)
}

# Setup Python environment
setup_python() {
    module load bright python3 2>/dev/null
    source $PROJ/../cs785-env/bin/activate
}

mkdir -p $BUILD $DATA_RAW $DATA_STRIPPED $DATA_BIR $DATA_GRAPHS $DATA_LABELS $DATA_EXT

# ============================================================
# Package list: name | url | tarball | src_dir | binaries
# Target: ~280K new functions to reach 500K total
# ============================================================

# Large packages (high function count)
PACKAGES=(
    # OpenSSL: ~18K functions per opt = ~72K total
    "openssl|https://www.openssl.org/source/openssl-3.2.1.tar.gz|openssl-3.2.1.tar.gz|openssl-3.2.1|apps/openssl"
    # GDB: ~10K functions per opt = ~40K total
    "gdb|https://ftp.gnu.org/gnu/gdb/gdb-14.2.tar.xz|gdb-14.2.tar.xz|gdb-14.2|gdb/gdb"
    # Vim: ~5K functions per opt = ~20K total
    "vim|https://github.com/vim/vim/archive/refs/tags/v9.1.0.tar.gz|vim-9.1.0.tar.gz|vim-9.1.0|src/vim"
    # Git: ~3K functions per opt = ~12K total
    "git|https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.43.0.tar.xz|git-2.43.0.tar.xz|git-2.43.0|git"
    # tmux: ~2K functions per opt = ~8K total
    "tmux2|https://github.com/tmux/tmux/releases/download/3.4/tmux-3.4.tar.gz|tmux-3.4.tar.gz|tmux-3.4|tmux"
    # nmap: ~3K functions per opt = ~12K total
    "nmap|https://nmap.org/dist/nmap-7.94.tar.bz2|nmap-7.94.tar.bz2|nmap-7.94|nmap"
    # coreutils v8 (different version than existing): ~4K × 4 = ~16K
    "coreutils3|https://ftp.gnu.org/gnu/coreutils/coreutils-8.32.tar.xz|coreutils-8.32.tar.xz|coreutils-8.32|src/cp src/rm src/chmod src/chown src/tail src/head src/wc src/tr src/tee src/env"
    # diffutils (move from demo to training): ~500 × 4 = ~2K
    "diffutils|https://ftp.gnu.org/gnu/diffutils/diffutils-3.10.tar.xz|diffutils-3.10.tar.xz|diffutils-3.10|src/diff src/cmp src/sdiff src/diff3"
    # GNU awk (larger version)
    "gawk2|https://ftp.gnu.org/gnu/gawk/gawk-5.2.2.tar.xz|gawk-5.2.2.tar.xz|gawk-5.2.2|gawk"
    # GNU libtool
    "libtool|https://ftp.gnu.org/gnu/libtool/libtool-2.4.7.tar.xz|libtool-2.4.7.tar.xz|libtool-2.4.7|libtoolize"
    # Readline
    "readline|https://ftp.gnu.org/gnu/readline/readline-8.2.tar.gz|readline-8.2.tar.gz|readline-8.2|examples/rl"
    # GNU Parallel
    "parallel|https://ftp.gnu.org/gnu/parallel/parallel-20240122.tar.bz2|parallel-20240122.tar.bz2|parallel-20240122|src/parallel"
    # Additional medium GNU packages
    "autoconf|https://ftp.gnu.org/gnu/autoconf/autoconf-2.72.tar.xz|autoconf-2.72.tar.xz|autoconf-2.72|bin/autoconf"
    "gnuplot|https://sourceforge.net/projects/gnuplot/files/gnuplot/6.0.0/gnuplot-6.0.0.tar.gz|gnuplot-6.0.0.tar.gz|gnuplot-6.0.0|src/gnuplot"
    # Non-GNU for diversity
    "xz|https://github.com/tukaani-project/xz/releases/download/v5.4.5/xz-5.4.5.tar.xz|xz-5.4.5.tar.xz|xz-5.4.5|src/xz/xz src/lzmainfo/lzmainfo"
    "file|https://github.com/file/file/archive/refs/tags/FILE5_45.tar.gz|file-FILE5_45.tar.gz|file-FILE5_45|src/file"
)

compile_package() {
    local pkg=$1 url=$2 tarball=$3 srcdir=$4 bins=$5

    echo "=== Processing $pkg ==="
    cd $BUILD

    # Download
    if [ ! -f "$tarball" ]; then
        echo "  Downloading $tarball..."
        wget -q "$url" -O "$tarball" || { echo "  DOWNLOAD FAILED"; return 1; }
    fi

    # Extract
    if [ ! -d "$srcdir" ]; then
        echo "  Extracting..."
        tar xf "$tarball" 2>/dev/null || { echo "  EXTRACT FAILED"; return 1; }
    fi

    for opt in O0 O1 O2 O3; do
        echo "  Compiling $pkg at -$opt..."
        cd "$BUILD/$srcdir"
        make clean >/dev/null 2>&1 || true
        make distclean >/dev/null 2>&1 || true

        # Special handling per package
        case "$pkg" in
            openssl)
                ./Configure linux-x86_64 -$opt -g no-shared >/dev/null 2>&1
                make -j8 >/dev/null 2>&1
                ;;
            vim)
                CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" ./configure --quiet --with-features=huge --disable-gui --without-x >/dev/null 2>&1
                make -j8 >/dev/null 2>&1
                ;;
            git)
                make CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" -j8 >/dev/null 2>&1
                ;;
            nmap)
                CFLAGS="-$opt -g -fno-pie -fno-PIE" CXXFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" ./configure --quiet --without-zenmap --without-ncat --without-ndiff --without-nping >/dev/null 2>&1
                make -j8 >/dev/null 2>&1
                ;;
            gdb)
                CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" ./configure --quiet --disable-tui --without-python >/dev/null 2>&1
                make -j8 >/dev/null 2>&1
                ;;
            *)
                # Standard autotools
                if [ -f configure ]; then
                    CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" ./configure --quiet >/dev/null 2>&1
                    make -j8 >/dev/null 2>&1
                elif [ -f Makefile ]; then
                    make CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" -j8 >/dev/null 2>&1
                fi
                ;;
        esac

        # Copy binaries
        for bin_path in $bins; do
            bin_name=$(basename "$bin_path")
            src_bin="$BUILD/$srcdir/$bin_path"
            if [ ! -f "$src_bin" ]; then
                # Try .libs/ for libtool
                alt="$BUILD/$srcdir/$(dirname $bin_path)/.libs/$bin_name"
                [ -f "$alt" ] && src_bin="$alt"
            fi
            if [ -f "$src_bin" ] && file "$src_bin" | grep -q "ELF"; then
                cp "$src_bin" "$DATA_RAW/${pkg}_${bin_name}_${opt}"
                cp "$src_bin" "$DATA_STRIPPED/${pkg}_${bin_name}_${opt}_stripped"
                strip "$DATA_STRIPPED/${pkg}_${bin_name}_${opt}_stripped"
                echo "    OK: ${pkg}_${bin_name}_${opt}"
            fi
        done
    done
    cd $PROJ
}

preprocess_binary() {
    local name=$1  # e.g. openssl_openssl_O0

    # Labels
    local raw="$DATA_RAW/$name"
    local labels="$DATA_LABELS/${name}.json"
    if [ -f "$raw" ] && [ ! -f "$labels" ]; then
        nm --defined-only "$raw" 2>/dev/null | awk '$2 ~ /[tT]/ {printf "{\"0x%s\": \"%s\"}\n", $1, $3}' | python3 -c "
import sys, json
labels = {}
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        labels.update(d)
    except: pass
with open('$labels', 'w') as f:
    json.dump(labels, f, indent=2)
print(f'  Labels: $name = {len(labels)}')
"
    fi

    # BAP lift
    local stripped="$DATA_STRIPPED/${name}_stripped"
    local bir="$DATA_BIR/${name}.bir"
    if [ -f "$stripped" ] && [ ! -f "$bir" ]; then
        echo -n "  BAP: $name ... "
        if bap "$stripped" --dump=bir:"$bir" 2>/dev/null; then
            echo "OK"
        elif bap "$stripped" --no-byteweight --dump=bir:"$bir" 2>/dev/null; then
            echo "OK (fallback)"
        else
            echo "FAILED"
            return 1
        fi
    fi

    # Parse graphs
    if [ -f "$bir" ]; then
        local existing=$(ls "$DATA_GRAPHS/${name}_"*.json 2>/dev/null | wc -l)
        if [ "$existing" -eq 0 ]; then
            echo "  Parse: $name"
            python3 -m src.preprocessing.parse_bap --bir "$bir" --binary-name "$name" --output-dir "$DATA_GRAPHS"
        fi

        # External calls
        local ext="$DATA_EXT/${name}_external.json"
        if [ ! -f "$ext" ]; then
            echo "  ExtCalls: $name"
            python3 -m src.preprocessing.extract_external --bir "$bir" --binary-name "$name" --output-dir "$DATA_EXT" --vocab-path "$DATA_EXT/external_vocab.json"
        fi
    fi
}

# Main
echo "═══════════════════════════════════════════════"
echo " Wulver Batch Preprocessing Pipeline"
echo " Target: 500K functions"
echo "═══════════════════════════════════════════════"

setup_bap
setup_python

# Compile all packages
for entry in "${PACKAGES[@]}"; do
    IFS='|' read -r pkg url tarball srcdir bins <<< "$entry"
    compile_package "$pkg" "$url" "$tarball" "$srcdir" "$bins"
done

# Preprocess all new binaries
echo ""
echo "═══════════════════════════════════════════════"
echo " Preprocessing new binaries"
echo "═══════════════════════════════════════════════"

for raw in $DATA_RAW/*_O[0-3]; do
    name=$(basename "$raw")
    # Only process if we don't have graphs yet
    existing=$(ls "$DATA_GRAPHS/${name}_"*.json 2>/dev/null | wc -l)
    if [ "$existing" -eq 0 ]; then
        preprocess_binary "$name"
    fi
done

echo ""
echo "═══════════════════════════════════════════════"
echo " Done! Run match_index rebuild next."
echo "═══════════════════════════════════════════════"
