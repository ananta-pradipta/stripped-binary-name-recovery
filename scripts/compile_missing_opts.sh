#!/bin/bash
# ============================================================
# Compile missing O1/O3 binaries for packages that only have O0/O2
# ============================================================
set +e  # Continue on errors
source ~/bfnr-project/activate.sh
cd ~/bfnr-project

BUILD_DIR="$HOME/bfnr-project/build_tmp"
DATA_RAW="$HOME/bfnr-project/data/raw"
DATA_STRIPPED="$HOME/bfnr-project/data/stripped"

# Packages missing O1/O3 and their binary paths (from packages.conf)
declare -A PKG_BINS
PKG_BINS[acct]="ac last lastcomm sa dump-utmp accton"
PKG_BINS[bc]="bc/bc dc/dc"
PKG_BINS[indent]="src/indent"
PKG_BINS[jq]="jq"
PKG_BINS[lua]="src/lua src/luac"
PKG_BINS[lz4]="programs/lz4"
PKG_BINS[rush]="src/rush"
PKG_BINS[sqlite]="sqlite3"
PKG_BINS[wget]="src/wget"

# Demo packages (don't need O1/O3 for training, but good for completeness)
declare -A DEMO_BINS
DEMO_BINS[dico]="dico/dico dicod/dicod"
DEMO_BINS[htop]="htop"
DEMO_BINS[strace]="src/strace"

declare -A PKG_DIRS
PKG_DIRS[acct]="acct-6.6.4"
PKG_DIRS[bc]="bc-1.07.1"
PKG_DIRS[indent]="indent-2.2.13"
PKG_DIRS[jq]="jq-1.7.1"
PKG_DIRS[lua]="lua-5.4.6"
PKG_DIRS[lz4]="lz4-1.9.4"
PKG_DIRS[rush]="rush-2.3"
PKG_DIRS[sqlite]="sqlite-autoconf-3450000"
PKG_DIRS[wget]="wget-1.21.4"
PKG_DIRS[dico]="dico-2.11"
PKG_DIRS[htop]="htop-3.3.0"
PKG_DIRS[strace]="strace-6.7"

compile_package() {
    local pkg=$1
    local src_dir=$2
    local bins=$3
    local opt=$4
    local full_dir="$BUILD_DIR/$src_dir"

    if [ ! -d "$full_dir" ]; then
        echo "  SKIP $pkg: source dir not found: $full_dir"
        return
    fi

    echo "  Compiling $pkg at -$opt..."
    cd "$full_dir"

    # Clean previous build
    make clean 2>/dev/null || true

    # Special handling for different build systems
    if [ "$pkg" = "lua" ]; then
        # Lua uses plain Makefile
        make MYCFLAGS="-$opt -g -fno-pie -fno-PIE" MYLDFLAGS="-no-pie" linux -j$(nproc) 2>/dev/null
    elif [ "$pkg" = "lz4" ]; then
        # lz4 uses plain Makefile
        make CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" -j$(nproc) 2>/dev/null
    elif [ -f configure ]; then
        # Autotools
        CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" ./configure --quiet 2>/dev/null
        make -j$(nproc) 2>/dev/null
    elif [ -f CMakeLists.txt ]; then
        mkdir -p build_$opt && cd build_$opt
        cmake -DCMAKE_C_FLAGS="-$opt -g -fno-pie -fno-PIE" -DCMAKE_EXE_LINKER_FLAGS="-no-pie" .. 2>/dev/null
        make -j$(nproc) 2>/dev/null
        cd ..
    else
        echo "  SKIP $pkg: no configure or CMakeLists.txt"
        return
    fi

    # Copy binaries
    for bin_path in $bins; do
        bin_name=$(basename "$bin_path")
        src_bin="$full_dir/$bin_path"
        if [ ! -f "$src_bin" ]; then
            echo "    WARN: $bin_path not found after compilation"
            continue
        fi

        # Check it's actually an ELF
        if ! file "$src_bin" | grep -q "ELF"; then
            echo "    WARN: $bin_path is not ELF"
            continue
        fi

        # Copy debug version
        cp "$src_bin" "$DATA_RAW/${pkg}_${bin_name}_${opt}"
        # Strip and copy
        cp "$src_bin" "$DATA_STRIPPED/${pkg}_${bin_name}_${opt}_stripped"
        strip "$DATA_STRIPPED/${pkg}_${bin_name}_${opt}_stripped"

        echo "    OK: ${pkg}_${bin_name}_${opt}"
    done

    cd ~/bfnr-project
}

echo "═══════════════════════════════════════════════"
echo " Compiling Missing O1/O3 Binaries"
echo "═══════════════════════════════════════════════"

MISSING_OPTS="O1 O3"

# Training packages
echo ""
echo "=== Training Packages ==="
for pkg in "${!PKG_BINS[@]}"; do
    for opt in $MISSING_OPTS; do
        # Check if already compiled
        existing=$(ls "$DATA_STRIPPED/${pkg}_"*"_${opt}_stripped" 2>/dev/null | wc -l)
        if [ "$existing" -gt 0 ]; then
            echo "  SKIP $pkg $opt: already have $existing binaries"
            continue
        fi
        compile_package "$pkg" "${PKG_DIRS[$pkg]}" "${PKG_BINS[$pkg]}" "$opt"
    done
done

# Demo packages (optional)
echo ""
echo "=== Demo Packages ==="
for pkg in "${!DEMO_BINS[@]}"; do
    for opt in $MISSING_OPTS; do
        existing=$(ls "$DATA_STRIPPED/${pkg}_"*"_${opt}_stripped" 2>/dev/null | wc -l)
        if [ "$existing" -gt 0 ]; then
            echo "  SKIP $pkg $opt: already have $existing binaries"
            continue
        fi
        compile_package "$pkg" "${PKG_DIRS[$pkg]}" "${DEMO_BINS[$pkg]}" "$opt"
    done
done

echo ""
echo "═══════════════════════════════════════════════"
echo " Done. Now run BAP preprocessing on new binaries."
echo "═══════════════════════════════════════════════"
