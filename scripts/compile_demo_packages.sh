#!/bin/bash
# Compile DEMO packages for cross-project evaluation
# These are skipped by 02_compile_dataset.sh because they have [DEMO] tags

set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="build_tmp"
RAW_DIR="data/raw"
STRIPPED_DIR="data/stripped"
OPT_LEVELS="O0 O2"

mkdir -p "$BUILD_DIR" "$RAW_DIR" "$STRIPPED_DIR"

compile_package() {
    local name="$1"
    local url="$2"
    local tarball="$3"
    local src_dir="$4"
    shift 4
    local bins=("$@")

    echo "══════════════════════════════════════════"
    echo "  Package: $name"
    echo "══════════════════════════════════════════"

    # Download if needed
    if [ ! -f "$BUILD_DIR/$tarball" ]; then
        echo "  Downloading $tarball..."
        wget -q -O "$BUILD_DIR/$tarball" "$url" || { echo "  FAILED to download"; return 1; }
    fi

    # Extract if needed
    if [ ! -d "$BUILD_DIR/$src_dir" ]; then
        echo "  Extracting..."
        cd "$BUILD_DIR"
        tar xf "$tarball" 2>/dev/null || { echo "  FAILED to extract"; cd ..; return 1; }
        cd ..
    fi

    # Compile at each optimization level
    for opt in $OPT_LEVELS; do
        local opt_flag="-${opt}"
        local build_marker="$BUILD_DIR/.built_${name}_${opt}"

        if [ -f "$build_marker" ]; then
            echo "  [$opt] Already built (cached)"
        else
            echo "  [$opt] Compiling with CFLAGS=$opt_flag..."
            cd "$BUILD_DIR/$src_dir"

            # Clean previous build
            make clean 2>/dev/null || true
            make distclean 2>/dev/null || true

            # Configure and build
            local project_root="$(cd ../.. && pwd)"
            if [ -f configure ]; then
                CFLAGS="-g $opt_flag" ./configure --quiet 2>/dev/null || { echo "  FAILED configure"; cd "$project_root"; continue; }
                make -j$(nproc) 2>/dev/null || { echo "  FAILED make"; cd "$project_root"; continue; }
            elif [ -f Makefile ]; then
                make -j$(nproc) CFLAGS="-g $opt_flag" 2>/dev/null || { echo "  FAILED make"; cd "$project_root"; continue; }
            else
                echo "  No configure or Makefile found"
                cd "$project_root"
                continue
            fi

            cd "$project_root"
            touch "$build_marker"
        fi

        # Copy binaries
        local count=0
        for bin in "${bins[@]}"; do
            local bin_path="$BUILD_DIR/$src_dir/$bin"
            local bin_name=$(basename "$bin")

            if [ -f "$bin_path" ] && file "$bin_path" | grep -q "ELF"; then
                local suffix="${opt}"
                local raw_name="${name}_${bin_name}_${suffix}_sym"
                local stripped_name="${name}_${bin_name}_${suffix}"

                cp "$bin_path" "$RAW_DIR/$raw_name"
                cp "$bin_path" "$STRIPPED_DIR/$stripped_name"
                strip --strip-all "$STRIPPED_DIR/$stripped_name"
                count=$((count + 1))
            fi
        done
        echo "  [$opt] $count binaries copied"
    done
    echo ""
}

echo "Compiling demo packages for cross-project evaluation"
echo "Optimization levels: $OPT_LEVELS"
echo ""

# ── Existing [DEMO] packages ──
# idutils-4.6 skipped: fails to compile on modern GCC (gnulib incompatibility)

compile_package "rcs" \
    "https://ftp.gnu.org/gnu/rcs/rcs-5.10.1.tar.lz" \
    "rcs-5.10.1.tar.lz" "rcs-5.10.1" \
    "src/ci" "src/co" "src/rcs" "src/rlog" "src/rcsdiff" "src/rcsmerge"

compile_package "tree" \
    "https://github.com/Old-Man-Programmer/tree/archive/refs/tags/2.1.3.tar.gz" \
    "tree-2.1.3.tar.gz" "tree-2.1.3" \
    "tree"

compile_package "dos2unix" \
    "https://waterlan.home.xs4all.nl/dos2unix/dos2unix-7.5.2.tar.gz" \
    "dos2unix-7.5.2.tar.gz" "dos2unix-7.5.2" \
    "dos2unix" "unix2dos"

# ── New packages ──
compile_package "curl" \
    "https://curl.se/download/curl-8.6.0.tar.xz" \
    "curl-8.6.0.tar.xz" "curl-8.6.0" \
    "src/curl"

compile_package "bzip2" \
    "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz" \
    "bzip2-1.0.8.tar.gz" "bzip2-1.0.8" \
    "bzip2"

compile_package "nginx" \
    "https://nginx.org/download/nginx-1.24.0.tar.gz" \
    "nginx-1.24.0.tar.gz" "nginx-1.24.0" \
    "objs/nginx"

echo ""
echo "══════════════════════════════════════════"
echo "  Done! Summary:"
for pkg in idutils rcs tree dos2unix curl bzip2 nginx; do
    raw=$(ls data/raw/${pkg}_* 2>/dev/null | wc -l)
    stripped=$(ls data/stripped/${pkg}_* 2>/dev/null | wc -l)
    echo "  $pkg: $raw raw, $stripped stripped"
done
echo "══════════════════════════════════════════"
echo ""
echo "Next step: Run BAP preprocessing"
echo "  bash scripts/03_preprocess.sh"
