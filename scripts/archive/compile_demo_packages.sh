#!/bin/bash
# Compile new demo packages for expanded evaluation
source ~/cs785-project/activate.sh
cd ~/cs785-project

BUILD_DIR="$HOME/cs785-project/build_tmp"
DATA_RAW="$HOME/cs785-project/data/raw"
DATA_STRIPPED="$HOME/cs785-project/data/stripped"
mkdir -p "$BUILD_DIR" "$DATA_RAW" "$DATA_STRIPPED"

OPT_LEVELS="O0 O2"

compile_package() {
    local pkg_name="$1"
    local url="$2"
    local tarball="$3"
    local src_dir="$4"
    shift 4
    local bins=("$@")

    echo ""
    echo "════════════════════════════════════════"
    echo " Compiling: $pkg_name"
    echo "════════════════════════════════════════"

    # Download if needed
    if [ ! -f "$BUILD_DIR/$tarball" ]; then
        echo "  Downloading $tarball..."
        wget --timeout=60 -O "$BUILD_DIR/$tarball" "$url" || {
            echo "  ✗ Download failed for $pkg_name"
            return 1
        }
    fi

    # Extract if needed
    if [ ! -d "$BUILD_DIR/$src_dir" ]; then
        echo "  Extracting..."
        cd "$BUILD_DIR"
        case "$tarball" in
            *.tar.xz) tar xf "$tarball" ;;
            *.tar.gz) tar xzf "$tarball" ;;
            *.tar.bz2) tar xjf "$tarball" ;;
            *.tar.lz) tar --lzip -xf "$tarball" 2>/dev/null || {
                echo "  ✗ lzip not available, installing..."
                sudo apt-get install -y lzip && tar --lzip -xf "$tarball"
            } ;;
        esac
        cd ~/cs785-project
    fi

    for opt in $OPT_LEVELS; do
        echo "  ── $opt ──"

        # Check if already compiled
        local already_done=true
        for bin_path in "${bins[@]}"; do
            local bin_base=$(basename "$bin_path")
            if [ ! -f "$DATA_STRIPPED/${pkg_name}_${bin_base}_${opt}_stripped" ]; then
                already_done=false
                break
            fi
        done
        if $already_done; then
            echo "    Already compiled, skipping"
            continue
        fi

        # Clean and configure
        cd "$BUILD_DIR/$src_dir"
        make clean 2>/dev/null || true
        make distclean 2>/dev/null || true

        if [ -f configure ]; then
            CFLAGS="-g -$opt" ./configure --quiet 2>&1 | tail -3 || {
                echo "    ✗ Configure failed"
                cd ~/cs785-project
                continue
            }
        fi

        # Build
        make -j$(nproc) CFLAGS="-g -$opt" 2>&1 | tail -5 || {
            echo "    ✗ Build failed"
            cd ~/cs785-project
            continue
        }

        # Copy and strip binaries
        for bin_path in "${bins[@]}"; do
            local bin_base=$(basename "$bin_path")
            local full_path="$BUILD_DIR/$src_dir/$bin_path"

            if [ -f "$full_path" ]; then
                # Debug (with symbols) → raw
                cp "$full_path" "$DATA_RAW/${pkg_name}_${bin_base}_${opt}_sym"
                # Stripped → stripped
                cp "$full_path" "$DATA_STRIPPED/${pkg_name}_${bin_base}_${opt}_stripped"
                strip -s "$DATA_STRIPPED/${pkg_name}_${bin_base}_${opt}_stripped"
                echo "    ✓ ${pkg_name}_${bin_base}_${opt}"
            else
                echo "    ✗ Binary not found: $full_path"
                # Try to find it
                local found=$(find "$BUILD_DIR/$src_dir" -name "$bin_base" -type f -executable 2>/dev/null | head -1)
                if [ -n "$found" ]; then
                    echo "      Found at: $found"
                    cp "$found" "$DATA_RAW/${pkg_name}_${bin_base}_${opt}_sym"
                    cp "$found" "$DATA_STRIPPED/${pkg_name}_${bin_base}_${opt}_stripped"
                    strip -s "$DATA_STRIPPED/${pkg_name}_${bin_base}_${opt}_stripped"
                    echo "    ✓ ${pkg_name}_${bin_base}_${opt} (auto-found)"
                fi
            fi
        done
        cd ~/cs785-project
    done
}

echo "Compiling new demo packages..."
echo "Opt levels: $OPT_LEVELS"
echo ""

# Tier 1: High gnulib overlap
compile_package "idutils" \
    "https://mirrors.kernel.org/gnu/idutils/idutils-4.6.tar.xz" \
    "idutils-4.6.tar.xz" "idutils-4.6" \
    "src/mkid" "src/lid" "src/fid" "src/fnid" "src/xtokid"

# rcs requires lzip to extract — skip if not available
if command -v lzip &>/dev/null; then
compile_package "rcs" \
    "https://mirrors.kernel.org/gnu/rcs/rcs-5.10.1.tar.lz" \
    "rcs-5.10.1.tar.lz" "rcs-5.10.1" \
    "src/ci" "src/co" "src/rcs" "src/rlog" "src/rcsdiff" "src/rcsmerge"
else
echo "  ⚠ Skipping rcs (lzip not installed)"
fi

compile_package "acct" \
    "https://mirrors.kernel.org/gnu/acct/acct-6.6.4.tar.bz2" \
    "acct-6.6.4.tar.bz2" "acct-6.6.4" \
    "ac" "last" "lastcomm" "sa" "dump-utmp" "accton"

compile_package "rush" \
    "https://mirrors.kernel.org/gnu/rush/rush-2.3.tar.xz" \
    "rush-2.3.tar.xz" "rush-2.3" \
    "src/rush"

# Tier 2: Non-gnulib controls
compile_package "htop" \
    "https://github.com/htop-dev/htop/releases/download/3.3.0/htop-3.3.0.tar.xz" \
    "htop-3.3.0.tar.xz" "htop-3.3.0" \
    "htop"

compile_package "strace" \
    "https://github.com/strace/strace/releases/download/v6.7/strace-6.7.tar.xz" \
    "strace-6.7.tar.xz" "strace-6.7" \
    "src/strace"

# Tier 3: Already in packages.conf, just need demo compilation
compile_package "sharutils_demo" \
    "https://mirrors.kernel.org/gnu/sharutils/sharutils-4.15.2.tar.xz" \
    "sharutils-4.15.2.tar.xz" "sharutils-4.15.2" \
    "src/shar" "src/unshar"

compile_package "dico" \
    "https://mirrors.kernel.org/gnu/dico/dico-2.11.tar.xz" \
    "dico-2.11.tar.xz" "dico-2.11" \
    "dico/dico"

echo ""
echo "════════════════════════════════════════"
echo " Compilation complete!"
echo "════════════════════════════════════════"
echo ""
echo "New binaries in $DATA_RAW and $DATA_STRIPPED:"
ls -la "$DATA_RAW" | grep -E "idutils|rcs_|acct|rush|htop|strace|sharutils_demo|dico" | wc -l
echo " new sym binaries"
ls -la "$DATA_STRIPPED" | grep -E "idutils|rcs_|acct|rush|htop|strace|sharutils_demo|dico" | wc -l
echo " new stripped binaries"
