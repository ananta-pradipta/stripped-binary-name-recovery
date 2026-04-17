#!/bin/bash
# ============================================================
# STEP 2: Download, Compile, and Strip Dataset
# ============================================================
# Reads package list from configs/packages.conf.
# Supports multiple optimization levels and platforms.
#
# Features:
#   - Multi-optimization: O0, O1, O2, O3 (configurable)
#   - Cross-platform: Linux ELF + Windows PE via MinGW
#   - Idempotent: skips already-compiled binaries
#
# Usage: bash scripts/02_compile_dataset.sh
# Time:  ~1-2 hours (all opts + both platforms)
# ============================================================
set -e
source ~/bfnr-project/activate.sh
cd ~/bfnr-project

echo "═══════════════════════════════════════════════"
echo " Step 2: Compiling Dataset"
echo "═══════════════════════════════════════════════"

BUILD_DIR="$HOME/bfnr-project/build_tmp"
DATA_RAW="$HOME/bfnr-project/data/raw"
DATA_STRIPPED="$HOME/bfnr-project/data/stripped"
PACKAGE_CONF="$HOME/bfnr-project/configs/packages.conf"
mkdir -p "$BUILD_DIR" "$DATA_RAW" "$DATA_STRIPPED"

if [ ! -f "$PACKAGE_CONF" ]; then
    echo "  ✗ Package config not found: $PACKAGE_CONF"
    exit 1
fi

# ── Read build settings from config ──
OPT_LEVELS=$(grep '^OPT_LEVELS=' "$PACKAGE_CONF" | tail -1 | cut -d= -f2)
PLATFORMS=$(grep '^PLATFORMS=' "$PACKAGE_CONF" | tail -1 | cut -d= -f2)
[ -z "$OPT_LEVELS" ] && OPT_LEVELS="O2"
[ -z "$PLATFORMS" ] && PLATFORMS="linux"

echo "  Optimization levels: $OPT_LEVELS"
echo "  Platforms:           $PLATFORMS"

# ── Check compilers ──
GCC_VERSION=$(gcc --version | head -1)
echo "  Linux compiler:  $GCC_VERSION"

HAS_MINGW=false
if echo "$PLATFORMS" | grep -q "windows"; then
    if command -v x86_64-w64-mingw32-gcc &>/dev/null; then
        MINGW_VERSION=$(x86_64-w64-mingw32-gcc --version | head -1)
        echo "  Windows compiler: $MINGW_VERSION"
        HAS_MINGW=true
    else
        echo "  ⚠ MinGW not found. Installing..."
        sudo apt-get install -y gcc-mingw-w64-x86-64 2>/dev/null && {
            HAS_MINGW=true
            echo "  ✓ MinGW installed"
        } || {
            echo "  ⚠ MinGW install failed. Skipping Windows binaries."
        }
    fi
fi

echo "  Strip:           strip -s (remove ALL symbols)"
echo "  Config:          $PACKAGE_CONF"
echo ""

# ══════════════════════════════════════════════
# Helper: compile one package at one opt level for one platform
# ══════════════════════════════════════════════
compile_one() {
    local pkg_name="$1"
    local src_dir_path="$2"
    local opt="$3"
    local platform="$4"
    shift 4
    local bins=("$@")

    local suffix="${opt}"
    local cc="gcc"
    local strip_cmd="strip"
    local host_flag=""
    local ext=""

    if [ "$platform" = "windows" ]; then
        suffix="${opt}_win"
        cc="x86_64-w64-mingw32-gcc"
        strip_cmd="x86_64-w64-mingw32-strip"
        host_flag="--host=x86_64-w64-mingw32"
        ext=".exe"
    fi

    local build_marker="$src_dir_path/.built_${suffix}"

    # Check if already compiled for this opt+platform
    if [ -f "$build_marker" ]; then
        # Just copy binaries that haven't been copied yet
        local count=0
        for bin_path in "${bins[@]}"; do
            local bin_name=$(basename "$bin_path" .exe)
            local name="${pkg_name}_${bin_name}_${suffix}"
            if [ -f "$DATA_RAW/${name}_sym" ]; then
                count=$((count + 1))
            elif [ -f "$src_dir_path/${bin_path}${ext}" ]; then
                cp "$src_dir_path/${bin_path}${ext}" "$DATA_RAW/${name}_sym"
                $strip_cmd -s "$DATA_RAW/${name}_sym" -o "$DATA_STRIPPED/${name}_stripped" 2>/dev/null || true
                count=$((count + 1))
            fi
        done
        [ $count -gt 0 ] && echo "    [$suffix] $count binaries (cached)"
        return 0
    fi

    # Clean and recompile
    cd "$src_dir_path"
    make distclean 2>/dev/null || make clean 2>/dev/null || true

    # Configure
    if [ "$platform" = "windows" ]; then
        CC="$cc" ./configure CFLAGS="-g -${opt}" $host_flag --quiet 2>/dev/null || {
            echo "    [$suffix] configure failed, skipping"
            return 1
        }
    else
        ./configure CFLAGS="-g -${opt}" --quiet 2>/dev/null || {
            echo "    [$suffix] configure failed, skipping"
            return 1
        }
    fi

    # Compile
    make -j$(nproc) --quiet 2>/dev/null || {
        echo "    [$suffix] compile failed, skipping"
        return 1
    }

    # Copy + strip binaries
    local count=0
    for bin_path in "${bins[@]}"; do
        local actual_path="$bin_path"
        # For Windows, check with .exe extension
        if [ "$platform" = "windows" ] && [ ! -f "$actual_path" ]; then
            actual_path="${bin_path}.exe"
        fi

        if [ -f "$actual_path" ]; then
            local bin_name=$(basename "$bin_path" .exe)
            local name="${pkg_name}_${bin_name}_${suffix}"
            cp "$actual_path" "$DATA_RAW/${name}_sym"
            $strip_cmd -s "$DATA_RAW/${name}_sym" -o "$DATA_STRIPPED/${name}_stripped" 2>/dev/null || {
                # If cross-strip fails, try native strip
                strip -s "$DATA_RAW/${name}_sym" -o "$DATA_STRIPPED/${name}_stripped" 2>/dev/null || true
            }
            count=$((count + 1))
        fi
    done

    touch "$build_marker"
    echo "    [$suffix] $count binaries"
    cd ~/bfnr-project
    return 0
}

# ══════════════════════════════════════════════
# Compile a package across all opt levels and platforms
# ══════════════════════════════════════════════
compile_package_full() {
    local pkg_name="$1"
    local url="$2"
    local tarball="$3"
    local src_dir="$4"
    local is_linux_only="$5"
    shift 5
    local bins=("$@")

    echo ""
    echo "── $pkg_name ──"
    cd "$BUILD_DIR"

    # Download
    if [ ! -f "$tarball" ]; then
        echo "  Downloading..."
        wget -q --show-progress "$url" -O "$tarball" 2>&1 || {
            echo "  ⚠ Download failed. Skipping."
            cd ~/bfnr-project
            return 1
        }
    fi

    # Extract
    if [ ! -d "$src_dir" ]; then
        tar xf "$tarball" 2>/dev/null || {
            echo "  ⚠ Extract failed. Skipping."
            cd ~/bfnr-project
            return 1
        }
    fi

    local src_path="$BUILD_DIR/$src_dir"

    # Compile for each optimization level × platform
    for opt in $OPT_LEVELS; do
        # Linux (always)
        compile_one "$pkg_name" "$src_path" "$opt" "linux" "${bins[@]}" || true

        # Windows (if enabled and not linux-only)
        if [ "$HAS_MINGW" = true ] && echo "$PLATFORMS" | grep -q "windows" && [ "$is_linux_only" != "true" ]; then
            compile_one "$pkg_name" "$src_path" "$opt" "windows" "${bins[@]}" || true
        fi
    done

    cd ~/bfnr-project
    return 0
}

# ══════════════════════════════════════════════
# Coreutils (special: many binaries)
# ══════════════════════════════════════════════
COREUTILS_BINS=(ls cat sort cp mv rm chmod chown head tail wc cut paste join uniq tr od fmt nl tsort ptx)

echo ""
echo "── coreutils-9.1 ──"
cd "$BUILD_DIR"
[ ! -f coreutils-9.1.tar.xz ] && wget -q --show-progress https://ftp.gnu.org/gnu/coreutils/coreutils-9.1.tar.xz
[ ! -d coreutils-9.1 ] && tar xf coreutils-9.1.tar.xz

SRC_PATH="$BUILD_DIR/coreutils-9.1"

for opt in $OPT_LEVELS; do
    BUILD_MARKER="$SRC_PATH/.built_${opt}"
    if [ ! -f "$BUILD_MARKER" ]; then
        cd "$SRC_PATH"
        make distclean 2>/dev/null || make clean 2>/dev/null || true
        echo "  Configuring [$opt]..."
        ./configure CFLAGS="-g -${opt}" --quiet 2>/dev/null
        echo "  Compiling [$opt]..."
        make -j$(nproc) --quiet 2>/dev/null
        touch "$BUILD_MARKER"
    fi

    COUNT=0
    cd "$SRC_PATH"
    for b in "${COREUTILS_BINS[@]}"; do
        name="coreutils_${b}_${opt}"
        if [ -f "src/$b" ] && [ ! -f "$DATA_RAW/${name}_sym" ]; then
            cp "src/$b" "$DATA_RAW/${name}_sym"
            strip -s "$DATA_RAW/${name}_sym" -o "$DATA_STRIPPED/${name}_stripped"
        fi
        [ -f "$DATA_RAW/${name}_sym" ] && COUNT=$((COUNT + 1))
    done
    echo "  [$opt] coreutils: $COUNT binaries"

    # Windows cross-compile for coreutils
    if [ "$HAS_MINGW" = true ] && echo "$PLATFORMS" | grep -q "windows"; then
        BUILD_MARKER_WIN="$SRC_PATH/.built_${opt}_win"
        if [ ! -f "$BUILD_MARKER_WIN" ]; then
            cd "$SRC_PATH"
            make distclean 2>/dev/null || make clean 2>/dev/null || true
            CC=x86_64-w64-mingw32-gcc ./configure CFLAGS="-g -${opt}" --host=x86_64-w64-mingw32 --quiet 2>/dev/null && {
                make -j$(nproc) --quiet 2>/dev/null && touch "$BUILD_MARKER_WIN"
            } || echo "  [${opt}_win] coreutils: configure failed, skipping"
        fi

        if [ -f "$BUILD_MARKER_WIN" ]; then
            COUNT_WIN=0
            cd "$SRC_PATH"
            for b in "${COREUTILS_BINS[@]}"; do
                name="coreutils_${b}_${opt}_win"
                for ext in "$b" "$b.exe"; do
                    if [ -f "src/$ext" ] && [ ! -f "$DATA_RAW/${name}_sym" ]; then
                        cp "src/$ext" "$DATA_RAW/${name}_sym"
                        x86_64-w64-mingw32-strip -s "$DATA_RAW/${name}_sym" -o "$DATA_STRIPPED/${name}_stripped" 2>/dev/null || \
                        strip -s "$DATA_RAW/${name}_sym" -o "$DATA_STRIPPED/${name}_stripped" 2>/dev/null || true
                        break
                    fi
                done
                [ -f "$DATA_RAW/${name}_sym" ] && COUNT_WIN=$((COUNT_WIN + 1))
            done
            echo "  [${opt}_win] coreutils: $COUNT_WIN binaries"
        fi
    fi
done
cd ~/bfnr-project

# ══════════════════════════════════════════════
# All other packages from config file
# ══════════════════════════════════════════════
PKG_NUM=0
FAIL_NUM=0

while IFS='|' read -r raw_name url tarball src_dir bins_str; do
    # Skip comments, empty lines, DEMO, settings
    [[ "$raw_name" =~ ^[[:space:]]*# ]] && continue
    [[ "$raw_name" =~ ^[[:space:]]*$ ]] && continue
    [[ "$raw_name" =~ DEMO ]] && continue
    [[ "$raw_name" =~ ^OPT_LEVELS ]] && continue
    [[ "$raw_name" =~ ^PLATFORMS ]] && continue

    # Check for LINUX_ONLY tag
    is_linux_only="false"
    if [[ "$raw_name" =~ LINUX_ONLY ]]; then
        is_linux_only="true"
        raw_name=$(echo "$raw_name" | sed 's/\[LINUX_ONLY\]//g')
    fi

    # Trim whitespace
    name=$(echo "$raw_name" | xargs)
    url=$(echo "$url" | xargs)
    tarball=$(echo "$tarball" | xargs)
    src_dir=$(echo "$src_dir" | xargs)
    bins_str=$(echo "$bins_str" | xargs)

    [ -z "$name" ] && continue

    IFS=' ' read -ra bins <<< "$bins_str"

    PKG_NUM=$((PKG_NUM + 1))
    compile_package_full "$name" "$url" "$tarball" "$src_dir" "$is_linux_only" "${bins[@]}" || {
        FAIL_NUM=$((FAIL_NUM + 1))
    }

done < "$PACKAGE_CONF"

# ══════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════"
echo " VERIFICATION"
echo "═══════════════════════════════════════════════"

NUM_RAW=$(find "$DATA_RAW" -name "*_sym" | wc -l)
NUM_STRIPPED=$(find "$DATA_STRIPPED" -name "*_stripped" | wc -l)

# Count by platform
NUM_LINUX=$(find "$DATA_RAW" -name "*_sym" ! -name "*_win_sym" | wc -l)
NUM_WINDOWS=$(find "$DATA_RAW" -name "*_win_sym" | wc -l)

# Count by opt level
for opt in $OPT_LEVELS; do
    OPT_COUNT=$(find "$DATA_RAW" -name "*_${opt}_sym" -o -name "*_${opt}_win_sym" | wc -l)
    echo "  $opt: $OPT_COUNT binaries"
done

echo ""
echo "  Total binaries:    $NUM_RAW"
echo "  Linux (ELF):       $NUM_LINUX"
echo "  Windows (PE):      $NUM_WINDOWS"
echo "  Stripped:           $NUM_STRIPPED"
echo "  Packages compiled: $PKG_NUM (+ coreutils)"
echo "  Packages failed:   $FAIL_NUM"

# Verify stripping
STRIP_OK=0
STRIP_FAIL=0
for f in "$DATA_STRIPPED"/*_stripped; do
    if [ -f "$f" ]; then
        T_COUNT=$(nm "$f" 2>/dev/null | grep -c ' T ' || echo 0)
        if [ "$T_COUNT" -eq 0 ]; then
            STRIP_OK=$((STRIP_OK + 1))
        else
            STRIP_FAIL=$((STRIP_FAIL + 1))
        fi
    fi
done
echo "  Fully stripped:    $STRIP_OK"
[ "$STRIP_FAIL" -gt 0 ] && echo "  ✗ Strip failures: $STRIP_FAIL"

# Function counts
echo ""
echo "── Function counts ──"
TOTAL_FUNCS=0
for f in "$DATA_RAW"/*_sym; do
    if [ -f "$f" ]; then
        COUNT=$(nm --defined-only "$f" 2>/dev/null | grep -c ' T ' || echo 0)
        TOTAL_FUNCS=$((TOTAL_FUNCS + COUNT))
    fi
done
echo "  TOTAL: $TOTAL_FUNCS functions across $NUM_RAW binaries"

# Save manifest
cat > "$HOME/bfnr-project/data/build_manifest.json" << EOF
{
    "optimization_levels": "$OPT_LEVELS",
    "platforms": "$PLATFORMS",
    "compiler_linux": "$GCC_VERSION",
    "compiler_windows": "$(x86_64-w64-mingw32-gcc --version 2>/dev/null | head -1 || echo 'N/A')",
    "num_binaries_total": $NUM_RAW,
    "num_binaries_linux": $NUM_LINUX,
    "num_binaries_windows": $NUM_WINDOWS,
    "total_functions": $TOTAL_FUNCS,
    "properly_stripped": $STRIP_OK,
    "packages_attempted": $PKG_NUM,
    "packages_failed": $FAIL_NUM
}
EOF

echo ""
echo "═══════════════════════════════════════════════"
echo " ✓ DATASET COMPILATION COMPLETE"
echo "   Binaries: $NUM_RAW (Linux: $NUM_LINUX, Windows: $NUM_WINDOWS)"
echo "   Functions: $TOTAL_FUNCS"
echo "   Opt levels: $OPT_LEVELS"
echo ""
echo " Next: bash scripts/03_preprocess.sh"
echo "═══════════════════════════════════════════════"
