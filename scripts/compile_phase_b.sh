#!/bin/bash
# ============================================================
# Phase B: Compile new non-GNU packages at O0-O3
# ============================================================
source ~/cs785-project/activate.sh
cd ~/cs785-project

BUILD_DIR="$HOME/cs785-project/build_tmp"
DATA_RAW="$HOME/cs785-project/data/raw"
DATA_STRIPPED="$HOME/cs785-project/data/stripped"

mkdir -p "$BUILD_DIR" "$DATA_RAW" "$DATA_STRIPPED"

compile_at_opt() {
    local pkg=$1
    local src_dir=$2
    local bins=$3
    local opt=$4
    local build_cmd=$5

    echo "  Compiling $pkg at -$opt..."
    cd "$BUILD_DIR/$src_dir"

    # Clean
    make clean 2>/dev/null || true

    # Build
    eval "$build_cmd" OPT="-$opt" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "    BUILD FAILED for $pkg $opt"
        cd ~/cs785-project
        return 1
    fi

    # Copy binaries
    for bin_path in $bins; do
        bin_name=$(basename "$bin_path")
        src_bin="$BUILD_DIR/$src_dir/$bin_path"
        if [ ! -f "$src_bin" ]; then
            echo "    WARN: $bin_path not found"
            continue
        fi
        if ! file "$src_bin" | grep -q "ELF"; then
            echo "    WARN: $bin_path not ELF"
            continue
        fi
        cp "$src_bin" "$DATA_RAW/${pkg}_${bin_name}_${opt}"
        cp "$src_bin" "$DATA_STRIPPED/${pkg}_${bin_name}_${opt}_stripped"
        strip "$DATA_STRIPPED/${pkg}_${bin_name}_${opt}_stripped"
        echo "    OK: ${pkg}_${bin_name}_${opt}"
    done
    cd ~/cs785-project
}

echo "═══════════════════════════════════════════════"
echo " Phase B: Non-GNU Package Compilation"
echo "═══════════════════════════════════════════════"

# ── 1. zlib ──
echo ""
echo "=== zlib ==="
ZLIB_DIR="zlib-1.3.1"
if [ ! -d "$BUILD_DIR/$ZLIB_DIR" ]; then
    echo "  Downloading zlib..."
    cd "$BUILD_DIR"
    wget -q https://zlib.net/zlib-1.3.1.tar.gz
    tar xf zlib-1.3.1.tar.gz
    cd ~/cs785-project
fi

for opt in O0 O1 O2 O3; do
    existing=$(ls "$DATA_STRIPPED/zlib_"*"_${opt}_stripped" 2>/dev/null | wc -l)
    if [ "$existing" -gt 0 ]; then
        echo "  SKIP zlib $opt: already exists"
        continue
    fi
    echo "  Compiling zlib at -$opt..."
    cd "$BUILD_DIR/$ZLIB_DIR"
    make clean 2>/dev/null || true
    CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" ./configure --static 2>/dev/null
    make -j$(nproc) 2>/dev/null
    # zlib builds minigzip and example as test programs
    if [ -f minigzip ]; then
        cp minigzip "$DATA_RAW/zlib_minigzip_${opt}"
        cp minigzip "$DATA_STRIPPED/zlib_minigzip_${opt}_stripped"
        strip "$DATA_STRIPPED/zlib_minigzip_${opt}_stripped"
        echo "    OK: zlib_minigzip_${opt}"
    fi
    if [ -f example ]; then
        cp example "$DATA_RAW/zlib_example_${opt}"
        cp example "$DATA_STRIPPED/zlib_example_${opt}_stripped"
        strip "$DATA_STRIPPED/zlib_example_${opt}_stripped"
        echo "    OK: zlib_example_${opt}"
    fi
    # Also build zpipe if possible
    if [ -f examples/zpipe.c ]; then
        gcc -$opt -g -fno-pie -fno-PIE -o zpipe examples/zpipe.c -L. -lz -no-pie 2>/dev/null
        if [ -f zpipe ]; then
            cp zpipe "$DATA_RAW/zlib_zpipe_${opt}"
            cp zpipe "$DATA_STRIPPED/zlib_zpipe_${opt}_stripped"
            strip "$DATA_STRIPPED/zlib_zpipe_${opt}_stripped"
            echo "    OK: zlib_zpipe_${opt}"
        fi
    fi
    cd ~/cs785-project
done

# ── 2. OpenSSL ──
echo ""
echo "=== OpenSSL ==="
OPENSSL_DIR="openssl-3.2.1"
if [ ! -d "$BUILD_DIR/$OPENSSL_DIR" ]; then
    echo "  Downloading OpenSSL..."
    cd "$BUILD_DIR"
    wget -q https://www.openssl.org/source/openssl-3.2.1.tar.gz
    tar xf openssl-3.2.1.tar.gz
    cd ~/cs785-project
fi

for opt in O0 O1 O2 O3; do
    existing=$(ls "$DATA_STRIPPED/openssl_"*"_${opt}_stripped" 2>/dev/null | wc -l)
    if [ "$existing" -gt 0 ]; then
        echo "  SKIP openssl $opt: already exists"
        continue
    fi
    echo "  Compiling OpenSSL at -$opt..."
    cd "$BUILD_DIR/$OPENSSL_DIR"
    make clean 2>/dev/null || true
    ./Configure linux-x86_64 -$opt -g -fno-pie -fno-PIE no-shared --prefix=/tmp/openssl_build 2>/dev/null
    make -j$(nproc) 2>/dev/null
    for bin in apps/openssl; do
        if [ -f "$bin" ] && file "$bin" | grep -q "ELF"; then
            bin_name=$(basename "$bin")
            cp "$bin" "$DATA_RAW/openssl_${bin_name}_${opt}"
            cp "$bin" "$DATA_STRIPPED/openssl_${bin_name}_${opt}_stripped"
            strip "$DATA_STRIPPED/openssl_${bin_name}_${opt}_stripped"
            echo "    OK: openssl_${bin_name}_${opt}"
        fi
    done
    cd ~/cs785-project
done

# ── 3. busybox ──
echo ""
echo "=== busybox ==="
BUSYBOX_DIR="busybox-1.36.1"
if [ ! -d "$BUILD_DIR/$BUSYBOX_DIR" ]; then
    echo "  Downloading busybox..."
    cd "$BUILD_DIR"
    wget -q https://busybox.net/downloads/busybox-1.36.1.tar.bz2
    tar xf busybox-1.36.1.tar.bz2
    cd ~/cs785-project
fi

for opt in O0 O1 O2 O3; do
    existing=$(ls "$DATA_STRIPPED/busybox_"*"_${opt}_stripped" 2>/dev/null | wc -l)
    if [ "$existing" -gt 0 ]; then
        echo "  SKIP busybox $opt: already exists"
        continue
    fi
    echo "  Compiling busybox at -$opt..."
    cd "$BUILD_DIR/$BUSYBOX_DIR"
    make clean 2>/dev/null || true
    make defconfig 2>/dev/null
    # Set optimization in .config
    sed -i "s/CONFIG_EXTRA_CFLAGS=\".*\"/CONFIG_EXTRA_CFLAGS=\"-$opt -g -fno-pie -fno-PIE\"/" .config
    sed -i "s/CONFIG_EXTRA_LDFLAGS=\".*\"/CONFIG_EXTRA_LDFLAGS=\"-no-pie\"/" .config
    # Disable PIE
    sed -i 's/CONFIG_PIE=y/# CONFIG_PIE is not set/' .config 2>/dev/null
    make -j$(nproc) 2>/dev/null
    if [ -f busybox ] && file busybox | grep -q "ELF"; then
        cp busybox "$DATA_RAW/busybox_busybox_${opt}"
        cp busybox "$DATA_STRIPPED/busybox_busybox_${opt}_stripped"
        strip "$DATA_STRIPPED/busybox_busybox_${opt}_stripped"
        echo "    OK: busybox_busybox_${opt}"
    else
        echo "    FAILED: busybox $opt"
    fi
    cd ~/cs785-project
done

# ── 4. curl (move from demo to training) ──
echo ""
echo "=== curl ==="
# curl is already compiled at O0/O2 as demo. Check if we need O1/O3.
CURL_DIR="curl-8.6.0"
if [ ! -d "$BUILD_DIR/$CURL_DIR" ]; then
    echo "  Downloading curl..."
    cd "$BUILD_DIR"
    wget -q https://curl.se/download/curl-8.6.0.tar.xz
    tar xf curl-8.6.0.tar.xz
    cd ~/cs785-project
fi

for opt in O0 O1 O2 O3; do
    existing=$(ls "$DATA_STRIPPED/curl_"*"_${opt}_stripped" 2>/dev/null | wc -l)
    if [ "$existing" -gt 0 ]; then
        echo "  SKIP curl $opt: already exists"
        continue
    fi
    echo "  Compiling curl at -$opt..."
    cd "$BUILD_DIR/$CURL_DIR"
    make clean 2>/dev/null || true
    CFLAGS="-$opt -g -fno-pie -fno-PIE" LDFLAGS="-no-pie" ./configure --quiet --without-ssl --without-libssh2 2>/dev/null
    make -j$(nproc) 2>/dev/null
    if [ -f src/curl ] && file src/curl | grep -q "ELF"; then
        cp src/curl "$DATA_RAW/curl_curl_${opt}"
        cp src/curl "$DATA_STRIPPED/curl_curl_${opt}_stripped"
        strip "$DATA_STRIPPED/curl_curl_${opt}_stripped"
        echo "    OK: curl_curl_${opt}"
    else
        echo "    FAILED: curl $opt"
    fi
    cd ~/cs785-project
done

echo ""
echo "═══════════════════════════════════════════════"
echo " Phase B compilation done."
echo " Next: run preprocess_new_bins.sh for BAP lifting"
echo "═══════════════════════════════════════════════"
