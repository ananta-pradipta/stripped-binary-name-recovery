#!/bin/bash
# ============================================================
# Phase 3: Push to 300K — more coreutils + diverse packages
# ============================================================
source ~/cs785-project/activate.sh
cd ~/cs785-project
set +e

BUILD_DIR="$HOME/cs785-project/build_tmp"
DATA_RAW="data/raw"
DATA_STRIPPED="data/stripped"
DATA_DEBUG="data/debug"

mkdir -p "$DATA_RAW" "$DATA_STRIPPED" "$DATA_DEBUG"
COMPILED=0

# Helper: compile, save debug+stripped, count
save_bin() {
    local actual="$1" name="$2"
    if [ -f "$actual" ] && file "$actual" 2>/dev/null | grep -q ELF; then
        cp "$actual" "$HOME/cs785-project/$DATA_RAW/${name}_sym"
        cp "$actual" "$HOME/cs785-project/$DATA_DEBUG/${name}"
        strip -s "$actual" -o "$HOME/cs785-project/$DATA_STRIPPED/${name}_stripped"
        COMPILED=$((COMPILED + 1))
        return 0
    fi
    # Try .libs fallback
    local libs="$(dirname "$actual")/.libs/$(basename "$actual")"
    if [ -f "$libs" ] && file "$libs" 2>/dev/null | grep -q ELF; then
        cp "$libs" "$HOME/cs785-project/$DATA_RAW/${name}_sym"
        cp "$libs" "$HOME/cs785-project/$DATA_DEBUG/${name}"
        strip -s "$libs" -o "$HOME/cs785-project/$DATA_STRIPPED/${name}_stripped"
        COMPILED=$((COMPILED + 1))
        return 0
    fi
    return 1
}

# ══ 1. More coreutils (coreutils4) — 25+ additional tools ══
echo "═══════════════════════════════════════════════"
echo " 1. coreutils4 — additional tools (O0-O3)"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
CU_DIR="coreutils-9.4"
[ ! -d "$CU_DIR" ] && tar xf "coreutils-9.4.tar.xz"

EXTRA_BINS="src/basename src/date src/echo src/factor src/groups src/hostid \
    src/install src/logname src/mkfifo src/numfmt src/pr src/printenv \
    src/printf src/pwd src/readlink src/rmdir src/sha1sum src/sha512sum \
    src/shuf src/sleep src/sum src/tac src/timeout src/truncate \
    src/tty src/uname src/unexpand src/unlink src/uptime src/users src/who src/whoami"

for opt in O0 O1 O2 O3; do
    # Check if already done
    first_bin=$(echo $EXTRA_BINS | awk '{print $1}')
    first_name="coreutils4_$(basename $first_bin)_${opt}"
    if [ -f "$HOME/cs785-project/$DATA_STRIPPED/${first_name}_stripped" ]; then
        echo "  [$opt] already compiled, skipping"
        continue
    fi

    cd "$BUILD_DIR/$CU_DIR"
    make distclean 2>/dev/null || make clean 2>/dev/null || true
    ./configure CFLAGS="-g -${opt}" --quiet 2>/dev/null && \
    make -j$(nproc) --quiet 2>/dev/null && {
        count=0
        for bin_path in $EXTRA_BINS; do
            bin_name=$(basename "$bin_path")
            name="coreutils4_${bin_name}_${opt}"
            [ -f "$HOME/cs785-project/$DATA_STRIPPED/${name}_stripped" ] && continue
            save_bin "$bin_path" "$name" && count=$((count + 1))
        done
        echo "  [$opt] $count binaries"
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/cs785-project

# ══ 2. zstd — manual build ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 2. zstd (O0-O3) [Makefile]"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
ZSTD_DIR="zstd-1.5.6"
if [ ! -d "$ZSTD_DIR" ]; then
    if [ ! -f "zstd-1.5.6.tar.gz" ]; then
        wget -q "https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz" -O "zstd-1.5.6.tar.gz"
    fi
    tar xf "zstd-1.5.6.tar.gz"
fi
for opt in O0 O1 O2 O3; do
    name="zstd_zstd_${opt}"
    if [ -f "$HOME/cs785-project/$DATA_RAW/${name}_sym" ]; then
        echo "  [$opt] already compiled"
        continue
    fi
    cd "$BUILD_DIR/$ZSTD_DIR"
    make -C programs clean 2>/dev/null; make -C lib clean 2>/dev/null
    make -j$(nproc) CFLAGS="-g -${opt}" CXXFLAGS="-g -${opt}" LDFLAGS="-g" 2>/dev/null && {
        save_bin "programs/zstd" "$name" && echo "  [$opt] compiled" || echo "  [$opt] binary not found"
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/cs785-project

# ══ 3. diffutils at O0/O1/O2/O3 (currently only default opt) ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 3. diffutils (O0-O3)"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
DIFF_TAR="diffutils-3.10.tar.xz"
DIFF_DIR="diffutils-3.10"
if [ ! -f "$DIFF_TAR" ]; then
    wget -q "https://ftp.gnu.org/gnu/diffutils/diffutils-3.10.tar.xz" -O "$DIFF_TAR"
fi
[ ! -d "$DIFF_DIR" ] && tar xf "$DIFF_TAR"
for opt in O0 O1 O2 O3; do
    name_check="diffutils2_diff_${opt}"
    if [ -f "$HOME/cs785-project/$DATA_STRIPPED/${name_check}_stripped" ]; then
        echo "  [$opt] already compiled"
        continue
    fi
    cd "$BUILD_DIR/$DIFF_DIR"
    make distclean 2>/dev/null || make clean 2>/dev/null || true
    ./configure CFLAGS="-g -${opt}" --quiet 2>/dev/null && \
    make -j$(nproc) --quiet 2>/dev/null && {
        count=0
        for bin in src/diff src/diff3 src/sdiff src/cmp; do
            bin_name=$(basename "$bin")
            name="diffutils2_${bin_name}_${opt}"
            save_bin "$bin" "$name" && count=$((count + 1))
        done
        echo "  [$opt] $count binaries"
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/cs785-project

# ══ 4. GNU bc (calculator — different domain) ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 4. GNU bc (O0-O3) — math/calculator domain"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
BC_TAR="bc-1.07.1.tar.gz"
BC_DIR="bc-1.07.1"
if [ ! -f "$BC_TAR" ]; then
    wget -q "https://ftp.gnu.org/gnu/bc/bc-1.07.1.tar.gz" -O "$BC_TAR"
fi
[ ! -d "$BC_DIR" ] && tar xf "$BC_TAR"
for opt in O0 O1 O2 O3; do
    name_check="bc2_bc_${opt}"
    if [ -f "$HOME/cs785-project/$DATA_STRIPPED/${name_check}_stripped" ]; then
        echo "  [$opt] already compiled"
        continue
    fi
    cd "$BUILD_DIR/$BC_DIR"
    make distclean 2>/dev/null || make clean 2>/dev/null || true
    ./configure CFLAGS="-g -${opt}" --quiet 2>/dev/null && \
    make -j$(nproc) --quiet 2>/dev/null && {
        for bin in bc/bc dc/dc; do
            bin_name=$(basename "$bin")
            name="bc2_${bin_name}_${opt}"
            save_bin "$bin" "$name" && echo "  [$opt] $bin_name"
        done
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/cs785-project

# ══ 5. GNU plotutils (graphics — very different domain) ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 5. GNU plotutils (O0-O3) — graphics domain"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
PLOT_TAR="plotutils-2.6.tar.gz"
PLOT_DIR="plotutils-2.6"
if [ ! -f "$PLOT_TAR" ]; then
    wget -q "https://ftp.gnu.org/gnu/plotutils/plotutils-2.6.tar.gz" -O "$PLOT_TAR"
fi
[ ! -d "$PLOT_DIR" ] && tar xf "$PLOT_TAR"
for opt in O0 O1 O2 O3; do
    name_check="plotutils_graph_${opt}"
    if [ -f "$HOME/cs785-project/$DATA_STRIPPED/${name_check}_stripped" ]; then
        echo "  [$opt] already compiled"
        continue
    fi
    cd "$BUILD_DIR/$PLOT_DIR"
    make distclean 2>/dev/null || make clean 2>/dev/null || true
    ./configure CFLAGS="-g -${opt}" CXXFLAGS="-g -${opt}" --quiet --without-x 2>/dev/null && \
    make -j$(nproc) --quiet 2>/dev/null && {
        for bin in graph/graph plot/plot tek2plot/tek2plot plotfont/plotfont spline/spline ode/ode double/double; do
            bin_name=$(basename "$bin")
            name="plotutils_${bin_name}_${opt}"
            save_bin "$bin" "$name" && echo "  [$opt] $bin_name"
        done
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/cs785-project

# ══ 6. GNU mailutils (email — different domain) ══
echo ""
echo "═══════════════════════════════════════════════"
echo " 6. GNU mailutils (O0-O3) — email domain"
echo "═══════════════════════════════════════════════"
cd "$BUILD_DIR"
MU_TAR="mailutils-3.16.tar.gz"
MU_DIR="mailutils-3.16"
if [ ! -f "$MU_TAR" ]; then
    wget -q "https://ftp.gnu.org/gnu/mailutils/mailutils-3.16.tar.gz" -O "$MU_TAR"
fi
[ ! -d "$MU_DIR" ] && tar xf "$MU_TAR"
for opt in O0 O2; do
    name_check="mailutils_mail_${opt}"
    if [ -f "$HOME/cs785-project/$DATA_STRIPPED/${name_check}_stripped" ]; then
        echo "  [$opt] already compiled"
        continue
    fi
    cd "$BUILD_DIR/$MU_DIR"
    make distclean 2>/dev/null || make clean 2>/dev/null || true
    ./configure CFLAGS="-g -${opt}" --quiet --without-guile --without-gssapi --without-gsasl --without-gnutls --without-mysql --without-postgres --without-ldap --without-fribidi 2>/dev/null && \
    make -j$(nproc) --quiet 2>/dev/null && {
        for bin in mail/mail readmsg/readmsg frm/frm from/from messages/messages; do
            bin_name=$(basename "$bin")
            name="mailutils_${bin_name}_${opt}"
            save_bin "$bin" "$name" && echo "  [$opt] $bin_name"
        done
    } || echo "  [$opt] FAILED"
    cd "$BUILD_DIR"
done
cd ~/cs785-project

echo ""
echo "═══════════════════════════════════════════════"
echo " Phase 3 done. Total new compiled: $COMPILED"
echo "═══════════════════════════════════════════════"
