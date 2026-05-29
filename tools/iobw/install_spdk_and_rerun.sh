#!/usr/bin/env bash
# ============================================================
# One-shot installer for the UMBP SPDK proxy backend, then
# re-runs the small-scale IOBW bench.
#
# Root cause this script fixes:
#   - `mori` was previously built without SPDK support, so the
#     `spdk_proxy` binary that LSM::SpawnProxyDaemon tries to
#     execlp() is missing  →  60s SpdkProxyTier timeout.
#   - `setup_spdk.sh --check` reports missing apt packages and
#     `3rdparty/spdk` submodule is uninitialized.
#
# What this script does:
#   1. apt-get install SPDK build-deps (nasm, meson, ...).
#   2. cd $MORI_DIR && ./tools/setup_spdk.sh
#      (configure + build + install SPDK to /usr/local; ldconfig).
#   3. BUILD_UMBP=ON pip install .
#      (rebuild + reinstall the `amd_mori` wheel so spdk_proxy
#       ends up in the wheel and on the PATH).
#   4. Locate the installed spdk_proxy binary.
#   5. Run tools/iobw/run_umbp_iobw_test.sh with
#      UMBP_SPDK_PROXY_BIN=<binary> + UMBP_LOG_LEVEL=0 so that
#      any further spdk_proxy stderr lands in server.log.
#
# Designed to be run in a detached tmux session, since steps 2-3
# take 15-30 minutes combined and we don't want the ssh dropping
# to kill the build.
#
# Env overrides:
#   MORI_DIR              default /sgl-workspace/mori
#   SGLANG_DIR            default /sgl-workspace/sglang
#   SKIP_APT_INSTALL      skip step 1
#   SKIP_SPDK_BUILD       skip step 2
#   SKIP_MORI_BUILD       skip step 3
#   SKIP_RUN_BENCH        skip step 5
# ============================================================
set -euo pipefail

MORI_DIR="${MORI_DIR:-/sgl-workspace/mori}"
SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"

log() { echo "[$(date '+%H:%M:%S')] [iobw-install] $*"; }
die() { log "ERROR: $*"; exit 1; }

[ -d "$MORI_DIR" ] || die "MORI_DIR=$MORI_DIR not found (set MORI_DIR=...)"
[ -d "$SGLANG_DIR" ] || die "SGLANG_DIR=$SGLANG_DIR not found"
[ -x "$MORI_DIR/tools/setup_spdk.sh" ] || die "$MORI_DIR/tools/setup_spdk.sh missing or not executable"

log "================================================================"
log "Mori repo:   $MORI_DIR"
log "Sglang repo: $SGLANG_DIR"
log "Mori HEAD:   $(git -C "$MORI_DIR" log --oneline -1 2>/dev/null || echo '(not a git repo)')"
log "================================================================"

# --- Step 1: apt-get build deps ----------------------------------------
if [ -z "${SKIP_APT_INSTALL:-}" ]; then
    log "Step 1/4: apt-get install SPDK build deps"
    export DEBIAN_FRONTEND=noninteractive
    if ! apt-get update; then
        log "WARN: apt-get update failed; continuing in case packages are cached"
    fi
    apt-get install -y --no-install-recommends \
        nasm meson help2man libaio-dev uuid-dev libcunit1-dev \
        libjson-c-dev libcmocka-dev libfuse3-dev python3-pyelftools
else
    log "Step 1/4: SKIPPED (SKIP_APT_INSTALL set)"
fi

# --- Step 2: build + install SPDK shared libraries ---------------------
if [ -z "${SKIP_SPDK_BUILD:-}" ]; then
    log "Step 2/4: building + installing SPDK shared libraries (10-20 min)"
    log "         (this also initialises the 3rdparty/spdk submodule)"
    cd "$MORI_DIR"
    # setup_spdk.sh does: configure --with-shared, make -j, make install, ldconfig
    ./tools/setup_spdk.sh
    log "         libspdk_env_dpdk should now be visible to ldconfig:"
    ldconfig -p 2>/dev/null | grep -i libspdk_env_dpdk || \
        die "libspdk_env_dpdk still missing after setup_spdk.sh"
else
    log "Step 2/4: SKIPPED (SKIP_SPDK_BUILD set)"
fi

# --- Step 3: rebuild mori with SPDK enabled ----------------------------
if [ -z "${SKIP_MORI_BUILD:-}" ]; then
    log "Step 3/4: BUILD_UMBP=ON pip install . (5-10 min)"
    cd "$MORI_DIR"
    BUILD_UMBP=ON pip install . 2>&1 | tail -n 80
else
    log "Step 3/4: SKIPPED (SKIP_MORI_BUILD set)"
fi

# --- Step 4: locate spdk_proxy -----------------------------------------
log "Step 4/4: locating spdk_proxy binary"
SPDK_BIN="$(command -v spdk_proxy 2>/dev/null || \
    find /usr/local /opt /sgl-workspace "$MORI_DIR/build_umbp" \
        -maxdepth 8 -type f -name spdk_proxy -executable 2>/dev/null | head -1)"
if [ -z "$SPDK_BIN" ]; then
    log "Locations searched:"
    log "  PATH (\`command -v spdk_proxy\`)"
    log "  find under /usr/local /opt /sgl-workspace $MORI_DIR/build_umbp"
    die "spdk_proxy still not found after rebuild — check Step 3 output above"
fi
log "Found spdk_proxy: $SPDK_BIN"
log "ldd $SPDK_BIN:"
ldd "$SPDK_BIN" 2>&1 | head -n 20 | sed 's/^/    /'

# --- Step 5: run the IOBW bench ----------------------------------------
if [ -z "${SKIP_RUN_BENCH:-}" ]; then
    log "Step 5: launching tools/iobw/run_umbp_iobw_test.sh"
    cd "$SGLANG_DIR"
    UMBP_SPDK_PROXY_BIN="$SPDK_BIN" \
    UMBP_LOG_LEVEL=0 \
    DO_CHECKOUT=false \
        tools/iobw/run_umbp_iobw_test.sh
else
    log "Step 5: SKIPPED (SKIP_RUN_BENCH set).  To run later:"
    log "        UMBP_SPDK_PROXY_BIN=$SPDK_BIN UMBP_LOG_LEVEL=0 \\"
    log "            DO_CHECKOUT=false tools/iobw/run_umbp_iobw_test.sh"
fi

log "All steps finished."
