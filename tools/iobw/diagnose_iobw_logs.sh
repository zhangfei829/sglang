#!/usr/bin/env bash
# ============================================================
# Diagnose UMBP IO-bandwidth telemetry for a single results dir.
#
# Usage:
#   diagnose_iobw_logs.sh [RESULTS_DIR]
#
# If RESULTS_DIR is omitted, picks the newest dir under
#   benchmark/hicache/results/tiered_cache_bench_dp8ep8/
#
# What it does:
#   1. Lists every server.log / bench.log / *.jsonl file under the dir.
#   2. Greps for the IO-BW markers (calling / done / IOBW / bandwidth_gib_s).
#   3. Reports counts and a short head/tail sample so we can tell whether
#      logs are *missing* or *just need a wider grep window*.
#   4. Invokes summarize_iobw_jsonl.py to recompute avg / max bandwidth
#      from the JSONL side-channel (the source of truth even when
#      server.log is truncated by SIGKILL or tee buffering).
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUMMARIZE_PY="$SCRIPT_DIR/summarize_iobw_jsonl.py"

log() { echo "[$(date '+%H:%M:%S')] [iobw-diag] $*"; }

target="${1:-}"
if [[ -z "$target" ]]; then
    base="$REPO_ROOT/benchmark/hicache/results/tiered_cache_bench_dp8ep8"
    if [[ ! -d "$base" ]]; then
        log "ERROR: no results base dir at $base"
        exit 1
    fi
    target="$(ls -1dt "$base"/*/ 2>/dev/null | head -n 1 | sed 's:/$::')"
    if [[ -z "$target" ]]; then
        log "ERROR: no timestamped run dir under $base"
        exit 1
    fi
fi

if [[ ! -d "$target" ]]; then
    log "ERROR: target dir does not exist: $target"
    exit 1
fi

log "Inspecting: $target"
log "----- File inventory ----------------------------------"
# Show .log and .jsonl files (size + mtime).
find "$target" -maxdepth 5 \
    \( -name "*.log" -o -name "*.jsonl" -o -name "summary.txt" \) \
    -printf "%TY-%Tm-%Td %TH:%TM  %10s  %p\n" 2>/dev/null \
    | sort -k1,2 || true

log "----- Server-log marker counts ------------------------"
server_logs=()
while IFS= read -r f; do server_logs+=("$f"); done < <(find "$target" -maxdepth 5 -name "server.log" 2>/dev/null)

if (( ${#server_logs[@]} == 0 )); then
    log "WARNING: no server.log under $target.  Was the server even launched?"
else
    for slog in "${server_logs[@]}"; do
        # NOTE: `grep -c` already prints "0" on zero matches (and exits 1).
        # Do NOT append `|| echo 0` or the count becomes "0\n0", which breaks
        # the `(( ... ))` arithmetic below with a syntax error.  Use :-0 only
        # to cover the missing-file case where grep prints nothing.
        local_calling=$(grep -c "batch_get_v1: calling UMBP BatchGet" "$slog" 2>/dev/null); local_calling=${local_calling:-0}
        local_get_done=$(grep -c "batch_get_v1: UMBP BatchGet done" "$slog" 2>/dev/null); local_get_done=${local_get_done:-0}
        local_set_calling=$(grep -c "batch_set_v1: calling UMBP BatchPut" "$slog" 2>/dev/null); local_set_calling=${local_set_calling:-0}
        local_set_done=$(grep -c "batch_set_v1: UMBP BatchPut done" "$slog" 2>/dev/null); local_set_done=${local_set_done:-0}
        local_iobw=$(grep -c "\[UMBPStore\]\[IOBW\]" "$slog" 2>/dev/null); local_iobw=${local_iobw:-0}
        local_bw_token=$(grep -c "bandwidth_gib_s" "$slog" 2>/dev/null); local_bw_token=${local_bw_token:-0}

        printf "  %s\n" "$slog"
        printf "    batch_get_v1 calling=%-7s done=%-7s\n" "$local_calling" "$local_get_done"
        printf "    batch_set_v1 calling=%-7s done=%-7s\n" "$local_set_calling" "$local_set_done"
        printf "    [UMBPStore][IOBW] lines = %s\n" "$local_iobw"
        printf "    bandwidth_gib_s tokens = %s\n" "$local_bw_token"

        if (( local_calling > 0 && local_get_done == 0 )); then
            log "    ALERT: BatchGet calling without matching done.  IO hang or process killed mid-call?"
        fi
        if (( local_set_calling > 0 && local_set_done == 0 )); then
            log "    ALERT: BatchPut calling without matching done.  IO hang or process killed mid-call?"
        fi
        if (( local_iobw == 0 && (local_get_done + local_set_done) > 0 )); then
            log "    NOTE: per-call done lines exist but no [UMBPStore][IOBW] summary."
            log "          That means atexit/close() didn't run (likely SIGKILL)."
            log "          The JSONL side-channel below should still have everything."
        fi
    done
fi

log "----- Sample done lines (last 5 per server.log) -------"
for slog in "${server_logs[@]}"; do
    printf "  %s:\n" "$slog"
    grep -E "batch_(get|set)_v1: UMBP Batch(Get|Put) done" "$slog" 2>/dev/null \
        | tail -n 5 \
        | sed 's/^/    /' \
        || true
done

log "----- JSONL side-channel ------------------------------"
jsonl_files=()
while IFS= read -r f; do jsonl_files+=("$f"); done < <(find "$target" -maxdepth 5 -name "iobw_*.jsonl" 2>/dev/null)

if (( ${#jsonl_files[@]} == 0 )); then
    log "No iobw_*.jsonl files inside $target."
    # Also probe common fallback locations.
    for cand in \
        "$REPO_ROOT/umbp_iobw_logs" \
        "$PWD/umbp_iobw_logs" \
        "/tmp/umbp_iobw_logs"; do
        if [[ -d "$cand" ]] && compgen -G "$cand/iobw_*.jsonl" >/dev/null; then
            log "Found JSONL files in fallback dir: $cand"
            while IFS= read -r f; do jsonl_files+=("$f"); done \
                < <(find "$cand" -maxdepth 1 -name "iobw_*.jsonl" 2>/dev/null)
        fi
    done
fi

if (( ${#jsonl_files[@]} == 0 )); then
    log "No JSONL side-channel files anywhere.  Possible reasons:"
    log "  - Old server code (without the JSONL patch) was still running."
    log "  - UMBP_IO_BW_STATS=false explicitly disabled the feature."
    log "  - UMBPStore never instantiated (case ran with non-UMBP backend)."
else
    log "Found ${#jsonl_files[@]} JSONL files:"
    printf "    %s\n" "${jsonl_files[@]}"

    log "----- Recomputed bandwidth from JSONL ------------------"
    if command -v python3 >/dev/null 2>&1; then
        python3 "$SUMMARIZE_PY" "${jsonl_files[@]}" || true
    elif command -v python >/dev/null 2>&1; then
        python "$SUMMARIZE_PY" "${jsonl_files[@]}" || true
    else
        log "WARNING: no python interpreter; cannot run $SUMMARIZE_PY"
        log "         Showing last 3 entries from each file instead:"
        for f in "${jsonl_files[@]}"; do
            printf "    %s\n" "$f"
            tail -n 3 "$f" | sed 's/^/      /'
        done
    fi
fi

log "----- summary.txt -------------------------------------"
if [[ -f "$target/summary.txt" ]]; then
    sed 's/^/    /' "$target/summary.txt"
else
    log "No $target/summary.txt"
fi
