# UMBP IO bandwidth telemetry tools

Helpers for the IO-BW patch in `python/sglang/srt/mem_cache/storage/umbp/umbp_store.py`
(see commit `5977d9f4e test(umbp): log batch io bandwidth stats` and the
follow-up JSONL side-channel patch).

This directory contains three things:

| File | Purpose |
|---|---|
| `run_umbp_iobw_test.sh` | One-shot runner: checkout via HTTPS, kill stale, run small bench, diagnose. |
| `diagnose_iobw_logs.sh` | Standalone: scan a results dir and recompute IO BW. |
| `summarize_iobw_jsonl.py` | Recompute avg / p50 / p95 / max from JSONL side-channel files. |

## What the telemetry produces

For every `batch_get_v1` / `batch_set_v1` call, `UMBPStore` now writes:

1. A `logger.info` line of the form

   ```
   [UMBPStore] batch_get_v1: UMBP BatchGet done: success=N/M elapsed_ms=... bandwidth_gib_s=...
   ```

2. A JSONL record into a side-channel file:

   ```
   <UMBP_IO_BW_STATS_DIR | SGLANG_LOG_DIR/umbp_iobw_logs | ./umbp_iobw_logs>/
     iobw_<host>_pid<PID>_dpx_tp<rank>_pp<rank>_<UTC>_<token>.jsonl
   ```

   Each record looks like:

   ```json
   {"type":"call","ts":1700000000.123,"op":"BatchGet",
    "requests":81,"expanded":81,"success":81,
    "total_bytes":364290048,"success_bytes":364290048,
    "elapsed_s":0.123,"bandwidth_gib_s":2.755}
   ```

   The file is `flush()`ed every record and (optionally) `fsync()`ed.
   It survives `SIGKILL`, OOM, and log-buffer truncation -- exactly the
   failure modes the original `logger.info` summary on `atexit` did not.

On shutdown the file also gets a `{"type":"summary", ...}` record with
the per-op totals.  An initial `{"type":"open", ...}` record identifies
the host / pid / rank.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `UMBP_IO_BW_STATS` | `true` if unset | Master switch.  Set `false` to disable everything. |
| `UMBP_IO_BW_STATS_MAX_RECORDS` | `100000` | Cap on the in-memory per-call list (used by the legacy logger summary). |
| `UMBP_IO_BW_STATS_JSONL` | `true` | Enable the JSONL side-channel. |
| `UMBP_IO_BW_STATS_JSONL_FSYNC` | `false` | `fsync(2)` every record.  Use `true` when SIGKILL is expected. |
| `UMBP_IO_BW_STATS_DIR` | (auto) | Override the side-channel directory.  Falls back to `${SGLANG_LOG_DIR}/umbp_iobw_logs`, then `./umbp_iobw_logs`, then `${TMPDIR}/umbp_iobw_logs`. |

The "extras" hash on `HiCacheStorageExtraInfo` also accepts the same
knobs under keys `io_bandwidth_stats`, `io_bandwidth_stats_max_records`,
`io_bandwidth_stats_jsonl`, `io_bandwidth_stats_jsonl_fsync`,
`io_bandwidth_stats_dir`.

## Quick recipe (server side)

```bash
cd /sgl-workspace/sglang

# Pre-clean.  Both the runner and the bench script do their own kill,
# but doing it once up front avoids racing with `tee`.
pkill -f "sglang\.launch_server|sglang serve|launch_server\.py" || true
pkill -f "umbp_master" || true

# Pull and run.  REPO_URL defaults to https://github.com/zhangfei829/sglang.git
# so no SSH key needs to be configured.
chmod +x tools/iobw/run_umbp_iobw_test.sh tools/iobw/diagnose_iobw_logs.sh
BRANCH=test/umbp-io-bandwidth-stats \
    tools/iobw/run_umbp_iobw_test.sh
```

This runs `case3:HBM_DRAM_SSD` with the small-scale parameters that have
already passed:

```
DUMMY_FORWARD=true
NUM_ROUNDS=20
NUM_CLIENTS=4
MAX_PARALLEL=4
REQUEST_RATE=10
HICACHE_SIZE=32
UMBP_SSD_BACKEND=spdk_proxy
UMBP_SPDK_NVME_PCI=0000:50:00.0
```

Override any of them on the env, e.g.:

```bash
NUM_ROUNDS=40 NUM_CLIENTS=8 MAX_PARALLEL=8 \
    tools/iobw/run_umbp_iobw_test.sh
```

## Just diagnose an existing results dir

```bash
# Latest run (default):
tools/iobw/diagnose_iobw_logs.sh

# Specific run:
tools/iobw/diagnose_iobw_logs.sh \
    benchmark/hicache/results/tiered_cache_bench_dp8ep8/20260529_120000
```

Sample output:

```
[12:00:00] [iobw-diag] Inspecting: .../20260529_120000
[12:00:00] [iobw-diag] ----- File inventory ----------------------------------
2026-05-29 12:00       12345  .../case3_HBM_DRAM_SSD/server.log
2026-05-29 12:00         567  .../case3_HBM_DRAM_SSD/umbp_iobw_logs/iobw_*.jsonl
...
[12:00:00] [iobw-diag] ----- Server-log marker counts ------------------------
  .../case3_HBM_DRAM_SSD/server.log
    batch_get_v1 calling=128     done=128
    batch_set_v1 calling=128     done=128
    [UMBPStore][IOBW] lines = 16
    bandwidth_gib_s tokens = 256

==> Per-op aggregated bandwidth (recomputed from call records)
op             calls    requests    expanded   success        success_bytes    elapsed_ms   avg_GiB/s   p50_GiB/s   p95_GiB/s   max_GiB/s
---------------------------------------------------------------------------------------------------------------------------------------
BatchGet         128         128         128       128         4 567 890 abc      1234.567       3.456       3.210       4.500       5.012
BatchPut         128         128         128       128         4 567 890 abc      2345.678       1.945       1.880       2.100       2.500
```

## "I checked out the new branch but `[UMBPStore][IOBW]` doesn't appear"

The diagnose script will tell you which of the following bucket you're in:

1. **`batch_*_v1: calling ...` lines exist but no `done` lines** →
   The IO call is hanging.  Process was probably killed (`timeout
   --kill-after`) mid-call.  Check `dmesg` and SPDK proxy state.
2. **`done` lines exist but no `[UMBPStore][IOBW]` summary** →
   Process was SIGKILLed before `atexit`.  The JSONL side-channel files
   still have everything; `summarize_iobw_jsonl.py` will reconstruct the
   summary.
3. **No `calling` lines either** →
   Either the run never hit UMBP (e.g. CASE wasn't case3), the server
   never started, or you're looking at a stale results dir.
   `latest_results_dir()` in the runner picks the newest one for you.
4. **JSONL files in `./umbp_iobw_logs/` but not in the results dir** →
   The runner moves them in for you, but if you ran the bench script
   directly (not through the runner) you have to do that by hand:

   ```bash
   mv umbp_iobw_logs RESULT_DIR/
   ```

## Server-side troubleshooting cheatsheet

```bash
# Is the running source actually the IOBW branch?
grep -nE "bandwidth_gib_s|UMBPStore\]\[IOBW" \
    /sgl-workspace/sglang/python/sglang/srt/mem_cache/storage/umbp/umbp_store.py | head

# Any stale processes left?
pgrep -af "sglang\.launch_server|launch_server\.py|umbp_master|spdk_proxy"

# Where did this benchmark land?
ls -td /sgl-workspace/sglang/benchmark/hicache/results/tiered_cache_bench_dp8ep8/*/ | head

# What's in the latest server.log?
RDIR=$(ls -td /sgl-workspace/sglang/benchmark/hicache/results/tiered_cache_bench_dp8ep8/*/ | head -1)
grep -nE "bandwidth_gib_s|\[UMBPStore\]\[IOBW\]|Batch(Get|Put) done" \
    "$RDIR"/case3_HBM_DRAM_SSD/server.log | head -n 30
```
