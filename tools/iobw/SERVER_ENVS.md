# UMBP IOBW — Per-Server Known-Good Environments

All values below are taken from runs that actually ran/PASSED in the transcript
history. Do not guess paths or PCI addresses — pick the matching server section.

Common to both servers:
- Repo (inside container): `/sgl-workspace/sglang`. Branch markers required:
  `bandwidth_gib_s` + `[UMBPStore][IOBW]` in `umbp_store.py`.
- Always pass `DO_CHECKOUT=false` (untracked ROCm hipify `*.hip` files trip the
  clean-repo guard; code is already on the right branch).
- `UMBP_LOG_LEVEL=0` to surface spdk_proxy child stderr in server.log.
- Telemetry: `UMBP_IO_BW_STATS=true` (default) writes per-call JSONL; recompute
  with `tools/iobw/summarize_iobw_jsonl.py "$RDIR"/**/iobw_*.jsonl`.
- **HICACHE_SIZE (L2) must be > device KV pool** (assert in memory_pool_host.py).
- **Read bandwidth (BatchGet) requires `READ_REPLAY=true`** — it flushes HBM+host
  DRAM (L3 survives) and replays prompts so reads come from L3. Without it you
  only get `BatchPut` (writes), `BatchGet=0`.
- Tiers: L1=HBM, L2=host DRAM (`--hicache-size`), L3=UMBP. Only `case3` uses UMBP.
  DRAM-only L3 (no SSD/SPDK): `UMBP_SSD_BYTES=0` + `UMBP_SSD_BACKEND=posix`.

---

## Server A: banff-ccs-aus-p19-38 (8x MI300X)

| Item | Value |
|---|---|
| Model | `/nfs/data/DeepSeek-V3` |
| SSD/SPDK NVMe PCI | `0000:50:00.0` |
| Container access | `sudo docker exec -i $(sudo docker ps --format '{{.Names}}' | head -1) bash -lc '...'` |
| Hugepages | `echo 32768 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages` (64 GiB; single global knob) |

VERIFIED PASSED (2026-05-29, SSD/spdk_proxy path):
- `CASES_OVERRIDE=case3:HBM_DRAM_SSD DUMMY_FORWARD=true NUM_ROUNDS=20 NUM_CLIENTS=1 MAX_PARALLEL=1 HICACHE_SIZE=32 UMBP_SSD_BACKEND=spdk_proxy`
- Result: BatchPut 26 calls / 2.75 GiB, **avg 1.405 GiB/s** (SSD via SPDK), BatchGet=0 (no READ_REPLAY).

---

## Server B: skyriver07 (8x MI300X)  [CURRENT]

| Item | Value |
|---|---|
| Repo location | inside container `storge` (host has NO `/sgl-workspace`); use `docker exec storge bash -lc '...'` from host |
| Model | `/mnt/nvme1/data/DeepSeekV3.1` (alt `/mnt/nvme1/data/deepseek-r1-FP8-Dynamic-from-BF16`) |
| SSD/SPDK NVMe PCI | `0000:88:00.0` |
| spdk_proxy bin | `/opt/venv/lib/python3.10/site-packages/mori/spdk_proxy` |
| Hugepages | per-NUMA node: `echo 16384 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages` and same for `node1` (=> 32768 total, 64 GiB) |
| cpuset | expand docker parent + container cpuset to `0-383` (nproc 384) if pinning needed |
| Device KV pool | **27.04 GB / 413056 tokens per rank** => `HICACHE_SIZE` must be > 27 (use 32) |

VERIFIED RUN (2026-06-09, DRAM-only L3, posix, SSD disabled):
- `CASES_OVERRIDE=case3:HBM_DRAM_SSD DUMMY_FORWARD=true NUM_ROUNDS=40 NUM_CLIENTS=1 MAX_PARALLEL=1 HICACHE_SIZE=32 UMBP_SSD_BACKEND=posix UMBP_SSD_BYTES=0`
- Result: BatchPut 66 calls / 7.8 GiB, **L3 DRAM write avg 15.09 GiB/s** (p50 14.56 / p95 17.79 / max 17.88). BatchGet=0 (was missing `READ_REPLAY=true`).
- To get L3 DRAM **read** bandwidth, add `READ_REPLAY=true`.

### Server B one-shot (DRAM-only, write + read via replay)

```bash
cd /sgl-workspace/sglang && \
MODEL_PATH=/mnt/nvme1/data/DeepSeekV3.1 \
UMBP_LOG_LEVEL=0 CASES_OVERRIDE=case3:HBM_DRAM_SSD DUMMY_FORWARD=true \
NUM_ROUNDS=40 NUM_CLIENTS=1 MAX_PARALLEL=1 HICACHE_SIZE=32 \
UMBP_SSD_BACKEND=posix UMBP_SSD_BYTES=0 READ_REPLAY=true \
DO_CHECKOUT=false tools/iobw/run_umbp_iobw_test.sh 2>&1 | tee /tmp/iobw_run.log; \
echo "==================== IOBW RESULT (inline) ====================" && \
RDIR="$(ls -1dt benchmark/hicache/results/tiered_cache_bench_dp8ep8/*/ 2>/dev/null | head -n1 | sed 's:/$::')" && \
{ [ -n "$RDIR" ] && echo "results_dir = $RDIR" && \
  python3 tools/iobw/summarize_iobw_jsonl.py "$RDIR"/**/iobw_*.jsonl 2>/dev/null \
    | grep -E "records|GiB/s|BatchGet|BatchPut|op " \
  || echo "NO DATA: see /tmp/iobw_run.log"; }
```
