#!/usr/bin/env python3
"""Isolated DRAM-tier batch-write (PUT) bandwidth microbenchmark.

Drives mori UMBPClient.batch_put_from_ptr() against a DRAM-only standalone
store. This routes through StandaloneClient::BatchPut ->
LocalStorageManager::WriteBatchFromPtr -> DRAMTier::BatchWrite, i.e. the
multi-threaded parallel non-temporal copy path on the backup (L2->L3) side.
Thread fan-out is controlled by UMBP_DRAM_WRITE_THREADS (read once at client
construction); UMBP_DRAM_NT_COPY=0 disables the AVX2 streaming-store path.

Each iteration calls client.clear() OUTSIDE the timed region so the keys are
absent and BatchPut actually copies every page (otherwise index_.MayExist would
short-circuit a repeated put).

Usage:
  UMBP_DRAM_WRITE_THREADS=8 python bench_dram_batch_write.py \
      --page-size 524288 --num-pages 512 --iters 20 --warmup 3

Reports best / median host-DRAM write bandwidth in GiB/s.
"""
import argparse
import os
import statistics
import time

import numpy as np


def import_umbp():
    import mori.umbp as umbp_mod

    return umbp_mod


def build_dram_only_client(umbp_mod, capacity_bytes):
    UMBPClient = umbp_mod.UMBPClient
    UMBPConfig = umbp_mod.UMBPConfig
    UMBPRole = umbp_mod.UMBPRole

    cfg = UMBPConfig.from_environment()
    cfg.role = UMBPRole.Standalone
    cfg.dram.capacity_bytes = int(capacity_bytes)
    cfg.dram.use_hugepages = False
    cfg.ssd.enabled = False
    return UMBPClient(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-size", type=int, default=512 * 1024,
                    help="bytes per page (KV block)")
    ap.add_argument("--num-pages", type=int, default=512,
                    help="number of pages written per batch")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    page_size = args.page_size
    num_pages = args.num_pages
    total_bytes = page_size * num_pages
    write_threads = os.environ.get("UMBP_DRAM_WRITE_THREADS", "(default 8)")
    nt = os.environ.get("UMBP_DRAM_NT_COPY", "1")

    umbp_mod = import_umbp()
    # 1.5x headroom so the offset allocator never trips the no-space path.
    client = build_dram_only_client(umbp_mod, int(total_bytes * 1.5) + (64 << 20))

    # Source: one big buffer, sliced per page; fill so the copy isn't elided.
    src = np.frombuffer(
        (np.arange(total_bytes, dtype=np.uint64) & 0xFF).astype(np.uint8).tobytes(),
        dtype=np.uint8,
    ).copy()
    src_base = src.ctypes.data

    keys = [f"page_{i}" for i in range(num_pages)]
    sizes = [page_size] * num_pages
    src_ptrs = [src_base + i * page_size for i in range(num_pages)]

    def one_batch_put():
        client.clear()  # drop keys so every put actually copies (timed OUTSIDE)
        t0 = time.perf_counter()
        res = client.batch_put_from_ptr(keys, src_ptrs, sizes)
        dt = time.perf_counter() - t0
        if not all(res):
            raise RuntimeError(f"batch_put miss: {sum(res)}/{num_pages} ok (DRAM full?)")
        return dt

    for _ in range(args.warmup):
        one_batch_put()

    # Correctness: read the data back and compare against src.
    client.clear()
    if not all(client.batch_put_from_ptr(keys, src_ptrs, sizes)):
        raise RuntimeError("batch_put failed during correctness check")
    chk = np.zeros(total_bytes, dtype=np.uint8)
    chk_base = chk.ctypes.data
    dst_ptrs = [chk_base + i * page_size for i in range(num_pages)]
    if not all(client.batch_get_into_ptr(keys, dst_ptrs, sizes)):
        raise RuntimeError("batch_get failed during correctness check")
    if not np.array_equal(src, chk):
        raise RuntimeError("data mismatch: batch write stored wrong bytes")

    bws = []
    for _ in range(args.iters):
        dt = one_batch_put()
        bws.append(total_bytes / dt / (1024 ** 3))

    client.clear()
    best = max(bws)
    med = statistics.median(bws)
    print(
        f"wthreads={write_threads:>11}  nt={nt}  page={page_size//1024}KiB  "
        f"pages={num_pages}  batch={total_bytes/(1024**2):.0f}MiB  "
        f"best={best:7.2f} GiB/s  median={med:7.2f} GiB/s"
    )


if __name__ == "__main__":
    main()
