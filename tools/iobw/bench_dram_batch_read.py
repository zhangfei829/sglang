#!/usr/bin/env python3
"""Isolated DRAM-tier batch-read bandwidth microbenchmark.

Drives mori UMBPClient.batch_get_into_ptr() against a DRAM-only standalone
store. This routes through LocalStorageManager::ReadBatchIntoPtr ->
DRAMTier::ReadBatchIntoPtr, i.e. the multi-threaded parallel-memcpy path we
optimized. Thread fan-out is controlled by the UMBP_DRAM_READ_THREADS env var,
which DRAMTier reads once at client construction.

Usage:
  UMBP_DRAM_READ_THREADS=8 python bench_dram_batch_read.py \
      --page-size 524288 --num-pages 512 --iters 20 --warmup 3

Reports best / median host-DRAM read bandwidth in GiB/s.
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
                    help="number of pages read per batch")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    page_size = args.page_size
    num_pages = args.num_pages
    total_bytes = page_size * num_pages
    read_threads = os.environ.get("UMBP_DRAM_READ_THREADS", "(default 8)")

    umbp_mod = import_umbp()
    # 1.5x headroom so the offset allocator never trips the no-space path.
    client = build_dram_only_client(umbp_mod, int(total_bytes * 1.5) + (64 << 20))

    # Source: one big buffer, sliced per page; fill so memcpy isn't elided.
    src = np.frombuffer(
        (np.arange(total_bytes, dtype=np.uint64) & 0xFF).astype(np.uint8).tobytes(),
        dtype=np.uint8,
    ).copy()
    dst = np.zeros(total_bytes, dtype=np.uint8)
    src_base = src.ctypes.data
    dst_base = dst.ctypes.data

    keys = [f"page_{i}" for i in range(num_pages)]
    sizes = [page_size] * num_pages
    dst_ptrs = [dst_base + i * page_size for i in range(num_pages)]

    for i in range(num_pages):
        ok = client.put_from_ptr(keys[i], src_base + i * page_size, page_size)
        if not ok:
            raise RuntimeError(f"put failed for {keys[i]} (DRAM full?)")

    def one_batch_get():
        res = client.batch_get_into_ptr(keys, dst_ptrs, sizes)
        if not all(res):
            raise RuntimeError(f"batch_get miss: {sum(res)}/{num_pages} hit")

    for _ in range(args.warmup):
        one_batch_get()

    # Correctness: dst must equal src after a fresh read.
    dst.fill(0)
    one_batch_get()
    if not np.array_equal(src, dst):
        raise RuntimeError("data mismatch: batch read returned wrong bytes")

    bws = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        one_batch_get()
        dt = time.perf_counter() - t0
        bws.append(total_bytes / dt / (1024 ** 3))

    client.clear()
    best = max(bws)
    med = statistics.median(bws)
    print(
        f"threads={read_threads:>11}  page={page_size//1024}KiB  pages={num_pages}  "
        f"batch={total_bytes/(1024**2):.0f}MiB  "
        f"best={best:7.2f} GiB/s  median={med:7.2f} GiB/s"
    )


if __name__ == "__main__":
    main()
