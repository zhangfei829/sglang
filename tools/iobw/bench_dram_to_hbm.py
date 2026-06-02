#!/usr/bin/env python3
"""Measure pinned-host(DRAM) -> device(HBM) copy bandwidth.

This is the ceiling for hicache's per-layer KV load (DRAM -> HBM).  Compare
against the SSD load bandwidth: if DRAM->HBM <= SSD read, the host<->GPU link is
the bottleneck and SSD can feed HBM at full speed (SSD can replace the DRAM tier
for the load path).

Per-layer KV (DeepSeek MLA): kv_cache_dim 576 * page_size 64 * bf16(2B) ~= 72 KiB
per page per layer.  We sweep from one page up to large transfers.
"""
import time

import torch


def bench(nbytes: int, iters: int = 50, warmup: int = 10) -> float:
    n = max(1, nbytes // 2)  # bf16 elements
    h = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
    d = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        d.copy_(h, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        d.copy_(h, non_blocking=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return (n * 2) / dt / (1024.0 ** 3)


if __name__ == "__main__":
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}")
    print(f"{'transfer':>14}  {'GiB/s':>8}  {'GB/s':>8}")
    sizes = [
        ("72 KiB (1page/layer)", 72 * 1024),
        ("576 KiB (8pages)", 576 * 1024),
        ("4.5 MiB (64pages)", 64 * 72 * 1024),
        ("18 MiB (256pages)", 256 * 72 * 1024),
        ("128 MiB", 128 * 1024 * 1024),
        ("512 MiB", 512 * 1024 * 1024),
        ("2 GiB", 2 * 1024 * 1024 * 1024),
    ]
    for label, nb in sizes:
        gibps = bench(nb)
        print(f"{label:>22}  {gibps:8.2f}  {gibps * 1.0737:8.2f}")
