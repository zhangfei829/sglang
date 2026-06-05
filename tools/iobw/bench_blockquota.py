#!/usr/bin/env python3
"""Sweep block_quota for the real sgl_kernel MLA kernel transfer op.

Isolates the hicache "kernel" backend grid-occupancy effect WITHOUT a server:
directly calls sgl_kernel.transfer_kv_per_layer_mla (host pinned -> device HBM,
one warp per token via device-accessible host pointer) for various block_quota
values and reports GiB/s. This pinpoints whether the default block_quota=2
starves the GPU (few warps) and what value fills it.

Run inside the sglang container (sgl_kernel installed), no model needed:
    python tools/iobw/bench_blockquota.py [num_tokens] [iters]
"""
import sys
import time

import torch
from sgl_kernel import transfer_kv_per_layer_mla

NUM_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 2432
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
WARMUP = 5

# DeepSeek-V3 MLA per-token KV: kv_lora(512) + rope(64) = 576 elems, bf16 (2B)
KV_DIM = 576
ELEM = 2
ITEM_SIZE = KV_DIM * ELEM  # 1152 bytes/token
CAP = NUM_TOKENS * 2  # buffer capacity (slots)

dev = torch.cuda.get_device_name(0)
print(f"GPU={dev} num_tokens={NUM_TOKENS} item_size={ITEM_SIZE}B iters={ITERS}")

# Host (pinned) source = L2 mirror; device dst = HBM pool.
src = torch.empty((CAP, KV_DIM), dtype=torch.bfloat16, pin_memory=True)
dst = torch.empty((CAP, KV_DIM), dtype=torch.bfloat16, device="cuda")
# Scattered indices (like real KV slots).
perm = torch.randperm(CAP, device="cuda")[:NUM_TOKENS].to(torch.int64)
src_idx = perm.clone()
dst_idx = perm.clone()

total_bytes = NUM_TOKENS * ITEM_SIZE


def run(block_quota: int, num_warps: int = 16) -> float:
    for _ in range(WARMUP):
        transfer_kv_per_layer_mla(src, dst, src_idx, dst_idx, ITEM_SIZE, block_quota, num_warps)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(ITERS):
        transfer_kv_per_layer_mla(src, dst, src_idx, dst_idx, ITEM_SIZE, block_quota, num_warps)
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / ITERS
    gibps = total_bytes / (ms / 1000.0) / (1024.0**3)
    return ms, gibps


print(f"{'block_quota':>12} {'ms/call':>10} {'GiB/s':>10}")
for bq in [2, 4, 16, 64, 152, 512, 4096]:
    ms, gibps = run(bq)
    print(f"{bq:>12} {ms:>10.4f} {gibps:>10.2f}")
