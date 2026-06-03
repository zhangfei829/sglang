#!/usr/bin/env python3
"""Analyze hicache host->device load from a server.log.

Parses the debug markers emitted with SGLANG_HICACHE_LOAD_AGG=1 (and optionally
SGLANG_HICACHE_LOAD_TIMING=1) plus the UMBPStore BatchGet telemetry, and reports:

  1. Per-load GiB/s distribution (load-bound rate, GPU-event timed).
  2. END-TO-END effective rate = sum(bytes) / wall-clock span of the loads.
     If SSD->host prefetch keeps up and overlaps load, end-to-end ~= per-load.
     If the load stream STARVES waiting for prefetch, end-to-end << per-load.
  3. Stream utilization = sum(load_dt) / wall span  (1.0 = never starved).
  4. UMBP BatchGet (SSD/storage -> host) bandwidth, for cold-read cross-check
     (compare against PCIe Gen4x4 RAID0 cold ceiling ~8.72 GiB/s; >that = cache).

Usage:
  python tools/iobw/analyze_hicache_load.py <server.log> [--since-frac 0.5]

--since-frac F: only consider AGG loads in the last fraction F of the wall-span
  (e.g. 0.5 = second half), to isolate the steady-state replay phase from the
  initial write/warmup phase.  Default 0.0 (use all).
"""
import argparse
import re
import sys

GIB = 1024.0**3

AGG_RE = re.compile(
    r"\[HICACHE-LOAD-AGG\]\s+tokens=(\d+)\s+layers=(\d+)\s+bytes=(\d+)\s+"
    r"dt_ms=([0-9.]+)\s+GiB/s=([0-9.]+)\s+wall=([0-9.]+)"
)
LAYER_RE = re.compile(
    r"\[HICACHE-LOAD\]\s+layer=\d+\s+tokens=(\d+)\s+bytes=(\d+)\s+"
    r"dt_ms=([0-9.]+)\s+GiB/s=([0-9.]+)"
)
BGET_RE = re.compile(r"BatchGet done:.*?bandwidth_gib_s=([0-9.]+)")


def stats(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "min": s[0],
        "p50": s[n // 2],
        "p95": s[min(n - 1, int(n * 0.95))],
        "max": s[-1],
        "avg": sum(s) / n,
    }


def fmt(d):
    if d is None:
        return "(none)"
    return (
        f"n={d['n']} min={d['min']:.2f} p50={d['p50']:.2f} "
        f"p95={d['p95']:.2f} avg={d['avg']:.2f} max={d['max']:.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--since-frac", type=float, default=0.0)
    args = ap.parse_args()

    agg = []  # (bytes, dt_ms, gibps, wall)
    layer_bw = []
    bget_bw = []
    with open(args.logfile, "r", errors="ignore") as f:
        for line in f:
            m = AGG_RE.search(line)
            if m:
                agg.append(
                    (int(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)))
                )
                continue
            m = LAYER_RE.search(line)
            if m:
                layer_bw.append(float(m.group(4)))
                continue
            m = BGET_RE.search(line)
            if m:
                bget_bw.append(float(m.group(1)))

    print(f"=== {args.logfile} ===")
    print(f"AGG loads parsed: {len(agg)}  per-layer lines: {len(layer_bw)}  "
          f"BatchGet lines: {len(bget_bw)}")

    if layer_bw:
        print(f"per-LAYER load GiB/s (each layer GPU-timed): {fmt(stats(layer_bw))}")

    if not agg:
        print("No HICACHE-LOAD-AGG lines (run with SGLANG_HICACHE_LOAD_AGG=1).")
        return

    # Optional time-window filter to isolate steady-state replay phase.
    w0 = min(a[3] for a in agg)
    w1 = max(a[3] for a in agg)
    span_all = w1 - w0
    if args.since_frac > 0.0 and span_all > 0:
        cutoff = w0 + span_all * args.since_frac
        agg_f = [a for a in agg if a[3] >= cutoff]
    else:
        agg_f = agg

    gibps = [a[2] for a in agg_f]
    total_bytes = sum(a[0] for a in agg_f)
    busy_s = sum(a[1] for a in agg_f) / 1000.0
    wall_lo = min(a[3] for a in agg_f)
    wall_hi = max(a[3] for a in agg_f)
    span = wall_hi - wall_lo

    print()
    print(f"per-LOAD GiB/s (all-layer, GPU-event timed): {fmt(stats(gibps))}")
    print(f"window: loads={len(agg_f)} total_bytes={total_bytes/GIB:.2f} GiB "
          f"busy(sum dt)={busy_s:.3f}s wall_span={span:.3f}s")
    if span > 0:
        eff = total_bytes / span / GIB
        util = busy_s / span
        print(f"END-TO-END effective KV->HBM = {eff:.2f} GiB/s "
              f"(sum_bytes / wall_span)")
        print(f"load-stream utilization = {util*100:.1f}%  "
              f"(100% = never starved by prefetch)")
        # Interpretation hint
        perload = stats(gibps)["p50"]
        if util >= 0.9:
            verdict = "pipeline FULL: prefetch keeps up, SSD source NOT the bottleneck"
        elif eff >= 0.8 * perload:
            verdict = "mostly overlapped"
        else:
            verdict = ("load STARVED: end-to-end << per-load -> prefetch (SSD->host) "
                       "is the bottleneck OR not overlapping")
        print(f"verdict: {verdict}")

    if bget_bw:
        print()
        print(f"UMBP BatchGet (storage->host) GiB/s: {fmt(stats(bget_bw))}")
        print("  cross-check: PCIe Gen4x4 RAID0 cold ceiling ~8.72 GiB/s; "
              "values >>that are controller/ring CACHE (not cold).")


if __name__ == "__main__":
    sys.exit(main())
