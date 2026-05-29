#!/usr/bin/env python3
"""Recompute UMBP IO-bandwidth stats from one or more JSONL side-channel files.

The JSONL files are produced by ``UMBPStore`` (see umbp_store.py) and contain
one record per BatchGet / BatchPut call, plus a final summary record when the
process shuts down cleanly.  Because each record is flushed (and optionally
fsync'd) immediately, this file survives SIGKILL / OOM / log buffer loss --
which is exactly when ``[UMBPStore][IOBW]`` lines in server.log get truncated.

Usage::

    summarize_iobw_jsonl.py FILE [FILE...]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def iter_records(paths: Iterable[Path]) -> Iterable[Tuple[Path, dict]]:
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(
                            f"WARN: {p}:{line_no} json decode failed: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    yield p, rec
        except OSError as exc:
            print(f"WARN: cannot open {p}: {exc}", file=sys.stderr)


def fmt_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    if n <= 0:
        return "0 B"
    e = min(int(math.log(n, 1024)), len(units) - 1)
    return f"{n / (1024 ** e):.3f} {units[e]}"


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.3f} ms"


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(paths: List[Path]) -> int:
    per_file_summary: List[Tuple[Path, dict]] = []
    per_file_open: List[Tuple[Path, dict]] = []

    # Per-op aggregates: op -> {requests, expanded, success, total_bytes,
    # success_bytes, elapsed_s, n_calls, bandwidth_samples}
    per_op_total: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "calls": 0,
            "requests": 0,
            "expanded": 0,
            "success": 0,
            "total_bytes": 0,
            "success_bytes": 0,
            "elapsed_s": 0.0,
            "samples_bw": [],
            "samples_elapsed_s": [],
            "samples_success_bytes": [],
        }
    )

    # Per-(file, op): for inter-file consistency reporting.
    per_file_op_calls: Dict[Tuple[str, str], int] = defaultdict(int)

    parse_errors = 0

    for p, rec in iter_records(paths):
        rtype = rec.get("type")
        if rtype == "open":
            per_file_open.append((p, rec))
            continue
        if rtype == "summary":
            per_file_summary.append((p, rec))
            continue
        if rtype != "call":
            parse_errors += 1
            continue
        op = rec.get("op", "?")
        agg = per_op_total[op]
        agg["calls"] += 1
        agg["requests"] += int(rec.get("requests", 0))
        agg["expanded"] += int(rec.get("expanded", 0))
        agg["success"] += int(rec.get("success", 0))
        agg["total_bytes"] += int(rec.get("total_bytes", 0))
        agg["success_bytes"] += int(rec.get("success_bytes", 0))
        elapsed_s = float(rec.get("elapsed_s", 0.0))
        agg["elapsed_s"] += elapsed_s
        agg["samples_bw"].append(float(rec.get("bandwidth_gib_s", 0.0)))
        agg["samples_elapsed_s"].append(elapsed_s)
        agg["samples_success_bytes"].append(int(rec.get("success_bytes", 0)))
        per_file_op_calls[(str(p), op)] += 1

    print(f"==> Files processed: {len(paths)}")
    print(f"==> open  records : {len(per_file_open)}")
    print(f"==> call  records : {sum(int(a['calls']) for a in per_op_total.values())}")
    print(f"==> summary recs  : {len(per_file_summary)}")
    if parse_errors:
        print(f"==> non-call/open/summary records: {parse_errors}")
    print()

    if not per_op_total:
        print("No call records.  Either UMBPStore never made a BatchGet/BatchPut,")
        print("or the JSONL side-channel was opened but no IO happened.")
        return 0

    print("==> Per-op aggregated bandwidth (recomputed from call records)")
    header = (
        f"{'op':<10}"
        f"{'calls':>10}"
        f"{'requests':>12}"
        f"{'expanded':>12}"
        f"{'success':>10}"
        f"{'success_bytes':>20}"
        f"{'elapsed_ms':>14}"
        f"{'avg_GiB/s':>12}"
        f"{'p50_GiB/s':>12}"
        f"{'p95_GiB/s':>12}"
        f"{'max_GiB/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for op in sorted(per_op_total):
        a = per_op_total[op]
        elapsed_s = max(float(a["elapsed_s"]), 1e-12)
        avg_bw = a["success_bytes"] / elapsed_s / (1024 ** 3)
        bws = sorted(a["samples_bw"])
        p50 = percentile(bws, 0.50)
        p95 = percentile(bws, 0.95)
        mx = max(bws) if bws else float("nan")
        print(
            f"{op:<10}"
            f"{int(a['calls']):>10d}"
            f"{int(a['requests']):>12d}"
            f"{int(a['expanded']):>12d}"
            f"{int(a['success']):>10d}"
            f"{int(a['success_bytes']):>20d}"
            f"{a['elapsed_s'] * 1000:>14.3f}"
            f"{avg_bw:>12.3f}"
            f"{p50:>12.3f}"
            f"{p95:>12.3f}"
            f"{mx:>12.3f}"
        )

    # Distribution of call sizes (helps explain bandwidth dips).
    print()
    print("==> Per-op call-size and per-call elapsed distribution")
    for op in sorted(per_op_total):
        a = per_op_total[op]
        ss = a["samples_success_bytes"]
        es = a["samples_elapsed_s"]
        if not ss:
            continue
        print(f"  {op}:")
        print(
            f"    success_bytes: min={fmt_bytes(min(ss))}  "
            f"p50={fmt_bytes(statistics.median(ss))}  "
            f"p95={fmt_bytes(percentile(ss, 0.95))}  "
            f"max={fmt_bytes(max(ss))}"
        )
        print(
            f"    per-call time: min={fmt_ms(min(es))}  "
            f"p50={fmt_ms(statistics.median(es))}  "
            f"p95={fmt_ms(percentile(es, 0.95))}  "
            f"max={fmt_ms(max(es))}"
        )

    # Per-file summary - useful when comparing TP/DP ranks.
    if per_file_summary:
        print()
        print("==> Per-file summary records (as written by UMBPStore on shutdown)")
        for p, rec in per_file_summary:
            ops = rec.get("ops") or {}
            collected = rec.get("records_collected")
            dropped = rec.get("records_dropped")
            print(f"  {p}: collected={collected} dropped={dropped}")
            for op, s in sorted(ops.items()):
                print(
                    f"    {op}: calls={s.get('calls')} "
                    f"avg_GiB/s={s.get('avg_bandwidth_gib_s'):.3f} "
                    f"max_GiB/s={s.get('max_call_bandwidth_gib_s'):.3f}"
                )

    # Per-file open records - useful when comparing TP/DP ranks coverage.
    if per_file_open:
        print()
        print("==> Per-file open records (one per UMBPStore instance)")
        for p, rec in per_file_open:
            print(
                f"  {p}: host={rec.get('host')} pid={rec.get('pid')} "
                f"local_rank={rec.get('local_rank')} pp_rank={rec.get('pp_rank')} "
                f"tp_size={rec.get('tp_size')}"
            )

    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    return summarize(args.paths)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
