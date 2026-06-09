#!/usr/bin/env bash
# Sweep UMBP_DRAM_READ_THREADS for the DRAM-tier batch-read microbenchmark.
# Each setting runs in a fresh process because DRAMTier reads the env var once
# at construction. Adjust PAGE_SIZE / NUM_PAGES / THREADS via env if needed.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
PAGE_SIZE="${PAGE_SIZE:-524288}"   # 512 KiB
NUM_PAGES="${NUM_PAGES:-512}"      # 256 MiB per batch
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-3}"
THREADS="${THREADS:-1 2 4 8 16}"

echo "=== DRAM-tier batch-read sweep  page=$((PAGE_SIZE/1024))KiB pages=${NUM_PAGES} iters=${ITERS} ==="
for t in ${THREADS}; do
  UMBP_DRAM_READ_THREADS="${t}" "${PY}" "${HERE}/bench_dram_batch_read.py" \
    --page-size "${PAGE_SIZE}" --num-pages "${NUM_PAGES}" \
    --iters "${ITERS}" --warmup "${WARMUP}" \
    || echo "threads=${t}  FAILED"
done
