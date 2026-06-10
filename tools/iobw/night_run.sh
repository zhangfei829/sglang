#!/usr/bin/env bash
# Overnight load-bound read_threads A/B (real forward + prewarm + wait_complete).
# T=1 vs T=2, 4 repeats each (8 runs), results appended to /tmp/night_results.txt.
# Each run records: prewarm-executed, replay-TTFT, BatchGet bw, replay #new/#cached
# (load-bound check), and prefill throughput (GPU-contention check ~1240).
cd /sgl-workspace/sglang || exit 1
RES=/tmp/night_results.txt
echo "NIGHT RUN START $(date)" > "$RES"

for i in 1 2 3 4; do
  for T in 1 2; do
    pkill -9 -f sglang 2>/dev/null
    sleep 5
    UMBP_DRAM_READ_THREADS="$T" \
    EXTRA_SERVER_ARGS="--hicache-storage-prefetch-policy wait_complete" \
    MODEL_PATH=/mnt/nvme1/data/DeepSeekV3.1 \
    UMBP_LOG_LEVEL=0 \
    CASES_OVERRIDE=case3:HBM_DRAM_SSD \
    DUMMY_FORWARD=false \
    NUM_ROUNDS=15 NUM_CLIENTS=16 MAX_PARALLEL=1 REQUEST_LENGTH=8192 \
    UMBP_SSD_BACKEND=posix UMBP_SSD_BYTES=0 \
    READ_REPLAY=true READ_REPLAY_PREWARM=1 \
    DO_CHECKOUT=false \
    tools/iobw/run_umbp_iobw_test.sh > "/tmp/night_${i}_t${T}.log" 2>&1

    RDIR=$(ls -1dt benchmark/hicache/results/tiered_cache_bench_dp8ep8/*/ 2>/dev/null | head -1)
    SLOG=$(ls "${RDIR}"case3_*/server.log 2>/dev/null | head -1)
    {
      echo "===== rep$i T=$T  $(date) ====="
      echo -n "prewarm_executed(>0=load-bound构造生效): "
      grep -c pre-warming "/tmp/night_${i}_t${T}.log"
      echo -n "replay-TTFT: "
      grep -hoE "TTFT. n=[0-9]+ avg=[0-9.]+s p50=[0-9.]+s p90=[0-9.]+s" "/tmp/night_${i}_t${T}.log" | tail -1
      echo -n "BatchGet: "
      grep -hE "^BatchGet +[0-9]" "/tmp/night_${i}_t${T}.log" | tail -1
      echo "replay末5 #new/#cached(load-bound: new应小,cached应大): "
      grep -oE "#new-token: [0-9]+, #cached-token: [0-9]+" "$SLOG" 2>/dev/null | tail -5
      echo "throughput末3(GPU独占应~1240,若~18=被抢占该轮作废): "
      grep -oE "input throughput .token/s.: [0-9.]+" "$SLOG" 2>/dev/null | tail -3
      echo ""
    } >> "$RES"
  done
done
echo "===== ALL DONE $(date) =====" >> "$RES"
