/* Microbench: DPDK rte_memcpy vs glibc memcpy at realistic KV block sizes.
 * Single-thread, to isolate per-core copy throughput.
 *
 * Build (container; DPDK headers come from the SPDK-bundled DPDK):
 *   DPDK_INC=/home/fizhang/spdk/dpdk/build/include
 *   gcc -O3 -mavx512f -mavx512bw -I"$DPDK_INC" -include rte_config.h \
 *       tools/iobw/bench_dpdk_memcpy.c -o /tmp/bdm
 *   (if rte_config.h not found, drop -include; if it needs more, see notes)
 * Run:
 *   taskset -c 0 /tmp/bdm
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <rte_memcpy.h>

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double gibps(size_t bytes, double sec) {
  return (double)bytes / sec / (1024.0 * 1024.0 * 1024.0);
}

int main(void) {
  const size_t sizes[] = {64ul * 1024, 256ul * 1024, 1ul << 20, 4ul << 20,
                          16ul << 20, 137ul << 20};
  const size_t nsz = sizeof(sizes) / sizeof(sizes[0]);
  printf("%-12s %14s %14s   (GiB/s, single-thread)\n", "block",
         "glibc_memcpy", "rte_memcpy");
  for (size_t k = 0; k < nsz; ++k) {
    size_t sz = sizes[k];
    char *src = aligned_alloc(64, sz);
    char *dst = aligned_alloc(64, sz);
    memset(src, 0xAB, sz);
    memset(dst, 0, sz);
    int iters = sz < (1u << 20) ? 2000 : (sz < (16u << 20) ? 200 : 30);

    for (int w = 0; w < 3; ++w) memcpy(dst, src, sz);
    double t0 = now_sec();
    for (int it = 0; it < iters; ++it) memcpy(dst, src, sz);
    double g = gibps(sz, (now_sec() - t0) / iters);

    for (int w = 0; w < 3; ++w) rte_memcpy(dst, src, sz);
    t0 = now_sec();
    for (int it = 0; it < iters; ++it) rte_memcpy(dst, src, sz);
    double r = gibps(sz, (now_sec() - t0) / iters);

    char lbl[32];
    if (sz >= (1u << 20)) snprintf(lbl, sizeof(lbl), "%zuMiB", sz >> 20);
    else snprintf(lbl, sizeof(lbl), "%zuKiB", sz >> 10);
    printf("%-12s %14.1f %14.1f\n", lbl, g, r);
    free(src);
    free(dst);
  }
  return 0;
}
