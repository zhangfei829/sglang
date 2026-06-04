// Microbenchmark: compare H2D copy throughput for many small per-page copies
// using (a) single stream, (b) N streams, (c) hipMemcpyBatchAsync (ROCm 7.1+).
//
// Mimics hicache page_first_direct L2->HBM load: NUM_COPIES copies of
// COPY_BYTES each (default 72 KiB = one MLA page-layer), pinned host -> device.
//
// Build:  hipcc -O3 -std=c++17 bench_batch_vs_streams.hip -o bench_bvs
// Run:    ./bench_bvs [num_copies] [copy_bytes] [iters]
//
// hipMemcpyBatchAsync is resolved at runtime via dlsym so the program still
// builds/links on ROCm < 7.1 (the batch case is simply skipped there).
#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define HIPCHECK(expr)                                                                  \
  do {                                                                                  \
    hipError_t _e = (expr);                                                             \
    if (_e != hipSuccess) {                                                             \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)_e, hipGetErrorString(_e),   \
              __FILE__, __LINE__);                                                      \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

static double gibps(size_t total_bytes, double seconds) {
  return (double)total_bytes / seconds / (1024.0 * 1024.0 * 1024.0);
}

int main(int argc, char** argv) {
  size_t num_copies = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 2318;  // ~38 pages * 61 layers
  size_t copy_bytes = (argc > 2) ? strtoull(argv[2], nullptr, 10) : 72ull * 1024;
  int iters = (argc > 3) ? atoi(argv[3]) : 50;
  int warmup = 5;

  printf("num_copies=%zu copy_bytes=%zu (%.1f KiB) iters=%d total/iter=%.1f MiB\n", num_copies,
         copy_bytes, copy_bytes / 1024.0, iters,
         (double)(num_copies * copy_bytes) / (1024.0 * 1024.0));

  // One big pinned host buffer + one big device buffer; each copy uses a
  // distinct slot so the access pattern is scattered like real KV slots.
  size_t total = num_copies * copy_bytes;
  void* h = nullptr;
  void* d = nullptr;
  HIPCHECK(hipHostMalloc(&h, total, hipHostMallocDefault));
  HIPCHECK(hipMalloc(&d, total));

  std::vector<void*> dsts(num_copies), srcs(num_copies);
  std::vector<size_t> sizes(num_copies, copy_bytes);
  for (size_t i = 0; i < num_copies; ++i) {
    srcs[i] = (char*)h + i * copy_bytes;
    dsts[i] = (char*)d + i * copy_bytes;
  }

  hipEvent_t t0, t1;
  HIPCHECK(hipEventCreate(&t0));
  HIPCHECK(hipEventCreate(&t1));

  // ---- (a) single stream ----
  {
    hipStream_t s;
    HIPCHECK(hipStreamCreate(&s));
    for (int w = 0; w < warmup; ++w)
      for (size_t i = 0; i < num_copies; ++i) HIPCHECK(hipMemcpyAsync(dsts[i], srcs[i], copy_bytes, hipMemcpyHostToDevice, s));
    HIPCHECK(hipStreamSynchronize(s));
    HIPCHECK(hipEventRecord(t0, s));
    for (int it = 0; it < iters; ++it)
      for (size_t i = 0; i < num_copies; ++i) HIPCHECK(hipMemcpyAsync(dsts[i], srcs[i], copy_bytes, hipMemcpyHostToDevice, s));
    HIPCHECK(hipEventRecord(t1, s));
    HIPCHECK(hipEventSynchronize(t1));
    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    printf("single-stream      : %.2f GiB/s  (%.3f ms/iter)\n", gibps(total * iters, ms / 1000.0), ms / iters);
    HIPCHECK(hipStreamDestroy(s));
  }

  // ---- (b) N streams ----
  for (int nstreams : {4, 8, 16}) {
    std::vector<hipStream_t> streams(nstreams);
    for (int k = 0; k < nstreams; ++k) HIPCHECK(hipStreamCreate(&streams[k]));
    for (int w = 0; w < warmup; ++w)
      for (size_t i = 0; i < num_copies; ++i)
        HIPCHECK(hipMemcpyAsync(dsts[i], srcs[i], copy_bytes, hipMemcpyHostToDevice, streams[i % nstreams]));
    for (auto& s : streams) HIPCHECK(hipStreamSynchronize(s));
    hipEvent_t b0, b1;
    HIPCHECK(hipEventCreate(&b0));
    HIPCHECK(hipEventCreate(&b1));
    HIPCHECK(hipEventRecord(b0, streams[0]));
    HIPCHECK(hipStreamWaitEvent(streams[0], b0, 0));
    double host_ms = 0;
    for (int it = 0; it < iters; ++it)
      for (size_t i = 0; i < num_copies; ++i)
        HIPCHECK(hipMemcpyAsync(dsts[i], srcs[i], copy_bytes, hipMemcpyHostToDevice, streams[i % nstreams]));
    for (auto& s : streams) HIPCHECK(hipStreamSynchronize(s));
    HIPCHECK(hipEventRecord(b1, streams[0]));
    HIPCHECK(hipEventSynchronize(b1));
    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, b0, b1));
    (void)host_ms;
    printf("%2d-streams         : %.2f GiB/s  (%.3f ms/iter)\n", nstreams, gibps(total * iters, ms / 1000.0), ms / iters);
    for (auto& s : streams) HIPCHECK(hipStreamDestroy(s));
  }

  // ---- (c) hipMemcpyBatchAsync (ROCm 7.1+, resolved via dlsym) ----
  using BatchFn = hipError_t (*)(void**, void**, size_t*, size_t, void*, size_t*, size_t, size_t*, hipStream_t);
  BatchFn batch = reinterpret_cast<BatchFn>(dlsym(RTLD_DEFAULT, "hipMemcpyBatchAsync"));
  if (batch == nullptr) {
    printf("hipMemcpyBatchAsync: NOT AVAILABLE (ROCm < 7.1)\n");
  } else {
    hipStream_t s;
    HIPCHECK(hipStreamCreate(&s));
    size_t fail_idx = 0;
    // attrs=nullptr, numAttrs=0 -> default attributes (works for plain H2D).
    for (int w = 0; w < warmup; ++w)
      batch(dsts.data(), srcs.data(), sizes.data(), num_copies, nullptr, nullptr, 0, &fail_idx, s);
    HIPCHECK(hipStreamSynchronize(s));
    HIPCHECK(hipEventRecord(t0, s));
    for (int it = 0; it < iters; ++it)
      batch(dsts.data(), srcs.data(), sizes.data(), num_copies, nullptr, nullptr, 0, &fail_idx, s);
    HIPCHECK(hipEventRecord(t1, s));
    HIPCHECK(hipEventSynchronize(t1));
    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    printf("hipMemcpyBatchAsync: %.2f GiB/s  (%.3f ms/iter)\n", gibps(total * iters, ms / 1000.0), ms / iters);
    HIPCHECK(hipStreamDestroy(s));
  }

  HIPCHECK(hipHostFree(h));
  HIPCHECK(hipFree(d));
  return 0;
}
