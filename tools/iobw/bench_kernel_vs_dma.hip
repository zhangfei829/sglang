// Strict isolation microbench: WHY is the hicache "kernel" backend slower than
// "direct" (DMA) for L2(host)->HBM page loads?
//
// Compares, for the SAME set of NUM_PAGES copies of PAGE_BYTES each:
//   (1) DMA  : hipMemcpyAsync, single stream
//   (2) DMA  : hipMemcpyAsync, N streams (overlap)
//   (3) KERNEL: a GPU gather kernel that mimics sgl-kernel transfer_kernel_impl
//               -- each warp copies one page, lanes stream 8-byte chunks with
//               nontemporal load/store, READING HOST memory over PCIe via a
//               device-accessible pointer (hipHostGetDevicePointer).
//
// If (3) is much slower than (2), it confirms the bottleneck is the GPU issuing
// fine-grained loads against host RAM over PCIe (not the DMA path / not layout).
//
// Build: hipcc -O3 -std=c++17 bench_kernel_vs_dma.hip -o bench_kvd
// Run:   ./bench_kvd [num_pages] [page_bytes] [iters]
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define WARP 64

#define HIPCHECK(expr)                                                                  \
  do {                                                                                  \
    hipError_t _e = (expr);                                                             \
    if (_e != hipSuccess) {                                                             \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)_e, hipGetErrorString(_e),   \
              __FILE__, __LINE__);                                                      \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

static double gibps(double bytes, double seconds) {
  return bytes / seconds / (1024.0 * 1024.0 * 1024.0);
}

// One warp per page; lanes cooperatively copy 8-byte chunks. src lives in host
// memory (accessed via device pointer) -> mimics transfer_item_warp exactly.
__global__ void gather_kernel(const uint64_t* __restrict__ src, uint64_t* __restrict__ dst,
                              int64_t chunks_per_page, int64_t num_pages) {
  int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = gid / WARP;
  int lane = (int)(gid % WARP);
  if (warp_id >= num_pages) return;
  const uint64_t* s = src + warp_id * chunks_per_page;
  uint64_t* d = dst + warp_id * chunks_per_page;
  for (int64_t j = lane; j < chunks_per_page; j += WARP) {
    uint64_t tmp = __builtin_nontemporal_load(s + j);
    __builtin_nontemporal_store(tmp, d + j);
  }
}

int main(int argc, char** argv) {
  int64_t num_pages = (argc > 1) ? strtoll(argv[1], nullptr, 10) : 2318;
  int64_t page_bytes = (argc > 2) ? strtoll(argv[2], nullptr, 10) : 72ll * 1024;
  int iters = (argc > 3) ? atoi(argv[3]) : 50;
  int warmup = 5;
  double total = (double)num_pages * page_bytes;

  printf("num_pages=%lld page_bytes=%lld (%.1f KiB) iters=%d total/iter=%.1f MiB\n",
         (long long)num_pages, (long long)page_bytes, page_bytes / 1024.0, iters,
         total / (1024.0 * 1024.0));

  void* h = nullptr;
  void* d = nullptr;
  HIPCHECK(hipHostMalloc(&h, (size_t)total, hipHostMallocDefault));
  HIPCHECK(hipMalloc(&d, (size_t)total));
  void* h_dev = nullptr;  // device-accessible pointer to host buffer
  HIPCHECK(hipHostGetDevicePointer(&h_dev, h, 0));

  std::vector<void*> srcs(num_pages), dsts(num_pages);
  for (int64_t i = 0; i < num_pages; ++i) {
    srcs[i] = (char*)h + i * page_bytes;
    dsts[i] = (char*)d + i * page_bytes;
  }

  hipEvent_t t0, t1;
  HIPCHECK(hipEventCreate(&t0));
  HIPCHECK(hipEventCreate(&t1));

  // (1)(2) DMA hipMemcpyAsync over nstreams (1 = single)
  for (int nstreams : {1, 8}) {
    std::vector<hipStream_t> streams(nstreams);
    for (int k = 0; k < nstreams; ++k) HIPCHECK(hipStreamCreate(&streams[k]));
    for (int w = 0; w < warmup; ++w)
      for (int64_t i = 0; i < num_pages; ++i)
        HIPCHECK(hipMemcpyAsync(dsts[i], srcs[i], page_bytes, hipMemcpyHostToDevice, streams[i % nstreams]));
    for (auto& s : streams) HIPCHECK(hipStreamSynchronize(s));
    HIPCHECK(hipEventRecord(t0, streams[0]));
    for (int it = 0; it < iters; ++it)
      for (int64_t i = 0; i < num_pages; ++i)
        HIPCHECK(hipMemcpyAsync(dsts[i], srcs[i], page_bytes, hipMemcpyHostToDevice, streams[i % nstreams]));
    for (auto& s : streams) HIPCHECK(hipStreamSynchronize(s));
    HIPCHECK(hipEventRecord(t1, streams[0]));
    HIPCHECK(hipEventSynchronize(t1));
    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    printf("DMA memcpy %2d-stream : %.2f GiB/s  (%.3f ms/iter)\n", nstreams,
           gibps(total * iters, ms / 1000.0), ms / iters);
    for (auto& s : streams) HIPCHECK(hipStreamDestroy(s));
  }

  // (3) KERNEL gather: GPU reads host over PCIe (mimics transfer_kernel_impl)
  {
    int64_t chunks_per_page = page_bytes / (int64_t)sizeof(uint64_t);
    int threads = 256;                                  // 4 warps/block
    int64_t warps_needed = num_pages;
    int64_t total_threads = warps_needed * WARP;
    int blocks = (int)((total_threads + threads - 1) / threads);
    hipStream_t s;
    HIPCHECK(hipStreamCreate(&s));
    for (int w = 0; w < warmup; ++w)
      hipLaunchKernelGGL(gather_kernel, dim3(blocks), dim3(threads), 0, s,
                         (const uint64_t*)h_dev, (uint64_t*)d, chunks_per_page, num_pages);
    HIPCHECK(hipStreamSynchronize(s));
    HIPCHECK(hipEventRecord(t0, s));
    for (int it = 0; it < iters; ++it)
      hipLaunchKernelGGL(gather_kernel, dim3(blocks), dim3(threads), 0, s,
                         (const uint64_t*)h_dev, (uint64_t*)d, chunks_per_page, num_pages);
    HIPCHECK(hipEventRecord(t1, s));
    HIPCHECK(hipEventSynchronize(t1));
    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
    printf("KERNEL gather       : %.2f GiB/s  (%.3f ms/iter)\n",
           gibps(total * iters, ms / 1000.0), ms / iters);
    HIPCHECK(hipStreamDestroy(s));
  }

  HIPCHECK(hipHostFree(h));
  HIPCHECK(hipFree(d));
  return 0;
}
