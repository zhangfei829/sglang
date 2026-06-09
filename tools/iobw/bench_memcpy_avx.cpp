// Microbench: compare host->host copy methods at realistic KV block sizes.
// Methods: glibc memcpy, AVX2 non-temporal store (current dram_tier NtCopy),
//          AVX-512 non-temporal store, AVX-512 streaming-load + NT-store.
// Single-thread, to isolate the per-instruction copy throughput (then multiply
// by thread count for the parallel ceiling). Run on the node where the DRAM
// tier lives (taskset to one NUMA node for clean numbers).
//
// Build (container, g++):
//   g++ -O3 -march=znver4 -mavx512f -mavx512bw tools/iobw/bench_memcpy_avx.cpp -o /tmp/bmc -lpthread
//   (if -march=znver4 unsupported: -mavx2 -mavx512f -mavx512bw)
// Run:
//   taskset -c 0 /tmp/bmc
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>

static inline void nt_copy_avx2(char* d, const char* s, size_t n) {
  size_t head = (32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31;
  if (head > n) head = n;
  std::memcpy(d, s, head);
  size_t i = head;
  for (; i + 128 <= n; i += 128) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 32));
    __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 64));
    __m256i e = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 96));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), a);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 32), b);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 64), c);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 96), e);
  }
  for (; i + 32 <= n; i += 32) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), a);
  }
  if (i < n) std::memcpy(d + i, s + i, n - i);
  _mm_sfence();
}

static inline void nt_copy_avx512(char* d, const char* s, size_t n) {
  size_t head = (64 - (reinterpret_cast<uintptr_t>(d) & 63)) & 63;
  if (head > n) head = n;
  std::memcpy(d, s, head);
  size_t i = head;
  for (; i + 256 <= n; i += 256) {
    __m512i a = _mm512_loadu_si512(s + i);
    __m512i b = _mm512_loadu_si512(s + i + 64);
    __m512i c = _mm512_loadu_si512(s + i + 128);
    __m512i e = _mm512_loadu_si512(s + i + 192);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i), a);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i + 64), b);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i + 128), c);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i + 192), e);
  }
  for (; i + 64 <= n; i += 64) {
    __m512i a = _mm512_loadu_si512(s + i);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i), a);
  }
  if (i < n) std::memcpy(d + i, s + i, n - i);
  _mm_sfence();
}

// streaming load (movntdqa) + NT store: source read once, bypass cache both ways
static inline void nt_copy_avx512_sl(char* d, const char* s, size_t n) {
  size_t head = (64 - (reinterpret_cast<uintptr_t>(d) & 63)) & 63;
  // streaming load requires 64B-aligned src; fall back to memcpy for head and
  // any unaligned src.
  if ((reinterpret_cast<uintptr_t>(s) & 63) != 0 || head > n) {
    nt_copy_avx512(d, s, n);
    return;
  }
  std::memcpy(d, s, head);
  size_t i = head;
  for (; i + 256 <= n; i += 256) {
    __m512i a = _mm512_stream_load_si512((void*)(s + i));
    __m512i b = _mm512_stream_load_si512((void*)(s + i + 64));
    __m512i c = _mm512_stream_load_si512((void*)(s + i + 128));
    __m512i e = _mm512_stream_load_si512((void*)(s + i + 192));
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i), a);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i + 64), b);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i + 128), c);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i + 192), e);
  }
  for (; i + 64 <= n; i += 64) {
    __m512i a = _mm512_stream_load_si512((void*)(s + i));
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i), a);
  }
  if (i < n) std::memcpy(d + i, s + i, n - i);
  _mm_sfence();
}

static double gibps(size_t bytes, double sec) {
  return (double)bytes / sec / (1024.0 * 1024.0 * 1024.0);
}

int main() {
  const size_t sizes[] = {64ul * 1024, 256ul * 1024, 1ul << 20, 4ul << 20,
                          16ul << 20, 137ul << 20};
  const int names_n = 4;
  const char* names[names_n] = {"glibc_memcpy", "avx2_nt", "avx512_nt",
                                "avx512_sl_nt"};
  printf("%-14s", "block");
  for (int m = 0; m < names_n; ++m) printf("%14s", names[m]);
  printf("   (GiB/s, single-thread)\n");

  for (size_t sz : sizes) {
    // 64B-aligned buffers
    char* src = static_cast<char*>(aligned_alloc(64, sz));
    char* dst = static_cast<char*>(aligned_alloc(64, sz));
    memset(src, 0xAB, sz);
    memset(dst, 0, sz);
    int iters = sz < (1u << 20) ? 2000 : (sz < (16u << 20) ? 200 : 30);
    double res[names_n];
    for (int m = 0; m < names_n; ++m) {
      // warmup
      for (int w = 0; w < 3; ++w) {
        if (m == 0) std::memcpy(dst, src, sz);
        else if (m == 1) nt_copy_avx2(dst, src, sz);
        else if (m == 2) nt_copy_avx512(dst, src, sz);
        else nt_copy_avx512_sl(dst, src, sz);
      }
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < iters; ++it) {
        if (m == 0) std::memcpy(dst, src, sz);
        else if (m == 1) nt_copy_avx2(dst, src, sz);
        else if (m == 2) nt_copy_avx512(dst, src, sz);
        else nt_copy_avx512_sl(dst, src, sz);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      double sec = std::chrono::duration<double>(t1 - t0).count() / iters;
      res[m] = gibps(sz, sec);
    }
    char lbl[32];
    if (sz >= (1u << 20)) snprintf(lbl, sizeof(lbl), "%zuMiB", sz >> 20);
    else snprintf(lbl, sizeof(lbl), "%zuKiB", sz >> 10);
    printf("%-14s", lbl);
    for (int m = 0; m < names_n; ++m) printf("%14.1f", res[m]);
    printf("\n");
    free(src);
    free(dst);
  }
  return 0;
}
