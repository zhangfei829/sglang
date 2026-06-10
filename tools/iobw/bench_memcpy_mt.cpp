// Multi-threaded copy microbench over a COLD, larger-than-cache working set,
// mirroring dram_tier ReadBatchIntoPtr (atomic work-stealing over N KV blocks).
//
// WHY THIS EXISTS:
//   bench_memcpy_avx.cpp measured single-thread throughput by copying the SAME
//   buffer 2000x -- i.e. HOT in L2/L3. There the cached AVX-512 copy hit ~45
//   GiB/s vs ~26 for memcpy/NT (1.7x), because its stores never reached DRAM.
//   The real path is different: each KV block is read ONCE from cold DRAM, the
//   working set (all cached pages) far exceeds L3, and there are many threads.
//   In that regime the cached copy's stores must writeback to DRAM (and may pay
//   a read-for-ownership on dst), while NT stores bypass cache and skip the RFO.
//   So the hot 1.7x may shrink or even invert. This bench settles it with the
//   number that actually matters: aggregate GiB/s on cold data, vs thread count.
//
// Build (container, g++):
//   g++ -O3 -mavx512f -mavx512bw tools/iobw/bench_memcpy_mt.cpp -o /tmp/bmt -lpthread
//   (drop -march; per-function target attr not needed here, we compile w/ avx512)
// Run:
//   /tmp/bmt                       # defaults below
//   BLOCK_KIB=4096 SET_MIB=4096 THREADS=1,2,4,8,16 PIN=1 /tmp/bmt
//
// Env knobs:
//   BLOCK_KIB  block size in KiB           (default 4096 = 4 MiB, ~real KV page)
//   SET_MIB    total working set in MiB    (default 4096 = 4 GiB, >> any L3)
//   THREADS    comma list of thread counts (default 1,2,4,8,16)
//   ROUNDS     timed rounds, report best   (default 5)
//   PIN        1 = pin thread t to cpu t   (default 0 = let OS schedule, like
//                                            the current remote ReadBatchIntoPtr)
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <sched.h>

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
  for (; i + 64 <= n; i += 64)
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + i), _mm512_loadu_si512(s + i));
  if (i < n) std::memcpy(d + i, s + i, n - i);
  _mm_sfence();
}

static inline void cached_copy_avx512(char* d, const char* s, size_t n) {
  size_t i = 0;
  for (; i + 256 <= n; i += 256) {
    __m512i a = _mm512_loadu_si512(s + i);
    __m512i b = _mm512_loadu_si512(s + i + 64);
    __m512i c = _mm512_loadu_si512(s + i + 128);
    __m512i e = _mm512_loadu_si512(s + i + 192);
    _mm512_storeu_si512(reinterpret_cast<void*>(d + i), a);
    _mm512_storeu_si512(reinterpret_cast<void*>(d + i + 64), b);
    _mm512_storeu_si512(reinterpret_cast<void*>(d + i + 128), c);
    _mm512_storeu_si512(reinterpret_cast<void*>(d + i + 192), e);
  }
  for (; i + 64 <= n; i += 64)
    _mm512_storeu_si512(reinterpret_cast<void*>(d + i), _mm512_loadu_si512(s + i));
  if (i < n) std::memcpy(d + i, s + i, n - i);
}

enum Method { M_MEMCPY = 0, M_CACHED, M_NT, M_N };
static const char* kMethodNames[M_N] = {"memcpy", "avx512_cached", "avx512_nt"};

static inline void copy_one(int m, char* d, const char* s, size_t n) {
  if (m == M_MEMCPY) std::memcpy(d, s, n);
  else if (m == M_CACHED) cached_copy_avx512(d, s, n);
  else nt_copy_avx512(d, s, n);
}

static size_t env_sz(const char* k, size_t def) {
  const char* v = std::getenv(k);
  return v ? static_cast<size_t>(std::strtoull(v, nullptr, 10)) : def;
}

static std::vector<int> parse_threads() {
  std::vector<int> out;
  const char* v = std::getenv("THREADS");
  if (!v) { return {1, 2, 4, 8, 16}; }
  std::string s(v), cur;
  for (char c : s) {
    if (c == ',') { if (!cur.empty()) out.push_back(std::atoi(cur.c_str())); cur.clear(); }
    else cur.push_back(c);
  }
  if (!cur.empty()) out.push_back(std::atoi(cur.c_str()));
  if (out.empty()) out.push_back(1);
  return out;
}

// One timed pass: T threads work-steal blocks via an atomic counter (exactly
// like ReadBatchIntoPtr). Returns aggregate GiB/s for the whole set.
static double run_pass(int method, int threads, char* src, char* dst,
                       size_t nblocks, size_t bs, bool pin) {
  std::atomic<size_t> next{0};
  auto worker = [&](int tid) {
    if (pin) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(tid, &set);
      sched_setaffinity(0, sizeof(set), &set);
    }
    size_t i;
    while ((i = next.fetch_add(1, std::memory_order_relaxed)) < nblocks)
      copy_one(method, dst + i * bs, src + i * bs, bs);
  };
  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<std::thread> pool;
  pool.reserve(threads);
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double bytes = static_cast<double>(nblocks) * bs;
  return bytes / sec / (1024.0 * 1024.0 * 1024.0);
}

int main() {
  const size_t bs = env_sz("BLOCK_KIB", 4096) * 1024;
  const size_t set_bytes = env_sz("SET_MIB", 4096) * (1024ull * 1024ull);
  const int rounds = static_cast<int>(env_sz("ROUNDS", 5));
  const bool pin = env_sz("PIN", 0) != 0;
  std::vector<int> tlist = parse_threads();
  size_t nblocks = set_bytes / bs;
  if (nblocks == 0) nblocks = 1;
  size_t total = nblocks * bs;

  char* src = static_cast<char*>(aligned_alloc(64, total));
  char* dst = static_cast<char*>(aligned_alloc(64, total));
  if (!src || !dst) { fprintf(stderr, "alloc failed (%.1f GiB x2)\n",
                              total / 1073741824.0); return 1; }
  std::memset(src, 0xAB, total);
  std::memset(dst, 0, total);

  printf("# cold-data multi-thread copy, block=%zuKiB set=%zuMiB nblocks=%zu "
         "rounds=%d pin=%d\n", bs >> 10, total >> 20, nblocks, rounds, pin ? 1 : 0);
  printf("# aggregate GiB/s (best of %d rounds), each block copied once per round\n",
         rounds);
  printf("%-8s", "threads");
  for (int m = 0; m < M_N; ++m) printf("%16s", kMethodNames[m]);
  printf("%14s\n", "cached/memcpy");

  for (int T : tlist) {
    double best[M_N] = {0, 0, 0};
    for (int m = 0; m < M_N; ++m) {
      // warm the buffers' page tables but keep data cold relative to L3 by
      // touching the whole (huge) set; one untimed pass also faults pages in.
      run_pass(m, T, src, dst, nblocks, bs, pin);
      for (int r = 0; r < rounds; ++r) {
        double g = run_pass(m, T, src, dst, nblocks, bs, pin);
        if (g > best[m]) best[m] = g;
      }
    }
    printf("%-8d", T);
    for (int m = 0; m < M_N; ++m) printf("%16.1f", best[m]);
    printf("%14.2f\n", best[M_MEMCPY] > 0 ? best[M_CACHED] / best[M_MEMCPY] : 0.0);
  }
  free(src);
  free(dst);
  return 0;
}
