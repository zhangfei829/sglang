// Minimal repro to find the correct method for letting a GPU kernel read a
// host (pinned/registered) buffer on ROCm (skyriver07 MI300X).
//
// sglang's hicache "kernel" IO backend launches a GPU kernel that dereferences
// the HOST KV pointer.  With cudaHostRegister(flags=0) AND flags=Mapped it still
// GPU-faults ("Memory access fault by GPU node-N").  This isolates the variables:
//   - register flags: 0 (Default) | 2 (Mapped) | 3 (Portable|Mapped)
//   - access pointer: HOST virtual addr  vs  hipHostGetDevicePointer() result
//   - buffer size: small .. large (find the GPU BAR / mapping limit)
//
// One (size,flags,mode) per process so a fault aborts only that probe.
//
// Build:  hipcc -O2 tools/iobw/test_host_register.hip -o /tmp/thr
// Usage:  /tmp/thr <size_mb> <flags> <mode>      mode: hostptr | devptr
//
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

__global__ void sum_kernel(const unsigned char* p, size_t n, unsigned long long* out) {
  // Touch one byte per 4 KiB page across the whole buffer -> forces the GPU to
  // dereference host addresses spanning the entire allocation.
  unsigned long long s = 0;
  for (size_t i = blockIdx.x * 4096ULL; i < n; i += (size_t)gridDim.x * 4096ULL) {
    s += p[i];
  }
  atomicAdd(out, s);
}

static const char* errs(hipError_t e) { return hipGetErrorString(e); }

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("usage: %s <size_mb> <flags:0|2|3> <mode:hostptr|devptr>\n", argv[0]);
    return 2;
  }
  size_t size_mb = strtoull(argv[1], nullptr, 10);
  unsigned int flags = (unsigned int)strtoul(argv[2], nullptr, 10);
  std::string mode = argv[3];
  size_t n = size_mb * 1024ULL * 1024ULL;

  printf("[probe] size=%zuMB flags=%u mode=%s\n", size_mb, flags, mode.c_str());
  fflush(stdout);

  // 1) Plain host allocation (like torch CPU tensor before pin/register).
  unsigned char* host = (unsigned char*)malloc(n);
  if (!host) { printf("  FAIL malloc\n"); return 1; }
  memset(host, 1, n);  // every byte = 1, so sum should == n

  // 2) hipHostRegister with the requested flags (what sglang does).
  hipError_t e = hipHostRegister(host, n, flags);
  printf("  hipHostRegister(flags=%u) -> %d (%s)\n", flags, (int)e, errs(e));
  fflush(stdout);
  if (e != hipSuccess) { printf("  REGISTER_FAILED\n"); free(host); return 1; }

  // 3) Choose the pointer the kernel will dereference.
  const unsigned char* kptr = host;
  if (mode == "devptr") {
    void* dptr = nullptr;
    hipError_t ed = hipHostGetDevicePointer(&dptr, host, 0);
    printf("  hipHostGetDevicePointer -> %d (%s) ptr=%p\n", (int)ed, errs(ed), dptr);
    fflush(stdout);
    if (ed != hipSuccess || !dptr) { printf("  GETDEVPTR_FAILED\n"); hipHostUnregister(host); free(host); return 1; }
    kptr = (const unsigned char*)dptr;
  }

  // 4) Launch a kernel that reads across the whole buffer.
  unsigned long long* d_out = nullptr;
  hipMalloc(&d_out, sizeof(*d_out));
  hipMemset(d_out, 0, sizeof(*d_out));
  hipLaunchKernelGGL(sum_kernel, dim3(256), dim3(1), 0, 0, kptr, n, d_out);
  hipError_t le = hipGetLastError();
  hipError_t se = hipDeviceSynchronize();
  printf("  launch=%d(%s) sync=%d(%s)\n", (int)le, errs(le), (int)se, errs(se));
  fflush(stdout);

  if (se == hipSuccess) {
    unsigned long long h_out = 0;
    hipMemcpy(&h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost);
    printf("  RESULT sum=%llu expected=%zu %s\n", h_out, n,
           (h_out == (unsigned long long)n ? "OK_GPU_READ_HOST_WORKS" : "MISMATCH"));
  } else {
    printf("  GPU_FAULT_OR_ERROR\n");
  }
  hipFree(d_out);
  hipHostUnregister(host);
  free(host);
  return 0;
}
