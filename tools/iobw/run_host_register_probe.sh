#!/usr/bin/env bash
# Build + sweep tools/iobw/test_host_register.hip to find the method that lets a
# GPU kernel read a host (registered) buffer on ROCm.  Each (size,flags,mode)
# runs in its own process so a GPU memory fault aborts only that probe.
#
# Run:  bash tools/iobw/run_host_register_probe.sh
set -u

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_host_register.hip"
BIN=/tmp/thr

echo "=== build ==="
if ! hipcc -O2 "$SRC" -o "$BIN" 2>&1; then
  echo "BUILD FAILED (need hipcc / ROCm). Trying CC=hipcc from rocm path..."
  /opt/rocm/bin/hipcc -O2 "$SRC" -o "$BIN" || { echo "still failed"; exit 1; }
fi
echo "built $BIN"
echo

# flags: 0=Default, 2=Mapped, 3=Portable|Mapped
# mode:  hostptr (sglang's current way) | devptr (hipHostGetDevicePointer)
for size in 64 1024 8192 65536; do
  for flags in 0 2 3; do
    for mode in hostptr devptr; do
      echo "----- size=${size}MB flags=${flags} mode=${mode} -----"
      "$BIN" "$size" "$flags" "$mode" 2>&1
      rc=$?
      echo "  exit_rc=$rc"
      echo
    done
  done
done
echo "=== done ==="
