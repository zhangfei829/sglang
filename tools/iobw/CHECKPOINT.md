# UMBP IO 带宽优化 — 进展存档（2026-06-10 更新）

一句话状态：**冷数据多线程实测定论 = AVX2 NT 最优（~1.47x over memcpy @8线程）。dram_tier 已改回 AVX2 NT 并推送（mori `3b98bd05`），等容器重编 + 全栈带宽验证（预期 31→~45+）。**

## ⚠️ 最终定论（2026-06-10 上午，决定性实测）

`tools/iobw/bench_memcpy_mt.cpp`（冷数据、4GiB 工作集、多线程、不绑核，复刻 ReadBatchIntoPtr）实测：

```
threads   memcpy  avx512_cached  avx512_nt   avx2_nt   rte_memcpy   nt/memcpy
1          14.9     16.7          23.4        24.3       16.9         1.57
2          28.3     31.5          45.3        47.1       32.4         1.60
4          52.2     50.4          80.5        87.8       55.7         1.54
8          88.4     76.5         130.2       133.5       82.6         1.47   <- 默认 8 线程甜点
16        126.0     99.1         148.3       148.4      100.8         1.18   <- 都逼近 DRAM 天花板~148
```

排序（8 线程，冷数据真实路径）：**avx2_nt 133 > avx512_nt 130 > memcpy 88 > rte_memcpy 83 ≈ cached 77**。

- **AVX2 NT 最优**，比 AVX-512 NT 还略快：Zen4 的 512-bit 是 256-bit 通路 double-pump，无宽度优势 + 可能触发 AVX-512 降频；NT 瓶颈是 WC-buffer drain，256-bit 已喂满。→ **原始 `e504bf50` 的 AVX2 NT 方向本来就对**。
- **cached / DPDK rte_memcpy 是同一类（缓存存）**，冷数据下比 memcpy 还慢（RFO+writeback = 3× 内存流量 vs NT 2×）。昨天单核 cached 45（1.7x）是**热 cache 假象**。DPDK 不仅不快还加依赖，**否决**。
- **PIN=1 全面更差**（16 线程 NT 146→50）：绑 cpu 0..T-1 挤在单 NUMA node 锁死带宽。**不要绑核**（远程版本来不绑，正确）。
- 16 线程 nt/memcpy 掉到 1.18：memcpy 也逼近 DRAM 带宽天花板 ~148。**8 线程是甜点（dram_tier 默认就是 8）**。

→ dram_tier `CopyBlock`：≥256KiB 用 **`NtCopyAvx2`（AVX2 streaming store + sfence）**，小块 memcpy；`UMBP_DRAM_NT_COPY=0` 可关。提交 `3b98bd05`（覆盖了中途错误的 `ecd20678` cached 和 `b1716a9a` avx512_nt）。

---

## 1. 最新改动（已 commit + push）

仓库：`zhangfei829/mori`，分支 `perf/dram-parallel-read`
- 提交：`3b98bd05`（**AVX2 NT，当前 tip，最终方案**）；中途的 `ecd20678`(cached) / `b1716a9a`(avx512_nt) 已被覆盖，**别用**。
- 文件：`mori/src/umbp/local/tiers/dram_tier.cpp`
- 现状：`CopyBlock` → ≥256KiB 用 `NtCopyAvx2`（256-bit streaming store + sfence），小块 memcpy；`UMBP_DRAM_NT_COPY=0` 关闭。

> 注意：dram_tier.cpp 属于 **mori 独立仓库**，不在 sglang 的 git 里。sglang 侧的工具分支是 `test/umbp-io-bandwidth-stats-tools`（已推 `iobw` remote）。

### 改了什么
远程版 `ReadBatchIntoPtr` 原本是纯 `std::memcpy` × 8 线程（默认 `UMBP_DRAM_READ_THREADS=8`），没有 NT/AVX 那套（那是我本地旧分支的实现，已被远程 reset 覆盖）。新增：

- `CachedCopyAvx512`：AVX-512 `loadu`/`storeu`，**非 NT（非 streaming）**。
  - 函数级 `__attribute__((target("avx512f")))` + 运行期 `__builtin_cpu_supports("avx512f")` 派发。
  - **不需要改 CMake 全局 `-march`**（CMakeLists 只有 `-O3`，没有 `-march`），默认编译即可生成 AVX-512 码。
- `CopyBlock(dst, src, size)` 派发器，按块大小选路径：
  - **≤ 16 MiB → 缓存拷贝（~45 GiB/s/core）**
  - **> 16 MiB → 回退 `std::memcpy`（glibc NT 路径，~24，避免 RFO）**
- 单线程 + 多线程两条 copy 路径都改用 `CopyBlock`。
- 开关：`UMBP_DRAM_CACHED_COPY=0` 关闭缓存拷贝、强制回退 memcpy（做 A/B 对照用）。

### 阈值依据（用户单核微基准实测，GiB/s）
```
block      glibc_memcpy  avx2_nt  avx512_nt  avx512_sl_nt  avx512_cached
64KiB         53.4        25.3     24.4       25.3          53.4
256KiB        53.7        26.4     26.3       26.3          53.6
1MiB          26.7        26.6     26.6       26.4          45.2
4MiB          26.7        26.6     26.7       26.7          45.3   <- 真实 KV page
16MiB         26.5        26.4     26.5       26.6          28.3   <- 缓存≈memcpy 交叉点
137MiB        15.8        24.4     22.9       22.0          17.4   <- 此时才该用 NT
```
- 真实 KV page（几 MB）→ 缓存拷贝 **1.7x**（45.3 vs 26.7）。
- `avx512_cached`（手写 loadu+storeu）= DPDK `rte_memcpy`，**不依赖 DPDK**。
- glibc 在几百 KB 后切到 NT，所以 1MiB+ 掉到 ~26；缓存拷贝避开这个切换。

---

## 2. 明天第一步：容器端重编 + 验证（在 GPU 机/容器里跑）

```bash
cd /sgl-workspace/mori          # 容器里 mori 路径，按实际改
git fetch origin perf/dram-parallel-read
git reset --hard origin/perf/dram-parallel-read   # -> ecd20678
BUILD_UMBP=ON pip install . --no-build-isolation -v   # 重编 libmori_pybinds.so
```

### 单核 A/B（最干净，先确认改动真生效）
```bash
# 新缓存拷贝：期望 4MB 块 ~45 GiB/s/core
UMBP_DRAM_READ_THREADS=1 <batch_get 带宽微基准>
# 对照（强制回退 memcpy）：期望 ~26
UMBP_DRAM_READ_THREADS=1 UMBP_DRAM_CACHED_COPY=0 <同一基准>
```

### 全栈多线程
- 8 线程跑全栈 `ReadBatchIntoPtr` 带宽，看是否从 **~31** 往上走。
- ⚠️ 注意：单核 1.7x 不一定线性放大到聚合——8 线程可能撞 **内存带宽 / NUMA 上限**。这正是下面待办「为什么 4 核只到 31 而非 104」要查的点。

---

## 3. 待办 / 下一步（按优先级）

1. **[首要]** 容器重编 + 全栈带宽 A/B：确认 31 → ?（cached on vs off，1/4/8/16 线程矩阵）。
2. **wait_complete 复测**：验证 L3 读 1.7x 在 `wait_complete` prefetch 策略下对 TTFT 的真实 E2E 收益（best_effort 异步预取会把 load 延迟藏掉，TTFT 看不到收益）。
3. **查多核扩展瓶颈**：4/8 核为什么只到 ~31 而非 ~104 —— 量 worker 的 NUMA 落点、每 worker 实际 memcpy 块大小占比。
4. 可选：真实计算（real-forward）+ wait_complete 复测，看之前 ~11% TTFT 收益在真实算力下还剩多少（很可能缩水）。
5. 可选：finer-grained Prefill 调度时序计时（radix 前缀匹配、KV 分配到叶子）。

---

## 4. 已确立的关键结论（避免明天重复踩）

- **主瓶颈不是 IO**：TTFT 大头是 DP-attention 同步（all_gather 的 straggler 等待，~11.5ms/prefill chunk）+ prefill 调度开销（~12ms），IO load ≈ 0ms。
- **prefetch 策略决定 load 优化是否可见**：
  - 默认 `best_effort` 异步预取 → load 延迟被隐藏，`prefetch_critical_path ≈ 0`，带宽优化对 TTFT **无感**。
  - `wait_complete` → prefetch 上关键路径，load 优化才在 TTFT 显形（实测注入 200ms 延迟能显著抬高 replay TTFT）。
- **kernel backend ≯ direct memcpy**：ROCm host→device 上 kernel gather 并不比 direct hipMemcpyAsync 快（曾被证伪）。
- **NT vs 缓存拷贝**：典型 KV 块（几 MB）缓存拷贝完胜 NT（1.7x）；只有 >L3（>16MiB，实测 137MiB）NT 才赢。Zen4 EPYC 上 DPDK rte_memcpy 用的就是缓存存，与 glibc 持平，真实负载无额外优势。

---

## 5. 相关文件位置

- mori 拷贝实现：`mori/src/umbp/local/tiers/dram_tier.cpp`（`CachedCopyAvx512` / `CopyBlock` / `ReadBatchIntoPtr`）
- UMBP store + JSONL 旁路遥测：`python/sglang/srt/mem_cache/storage/umbp/umbp_store.py`
- 调度器计时埋点：`python/sglang/srt/managers/scheduler.py`、`scheduler_dp_attn_mixin.py`
- 工具脚本：`tools/iobw/`（`run_umbp_iobw_test.sh` 等）
- TTFT 分解大文档：`tools/iobw/ttft_decomposition.md`
- 微基准（memcpy/AVX2/AVX512/DPDK 对比）：`tools/iobw/` 下的 C++ 微基准
