# UMBP / hicache IO 带宽实测结果（skyriver07）

机器：skyriver07，8×MI300X，DP8+TP8+EP8，mori MoE，DeepSeek-V3.1（dummy-forward）。
盘：2× Samsung PM9A3 7.68TB（`0000:88:00.0` + `0000:c3:00.0`），**PCIe Gen4 x4**（sysfs `current_link_speed=16 GT/s width=4`）。
组网：SPDK `bdev_raid_create` RAID0（strip 128KB），逗号分隔 `UMBP_SPDK_NVME_PCI="0000:88:00.0,0000:c3:00.0"`。

## 物理上限（硬数据，sanity-check 基准）
- 单盘 PCIe Gen4 x4：16×4×(128/130)/8 ≈ **7.88 GB/s**。
- RAID0 双盘聚合：≈ **15.76 GB/s**。
- **任何读数 > 对应上限 = cache，不是介质带宽，作废。**

## 带宽矩阵（全部过 sanity check）

| 指标 | 单盘 | RAID0 | 来源 / 方法 |
|---|---|---|---|
| RAW 顺序写 | 4.17 GB/s | **8.30 GB/s** (1.99×) | bdevperf 直打 bdev，1MiB/qd256 |
| RAW 顺序读 | 6.75 GB/s | **13.56 GB/s** (2.01×) | bdevperf |
| Tier batch 写 | 3.49 GB/s | **5.59 GB/s** (1.60×) | bench_umbp_micro 64GB 冷读，io_workers=4 |
| Tier batch 读 | 5.27 GB/s | **9.36 GB/s** (1.78×) | bench_umbp_micro 64GB 冷读 |
| 全栈 SSD load（读，p50） | — | **~8.9 GB/s** | 大工作集(63GB)+DRAM=256MB 逐出+read-replay，p50；与 tier 9.36 互证 |
| 全栈 DRAM 写（热路径） | — | 3.4 GiB/s | umbp_store telemetry（local 模式 Phase1，未触 SSD） |

## 关键认知
- **local 模式（非分布式）前台 `BatchPutFromPtr` 只写 DRAM tier**（`umbp_client.cpp:272` Phase1）；Phase2 写 SSD 仅 `role_==SharedSSDLeader`（分布式）才走。所以前台 telemetry 的 ~3.4 GiB/s 是 **DRAM 写**，不是 SSD。
- **SSD 写**在 local 模式靠**逐出**（DRAM 满→demote 单键到 SSD，`local_storage_manager.cpp:630`）。
- 测真 SSD 带宽必须：① 工作集 >> 所有 cache（SSD 控制器 DRAM、**proxy ring 32GB**）；② 冷读。小工作集读出的 14-18 GiB/s 全是 ring/控制器 cache。
- 全栈测 SSD load 配方：`UMBP_DRAM_BYTES=256MB`（逼逐出下盘）+ 工作集 32GB(>ring)~96GB(<SSD容量) + `READ_REPLAY=true`。
- RAID0 写 1.6×、读 ~2× 真扩展（裸盘精确 2×；tier 层因 pipeline 开销略低）。
- `--hicache-io-backend direct` 必加（kernel backend 在 skyriver07 GPU permission fault）。

## DRAM→HBM load 带宽（host→device 链路，bench_dram_to_hbm.py）

| transfer | DRAM→HBM |
|---|---|
| 72 KiB（1 page/layer） | 0.44 GiB/s（延迟受限） |
| 4.5 MiB（64 pages） | 26.75 GiB/s |
| 18 MiB（256 pages） | 39.43 GiB/s |
| 128 MiB+（饱和） | **53.5 GiB/s ≈ 57.5 GB/s**（PCIe Gen5 x16 量级） |

## 架构结论：SSD 能否取代 DRAM 喂 HBM？

| 路径 | 带宽 |
|---|---|
| DRAM→HBM 饱和 | 57.5 GB/s |
| DRAM→HBM 实际 per-layer（多页 4.5-18MiB） | 27-39 GB/s |
| SSD load（RAID0 2 盘） | 13.56 GB/s (raw) / 9.36 (tier) |

**DRAM→HBM 比 SSD load 快 2-4×** → **当前 2 盘 RAID0 的 SSD 不能透明取代 DRAM**：用 SSD 当源会把 load 卡在 ~13.6 GB/s（DRAM 能 27-57）。
要让 SSD 追平：需 **4 盘 RAID0（~27 GB/s，追平实际 per-layer DRAM 速率）甚至 6-8 盘（追平饱和 57）**；或 workload per-layer 传输都很小（DRAM 也降到 0.4-27，差距缩小）。

---

## L3 UMBP DRAM tier 读优化（mori perf/dram-parallel-read-v2）

`DRAMTier::ReadBatchIntoPtr` 原单线程 memcpy + 独占锁。改：shared_mutex 读读并发 + 多线程并行 memcpy + NT-store(跳 RFO) + 跨 CCD 物理核绑定。

### 组件级带宽（微基准 bench_dram_batch_read.py，256MiB/批，taskset node0）

| 线程 | memcpy(NT off) | **NT(默认)** |
|---|---|---|
| 1 | 7.85（基线） | 14.8（裸单核） |
| 4 | 28.5 | **76.7** |
| 8 | 44.8 | 133.8 |
| 16 | 91.6 | 138.7 |
| 32 | 113.7 | 140(峰值) |

- 单 socket 拷贝上限 ~115（memcpy）/~140（NT），拐点 32-48 线程；>48 反降（内存控制器饱和 + spawn 开销）。
- **绑核必须跨 CCD**：连续核堆单 GMI 链路反而慢 2x；跨 CCD 物理核散布才线性 scale。
- **NT-store**：dst 是 staging、写完 DMA 走、CPU 不再读 → 跳 RFO，单核 14.8→23.7（1.67x），省核 + 抬天花板。
- 默认 `UMBP_DRAM_READ_THREADS=4` + `UMBP_DRAM_NT_COPY` 开 + 绑核开。

### E2E（全栈 DeepSeek-V3.1 MLA, DP8, case3 kernel backend）

L3 读组件 E2E（BatchGet）：baseline 1核noNT **14.1** → 优化 4核NT **44.8**（3.2x，单批 38ms→12ms）。

**但 E2E TTFT/吞吐：优化无效，穷尽验证：**

| 场景 | Get | A(优化) vs B(baseline) |
|---|---|---|
| 8c/64c read-replay | 18 | prefetch_critical_path=0，相同 |
| 128c rate200 饱和 | — | 都 87 req/s，TTFT 相同 |
| 小池频繁读(mem-frac0.65) | ~680 | Median 0.15 vs 0.13、吞吐 15.3 vs 15.5，相同 |

**根因**：① L3→host prefetch 被 hicache **异步提前发起**（请求入队即 prefetch、后台跑完），从不在关键路径（`prefetch_critical_path_sum=0`，连慢 baseline 都不阻塞）；prefill 类负载真正的墙是 compute/调度吞吐，不是 L3 读。② host→HBM 散列 gather（历史 direct ~6.0）是另一独立下游瓶颈。

**定位**：组件级真 3.2x（省核/余量/CPU 效率）；prefill 服务 E2E 中性。唯一可能兑现的是 **decode-KV-offload**（每步 compute 极小、load 海量、结构性在关键路径），尚未测。

### 本轮干净 A/B（2026-06-09, skyriver07, DRAM-only, READ_REPLAY, page_first posix）

条件：`case3` DRAM-only（`UMBP_SSD_BYTES=0` posix）、`DUMMY_FORWARD=true`、`NUM_ROUNDS=20 NUM_CLIENTS=16 MAX_PARALLEL=16`、`HICACHE_SIZE=32`(L2)、`UMBP_DRAM_BYTES=64GB`(L3)、`READ_REPLAY=true`、device KV pool=27.04GB/413056 tokens/rank。
A/B 唯一变量 = `UMBP_DRAM_READ_THREADS`（1=优化前 vs 4=优化后）。E2E TTFT 取 replay 阶段（L3 载入在 prefill 关键路径）的 `[read-replay][TTFT]`。

| 指标 | 优化前 threads=1 | 优化后 threads=4 | 变化 |
|---|---|---|---|
| **L3 DRAM 读带宽 avg** | 21.74 GiB/s | **34.40 GiB/s** | **+58% (1.58x)** |
| 读带宽 p50 | 21.65 | 36.01 | +66% |
| 读带宽 p95 / max | 24.39 / 24.72 | 46.08 / 46.20 | +89% / +87% |
| L3 读总耗时（聚合） | 355.4 ms | 196.6 ms | −45% |
| BatchGet 调用 / 数据量 | 15 / 8.29 GB | 14 / 7.26 GB | — |
| L3 DRAM 写带宽 (BatchPut) | 15.99 GiB/s | 15.91 GiB/s | ~0%（读优化不影响写，预期） |
| **E2E replay TTFT avg** | 0.2037 s | 0.2018 s | **−0.9%（噪声内）** |
| E2E replay TTFT p50 | 0.2208 s | 0.2035 s | −7.8% |
| E2E replay TTFT p99 | 0.2360 s | 0.2252 s | −4.6% |

**结论（数据支撑，不外推）**：读优化组件级 1.58x 有效；E2E TTFT 基本不动（~1%，噪声内）。量化原因：每请求 L3 读 ≈ 8.3GB/16/8rank ≈ 65MB/rank，在 21.7→34.4 GiB/s 下耗时 ~3ms→~1.8ms，占 204ms TTFT 仅 ~1% → 读带宽翻倍在 E2E 不可见。**L3 DRAM 读不是 TTFT 关键路径。**

### L2 DRAM->HBM 后端优化前后对比（kernel 自比 + direct 自比，2026-06-09 定稿）

skyriver07 / MI300X / DeepSeekV3.1 MLA。各后端自比，page_size 固定（不改 KV 粒度避免影响命中率）。

| 后端 | 优化旋钮 | 优化前 | 优化后 | 提升 |
|---|---|---|---|---|
| kernel | `SGLANG_HICACHE_BLOCK_QUOTA` 2->4 | 5.38 GiB/s | 10.31 GiB/s | 1.92x |
| direct | `SGLANG_HICACHE_DIRECT_STREAMS`（多流） | 6.0 GiB/s | 8.28 GiB/s | 1.38x |

- kernel 数据来自 `bench_blockquota.py` 隔离微基准（一次 kernel 搬 2432 token）。完整 sweep：
  block_quota 2=5.38 / 4=10.31 / 16=34.75 / 64=44.88 / 152~4096≈47（饱和）GiB/s。
  按工作点取 block_quota=4=10.31（per-layer 真实负载每层页数少，4 是实际工作点；4096 是过度配置的饱和上限）。
- direct 数据为全栈 per-layer AGG：1 流 6.0 -> 多流 8.28（隔离 8 流可达 21.74，但全栈受 per-layer 碎片限制）。
- **E2E 注脚（实测）**：真实 forward 下 `load_back_critical_path_seconds_sum` 整轮仅 0.171s（543 次，占 TTFT 12-21s/轮 的 <0.1%）。load 已被逐层计算遮住、不在关键路径 -> 上述组件级提升（1.92x / 1.38x）对 E2E TTFT 几乎零收益。

### Grafana/Prometheus 实测坐实「load 不在关键路径」（2026-06-09）

常驻 server（真实 forward，无 dummy）+ 32 client × 30 round 多轮复用，Prometheus 抓 `:30000/metrics`。
查询 `sum(sglang:load_back_critical_path_seconds_sum) / sum(sglang:time_to_first_token_seconds_sum)`：

- 结果 = **1.82e-6 ≈ 0.00018%**（load 阻塞累计 0.0242s vs TTFT 总量 ~13000s）。
- 即 compute 阻塞在 host->device load 上的时间占 TTFT 的百万分之二，TTFT 的 99.9998% 是真实 compute。
- **坐实**：L3 读优化（1.58x）、L2->HBM load 优化（kernel 1.92x / direct 1.38x）组件级有效，但 **E2E TTFT 收益 ≈ 0**——load 不在关键路径（per-layer load 与逐层 compute 流水重叠 + prefetch 异步提前发起，load 被 compute 完全遮住）。
- 与历史 dummy-forward 下 `load_back_critical_path=0.171s/543次（<0.1%）` 完全一致，两种测法互证。
- **降 TTFT 的正确方向 = prefill compute / 调度吞吐，不是 load 带宽。**

### 附带修复
- **kernel backend ROCm 跑通**：`SGLANG_HICACHE_HOST_REGISTER_FLAGS=2`（cudaHostRegisterMapped）让 host 池映射进 device 空间，`host_to_device_accessible_ptr`(memory_pool_host.py:835) 的 hipHostGetDevicePointer 才返回有效 devptr。默认 0（仅 page-lock）→ kernel GPU fault。CUDA 靠 UVA 无需此 flag——ROCm 特有配置，非代码 bug。
- **umbp_store.py**：`cfg.ssd_backend → cfg.ssd.ssd_backend` + 所有 `cfg.spdk_* → cfg.ssd.spdk_*`（refactor mori config 把 spdk 字段全迁到 ssd 子配置）。
