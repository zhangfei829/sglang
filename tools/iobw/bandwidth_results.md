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

### 附带修复
- **kernel backend ROCm 跑通**：`SGLANG_HICACHE_HOST_REGISTER_FLAGS=2`（cudaHostRegisterMapped）让 host 池映射进 device 空间，`host_to_device_accessible_ptr`(memory_pool_host.py:835) 的 hipHostGetDevicePointer 才返回有效 devptr。默认 0（仅 page-lock）→ kernel GPU fault。CUDA 靠 UVA 无需此 flag——ROCm 特有配置，非代码 bug。
- **umbp_store.py**：`cfg.ssd_backend → cfg.ssd.ssd_backend`（refactor mori config 迁移漏改字段）。
