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
