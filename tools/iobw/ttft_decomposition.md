# TTFT 分解：为什么 L3 读 / L2→HBM load 优化对 E2E 无效（数据实测）

skyriver07 / MI300X / DeepSeekV3.1 MLA / DP8+EP8 / dummy-forward / case3 DRAM-only(L3 DRAM tier) /
kernel backend(block_quota=4) / `--disable-overlap-schedule`(normal loop) / 16 client × 20 round /
read-replay。全部为 server 端 env-gated 实测 + Prometheus per-stage。

## 1. 实测数据（每段都有数）

单个 prefill batch（`[LOOP-TIMING]`，bs=1，cuda-synced）：

| 段 | ms | 含义 |
|---|---|---|
| getbatch | ~1.5–3.5 | 取下一批 = 调度 + load 准备 |
| runbatch（forward 全程） | ~2.0 | 一次前向(dummy) |
| └ kv_write（`[DUMMY-FWD]`） | ~1.5 | dummy 写 KV 槽 |
| process | ~0.8 | 写穿 + radix insert |
| 单 batch 合计 | ~5 | |

Prometheus per-stage（请求级平均）：

| stage | ms |
|---|---|
| request_process | 0.48 |
| chunked_prefill（每分块段，含块间等待） | 22 |
| prefill_forward（整次 prefill 全程） | 140 |
| load_back_critical_path（L2→HBM 卡关键路径） | ≈ 0 |
| prefetch_critical_path（L3 读卡关键路径） | ≈ 0 |

## 2. 自洽校验（数字互相对上）

- prompt ≈ 18000 token，cache hit ≈ 0.47 → 新 token ≈ 9000 → 按 `chunked_prefill_size=2048` 切 ≈ 5 块。
- `5 块 × 22 ms/块 ≈ 110–140 ms ≈ prefill_forward(140 ms)` ✓ → **prefill_forward = Σ 各分块**。
- 每块 22 ms 里真正 forward(runbatch)只有 ~2 ms → **20 ms 是块间等待**（被其它 15 个并发请求的块插队调度）。

## 3. 时间线图（单个请求的 prefill = TTFT 主体）

```
prefill_forward ≈ 140 ms
├─块1: [等其它请求 ~20ms][forward 2ms]  = 22ms
├─块2: [等 ~20ms][forward 2ms]          = 22ms
├─块3: [等 ~20ms][forward 2ms]          = 22ms
├─块4: [等 ~20ms][forward 2ms]          = 22ms
└─块5: [等 ~20ms][forward 2ms]          = 22ms

单块 forward 2ms 内部：
  [kv_write 1.5ms][load_back ≈0ms][其它 ~0.5ms]
                        ↑ L3读/L2→HBM 优化作用点 = 0ms
```

## 4. 结论（数据闭环）

- TTFT(~140ms) = **5 分块 × (≈20ms 调度等待 + ≈2ms forward)** = chunked-prefill 分片 + 16 路并发交错的**调度吞吐墙**。
- IO（load_back + prefetch）= **0 ms**，且仅存在于每块那 2ms forward 内、本身≈0。
- ⇒ L3 读优化(21.7→34.4 GiB/s)、L2→HBM 优化(kernel 5.38→10.31 / direct 6.0→8.28)动的是 **0ms 段**，对 140ms 的调度墙不可见 → **E2E TTFT 零收益**（real-forward 下更是被 compute 淹没）。
- 降 TTFT 的有效方向：**减少分块数（增大 chunk）/ 提高调度吞吐 / 降并发争用 / 减小长 prompt 的 O(N²) 成本**，与 KV load 带宽无关。

## 4b. 决定性实测：load 优化是否有效取决于 prefetch policy（注入可控延迟扫描）

给 L3 prefetch 读注入可控延迟（`SGLANG_UMBP_GET_DELAY_MS`），扫 0/100/200ms，看 replay-TTFT。
两种 prefetch policy 对比（case3 DRAM-only, dummy, 16c×20r, read-replay；reads 均触发 BatchGet 15-16 次）：

| 每读注入延迟 | best_effort(默认,异步) | wait_complete(同步) |
|---|---|---|
| 0ms | 0.2155s | 0.234s |
| 100ms | 0.2036s | 0.333s (+0.10s) |
| 200ms | 0.1942s | 0.539s (+0.31s) |
| 行为 | 平(延迟被完全吸收) | 随延迟单调上升(~1:1.5 传导) |

**结论（数据闭环）**：
- **best_effort（默认）**：prefetch 异步提前发起，load 被藏在关键路径外 → 加 200ms/读 TTFT 不动 → **load 带宽优化(L3读 6→34、L2→HBM 1.4~1.9x) E2E 零收益**。前面所有"无影响"实测都是 best_effort，根因在此。
- **wait_complete**：调度器等 prefetch 完成才推进 → load 延迟近 1:1.5 传导进 TTFT → **此模式下 L3读/L2→HBM 优化直接降 TTFT**。

即"load 优化无效"不是普适——它是 **best_effort 异步策略掩盖**的结果；切到 wait_complete，load 优化重新有效。

## 4c. 真实 L3 读优化在 wait_complete 下的 E2E 收益（UMBP_DRAM_READ_THREADS 1 vs 4）

wait_complete + read-replay，真实切 DRAM 读线程数（非注入延迟）：

| 配置 | L3 读带宽 | BatchGet 总耗时 | replay-TTFT avg |
|---|---|---|---|
| threads=1 | 17.95 GiB/s | 434 ms | 0.2651s |
| threads=4 | 31.45 GiB/s | 260 ms | 0.2359s |
| 变化 | +75% (1.75x) | −174 ms | **−29ms (−11%)** |

**结论**：wait_complete 下，把 L3 读 18→31 GiB/s → TTFT 真实降 ~11%（−29ms），带宽/读时间/TTFT 三者方向一致。**load 优化在 wait_complete 下兑现 E2E 收益**；best_effort 下为 0（见 4b）。

## 最终总结
- "load 优化无 E2E 影响" 仅在 **best_effort（默认异步 prefetch）** 成立——load 被藏在关键路径外。
- 切 **wait_complete**：load 进关键路径，L3 读 1.75x → TTFT −11%；注入延迟 ~1:1.5 传导进 TTFT。
- ⇒ 要让 L3读/L2→HBM 带宽优化对 TTFT 有效，部署需 wait_complete prefetch policy。

## 5. 复现

server（容器 stor，常驻）：
`SGLANG_HICACHE_BLOCK_QUOTA=4 SGLANG_DUMMY_FWD_TIMING=1 SGLANG_LOOP_TIMING=1 python -m sglang.launch_server ... --hicache-storage-backend umbp --hicache-storage-backend-extra-config '{"dram_capacity_bytes":68719476736,"ssd_enabled":false,...}' --hicache-io-backend kernel --dummy-forward --disable-overlap-schedule --max-total-tokens 40000`

压测：`bench_multiturn.py --num-clients 16 --num-rounds 20 --max-parallel 16 --request-length 2048 --read-replay`

采集：`grep [LOOP-TIMING]/[DUMMY-FWD] /tmp/srv.log` + Prometheus `per_stage_req_latency_seconds` / `load_back_critical_path_seconds` / `prefetch_critical_path_seconds`。
