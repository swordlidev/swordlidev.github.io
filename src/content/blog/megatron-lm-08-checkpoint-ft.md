---
title: "Megatron-LM 分布式训练（八）：Checkpoint、容错与弹性训练"
description: "几百卡训几周，硬件一定会出故障。本文讲分布式 Checkpoint 为什么能换并行配置重载（ShardedTensor 的 global_offset）、异步存档、容错（ft_launcher + 心跳 + 分段超时抓“沉默 hang”）与弹性训练的区别，以及落到集群的实战 checklist。"
pubDate: 2025-09-29
tags: ["Megatron-LM", "分布式训练", "Checkpoint", "容错", "弹性训练"]
---

> **本文是《Megatron-LM 分布式训练》系列第 8 篇**。前七篇解决了"怎么把模型高效地切到几百卡上算"（[总览](/blog/megatron-lm-01-overview/) / [TP](/blog/megatron-lm-02-tensor-parallel/) / [PP](/blog/megatron-lm-03-pipeline-parallel/) / [DP](/blog/megatron-lm-04-data-parallel-zero/) / [通信](/blog/megatron-lm-05-communication/) / [MoE·CP](/blog/megatron-lm-06-moe-ep-cp/) / [性能调优](/blog/megatron-lm-07-perf-tuning/)）。本篇解决最后一个工程现实：**几百卡训几周，硬件一定会出故障——怎么不让几周心血白费。**

## 背景：为什么大规模训练必须谈容错

单卡训练挂了，重跑就行。但 64 卡（甚至几千卡）同步训练完全不同：

- **故障率随规模放大**：GPU 掉卡、ECC 错误、网络抖动、节点 OOM/宕机、RoCE 链路闪断……卡越多，"某处出事"的概率越接近必然。几千卡集群常常几小时就有一次故障。
- **同步训练是"一损俱损"**：所有 rank 在每一步都要 all-reduce 同步，**任何一张卡挂了/卡住了，整个 job 立刻 hang 或崩**。
- **没有 checkpoint = 从头再来**：训了 3 周突然节点宕机，若没存档，3 周算力（可能几十万人民币）直接蒸发。

所以大规模训练的工程铁律：**定期存档（checkpoint）+ 自动检测故障 + 从最近存档恢复**，三者缺一不可。

## 分布式 Checkpoint 的核心难题

模型被 TP/PP/DP 切到几百卡上，**每张卡只持有一小片参数**。怎么存？

- **naive 做法**：把所有分片汇总到 rank0 存成一个大文件 → rank0 内存爆炸、写盘串行慢、几百 GB 单文件。几百卡时根本不可行。
- **Megatron 做法**（`dist_checkpointing/`）：**每张卡只存自己那一片**，但每片都附带元数据说明"我这片在完整张量的哪个位置"。并行写盘，各存各的。

## `ShardedTensor`：分布式 ckpt 的灵魂

> "Represents a mapping between a local tensor and a global tensor. Global tensor is assumed to consist of many local tensors distributed between different processes." —— `mapping.py:48`

每个被切开的参数用一个 `ShardedTensor` 描述，关键字段：

| 字段 | 含义 |
|---|---|
| `key` | 全局张量的唯一名字（如某层的权重） |
| `data` | **本卡持有的那一片**真实数据 |
| `local_shape` | 本地分片的形状 |
| `global_shape` | 完整张量的形状 |
| `global_offset` | **本地分片在全局张量里的偏移**（关键！） |
| `replica_id` | 副本编号：同一片在多卡上重复时（如 DP 副本）用它去重，只存一份 |

**精髓**：存档存的不是"卡 17 的文件"，而是**"全局张量 + 每片的全局坐标（`global_offset`）"**。这就带来了杀手锏——

## 杀手锏：换并行配置也能重载（parallelism-agnostic）

因为 checkpoint 记录的是**全局逻辑视图**（完整张量长什么样、每片在哪），而不是**物理视图**（哪张卡存了什么），所以：

> **存档时用 `TP=8 PP=4`，恢复时可以改成 `TP=4 PP=8`（甚至改 DP）——只要全局形状对得上，框架会自动把分片重新"裁剪/拼接"到新的并行布局上。**

这对集群实战意义巨大：

- **扩缩容**：从 64 卡扩到 128 卡继续训，不用重训；
- **调并行策略**：发现 `TP=8` 不如 `TP=4 PP=8` 快，直接改配置从 ckpt 续训；
- **换硬件**：从 8 卡节点迁到 4 卡节点。

加载时 `load()`（`serialization.py:3`）需要你传入**当前并行配置下的 sharded state dict 作为"向导"**，框架按这个向导去 ckpt 里取对应的全局区域填进来。

> 这正是上一篇系列里 Qwen3-235B SFT 实战中"转换用 TP=4/PP=2、训练自动重切到 TP=8/PP=2/EP=4"能成立的底层原因。

## 保存后端：`torch_dist`

默认策略 `TorchDistSaveShardedStrategy`（`backend="torch_dist"`，`serialization.py:63`），基于 PyTorch 原生 Distributed Checkpoint：

- **并行写**：各 rank/各节点同时写自己的分片到共享存储（Lustre 这类并行文件系统正好派上用场，全节点路径一致）；
- **可移植**：生成标准化的分片文件 + 全局元数据，支持上一节的重切分。

## 异步 Checkpoint

**问题**：同步存档时训练必须停住等写盘完成。大模型一次 ckpt 几十~几百 GB，写共享存储可能几分钟——频繁存档 → 大量时间浪费在"干等写盘"。

**异步方案**（`--async-save`）：拆成两段

1. **短暂阻塞**：把 ckpt 从 GPU 显存快速拷到 CPU 内存（秒级）；
2. **后台写盘**：`schedule_async_save`（`async_utils.py:190`）把写盘任务丢给后台进程，**训练立刻继续**；之后用 `maybe_finalize_async_save`（`:200`）在合适时机收尾。

训练结束时会 `maybe_finalize_async_save(blocking=True, terminate=True)`（`training.py:2273`）确保最后一个异步存档落盘。效果：**存档几乎不打断训练**，可以存得更勤、丢的进度更少。

## 容错 Fault Tolerance

Megatron 集成 **NVIDIA nvidia-resiliency-ext** 的 FT 包（`ft_integration.py`），用 **`ft_launcher`** 替代 `torchrun` 启动：

```bash
ft_launcher --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes=${NUM_NODES} --nproc-per-node=${NUM_GPUS_PER_NODE} \
    --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \
    --ft-param-rank_out_of_section_timeout=300 \
    train_script_with_ft.py
```

机制：

- **`RankMonitorClient`**（`ft_integration.py:83`）：监控每个 rank 的"心跳"；
- **分段超时（section）**：把训练切成 `setup` / `step` / `checkpointing` 几个阶段，各设超时阈值。某 rank 在 `step` 段超过 180s 没动静 → 判定 **hang**（比如死锁、某卡掉了但没报错）；
- **自动恢复**：检测到 hang/crash → 触发 **in-job restart**（不重新排队，直接在本 job 内重启），从最近的 ckpt 恢复继续训；
- **`--calc-ft-timeouts`**：根据实际观测到的耗时自动校准超时阈值，避免误杀。

> 为什么需要专门的 hang 检测？因为最可怕的不是"崩溃报错"（这个 torchrun 能发现），而是**某张卡悄悄卡住、整个集群跟着一起 hang 但不报错**——干等到天荒地老。section 超时就是来抓这种"沉默故障"的。

## 弹性训练 Elastic（与容错的区别）

两个常被混淆的概念：

| | 容错 Fault Tolerance | 弹性 Elastic |
|---|---|---|
| 解决 | 出故障后**恢复**（从 ckpt 重启继续） | 运行中**动态增减节点**仍能继续 |
| 依赖 | ckpt + hang 检测 + 自动重启 | rendezvous（会合）机制 + 可重切分 ckpt |
| 场景 | 节点挂了→备用节点顶上，从存档恢复 | 临时拿到更多卡就扩容、被抢占就缩容 |

两者都建立在**可重切分的分布式 ckpt**（见前文）之上——节点数变了，并行布局跟着变，ckpt 能自动适配，这才谈得上弹性。rendezvous 就是第 5 篇里见过的 `--rdzv_backend=c10d --rdzv_endpoint=...`：节点启动时先到会合点"报到"，凑齐就开训，有节点进出就重新会合。

## 落到集群（实战 checklist）

1. **ckpt 存共享存储**（如 Lustre），用默认 `torch_dist` 后端，天然并行写 + 可重切分；
2. **开 `--async-save`**：存档不打断训练，可以存得更勤；
3. **设合理 `--save-interval`**：权衡——太频繁浪费 IO，太稀疏故障时丢的进度多；经验是"存一次的开销 ≪ 两次存档间可能损失的时间"；
4. **用 `ft_launcher` + FT**：自动抓 hang、自动从最近 ckpt 重启；
5. **保留多个 ckpt**：防止最新的那个存到一半时崩了（存档本身也可能失败）；
6. 续训/调参时善用重切分能力：改并行配置直接从老 ckpt 加载，不用重训。

## 小结

- 大规模同步训练"一损俱损"，必须 **定期存档 + 故障检测 + 自动恢复**；
- 分布式 ckpt 让**每张卡只存自己那片**，用 `ShardedTensor` 的 `global_offset` 记录全局坐标（`mapping.py:48`）；
- 杀手锏：记录全局逻辑视图 → **换并行配置（TP/PP/DP）也能重载**，扩缩容/调策略不用重训；
- `--async-save` 把写盘丢后台，存档几乎不打断训练；
- FT 包（`ft_launcher` + `RankMonitorClient` + section 超时）抓"沉默 hang"并自动 in-job 重启；
- 容错=挂了能恢复，弹性=节点数可动态变，二者都依赖可重切分的 ckpt + rendezvous。

---

## 主线回顾

至此，从"把 GPU 排成网格"到"几百卡容错长训"的主线走完了：

```
第1篇 并行网格(TP/PP/DP/CP/EP)  ──┐
第1篇 一次迭代生命周期            │  怎么把一个大模型
第2篇 TP(切层内,NVLink)          │  切到几百张卡上，
第3篇 PP(切层间,1F1B,可跨机)     │  并让每一步高效运转
第4篇 DP+分布式优化器(ZeRO)      ──┘
第5篇 通信与网络(NCCL/RoCE/busbw) ── 跨机通信扛不扛得住
第6篇 EP(MoE)+CP(长序列)         ── 另两个并行维度
第7篇 重计算/FP8-BF16/MFU        ── 省显存+提速,用MFU记分
第8篇 Checkpoint/容错/弹性       ── 几周长训不白费
```

一条主线串起来：**切得下（并行）→ 通得畅（通信）→ 装得下（显存）→ 跑得快（精度/MFU）→ 死不了（容错）**。这套心智模型如何落到一次真实任务，可以参考[《Qwen3-235B-A22B 64 卡全参 SFT 实战复盘》](/blog/qwen3-235b-sft-roleplay/)。
