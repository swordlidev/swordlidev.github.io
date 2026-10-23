---
title: "Megatron-LM 分布式训练（六）：MoE 专家并行 EP 与上下文并行 CP"
description: "把 3D 并行扩展到五维。本文讲 MoE 为什么能解耦参数量与计算量、专家并行 EP 的两次 all-to-all、router 的负载均衡，以及上下文并行 CP 如何用 Ring Attention + zigzag 切法处理超长序列。"
pubDate: 2025-09-23
tags: ["Megatron-LM", "分布式训练", "MoE", "专家并行", "上下文并行"]
---

> **本文是《Megatron-LM 分布式训练》系列第 6 篇**。前五篇把稠密模型的 **TP/PP/DP** 讲透了（[总览](/blog/megatron-lm-01-overview/)起），本篇补齐另两个维度：**EP**（给 MoE 稀疏模型）和 **CP**（给超长序列），合起来就是 Megatron 的**五维并行**。

## 背景：稀疏模型与超长序列带来的新维度

前五篇的 TP/PP/DP 适用于**稠密模型**（每个 token 都过全部参数）。但近两年两个趋势催生了新的并行维度：

- **MoE 稀疏模型**（Mixtral、DeepSeek-V3、Qwen3 等）：参数堆到几百 B 甚至上 T，但每个 token 只激活一小部分。需要 **专家并行 EP** 把专家分散到多卡。
- **超长上下文**（128K、1M token）：attention 的激活随序列长度暴涨，单卡放不下。需要 **上下文并行 CP** 沿序列维切分。

## MoE 是什么，为什么需要专家并行

**稠密模型**：每个 token 都过同一个 FFN，参数量 = 计算量，想变大就得算得更多。

**MoE（Mixture-of-Experts）**：把一个 FFN 换成 **N 个专家 FFN + 一个路由器（router）**。每个 token 只被路由到 **top-k 个专家**（典型 k=2），其余专家不参与计算。

$$\text{总参数量} \propto N \quad\text{但}\quad \text{单 token 计算量} \propto k \ll N$$

好处：**参数量（模型容量）和计算量解耦** —— 可以把参数堆到几百 B 甚至上 T，而每个 token 的 FLOPs 只相当于一个小稠密模型。代价：N 个专家的权重**放不进一张卡** → 需要**专家并行 EP**，把不同专家分到不同卡上。

## MoE 层的数据流

```
hidden_states
   │
   ├─► router          ── 算每个 token 去哪些专家 + 权重
   │
   ├─► token_dispatcher.dispatch   ── all-to-all：把 token 发到专家所在的卡
   │
   ├─► experts         ── 各卡用本地专家算 FFN (GroupedGEMM)
   │
   ├─► token_dispatcher.combine    ── all-to-all：把结果收回原来的卡
   │
   └─(+ shared_experts)  ── 可选：所有 token 都过的共享专家，可与路由专家 overlap
```

Megatron 按 `--moe-token-dispatcher-type` 选分发器：

| dispatcher | 通信方式 | 适用 |
|---|---|---|
| `allgather` | all-gather 全部 token | EP 小、实现简单 |
| `alltoall` | **all-to-all** | 标准方案 |
| `flex`（+ DeepEP） | 融合 all-to-all | 大规模、最优性能 |

## Router：token 怎么选专家

两步：

1. **gating**：`logits = hidden_states @ W_router`，得到每个 token 对 N 个专家的打分。
2. **routing**：对 logits 做 top-k，选出每个 token 的 k 个专家及其权重（softmax/sigmoid）。

**核心难题：负载均衡（load balancing）。** 若放任不管，router 会"偷懒"把大多数 token 都发给少数几个专家 → 其余专家饿死（**专家塌缩**），模型容量浪费，且热门专家所在卡成为瓶颈。Megatron 提供多种均衡手段：

| 手段 | 思路 |
|---|---|
| **aux_loss**（主流） | 加一个辅助损失，惩罚"token 分布"和"专家被选概率"的不均，鼓励均摊；系数 `--moe-aux-loss-coeff` |
| seq/global aux_loss | 在序列级 / 全局范围算均衡损失 |
| **sinkhorn** | 用 Sinkhorn 迭代把路由矩阵归一成双随机矩阵，天然均衡（不配 aux_loss） |

> aux_loss 是"软约束"：它不保证绝对均衡，而是在主 loss 之外加一个拉力。训练日志里会看到 `load_balancing_loss` 指标，盯着它能判断专家是否均衡。

## 专家并行 EP：把专家分到不同卡

设 N=8 个专家、EP=4，则每张卡放 **N/EP = 2 个专家**。问题来了：一个 token 被路由到的专家，**很可能不在它当前所在的卡上**。怎么办？——**两次 all-to-all**：

```
dispatch (all-to-all):  按"每个 token 要去的专家"把 token 重排、发送到专家所在的 EP rank
         ↓
experts:                每张卡只用本地 2 个专家，对收到的 token 批量算 FFN
         ↓
combine  (all-to-all):  把算完的结果按原路发回 token 原来所在的卡
```

**为什么用 all-to-all 而不是 all-reduce？** 因为每个 token 去哪个专家是**不规则、点对点**的——卡 0 可能要给卡 3 发 100 个 token、给卡 1 发 20 个。all-to-all 正是"每张卡给每张卡发不同量数据"的原语（见第 1 篇），天生契合这种"洗牌"。

## EP 的通信特征与放置

| | TP | EP |
|---|---|---|
| 原语 | all-reduce | **all-to-all** |
| 触发 | 每层 2 次（固定量） | 每 MoE 层 2 次（**量随路由波动**） |
| 痛点 | 带宽 | **负载不均 + 延迟**（token 分布不均→某些链路拥塞） |

- **放置**：EP 常和 DP 共享物理维度（从数据并行维度里"借"出来做专家并行），默认排列 `order="tp-cp-ep-dp-pp"` 里 ep 紧跟 cp。Megatron 用 `--expert-model-parallel-size`(EP) 和 `--expert-tensor-parallel-size`(专家部分的 TP) **独立配置专家部分的并行**——因为专家部分和 attention 部分的最优并行度往往不同（这叫 **Parallel Folding**）。
- **overlap**：`--overlap-moe-expert-parallel-comm` 可把 EP 的 all-to-all 藏进计算；shared expert 可与路由专家并发执行。
- **省显存**：MoE 也支持分布式优化器（ZeRO-1 equivalent），对专家权重同样切优化器状态。

## 上下文并行 CP：切序列长度

**问题**：序列长度 s 翻倍，attention 的激活（尤其 attention 矩阵）和 KV cache 显存会**平方 / 线性增长**。128K、1M 这种超长 context，单卡放不下激活。TP 切不了序列维，PP 也不行。

**CP 的解法**：沿**序列维**把输入切成 CP 份，每个 CP rank 只持有序列的一段：

```python
# 每个 cp_rank 只拿自己那一段 token/labels/loss_mask/position_ids
local_seq_len = val.shape[seq_dim] // cp_size
batch[key] = val.narrow(seq_dim, cp_rank * local_seq_len, local_seq_len)
```

**难点：attention 需要"全局"的 K、V。** token i 要和它前面所有 token 做 attention，但那些 token 分散在别的 CP rank 上。解法是 **Ring Attention**：KV 分块在 CP 组内**环形传递**（像第 4 篇的 ring all-reduce），每张卡轮流拿到别人的 KV 块算一轮 partial attention，转一圈后每个 query 就和全部 KV 都算过了。这部分在 Transformer Engine 里实现，Megatron 负责切 batch + 建 CP 组。

## CP 的精髓：zigzag 负载均衡

**因果（causal）mask 导致负载天然不均**：序列靠前的 token 只需和少数前文算 attention（计算量小），靠后的 token 要和大量前文算（计算量大）。若简单按"前 1/N、第 2 个 1/N……"切，拿到序列尾段的 CP rank 会累死，拿到头段的闲死。

**zigzag 切法**：每个 CP rank 同时拿"一个靠前的段 + 一个靠后的段"，让负载均摊：

```python
segment_len = seq_len // (2 * cp_size)
front = val.narrow(seq_dim, cp_rank * segment_len, segment_len)                 # 靠前的段
back  = val.narrow(seq_dim, (2*cp_size - cp_rank - 1) * segment_len, segment_len)  # 对称的靠后段
batch[key] = torch.cat((front, back), dim=seq_dim)
```

所以要求 **序列长度能被 `2 × CP` 整除**（代码里有断言）。以 CP=2 为例，4 段 `[0,1,2,3]`：rank0 拿 `{段0, 段3}`，rank1 拿 `{段1, 段2}` —— 每卡都是"一头一尾"，计算量相当。

## 五维并行全景

扩展第 1 篇的公式：

$$\text{world\_size} = TP \times CP \times EP \times DP \times PP$$

（对 MoE，专家部分可用独立的 expert-TP/EP/expert-DP，稠密部分用 TP/CP/DP；两者在同一组卡上"折叠"复用。）

各维度放置总原则（把前六篇缝起来）：

| 维度 | 通信原语 | 频率 | 放哪 |
|---|---|---|---|
| **TP** | all-reduce | 极高 | 机内 NVLink |
| **CP** | P2P（ring KV） | 中（每 attention） | 机内优先，可跨机 |
| **EP** | all-to-all | 中（每 MoE 层，波动） | 机内优先，跨机需高带宽 |
| **DP** | reduce-scatter/all-gather | 低 | 跨机 RoCE |
| **PP** | P2P send/recv | 低 | 跨机 RoCE |

## 小结

- MoE 用 router + top-k 把"参数量"和"计算量"解耦，稀疏激活 → 能堆超大模型；
- **EP** 把专家分到不同卡，靠**两次 all-to-all**（dispatch/combine）把 token 洗到专家所在卡再洗回来；
- router 必须做**负载均衡**（aux_loss / sinkhorn），否则专家塌缩；
- **CP** 沿序列维切，解决超长 context 的激活显存，attention 靠 **Ring Attention** 传 KV；
- CP 用 **zigzag** 切法对冲 causal mask 的负载不均（要求 seq 能被 2×CP 整除）；
- 五维并行 `world_size = TP×CP×EP×DP×PP`，放置原则：高频 all-reduce（TP）绑 NVLink，低频 P2P（PP/DP）走跨机。

最后一篇讲**显存与性能调优**：同样的卡，怎么塞下更大的模型、跑得更快，以及如何用 MFU 量化效率。
