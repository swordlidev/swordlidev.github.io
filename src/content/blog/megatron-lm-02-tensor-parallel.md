---
title: "Megatron-LM 分布式训练（二）：张量并行 TP 的内部原理"
description: "张量并行把单层的权重矩阵切到多张卡。本文从矩阵乘法的行/列切分讲起，解释 Column/Row Parallel Linear、f 与 g 镜像算子，以及为什么 TP 必须绑定 NVLink 同机。"
pubDate: 2026-10-19
tags: ["Megatron-LM", "分布式训练", "张量并行", "LLM"]
---

> **本文是《Megatron-LM 分布式训练》系列第 2 篇**。上一篇讲了[并行维度与训练生命周期](/blog/megatron-lm-01-overview/)，本篇聚焦**张量并行（Tensor Parallelism, TP）**。

## 背景：一层 Transformer 里在算什么

Transformer 的主要算力都花在**矩阵乘法**上。无论是 attention 里的 QKV 投影、输出投影，还是 FFN/MLP 里的两个线性层，本质都是 $Y = XA$ 这样的稠密矩阵乘：

- $X$ 是输入激活，形状约 $[b\cdot s,\ h]$（batch×序列，隐藏维）；
- $A$ 是权重矩阵，形状约 $[h,\ h']$。

当 $h$ 很大（几千到上万）时，单个权重矩阵就可能几 GB，单卡存不下，单卡算也慢。**张量并行**的思路就是：把这一个大矩阵乘法，按行或按列切开，分给 TP 组内的多张卡并行计算，再用一次通信把结果缝合。

关键问题是：切了之后结果还对不对？怎么切能让**通信次数最少**？这正是本篇要讲清楚的。

## 两种切法：列并行与行并行

对 Linear 层 $Y = XA$：

- **列并行 `ColumnParallelLinear`**：把 $A$ 按列切 $[A_1, A_2]$，每卡拿完整 $X$ 各算一块，输出 $Y=[Y_1,Y_2]$ 按列切开。
- **行并行 `RowParallelLinear`**：把 $A$ 按行切 $[A_1;A_2]$、输入 $X$ 按列切 $[X_1,X_2]$，每卡算部分和 $X_1A_1$、$X_2A_2$，**需 all-reduce 相加**。

直觉上，列并行算完得到的是"结果的不同列"，天然不需要求和；行并行算完每张卡只得到"完整结果的一部分贡献"，必须相加才完整。

## 列+行串联，把通信降到最少

MLP 是两个线性层夹一个激活：$Y = \text{GeLU}(XA)B$。Megatron 这样接：

```
X ──f──► [ColumnParallel A] ──► GeLU(局部) ──► [RowParallel B] ──g──► Y
         (A按列切,输出按列切)   (逐元素,无需通信)  (B按行切,部分和)   (all-reduce求和)
```

- 第一层列切：输出按列分，GeLU 逐元素独立做，**零通信**；
- 第二层行切：输入正好要"按列切的 X"，无缝衔接；
- 结果：**整个 MLP 块前向只在 `g` 处 1 次 all-reduce、反向在 `f` 处 1 次 all-reduce**。Attention 块（QKV 列切、输出投影行切）同理。

这个"列切接行切"的设计是 TP 的精髓：中间的激活一直保持切分状态流动，只在块的出口汇总一次。

## `f` 与 `g`：一对镜像算子（TP 灵魂）

在 `mappings.py` 中：

- **`f` = `_CopyToModelParallelRegion`**（列并行入口）：前向恒等，反向 all-reduce。因前向把 $X$ 复制到每卡，反向每卡各产生一份 $X$ 的梯度，必须加起来。
- **`g` = `_ReduceFromModelParallelRegion`**（行并行出口）：前向 all-reduce，反向恒等。因 $Y$ 是各卡部分和相加，加法的梯度原样回传。

一句话记忆：**`f`（前向恒等 / 反向 all-reduce），`g`（前向 all-reduce / 反向恒等）**。一个 Transformer 层 = 前向 2 次 + 反向 2 次 all-reduce。`_reduce` 本体就是 `torch.distributed.all_reduce(input_, group=TP组)`。

## 为什么 TP 必须 NVLink 同机

$$\text{每次迭代 TP 通信次数} = L \times 4 \times \text{num\_microbatches}$$

几十层模型 + 多 microbatch = 每次迭代上千次 all-reduce，**全在计算关键路径上**（无法隐藏）。

| 互联 | 带宽量级 | 适合 |
|---|---|---|
| NVLink（机内） | 数百 GB/s~TB/s | ✅ TP 高频关键路径 |
| RoCE/IB（机间） | 几十~100+ GB/s | ❌ 扛不住 TP；✅ 只够 DP/PP |

所以 TP 组一定要落在同一台机器、走 NVLink。这也呼应了第 1 篇的"工程铁律：TP 不跨机"。

## 序列并行 SP（顺带）

`--sequence-parallel`（必须配 TP）把 TP 里"复制的区域"（LayerNorm、Dropout）沿序列维切开，把 `g` 的 all-reduce 拆成 `reduce-scatter + all-gather` —— **总通信量不变，但省显存**（激活不再在 TP 组内重复存储）。第 7 篇调优清单里，SP 是"几乎免费"的第一优先项。

## 小结

- TP 把单层的大矩阵乘法切到多卡：**列并行**输出按列切、无需求和；**行并行**各卡算部分和、需 all-reduce。
- "列切接行切"让中间激活保持切分流动，整个 MLP/Attention 块前向只需 1 次 all-reduce。
- `f` / `g` 是一对镜像算子（前向恒等↔反向 all-reduce），一层 = 前向 2 + 反向 2 次 all-reduce。
- TP 通信极高频且在关键路径上，**必须 NVLink 同机**；SP 可在不增加通信量的前提下进一步省激活显存。

下一篇讲**流水线并行 PP**：当模型层数多到一台机器也放不下时，怎么把层切到多机，又用 1F1B 调度把"气泡"压到最小。
