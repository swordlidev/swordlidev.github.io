---
title: "Megatron-LM 分布式训练（三）：流水线并行 PP 与 1F1B 调度"
description: "流水线并行把模型的层切到不同卡。本文讲清 microbatch、流水线气泡（bubble），对比朴素 GPipe 与 1F1B 调度，推导 bubble 公式，并给出 64 卡 3D 并行蓝图。"
pubDate: 2025-09-12
tags: ["Megatron-LM", "分布式训练", "流水线并行", "LLM"]
---

> **本文是《Megatron-LM 分布式训练》系列第 3 篇**。前两篇讲了[总览](/blog/megatron-lm-01-overview/)与[张量并行 TP](/blog/megatron-lm-02-tensor-parallel/)，本篇聚焦**流水线并行（Pipeline Parallelism, PP）**。

## 背景：TP 之外，还需要切层

第 2 篇的 TP 解决了"单层太大"的问题，但它受限于单机卡数（TP 不跨机）。当模型有几十上百层、一台机器（8 卡）也放不下全部层时，就需要**沿层的方向**把模型切成几段，分到不同机器上 —— 这就是流水线并行。

类比一条工厂流水线：工位 1 负责第 1~8 层，工位 2 负责第 9~16 层……数据像半成品一样依次流过各工位。相邻工位之间只需传递一个激活张量（P2P），通信量小、频率低，所以 **PP 可以跨机**，和 TP 正好互补。

但流水线有个天然问题：**气泡（bubble）**。

## 问题与解法

PP 把层切成段（stage），数据像流水线穿过各工位。只喂 1 个 batch 会产生**气泡**——当工位 1 在干活时，工位 2/3/4 还没拿到数据只能干等；收尾时反过来。解法：把 global batch 切成多个 **microbatch** 连续喂入，让各工位尽量同时忙起来。

## 朴素 vs 1F1B

- **朴素（GPipe）**：先把所有 microbatch 全部前向，再全部反向 → 所有 microbatch 的激活都得同时存着 → 显存峰值 = m 份（m 为 microbatch 数）。
- **1F1B（Megatron 默认）**：进入稳定期后，每做 **1 个前向（1 Forward）紧跟 1 个反向（1 Backward）**，反向一做完就立刻释放该 microbatch 的激活 → 显存峰值 ≈ **p 份**（p=PP stage 数），与 m 无关。

1F1B 的意义：在不牺牲吞吐的前提下，把激活显存从 $O(m)$ 降到 $O(p)$，这样你就可以放心增大 m 去压气泡（见下）。

## 三阶段

热身数公式：

```
num_warmup_microbatches = total_stages - current_stage - 1
```

PP=4 时各 stage 热身数 = `4 - rank - 1`：stage0=3、stage1=2、stage2=1、stage3=0。

时间轴（PP=4，F=前向，B=反向，数字是 microbatch 编号）：

```
GPU0: F1 F2 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 B7 B8
GPU1:    F1 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 F8 B7 B8
GPU2:       F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8
GPU3:          B1 F2 B2 F3 B3 F4 B4 ...
      └─warmup─┘└──── steady 1F1B ────┘└─cooldown─┘
```

- **warmup**：只前向填满管道，`input/output` 存入列表留给反向；
- **steady**：`send_forward_recv_backward` **融合通信**（同时发前向输出、收反向梯度），`pop(0)` 取最早的 microbatch 做反向并释放其激活；
- **cooldown**：只反向，排空管道。

## Bubble 公式

$$\text{bubble fraction} = \frac{p-1}{m + (p-1)} \approx \frac{p-1}{m}$$

**结论：microbatch 数 m 越多，气泡越小**（经验 m ≥ 4p）。这是开 PP 后第一个要调的旋钮。直觉：warmup+cooldown 的"空转"长度由 p 决定，而稳定期长度由 m 决定，m 相对 p 越大，空转占比越小。

## 为什么 PP 可以跨机

| | TP | PP |
|---|---|---|
| 原语 | all-reduce（集合） | send/recv（P2P） |
| 搬运 | 每层激活，全组参与 | 仅 stage 边界一个激活张量 |
| 频率 | 每层 2 次（极高） | 每 microbatch 1 次（低） |
| 关键路径 | 是，难隐藏 | 可融合通信 overlap |

PP 低频、量小、可 overlap → **能跨机走 RoCE**，与 TP 互补。

## 3D 并行蓝图（64 卡）

把前三篇缝起来，一个典型的 64 卡（8 节点 × 8 卡）布置：

```
TP=8  → 节点内 8 卡 NVLink 做张量并行
PP=4  → 跨 4 组节点做流水线，走 RoCE 点对点
DP=2  → 剩 2 份完整副本做数据并行，走 RoCE all-reduce
校验:  TP×PP×DP = 8×4×2 = 64 ✓
```

布置原则：**TP 放最内层绑 NVLink；PP/DP 放跨机。** 这正是默认排列 `order='tp-cp-ep-dp-pp'` 的意义——让变化最快的 TP 落在相邻 rank（同机）上。

## 小结

- PP 沿**层**把模型切成 stage，用 microbatch 流水填充，相邻 stage 只传一个激活（P2P），因此可跨机。
- **1F1B** 把激活显存峰值从 $O(m)$ 降到 $O(p)$，让你能放心增大 m。
- 气泡占比 $\approx (p-1)/m$，**增大 microbatch 数 m 是压气泡的首要手段**（经验 m ≥ 4p）。
- 3D 并行蓝图：TP 绑 NVLink 放机内，PP/DP 走 RoCE 放跨机。

下一篇讲**数据并行 DP 与分布式优化器**，看 ZeRO 思想如何把那"重复存 N 份"的优化器状态切开省显存。
