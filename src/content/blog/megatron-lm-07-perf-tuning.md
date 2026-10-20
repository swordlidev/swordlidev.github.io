---
title: "Megatron-LM 分布式训练（七）：显存与性能调优（重计算 / FP8-BF16 / MFU）"
description: "系列收官：显存都花在哪、重计算如何用算力换激活显存、BF16 与 FP8 混合精度怎么选、MFU 为什么是训练效率的金标准，以及一份按性价比排序的实战调优清单。"
pubDate: 2026-10-14
tags: ["Megatron-LM", "分布式训练", "性能调优", "混合精度", "MFU"]
---

> **本文是《Megatron-LM 分布式训练》系列第 7 篇（完结）**。前六篇讲"怎么把模型切到多卡"（[总览](/blog/megatron-lm-01-overview/) / [TP](/blog/megatron-lm-02-tensor-parallel/) / [PP](/blog/megatron-lm-03-pipeline-parallel/) / [DP](/blog/megatron-lm-04-data-parallel-zero/) / [通信](/blog/megatron-lm-05-communication/) / [MoE·CP](/blog/megatron-lm-06-moe-ep-cp/)），本篇换个视角：**同样的卡，怎么塞下更大的模型、跑得更快**。

## 背景：调优的三把武器

把模型切到多卡之后，真正决定训练体验的是两个指标：**能不能跑起来（显存）** 和 **跑得快不快（吞吐）**。本篇的三把武器分别对应：

- **重计算（recomputation）**：用算力换显存，解决激活放不下。
- **混合精度（BF16/FP8）**：既省显存又提速。
- **MFU**：量化效率、定位瓶颈，是所有调优的"记分牌"。

## 先搞清楚显存花在哪（四大块）

| 块 | 大小量级 | 谁来省 |
|---|---|---|
| **参数 P / 梯度 G / 优化器状态 Os** | 16 字节/参数（第 4 篇） | ZeRO / 分布式优化器、TP/PP 切模型 |
| **激活 Activation** | ∝ `batch × seq × hidden × layers` | **重计算（本篇）**、SP、CP |
| 临时 buffer（通信/GEMM workspace） | 中 | 分桶、固定 buffer |
| 显存碎片 | 不定 | `PYTORCH_CUDA_ALLOC_CONF` 等 |

**关键认知**：参数相关的三块被 TP/PP/ZeRO 解决后，**激活常常成为新的显存瓶颈**——尤其长序列、大 micro-batch 时。激活显存粗略正比于：

$$M_{\text{act}} \propto b \times s \times h \times L$$

（b=micro-batch, s=seq, h=hidden, L=层数）。这就是重计算要对付的对象。

## 重计算（activation recomputation / gradient checkpointing）

**思想**：前向时**不保存**中间激活，反向需要时**重新前向算一遍**。用**算力换显存**——典型代价是多一次前向（约 +33% 计算量），但激活显存大降。

Megatron 的配置：

| 参数 | 取值 | 含义 |
|---|---|---|
| `recompute_granularity` | `full` / `selective` | 整层 checkpoint vs 只选关键子模块 |
| `recompute_method` | `uniform` / `block` | 均匀分块 vs 每个 PP stage 只重算前 N 层 |
| `recompute_num_layers` | int | 配合上面，重算多少层 |
| `recompute_modules` | `["core_attn", ...]` | selective 时选哪些子模块（默认 `core_attn`） |
| `distribute_saved_activations` | bool | 把保存的激活沿 model-parallel 组再切开（配合 SP） |

**两档的本质区别（重点）**：

- **`full`**：checkpoint 整个 transformer layer，省显存最多，但要多算一整次前向，算力开销最大。显存实在不够时才用。
- **`selective`**（推荐）：只重算**"显存占用大、但计算量小"**的部分——默认是 `core_attn`（attention 分数矩阵是 $O(s^2)$ 的显存大户，但重算它的 FLOPs 相对便宜）。这是性价比最高的方案，出自论文 *Reducing Activation Recomputation in Large Transformer Models*（arXiv:2205.05198）。20B+ 模型几乎必开。

> 直觉：selective 专挑"存起来亏、重算起来便宜"的激活丢掉，把宝贵显存留给别的。`full` 是"全都不存"的暴力省法。

## 混合精度：BF16 / FP16 / FP8

**为什么不用 FP32 训练**：慢、费显存。现代大模型用混合精度——**低精度算、高精度存 master 权重**（回忆第 4 篇那 16 字节里的 `fp32 master`）。

**BF16 vs FP16**（都 2 字节，区别在指数/尾数位分配）：

| | 指数位 | 尾数位 | 动态范围 | 精度 |
|---|---|---|---|---|
| FP16 | 5 | 10 | 窄（易溢出） | 高 |
| **BF16** | 8 | 7 | **宽（和 FP32 一样）** | 低 |

大模型训练**首选 BF16**：它的动态范围和 FP32 相同，**几乎不会溢出**，所以**不需要 loss scaling**（FP16 因范围窄，必须动态 loss scaling 防梯度下溢，就是第 1 篇 `optimizer.step()` 里那个 inf/nan 检查 + 跳步机制）。

**FP8**（Hopper/Blackwell 硬件支持，Transformer Engine 实现）：

- 两种格式：**E4M3**（精度高，前向/权重用）、**E5M2**（范围大，反向梯度用）；
- **recipe**（缩放策略）：`delayed scaling`（用历史统计定缩放因子）、`current scaling`（当前张量实时定）、`mxfp8`（block-wise 微缩放，更鲁棒）；
- **只有矩阵乘（Linear/GEMM）走 FP8**，而 **LayerNorm、softmax、残差累加、优化器更新仍保持 BF16/FP32**——因为这些对精度敏感，FP8 会训崩；
- 收益：GEMM 吞吐翻倍、激活/通信字节减半；代价：需要 TE、调 recipe，数值稳定性要盯 loss 曲线。

## MFU：训练效率的金标准

**MFU（Model FLOPs Utilization）** = 实际跑出的"有效模型算力" ÷ 硬件理论峰值算力：

$$\text{MFU} = \frac{\text{实测 TFLOP/s/GPU}}{\text{硬件峰值 TFLOP/s/GPU}}$$

**分子：模型 FLOPs** 的经典公式（N=参数量，D=处理的 token 数）：

$$\text{FLOPs} \approx \underbrace{6ND}_{\text{前向2 + 反向4}} + \underbrace{O(\text{attention 的 } s^2 \text{ 项})}_{\text{长序列不可忽略}}$$

Megatron 真实实现（`num_floating_point_operations`）就是这个结构：`mlp_layer_flops`（含 `expansion`、swiglu）、`attn_layer_flops`（含 `seqlen` 平方项）、`moe_layer_flops`（按**激活的** top-k 专家折算，而非全部专家）等分项求和。

日志里的吞吐：

```python
throughput = num_floating_point_operations(...) / (elapsed_time_per_iteration * 1e12 * world_size)
# → 日志打印 "throughput per GPU (TFLOP/s/GPU)"
```

> **一个极其重要的细节**：`num_floating_point_operations` 算的是**模型本身的有效 FLOPs**，**不包含重计算多出来的那次前向**。所以：
> - 开了重计算 → 实际算力花得更多，但分子不变、每步耗时变长 → **MFU 会下降**。这是对的——MFU 本就该反映"重计算是为省显存付出的效率代价"。
> - 这也是 MFU 和 **HFU（Hardware FLOPs Utilization）** 的区别：HFU 含重计算，MFU 不含。**看 MFU 评估"模型训练效率"，看 HFU 评估"硬件忙不忙"。**

**硬件峰值参考**（稠密，不含稀疏加速）：

| GPU | BF16 | FP8 |
|---|---|---|
| H100 (SXM) | ~989 TFLOP/s | ~1979 TFLOP/s |
| H800 | ~989 TFLOP/s（算力同 H100，NVLink 带宽被阉割） | ~1979 |

**MFU 经验值**：大模型训练 **40%~50% 算正常，50%+ 算优秀**，<30% 通常有明显瓶颈（通信没 overlap、bubble 太大、数据加载慢等）。

> 结合第 5 篇：如果你的 DP/PP 跨机通信没被计算隐藏（busbw 不够），MFU 会直接掉下来——**MFU 就是前六篇所有并行/通信决策的最终记分牌。**

## 调优优先级（实战 checklist）

按"性价比从高到低"：

1. **`--sequence-parallel`**（配 TP）：几乎免费省激活显存（第 2 篇），必开；
2. **`--use-distributed-optimizer`**（ZeRO-1）：省优化器状态（第 4 篇），必开；
3. **`--recompute-granularity selective`**：性价比最高的激活省法，20B+ 必开；显存还不够再上 `full`；
4. **`--overlap-grad-reduce --overlap-param-gather`**：把 DP 通信藏进计算（第 4/5 篇）；
5. **FP8**（Hopper+）：GEMM 提速、省带宽，盯 loss 稳定性；
6. **调 batch**：增大 `global-batch`（更多 microbatch）压 PP bubble（第 3 篇 `m≥4p`），但受激活显存约束——和重计算联动权衡；
7. 用 `throughput per GPU (TFLOP/s/GPU)` 日志 + 显存快照持续量化，改一项看一项。

## 小结

- 显存四大块：参数/梯度/优化器状态（ZeRO 切）+ **激活（重计算切）** + buffer + 碎片；
- **重计算**用算力换激活显存：`selective`（只重算 core_attn，性价比最高）优先，`full`（整层）兜底；
- **BF16** 动态范围等于 FP32、免 loss scaling，是大模型首选；**FP8** 只给 GEMM 用、norm/softmax 保持高精度；
- **MFU = 实测 TFLOP/s ÷ 硬件峰值**，Megatron 的 throughput 不含重计算 FLOPs（故重计算会降 MFU）；40-50% 正常；
- MFU 是前六篇所有并行/通信决策的最终记分牌；调优按 SP → 分布式优化器 → selective 重计算 → overlap → FP8 的优先级来。

---

至此，《Megatron-LM 分布式训练》系列完结。从"为什么要并行"到"五维并行怎么切"，再到"网络与性能怎么调"，希望这条主线能帮你建立起大模型分布式训练的完整心智模型。有问题欢迎交流。
