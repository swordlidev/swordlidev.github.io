---
title: "Megatron-LM 分布式训练（一）：并行维度与训练生命周期"
description: "分布式训练系列开篇：先补齐为什么需要多卡训练、集合通信原语等背景，再讲 Megatron 的 GPU 网格世界观（TP/PP/DP/CP/EP）和一次训练迭代的完整生命周期。"
pubDate: 2025-09-05
tags: ["Megatron-LM", "分布式训练", "LLM", "并行计算"]
---

> **本文是《Megatron-LM 分布式训练》系列第 1 篇**。本系列结合 Megatron-LM 0.19.0 / Megatron-Core 真实代码与 64 卡（8 节点 × 8 卡）RoCE 集群实战整理，面向想系统理解大模型并行训练的读者。
>
> 系列目录：
> 1. **总览：并行维度与训练生命周期**（本文）
> 2. [张量并行 TP 的内部原理](/blog/megatron-lm-02-tensor-parallel/)
> 3. [流水线并行 PP 与 1F1B 调度](/blog/megatron-lm-03-pipeline-parallel/)
> 4. [数据并行 DP 与分布式优化器（ZeRO）](/blog/megatron-lm-04-data-parallel-zero/)
> 5. [通信与网络实测（NCCL / RoCE）](/blog/megatron-lm-05-communication/)
> 6. [MoE 专家并行 EP 与上下文并行 CP](/blog/megatron-lm-06-moe-ep-cp/)
> 7. [显存与性能调优](/blog/megatron-lm-07-perf-tuning/)
> 8. [Checkpoint、容错与弹性训练](/blog/megatron-lm-08-checkpoint-ft/)

## 背景：为什么单卡训练不了大模型

先算一笔显存账。用混合精度 + Adam 训练，**每个参数**大约要占 16 字节（后面第 4 篇会拆解）：bf16 权重 2B + bf16 梯度 2B + fp32 master 权重 4B + Adam 一阶动量 4B + 二阶动量 4B。

一个 7B 模型光这部分就要 $7\times10^9 \times 16 \approx 112\ \text{GB}$，已经超过一张 H100（80GB）；再加上**激活（activation）**——前向过程中为反向保留的中间结果，与 `batch × seq × hidden × layers` 成正比——单卡根本放不下。更别说 70B、几百 B 的模型。

于是有了两类基本思路：

- **切数据**：每张卡拿完整模型、各算一批数据，再同步梯度 —— 这是**数据并行（DP）**，解决"算得更快"。
- **切模型**：把一个模型拆到多张卡上 —— 这是**模型并行**，解决"单卡放不下"。模型并行又细分为切层内权重的**张量并行（TP）**和切层的**流水线并行（PP）**。

Megatron-LM 的价值，就在于把这些并行方式组合起来（3D 乃至 5D 并行），高效地训练超大模型。

## 预备：集合通信原语速查

分布式训练的一切协同都建立在几个**集合通信（collective communication）**原语之上，由 NCCL 等库实现。先混个眼熟，后面每一篇都会反复用到：

| 原语 | 语义 | 典型用途 |
|---|---|---|
| **all-reduce** | 每张卡各有一份数据，输出"所有卡对应位置求和（或平均）"的结果，人人拿到 | TP 层内求和、DP 梯度平均 |
| **reduce-scatter** | 先求和，再把结果**切片**分给各卡（每卡只拿自己那片） | 分布式优化器梯度同步 |
| **all-gather** | 每卡各持一片，互相收集拼成完整数据 | 分布式优化器参数回收 |
| **all-to-all** | 每张卡给每张卡发送**不同**的数据（洗牌） | MoE 把 token 分发到专家卡 |
| **P2P send/recv** | 点对点，一张卡发给另一张卡 | 流水线相邻 stage 传激活 |
| **broadcast** | 一张卡把数据复制给组内所有卡 | TP 组内分发输入数据 |

一个关键事实：**`all-reduce = reduce-scatter + all-gather`**。这个等式后面会多次出现，是理解分布式优化器省显存、以及 Ring All-Reduce 算法的基础。

---

## 并行维度心智模型 —— 把 GPU 排成一个"网格"

一句话抓本质：

> Megatron 不是把"模型"发给每张卡，而是先把所有 GPU 在逻辑上排成一个多维网格，再规定"模型的哪一部分、数据的哪一份"落在网格的哪个坐标上。

### 五个并行维度

| 维度 | 切的是什么 | 直觉比喻 |
|---|---|---|
| **DP** 数据并行 | 切**数据批次**，每组持有完整模型副本 | 多个工人各做一批作业，最后对梯度取平均 |
| **TP** 张量并行 | 切**单层内部的权重矩阵**（层内切） | 一道大矩阵乘法，几个人各算一列/一行 |
| **PP** 流水线并行 | 切**模型的层**（层间切） | 流水线，工位1做第1~8层，工位2做第9~16层 |
| **CP** 上下文并行 | 切**序列长度**（超长 context 用） | 一篇超长文章按段落分给不同人读 |
| **EP** 专家并行 | 切 **MoE 的专家** | 不同专家住在不同卡上 |

记忆点：**TP 和 PP 都是"切模型"（model parallel），DP 是"切数据"**。切模型因为单卡放不下；切数据为了更快。

### 核心公式

$$\text{world\_size} = TP \times PP \times DP \times CP \quad(\text{暂不含 EP})$$

以 8 卡、TP=2、PP=1 为例：

$$DP = \frac{\text{world\_size}}{TP \times PP} = \frac{8}{2 \times 1} = 4$$

即：每 2 张卡放一份完整模型（TP=2），这样的副本共 4 份（DP=4），各吃不同数据、算完梯度 all-reduce 求平均。

### rank → 坐标的映射（很多人栽这）

默认排列顺序 `order="tp-cp-ep-dp-pp"`（TP 变化最快，相邻 rank 优先凑成 TP 组）。对 8 卡（TP=2, PP=1, DP=4）：

```
TP 组（层内切，NVLink 高频通信）: {0,1} {2,3} {4,5} {6,7}
DP 组（梯度 all-reduce）        : {0,2,4,6} {1,3,5,7}
```

**工程铁律：TP 一定不要跨机。** TP 组内每次前向/反向都要 all-reduce，跨机走 RoCE 会慢到无法接受，所以 TP ≤ 单机卡数（通常 ≤8）。`CUDA_DEVICE_MAX_CONNECTIONS=1` 就是为了让 TP 通信走单一队列、避免乱序。

### 代码落点

`parallel_state.py` 开头的全局变量为"当前卡属于哪些组"占坑：

- `_TENSOR_MODEL_PARALLEL_GROUP`：TP 组（层内）
- `_PIPELINE_MODEL_PARALLEL_GROUP`：PP 组（层间）
- `_DATA_PARALLEL_GROUP`：DP 组（梯度平均）

`initialize_model_parallel(...)` 在启动时按传入的 TP/PP/DP 大小，把这些 `None` 替换成真正的 `torch.distributed` 通信组。任何地方想做 TP 通信都通过 `get_tensor_model_parallel_group()` 拿到组——这是统一入口。

---

## 一次训练迭代的完整生命周期

### 调用栈全景

```
pretrain_gpt.py  →  pretrain()                  # 入口：建模型/优化器/数据集
                      └─ train()                # 大循环：while iteration < train_iters
                          └─ train_step()       # ★ 一次迭代
                              └─ forward_backward_func()   # 调度器：安排前向/反向
                                   └─ 循环每个 microbatch:
                                        ├─ forward_step()  # 你在 pretrain_gpt.py 里写的
                                        │    ├─ get_batch()      # 取数据
                                        │    ├─ model(...)       # 前向 → logits/loss
                                        │    └─ loss_func        # 算 loss
                                        └─ backward_step()       # 反向 → 梯度
                              └─ optimizer.step()              # 更新参数
                              └─ opt_param_scheduler.step()    # 更新学习率
```

关键认知：**`pretrain_gpt.py` 只负责"模型长什么样"和"一个 microbatch 怎么算 loss"，并行协同全在 `megatron.core` / `megatron.training` 里。** 换模型只改 `pretrain_gpt.py`，并行策略不用动。

### `train_step()` 的四个阶段

1. **清梯度**：`model` 是列表（为虚拟流水线准备），`zero_grad_buffer()` 清的是 Megatron 自管的**连续梯度缓冲区**（不是 PyTorch 默认 per-param `.grad`），这块缓冲区是 DP reduce-scatter/all-reduce 的基础。
2. **决定 microbatch 数**：

   $$\text{num\_microbatches} = \frac{\text{global\_batch}}{\text{micro\_batch} \times DP} = \frac{32}{1 \times 4} = 8$$

   即每卡串行跑 8 个 microbatch、梯度累加 —— 这就是**梯度累加（gradient accumulation）**。
3. **前向+反向**：委托给 `forward_backward_func`（按 PP 大小动态选），返回各 microbatch 的 loss 列表。
4. **更新参数 + 学习率**：`optimizer.step()` 内部做梯度裁剪 → loss scaling 检查（有 inf/nan 则 `update_successful=False` 跳过）→ 更新权重；`grad_norm` 要跨 model-parallel 组汇总（因参数被 TP/PP 切开）。

### 调度器是并行的分水岭

| PP 情况 | 调度实现 |
|---|---|
| PP = 1 | `forward_backward_no_pipelining`（朴素循环） |
| PP > 1 | `forward_backward_pipelining_without_interleaving`（1F1B） |
| PP > 1 + 虚拟流水线 | `..._with_interleaving` |

PP=1 时就是：跑一个 microbatch 前向 → 立刻反向 → 下一个，重复 N 次（最后一个单独处理，配合 DP `no_sync` 优化）。

### `forward_step()` / `loss_func()`

- `get_batch` 取数据：只有 **TP rank 0** 真正读盘，再在 TP 组内 broadcast（省 IO）。
- `model(...)` 传了 `labels`，内部直接算 per-token loss。
- 返回 `(output_tensor, partial(loss_func, ...))`——返回**还没执行的 loss_func**，因为只有流水线最后一级才需算 loss。
- `loss_func` 固定返回三元组：`loss`（masked loss）、`num_tokens`（用于按全局 token 归一化）、`report`（日志指标，如 `lm loss`）。

### 并行维度钉到时间轴（本篇精华）

| 时机 | 谁通信 | 类型 | 频率 |
|---|---|---|---|
| `model(...)` 每层内部 | **TP 组** | all-reduce（NVLink） | 极高（每层 2 次） |
| 流水线工位之间 | **PP 组** | P2P send/recv | 中（每 microbatch 边界） |
| 反向梯度就绪 | **DP 组** | reduce-scatter / all-reduce | 低（每迭代 1 次，可 overlap） |
| 参数更新 | DP 组（分布式优化器） | all-gather 参数 | 低（每迭代 1 次） |

这张表就是性能调优地图：**TP 必须 NVLink 同机；DP/PP 频率低可跨机 RoCE。** 后面几篇会逐一展开每个维度。

## 小结

- 单卡放不下大模型（参数 + 激活），所以要么**切数据**（DP，为了快），要么**切模型**（TP/PP，为了放得下）。
- Megatron 的世界观是先把 GPU 排成多维网格，再把模型/数据映射到网格坐标；`world_size = TP×PP×DP×CP`。
- 一次迭代 = 清梯度 → 定 microbatch 数（梯度累加）→ 前向反向（调度器按 PP 选）→ 更新参数/学习率。
- 记住那张"通信时间轴"表：TP 极高频必须 NVLink，DP/PP 低频可跨机 —— 这是贯穿全系列的主线。

下一篇进入**张量并行 TP**，看一道矩阵乘法是怎么被切到多张卡上、又只用两次 all-reduce 就缝合回来的。
