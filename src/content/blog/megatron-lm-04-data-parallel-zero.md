---
title: "Megatron-LM 分布式训练（四）：数据并行 DP 与分布式优化器（ZeRO）"
description: "数据并行为什么会重复浪费显存？本文讲清混合精度下的 16 字节/参数构成、ZeRO 思想、Megatron 分布式优化器（reduce-scatter + all-gather）、梯度缓冲区 overlap，并厘清 ZeRO 各档与 Megatron / FSDP / DeepSpeed 的关系。"
pubDate: 2026-10-17
tags: ["Megatron-LM", "分布式训练", "数据并行", "ZeRO", "LLM"]
---

> **本文是《Megatron-LM 分布式训练》系列第 4 篇**。前面讲了[总览](/blog/megatron-lm-01-overview/)、[TP](/blog/megatron-lm-02-tensor-parallel/)、[PP](/blog/megatron-lm-03-pipeline-parallel/)，本篇聚焦**数据并行（DP）与分布式优化器**。

## 背景：数据并行与优化器状态

数据并行（DP）最朴素：每张卡持有一份**完整模型副本**，各吃不同数据算梯度，再用 all-reduce 对梯度求平均，保证各副本参数一致。它解决"算得快"，但不解决"放得下"——因为每张卡都存了完整的一套。

这"一套"到底有多大？用混合精度 + Adam，每个参数的"随行物料"远不止权重本身。先建立这个账本，才能理解 ZeRO 为什么能省那么多。

## 朴素 DP 的致命浪费

混合精度 + Adam，每个参数占 16 字节：

| 内容 | 字节/参数 |
|---|---|
| bf16 权重 | 2 |
| bf16 梯度 | 2 |
| fp32 master 权重 | 4 |
| Adam 动量 m | 4 |
| Adam 方差 v | 4 |
| **合计** | **16** |

**浪费**：后 12 字节（fp32 master + m + v，即**优化器状态**）在 DP 组里每张卡都重复存了一份。DP=8 就是同样的优化器状态存了 8 遍。

## 分布式优化器（ZeRO-1）

**ZeRO（Zero Redundancy Optimizer）** 的核心思想：既然 DP 组里每张卡存的优化器状态是一样的，那就**切开**，每卡只负责 $1/DP$，用到时再通信拿。Megatron 的分布式优化器相当于 ZeRO-1，用 `--use-distributed-optimizer` 开启。

| 方案 | 每卡优化器状态显存 |
|---|---|
| 朴素 DP | $12\Psi$（全量重复，$\Psi$ 为参数量） |
| 分布式优化器 | $12\Psi / DP$ |

## 通信模式：all-reduce → reduce-scatter + all-gather

```
分布式优化器:
  反向出梯度
    → reduce-scatter（每卡只拿"自己负责那片"的求和梯度）
    → 本卡用 fp32 master 更新自己这 1/DP 参数
    → all-gather（把各卡更新后的参数拼回完整权重，供下次前向）
```

精妙之处：回忆第 1 篇的等式 **`all-reduce = reduce-scatter + all-gather`**。朴素 DP 做一次 all-reduce；分布式优化器把它拆成两半，中间插入"各管一片的参数更新"，**总通信量不变，显存却省了 DP 倍**。这是近乎免费的优化，必开。

## 梯度缓冲区 + 分桶 + overlap

Megatron 预分配一整块**连续的 fp32 梯度缓冲区**，参数分桶（bucket）。**一个桶里所有参数的梯度就绪 → 立刻异步发起该桶的通信，同时继续算前面的层** → 把通信藏进计算。

- `--overlap-grad-reduce`：梯度 reduce-scatter 藏进反向；
- `--overlap-param-gather`：参数 all-gather 藏进前向（prefetch）；
- `--ddp-bucket-size`：桶太小→通信次数多；太大→overlap 窗口小，需要权衡。

## 跨 model-parallel 的梯度收尾

有些梯度在 DP 同步之外还需要额外处理：

- **tied embedding**：输入 embedding 与输出 lm-head 权重共享，但分处 PP 第一级 / 最后一级两张卡 → 需在这两级间额外 all-reduce 梯度；
- **SP 切开的 LayerNorm 等**：需在 TP 组内 all-reduce 补齐。

顺序很重要：**先 DP 同步 → 再 model-parallel 梯度收尾 → 才 `optimizer.step()`**。

---

## 深入：朴素 DP 的 all-reduce 到底怎么做（四步接力）

```
[反向计算]          [反向钩子]              [凑齐一桶]           [集合通信]           [更新]
autograd 算出     param.grad 累加进      一个 bucket 所有     对 grad_data 做     optimizer.step()
每个 param.grad → param.main_grad  →    梯度 ready 时  →    all_reduce(原地) → 读 main_grad 更新
                  (fp32 连续缓冲区)       自动触发 sync        每卡得求和梯度       全部参数
```

1. **挂钩子**：每个参数在其梯度累加节点注册 backward post-hook。`param.main_grad` 是大缓冲区的一个**视图**。
2. **钩子搬梯度**：`param.main_grad.add_(param.grad.data)`（用 `add_` 因为要梯度累加；用 fp32 因 bf16 多次累加会丢精度）→ 置 `param.grad = None`（省显存）→ `register_grad_ready(param)`。
3. **凑齐一桶自动触发** `start_grad_sync()`（这是 overlap 的来源）。
4. **真正 all-reduce**：
   - 先缩放：`grad_data *= gradient_scaling_factor`（典型 `1/DP`），把"求和"变"求平均"（或直接用 `ReduceOp.AVG`）；
   - 选组：开分布式优化器用 `reduce_scatter`，否则 `all_reduce`，对象是整个桶 `bucket.grad_data`（一次通信搞定成百上千个参数），组是 DP 组，**原地写回**；`async_op` 配合 overlap。
5. **各自更新全部参数**：每卡 `main_grad` 都是同一份平均梯度 → 因为起点权重相同 + 梯度相同 + 优化器相同 → 更新后天然保持一致，无需再同步参数。

## 深入：all-reduce 底层算法（NCCL）

语义：每卡各有一个向量，输出让每卡都拿到"所有卡对应元素求和"。等价于 reduce + broadcast。

**Ring All-Reduce**（大消息默认）：N 卡排成环，向量切 N 块，两阶段各 N-1 步：

- 阶段 1 **Reduce-Scatter**：边传边加，N-1 步后第 k 卡持有第 k 块的完整求和；
- 阶段 2 **All-Gather**：再传一圈，让每卡收齐全部块。

每卡网线收发量 = 总大小 × $2(N-1)/N$。这就是带宽基准脚本里那个系数的来历：

```python
busbw = algbw * 2.0 * (world - 1) / world
```

- **algbw（算法带宽）** = 数据量 / 时间（用户视角，不同 N 不可直接比硬件峰值）；
- **busbw（总线带宽）** = algbw × $2(N-1)/N$（硬件视角，可直接对比 NVLink/RoCE 线速）。

ring 下每条链路负载 $2(N-1)/N \to 2$（几乎与 N 无关）→ 能扩展到成百上千卡。

**Tree 算法**：小消息延迟主导，约 $2\log_2 N$ 步。NCCL 会自动按消息大小 / 卡数 / 拓扑在 Ring / Tree 间切换。看 busbw 只取**大消息的平台值**。

## ZeRO 各档（Megatron / FSDP / DeepSpeed 关系）

ZeRO 逐级沿 DP 组（N 卡）切分 {参数 P、梯度 G、优化器状态 Os}：

| 方案 | P | G | Os | 每卡显存 | 通信量 | Megatron 对应 |
|---|---|---|---|---|---|---|
| 朴素 DP | 全 | 全 | 全 | 16 | 1× | 默认 |
| ZeRO-1 | 全 | 全 | /N | 2+2+12/N | 1× | — |
| ZeRO-1.5（Megatron） | 全 | /N | /N | 2+2/N+12/N | 1× | `--use-distributed-optimizer` |
| ZeRO-3 | /N | /N | /N | 16/N | ~1.5× | `--use-megatron-fsdp` / `--use-torch-fsdp2` |

附加件：**ZeRO-R**（激活/buffer 优化）、**ZeRO-Offload**（优化器卸载 CPU）、**ZeRO-Infinity**（卸载 NVMe）、**ZeRO++**（量化权重/梯度、分层切分降通信）。

三者关系一句话：

> **ZeRO 是"显存切分思想"（DeepSpeed 首创）；DeepSpeed / FSDP / Megatron 是三套落地实现。**

```
         【思想层】  ZeRO (切 P/G/Os)  —— DeepSpeed 首创
   ┌────────┼─────────────────────┐
【实现层】 DeepSpeed      PyTorch FSDP       Megatron DistributedOptimizer
         (ZeRO-1/2/3)   (ZeRO-3)           (≈ZeRO-1，切 Os+G)
【定位】  纯DP省显存库   纯DP省显存(官方)    3D并行里的DP一环
```

- **FSDP ≈ PyTorch 官方重写的 ZeRO-3**；
- **Megatron 的看家本领是 TP+PP**（DeepSpeed/FSDP 本身不提供），它的 ZeRO 只是 DP 维度的增强；
- 工业现实：`Megatron-DeepSpeed`（训练 BLOOM-176B）、Megatron 内置 FSDP 都是"你中有我"。

**实战选型**：

| 场景 | 选什么 |
|---|---|
| 模型大、必须切层（主场景） | **Megatron TP×PP + DistributedOptimizer(ZeRO-1)** |
| 模型中等、只用 DP、想简单 | FSDP(ZeRO-3) 或 DeepSpeed ZeRO-3 |
| 单卡/少卡硬塞 | DeepSpeed ZeRO-Offload/Infinity |

**铁律**：3D 并行里默认**不叠 ZeRO-3**，只叠 ZeRO-1 —— 因为 TP/PP 已经切了参数/激活，ZeRO-3 额外的 all-gather 会和 TP/PP 抢通信带宽。

## 小结

- 朴素 DP 把 12 字节/参数的**优化器状态**在 DP 组里重复存了 N 份，这是最大浪费。
- **分布式优化器（ZeRO-1）** 把优化器状态切成 $1/DP$，用 `all-reduce = reduce-scatter + all-gather` 的拆分实现，**通信量不变、显存省 DP 倍**。
- 梯度走连续缓冲区 + 分桶 + 异步通信，可把 DP 通信 **overlap** 进计算。
- ZeRO 是思想，Megatron/FSDP/DeepSpeed 是实现；3D 并行场景只叠 ZeRO-1，不叠 ZeRO-3。

下一篇把视角转到**物理网络**：NCCL 在"机内 NVLink + 机间 RoCE"上到底怎么跑，busbw 多少才算健康，以及带宽如何决定 DP/PP 跨机通信能否被计算隐藏。
