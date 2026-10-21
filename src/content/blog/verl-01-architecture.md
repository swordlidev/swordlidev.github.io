---
title: "verl 源码解读（一）：入门与整体架构"
description: "verl 是什么、解决什么问题、核心概念有哪些、一次 RL 训练的数据流长什么样。本系列第 1 篇建立对 HybridFlow、single-controller/multi-controller、DataProto 与 WorkerGroup 的整体认知。"
pubDate: 2026-02-03
tags: ["verl", "强化学习", "RLHF", "HybridFlow", "源码解读", "LLM"]
---

> **本文是《verl 源码解读》系列第 1 篇**。本系列基于 verl（字节 Seed 团队发起的 LLM RL 训练库）源码，自顶向下拆解它如何用「几行代码」表达复杂的 PPO/GRPO 数据流并高效跑在上百张 GPU 上。
>
> 系列目录：
> 1. **入门与整体架构**（本文）
> 2. [HybridFlow 编程模型与单控制器](/blog/verl-02-hybridflow-single-controller/)
> 3. [DataProto：贯穿 verl 的数据协议](/blog/verl-03-dataproto/)
> 4. [PPO / GRPO 训练主循环与核心算法](/blog/verl-04-ppo-grpo-core/)
> 5. [Rollout 生成与训练推理混合引擎](/blog/verl-05-rollout-hybrid-engine/)

本篇目标：建立对 verl 的整体认知——它是什么、解决什么问题、核心概念有哪些、一次训练的数据流长什么样。后续几篇逐个深入核心模块。

## 1. verl 是什么

verl（Volcano Engine Reinforcement Learning for LLMs）是字节跳动 Seed 团队发起、由社区维护的**大语言模型强化学习（RL）训练库**，是论文 **HybridFlow: A Flexible and Efficient RLHF Framework**（arXiv:2409.19256）的开源实现。

一句话概括它的定位：**用几行代码就能搭出 PPO / GRPO 这类复杂的后训练（post-training）数据流，并且能高效地跑在几十上百张 GPU 上。**

它主要面向三类需求：

- **SFT（监督微调）**：`verl/trainer/sft_trainer.py`
- **RLHF / RLVR（强化学习）**：PPO、GRPO、GSPO、ReMax、REINFORCE++、RLOO、DAPO 等
- **多模态 / 多轮 Agent RL**：VLM、工具调用、多轮对话

## 2. verl 想解决的核心矛盾

LLM 的 RL 训练（以 PPO 为例）在一个训练 step 里会涉及多种“角色模型”：

| 角色 | 作用 | 典型后端 |
|------|------|----------|
| Actor（策略模型） | 被训练、参数会更新 | FSDP / Megatron |
| Rollout（生成） | 用当前策略采样生成回答 | vLLM / SGLang |
| Critic（价值模型） | 估计状态价值，PPO 需要 | FSDP / Megatron |
| Reference（参考模型） | 计算 KL 约束，防止跑偏 | FSDP / Megatron |
| Reward Model（奖励模型） | 给回答打分（也可用函数式奖励） | FSDP / Megatron |

这里有两个本质矛盾：

1. **训练引擎和推理引擎需求不同**：训练（反向传播）适合 FSDP / Megatron；生成（自回归解码）用 vLLM / SGLang 快一个数量级。同一个 Actor 的权重要在两种引擎之间来回“搬家”。
2. **多个模型要共享有限的 GPU**：不可能给每个角色都独占一组卡。需要灵活地把不同模型“摆放”（placement）到不同或相同的 GPU 上。

verl 用 **HybridFlow（混合控制器）** 编程模型来解决这些问题，这是它最核心的设计，会在第 2 篇详细讲。

> 其中训练后端涉及的 FSDP / Megatron 的 TP/PP/DP/EP、ZeRO 等概念，可参考我的[《Megatron-LM 分布式训练》系列](/blog/megatron-lm-01-overview/)。

## 3. 核心设计思想：single-controller + multi-controller

verl 的编程模型可以概括为“**单控制器下发、多控制器执行**”：

- **Single-Controller（单控制器 / Driver）**：一个中心进程（driver）掌握整个 RL 数据流的“剧本”。它按顺序决定“先生成、再算 log_prob、再算 reward、再算 advantage、最后更新 actor”。轻量计算（如 advantage）就在 driver 上做。
- **Multi-Controller（多控制器 / Workers）**：每个角色模型由一组分布式 worker 承载（基于 Ray）。driver 只需通过 RPC 调用 worker group 的方法，无需关心底层是 8 卡还是 64 卡、是 FSDP 还是 Megatron。

对应到代码就是：

```python
# driver 侧（verl/trainer/ppo/ray_trainer.py 的 fit 方法）只需这样写：
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)   # 生成
old_log_prob    = self.actor_rollout_wg.compute_log_prob(batch)          # 算 log_prob
values          = self.critic_wg.compute_values(batch)                   # 算 value
# ... advantage 在 driver 本地算 ...
self.actor_rollout_wg.update_actor(batch)                                # 更新策略
```

每一行 `xxx_wg.method()` 背后，都是对一组 GPU worker 的分布式调用。**复杂的数据切分、分发、聚合，被封装在装饰器里**（第 2 篇讲）。

## 4. 目录结构速览

```
verl/
├── trainer/              # 训练入口与主循环
│   ├── main_ppo.py       # PPO/GRPO 程序入口
│   ├── ppo/
│   │   ├── ray_trainer.py   # 核心：RayPPOTrainer，fit() 主循环
│   │   ├── core_algos.py    # PPO/GRPO 的 loss、advantage 等纯算法
│   │   └── reward.py        # 奖励计算
│   ├── sft_trainer.py    # SFT 训练
│   └── config/           # 大量 yaml 配置（Hydra/OmegaConf）
├── single_controller/    # HybridFlow 核心：单控制器 + WorkerGroup + Dispatch
│   ├── base/
│   │   ├── decorator.py     # @register 分发装饰器、Dispatch 模式
│   │   ├── worker.py        # Worker 基类
│   │   └── worker_group.py  # WorkerGroup / ResourcePool
│   └── ray/                 # Ray 后端实现
├── workers/              # 各角色的具体实现
│   ├── rollout/             # vLLM / SGLang 生成
│   ├── engine/              # 训练引擎（FSDP / Megatron）
│   ├── reward_manager/      # 奖励管理
│   └── config/
├── protocol.py           # DataProto：贯穿全程的数据协议
├── models/               # 模型定义与适配
└── utils/                # 各类工具
```

## 5. 一个 RL 训练 step 的数据流

以 PPO 为例，`RayPPOTrainer.fit()` 中一个 step 的大致流程：

```
┌────────────────────────────────────────────────────────────┐
│ 1. 从 dataloader 取一批 prompt → 封装成 DataProto           │
│ 2. gen_batch.repeat(n)  →  每个 prompt 采样 n 条回答         │
│ 3. Rollout 生成：actor_rollout_wg.generate_sequences()       │  ← vLLM/SGLang
│ 4. 重新计算 old_log_prob（训练引擎口径，用于重要性采样）     │  ← FSDP/Megatron
│ 5. Reference 计算 ref_log_prob（用于 KL）                    │
│ 6. Reward：函数式奖励 / 奖励模型打分                         │
│ 7. 计算 advantage（GAE / GRPO 分组归一化）  ← 在 driver 本地  │
│ 8. Critic 更新（PPO 需要；GRPO 无 critic）                   │
│ 9. Actor 更新：update_actor()                               │  ← FSDP/Megatron
│ 10. 权重同步：训练后的新权重 → Rollout 引擎                   │
└────────────────────────────────────────────────────────────┘
```

其中几个关键点：

- **第 3 步和第 9 步的 Actor 是同一份权重，但跑在不同引擎上**，所以需要“权重 resharding / 同步”（第 5 篇讲）。
- **第 4 步为什么要重新算 log_prob？** 因为生成引擎（vLLM）和训练引擎（FSDP）的数值实现有差异，PPO 需要用训练引擎口径的 `old_log_prob` 来保证重要性采样比值的正确性。
- **第 7 步在 driver 本地做**，因为 advantage 计算是轻量的张量运算，没必要分发到 worker。

## 6. verl 支持的主要能力（用于选型）

- **训练后端**：FSDP、FSDP2、Megatron-LM
- **生成后端**：vLLM、SGLang、HF Transformers
- **RL 算法**：PPO、GRPO、GSPO、ReMax、REINFORCE++、RLOO、DAPO、DrGRPO 等
- **奖励**：函数式（可验证奖励，如数学/代码）+ 模型式
- **模型**：Qwen 系列、Llama3、Gemma2、DeepSeek 等 HF 模型；VLM 多模态
- **规模**：可扩展到 671B 模型、数百张 GPU（专家并行）
- **其他**：LoRA、序列打包、序列并行（Ulysses）、Flash Attention 2
- **硬件**：NVIDIA、AMD ROCm、Ascend NPU

## 7. 如何快速上手（从哪跑第一个例子）

典型的 PPO / GRPO 例子位于 `examples/` 下，例如：

- `examples/grpo_trainer/run_qwen2-7b.sh`
- `examples/ppo_trainer/`

这些脚本本质上是给 `verl/trainer/main_ppo.py` 传一大堆 Hydra 配置（`actor_rollout_ref.xxx`、`critic.xxx`、`algorithm.xxx`、`trainer.xxx`）。读懂一个 example 脚本的配置项，再对照 `verl/trainer/config/` 下的 yaml，就能理解整个训练的“旋钮”。

## 小结

- verl 是一个“灵活 + 高效 + 生产可用”的 LLM RL 训练库，核心是 HybridFlow 编程模型。
- 它用 **single-controller 描述数据流 + multi-controller 执行计算** 的方式，把“复杂的分布式 RL 流程”写成“几行同步代码”。
- 贯穿始终的两个核心抽象：**DataProto（数据）** 和 **WorkerGroup + Dispatch（计算）**。
- 它天然要处理“训练引擎 vs 推理引擎”的混合，这决定了它的很多独特设计。

下一篇我们深入 HybridFlow 的核心：单控制器、WorkerGroup 和那套 `@register` 分发装饰器。
