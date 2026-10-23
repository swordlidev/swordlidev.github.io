---
title: "verl 源码解读（五）：Rollout 生成与训练推理混合引擎"
description: "verl 最有工程含量的部分：生成为什么单独用 vLLM/SGLang，BaseRollout 的 resume/generate/release/update_weights 生命周期，以及训练后新权重如何经 ShardingManager 重分片同步回推理引擎、显存如何错峰。"
pubDate: 2026-02-24
tags: ["verl", "vLLM", "推理引擎", "强化学习", "源码解读"]
---

> **本文是《verl 源码解读》系列第 5 篇**。前四篇讲清了“数据流怎么编排、数据怎么流转、算法怎么算”。本篇聚焦 verl 最有工程含量的部分：**生成（Rollout）如何用 vLLM/SGLang 加速，以及训练后的新权重如何同步回推理引擎**。对应代码在 `verl/workers/rollout/`。
>
> 系列目录：[1 架构](/blog/verl-01-architecture/) · [2 HybridFlow](/blog/verl-02-hybridflow-single-controller/) · [3 DataProto](/blog/verl-03-dataproto/) · [4 PPO/GRPO](/blog/verl-04-ppo-grpo-core/) · **5 Rollout 混合引擎** · [6 RL 算法详解](/blog/verl-06-rl-algorithms/)

## 1. 为什么 Rollout 要单独用一套引擎

RL 训练每个 step 都要“用当前策略生成一批回答”。生成是**自回归解码**，和训练（一次 forward+backward）是完全不同的负载：

- **训练引擎**（FSDP / Megatron）：为反向传播和参数分片优化，直接拿来做逐 token 解码非常慢。
- **推理引擎**（vLLM / SGLang）：有 PagedAttention、continuous batching、KV cache 复用等，生成吞吐能快一个数量级。

所以 verl 让 **Actor 用训练引擎存在，但生成时切换到 vLLM/SGLang**。这就带来了全篇的核心难题：**同一份 Actor 权重，要在两套引擎之间来回同步。** 这正是 HybridFlow 论文里 "3D-HybridEngine" 要解决的问题。

## 2. BaseRollout：统一的生成抽象

所有生成后端都实现 `verl/workers/rollout/base.py` 的 `BaseRollout`，它只有几个抽象方法，把“生成 + 权重/显存管理”抽象得非常干净：

```python
class BaseRollout(ABC):
    @abstractmethod
    async def resume(self, tags: list[str]):
        """把权重或 KV cache 恢复到 GPU 显存（tags: 'weights' / 'kv_cache'）"""

    @abstractmethod
    async def update_weights(self, weights, wire_format="named_tensors", **kwargs):
        """用训练侧传来的新权重更新推理引擎"""

    @abstractmethod
    async def release(self):
        """释放显存里的权重和 KV cache"""

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """同步批量生成"""
```

这四个方法恰好对应一个 RL step 里 Rollout 的生命周期：
**resume（唤醒）→ generate（生成）→ release（释放显存给训练用）→ update_weights（训练后更新权重）**。

## 3. 后端注册表：vLLM / SGLang / TRT-LLM

verl 用注册表选择后端，默认都是 **async（server）模式**：

```python
_ROLLOUT_REGISTRY = {
    ("vllm",   "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter",
}

def get_rollout_class(rollout_name, mode="async"):
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]
    module_name, class_name = fqdn.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)
```

配置 `actor_rollout_ref.rollout.name=vllm`（或 `sglang`）就切换后端。`rollout/` 目录下还有 `naive/`（纯 HF）、`hf_rollout.py` 等用于调试/对拍。

## 4. 两种工作模式：sync vs server(async)

### 4.1 同步模式

driver 调 `actor_rollout_wg.generate_sequences(gen_batch)`，走第 2 篇的 `DP_COMPUTE_PROTO`：prompt 被切分到各 rollout worker，并行生成后 concat 回来。适合“一问一答”的单轮场景。

### 4.2 Server / async 模式（多轮 Agent 必需）

对于**多轮对话、工具调用、Agent**，一次“生成”中间可能要调外部工具、再继续生成，无法简单地一次性批处理。verl 提供 **server 模式**：把推理引擎拉起成一个服务（`llm_server.py` / `replica.py` / `vllm_async_server.py`），用 `AgentLoop` 之类的驱动按 OpenAI-like 接口多轮交互。

主循环里对应的是 `async_rollout_manager.generate_sequences(...)`（见第 4 篇 fit() 片段），底层用 `replica.py` 管理多个推理副本（replica），实现请求级的并发调度。

## 5. 权重同步：整篇的核心

训练一个 step 后，Actor 权重变了，必须把新权重灌进推理引擎，否则下一轮还在用旧策略采样。难点在于：

- 训练侧权重可能是 **FSDP 分片** 或 **Megatron 的 TP/PP 切分**；
- 推理侧（vLLM）有自己的并行与权重布局；
- 两者的 **sharding（分片方式）通常不一致**，需要“重新分片 / resharding”。

### 5.1 ShardingManager：训练布局 ↔ 推理布局

verl 为每种组合提供了 sharding manager（如 FSDP↔vLLM、Megatron↔vLLM），负责在生成前后转换权重布局，并处理设备 mesh 的映射。它是“3D-HybridEngine 消除内存冗余、降低通信开销”的落地点。

> 训练侧 FSDP / Megatron 的分片方式（ZeRO 切参数/梯度/优化器、TP/PP 切模型），详见[《Megatron-LM 分布式训练》系列](/blog/megatron-lm-04-data-parallel-zero/)。

### 5.2 update_weights 的传输格式

`update_weights(weights, wire_format=...)` 接收一个 **生成器**，逐个 yield `(name, tensor)`：

```python
async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None],
                         wire_format="named_tensors", **kwargs): ...
```

- `wire_format="named_tensors"`：传完整张量（默认）。
- `wire_format="delta_flush"`：只传增量（稀疏 delta，SGLang 专用），进一步省带宽。

用生成器而非一次性字典，是为了**边生成边传输、边释放**，避免把整份权重同时驻留显存。相关实现见 `vllm_rollout/weight_update_utils.py`、`bucketed_weight_transfer.py`（分桶传输）。

### 5.3 在主循环里的落点

第 4 篇 fit() 里的 `self.checkpoint_manager.update_weights(self.global_steps)` 以及生成前后的 `sleep_replicas()` / resume，就是这条链路的触发点。

## 6. 显存错峰：sleep / wake / offload

Actor（训练）、Critic、Ref、Rollout 常常 colocate 在同一组 GPU（第 2 篇）。显存不可能同时装下所有引擎的全部状态，verl 用“错峰”策略：

```
┌── 生成阶段 ─────────────────────────────┐
│ resume rollout 权重 + KV cache 到 GPU   │
│ 训练引擎的优化器状态 offload 到 CPU      │
│ → 生成                                   │
└─────────────────────────────────────────┘
          │ 生成完成
          ▼
┌── 训练阶段 ─────────────────────────────┐
│ release / sleep rollout（吐出显存）      │
│ 训练引擎权重/优化器状态 reload 回 GPU    │
│ → 前向/反向/更新                         │
└─────────────────────────────────────────┘
          │ 更新完成
          ▼
     update_weights → 新权重灌回 rollout
```

`BaseRollout.resume("weights"/"kv_cache")` 和 `release()` 就是这套机制的接口。vLLM 的 sleep/wake、FSDP 的 CPU offload 都服务于此。理解了这个“显存一会儿给生成、一会儿给训练”的错峰，就理解了 verl 能在有限 GPU 上塞下这么多模型的关键。

## 7. 一个完整 step 的引擎视角

把前五篇串成一条时间线（单机单组 GPU，colocate）：

```
[t0] resume rollout 权重 → vLLM 就绪
[t1] generate_sequences：vLLM 高速解码出 n 条回答        （推理引擎）
[t2] release rollout / sleep，吐出显存
[t3] reload 训练引擎 → compute_log_prob / ref / values  （训练引擎）
[t4] driver 本地算 reward + advantage                   （driver）
[t5] update_critic / update_actor：反向传播更新参数      （训练引擎）
[t6] ShardingManager 转换布局 → update_weights 灌回 vLLM
[t7] 下一个 step，回到 t0（vLLM 已是最新策略）
```

- t1 用**推理引擎**，t3/t5 用**训练引擎**，t6 做**布局转换 + 权重同步**；
- t0/t2 的 resume/release 是**显存错峰**；
- 全程数据用 **DataProto** 承载，调用用 **WorkerGroup** 分发——正好把整个系列串起来。

## 8. 小结

- Rollout 之所以单独成体系，是因为“生成”适合用 vLLM/SGLang，而“训练”适合用 FSDP/Megatron，两者负载与分片方式都不同。
- `BaseRollout` 用 `resume / generate / release / update_weights` 四个方法，把“生成 + 显存管理 + 权重同步”抽象得很干净；后端通过注册表可插拔。
- **权重同步（ShardingManager + update_weights 生成器 + 分桶/增量传输）** 和 **显存错峰（sleep/wake/offload）** 是 verl 工程上最核心、最能体现 HybridEngine 价值的两块。
- server/async 模式让多轮 Agent、工具调用这类复杂生成成为可能。

## 系列回顾

| 篇 | 回答的问题 |
|----|-----------|
| [01](/blog/verl-01-architecture/) | verl 是什么、整体架构与数据流 |
| [02](/blog/verl-02-hybridflow-single-controller/) | 复杂 RL 数据流怎么写成几行代码（HybridFlow / WorkerGroup / Dispatch） |
| [03](/blog/verl-03-dataproto/) | 数据怎么统一承载与流转（DataProto） |
| [04](/blog/verl-04-ppo-grpo-core/) | PPO/GRPO 具体怎么算（fit 主循环 + core_algos） |
| 05 | 训练与推理两套引擎怎么协同（Rollout + 权重同步 + 显存错峰） |

至此，verl 的核心脉络——**single-controller 编排、DataProto 数据总线、分布式分发、RL 算法、训练推理混合引擎**——就完整地串起来了。接下来可以挑一个 `examples/` 脚本实际跑一遍，对照这五篇把每个环节落到实处。
