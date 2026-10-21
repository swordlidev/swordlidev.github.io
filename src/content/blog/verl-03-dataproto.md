---
title: "verl 源码解读（三）：DataProto 数据协议"
description: "verl 里贯穿全程的数据总线 DataProto：batch / non_tensor_batch / meta_info 三段式结构，以及 chunk / concat / union / select / repeat 五个核心操作如何支撑分布式分发与 RL 多阶段数据流。"
pubDate: 2026-02-13
tags: ["verl", "强化学习", "源码解读", "LLM"]
---

> **本文是《verl 源码解读》系列第 3 篇**。第 2 篇（[HybridFlow 与单控制器](/blog/verl-02-hybridflow-single-controller/)）讲了 WorkerGroup 如何把数据“切分下发、聚合回收”——而能被这样切分和拼接的前提，是所有数据都装在一个统一的容器里：**DataProto**。本篇深入 `verl/protocol.py`。
>
> 系列目录：[1 架构](/blog/verl-01-architecture/) · [2 HybridFlow](/blog/verl-02-hybridflow-single-controller/) · **3 DataProto** · [4 PPO/GRPO](/blog/verl-04-ppo-grpo-core/) · [5 Rollout 混合引擎](/blog/verl-05-rollout-hybrid-engine/)

## 1. 为什么需要 DataProto

RL 训练里流动的数据非常“杂”：

- **张量**：`input_ids`、`attention_mask`、`responses`、`log_probs`、`advantages`、`values`……形状各异但第 0 维都是 batch。
- **非张量**：`uid`（每条样本的唯一 id）、原始问题文本、数据来源、ground-truth 答案……这些是字符串/对象，塞不进一个规整的张量。
- **元信息**：`temperature`、`global_steps`、各种开关标志等，对整个 batch 全局有效，不随样本切分。

如果用裸 `dict[str, Tensor]` 传来传去，很快就会遇到：怎么统一切分 batch？非张量怎么切？哪些是全局信息不该被切？不同阶段产生的新字段怎么合并回去？

**DataProto 就是把这三类数据统一打包的协议容器**，并提供一套 `chunk / concat / union / select / repeat` 等操作，让“分布式分发”和“数据流拼装”都有明确语义。

## 2. DataProto 的三段式结构

```python
@dataclass
class DataProto:
    batch: TensorDict = None          # 张量数据（基于 tensordict.TensorDict）
    non_tensor_batch: dict = field(default_factory=dict)   # 非张量，np.array(dtype=object)
    meta_info: dict = field(default_factory=dict)          # 全局元信息
```

| 字段 | 装什么 | 切分行为 |
|------|--------|----------|
| `batch` | 第 0 维对齐的张量，用 `TensorDict` 存 | 沿 batch 维切分 |
| `non_tensor_batch` | 每条样本一个对象，存成 `np.ndarray(dtype=object)` | 沿 batch 维切分 |
| `meta_info` | 全局标量/配置 | **不切分**，原样复制 |

关键约束（见 `check_consistency`）：`batch` 和 `non_tensor_batch` 的**第 0 维长度必须一致**（都等于 batch size），而且只支持 `num_batch_dims=1`。`non_tensor_batch` 里的值必须是 `dtype=object` 的 numpy 数组。

`__len__` 就是 batch size：优先取 `batch` 的 `batch_size[0]`，否则取任一 `non_tensor_batch` 数组的长度。

## 3. 构造：从 dict 到 DataProto

训练循环里最常见的构造方式：

```python
# 从 dataloader 出来的一个 dict 直接构造
batch: DataProto = DataProto.from_single_dict(batch_dict)
```

`from_single_dict` 会自动判断 dict 里每个值是张量还是 numpy 数组，分别放进 `batch` 和 `non_tensor_batch`，再调用 `from_dict`。`from_dict` 负责把张量们组装成一个 `TensorDict`，并校验长度一致。

还有 `from_dict(tensors=..., non_tensors=..., meta_info=...)` 的显式构造方式，以及 `auto_padding` 开关（配合第 2 篇讲的自动 padding 分发）。

## 4. 五个核心操作

这几个操作撑起了整个 RL 数据流，务必理解。

### 4.1 chunk：切分（分发的基础）

```python
chunks = data.chunk(chunks=world_size)   # 返回 list[DataProto]
```

沿 batch 维把 `batch` 和 `non_tensor_batch` 都切成 `chunks` 份，`meta_info` 原样复制给每一份。第 2 篇里 `DP_COMPUTE_PROTO` 的 dispatch 就是靠它（外加 auto-padding）把数据发给各 worker。

### 4.2 concat：拼接（聚合的基础）

```python
merged = DataProto.concat([dp0, dp1, ...])
```

把多个 DataProto 沿 batch 维拼回一个。worker 返回结果后的 collect 就是靠它。`chunk` 和 `concat` 互为逆操作。

### 4.3 union：按列合并新字段 ★

这是 RL 数据流最精妙的一步。训练中不同阶段会往同一批样本上**追加新列**：

```
rollout 产出 responses ──► union 回 batch
compute_log_prob 产出 old_log_prob ──► union
compute_ref 产出 ref_log_prob ──► union
compute_values 产出 values ──► union
reward 产出 token_level_scores ──► union
compute_advantage 产出 advantages / returns ──► union
```

```python
batch = batch.union(gen_batch_output)   # 把新算出的列合并进来
```

`union` 要求两边 batch size 相同，且不能有冲突的 key（相同 key 必须值相同）。底层用 `union_tensor_dict` / `union_numpy_dict` 分别合并张量和非张量部分。可以把一个 batch 想象成一张“宽表”，每个计算阶段往右边加几列。

### 4.4 select / pop：选列与弹出

```python
gen_batch = batch.select(batch_keys=["input_ids", "attention_mask", "position_ids"],
                         non_tensor_batch_keys=["raw_prompt"])
```

`select` 取出子集（比如生成只需要 prompt 相关的列，不需要带上 advantages）；`pop` 则是取出并从原对象删除某些列（用完就清理，省显存）。

### 4.5 repeat：按样本复制 ★（GRPO 的关键）

GRPO/PPO 要对每个 prompt 采样 `n` 条回答，就靠 `repeat`：

```python
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
```

`interleave=True` 时，样本排列是 `[p0, p0, p0, p1, p1, p1, ...]`（同一 prompt 的 n 条相邻），这对 GRPO 的“按组归一化 advantage”非常重要——同一个 prompt 的 n 条回答构成一个 group。

## 5. 它如何支撑分布式与 Ray

### 5.1 序列化

DataProto 实现了 `__getstate__` / `__setstate__`：张量部分会先 `consolidate` 再按需做 CPU 序列化，非张量和 meta_info 直接走 pickle。这样它才能作为 Ray RPC 的参数在进程/节点间传输。

### 5.2 DataProtoFuture：异步与 overlap

`protocol.py` 里还有 `DataProtoFuture`。当 `@register(..., blocking=False)` 时，worker 方法立刻返回 future 而非真实数据，driver 可以继续发起下一个调用，等真正需要时再 `.get()`。这让“生成”和“训练”等阶段可以部分重叠，提升吞吐。第 2 篇装饰器里的 `materialize_futures` / `_materialize_futures` 就是在调用前把 future “兑现”成真实数据。

### 5.3 padding 工具

`pad_dataproto_to_divisor` / `unpad_dataproto` 把 batch 补齐到某个除数的整数倍，再在聚合后去掉。配合第 2 篇的自动 padding，解决“batch 不整除 GPU 数”的问题。

## 6. 在训练循环里的真实轨迹

对照第 1 篇的数据流，用 DataProto 的视角再走一遍：

```python
# 1. 取 prompt
batch = DataProto.from_single_dict(batch_dict)
batch.non_tensor_batch["uid"] = np.array([uuid4() ...])      # 加非张量列
batch.meta_info["temperature"] = ...                         # 加全局信息

# 2. 选出生成需要的列，并按 n 复制
gen_batch = self._get_gen_batch(batch)          # select
gen_batch_output = gen_batch.repeat(n, interleave=True)      # repeat

# 3. 生成（分布式，DP_COMPUTE_PROTO：内部 chunk → RPC → concat）
gen_batch_output = actor_rollout_wg.generate_sequences(gen_batch_output)

# 4~6. 依次算 old_log_prob / ref / reward，并 union 回主 batch
batch = batch.union(gen_batch_output)
old_log_prob = actor_rollout_wg.compute_log_prob(batch)
batch = batch.union(old_log_prob)
# ... ref / values / reward 同理，不断往宽表加列 ...

# 7. 在 driver 本地算 advantage，再 union
batch = compute_advantage(batch, ...)

# 9. 更新 actor（同样是分布式分发）
actor_rollout_wg.update_actor(batch)
```

可以看到：**DataProto 像一条贯穿始终的“数据总线”，每个阶段从上面取列、往上面加列；而 chunk/concat 让它能无缝地在分布式 worker 间切分与聚合。**

## 7. 小结

- DataProto = `batch`（张量）+ `non_tensor_batch`（非张量对象）+ `meta_info`（全局信息）三段式容器。
- 核心操作：`chunk`（切分）/`concat`（拼接）是分布式分发的基础；`union`（加列）组织 RL 多阶段数据流；`repeat`（复制）支撑 GRPO 分组采样；`select`/`pop` 控制显存与传输量。
- 它可序列化、支持 `DataProtoFuture` 异步、内置 padding 工具，是 HybridFlow 分发机制能工作的前提。

有了数据总线（DataProto）和分发机制（WorkerGroup/Dispatch），下一篇我们看真正的“剧本”：PPO/GRPO 主循环与核心算法。
