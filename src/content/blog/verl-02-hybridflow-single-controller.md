---
title: "verl 源码解读（二）：HybridFlow 编程模型与单控制器"
description: "深入 verl 的 single-controller 实现核心：WorkerGroup、@register 装饰器与 Dispatch 分发模式。看它如何把复杂的分布式 RL 数据流封装成 driver 侧几行同步代码。"
pubDate: 2026-02-08
tags: ["verl", "HybridFlow", "Ray", "分布式训练", "源码解读"]
---

> **本文是《verl 源码解读》系列第 2 篇**。第 1 篇（[入门与整体架构](/blog/verl-01-architecture/)）我们知道了 verl 用“single-controller 描述数据流 + multi-controller 执行计算”来组织 RL 训练。本篇深入这套机制的实现核心：`WorkerGroup`、`@register` 装饰器和 `Dispatch` 分发模式。对应代码在 `verl/single_controller/`。
>
> 系列目录：[1 架构](/blog/verl-01-architecture/) · **2 HybridFlow** · [3 DataProto](/blog/verl-03-dataproto/) · [4 PPO/GRPO](/blog/verl-04-ppo-grpo-core/) · [5 Rollout 混合引擎](/blog/verl-05-rollout-hybrid-engine/)

## 1. 为什么需要 HybridFlow

写过多卡分布式训练的人都知道两种典型范式：

- **单控制器（single-controller）**：一个 driver 发号施令，逻辑清晰，但如果每一步张量操作都走 driver，通信会成为瓶颈。
- **多控制器（multi-controller, SPMD）**：每个 rank 跑同一份代码（如 `torchrun`），高效，但要表达“多个模型交替计算”的复杂 RL 数据流会非常痛苦——你得手动协调谁先算、数据怎么在模型间传递。

HybridFlow 的思路是**把两者结合**：

- 用**单控制器**表达 RL 数据流的“剧本”（先生成、再打分、再更新……），逻辑直观；
- 每个计算密集的角色，用**多控制器（一组 SPMD worker）**去执行，高效；
- 中间的“数据如何从 driver 切分下发到 workers、结果如何聚合回来”，用**装饰器自动完成**，对用户透明。

于是 driver 代码看起来就像在调用一个普通对象的方法：

```python
output = self.actor_rollout_wg.generate_sequences(gen_batch)
```

但 `actor_rollout_wg` 其实是一组分布在多张 GPU 上的 worker。

## 2. 三个核心概念

### 2.1 Worker：单个计算单元

`verl/single_controller/base/worker.py` 中的 `Worker` 是所有角色 worker 的基类。一个 worker 通常对应一个 GPU 进程（一个 rank），持有 `rank` / `world_size` 等分布式信息。实际的 Actor、Critic 等 worker 都会继承它，并把自己的方法用 `@register` 标注。

### 2.2 ResourcePool：GPU 资源池

`worker_group.py` 中的 `ResourcePool` 描述“有多少节点、每个节点多少进程”：

```python
class ResourcePool:
    def __init__(self, process_on_nodes=None, max_colocate_count=10, n_gpus_per_node=8):
        self._store = process_on_nodes      # 如 [8, 8] 表示两个节点各 8 进程
    @property
    def world_size(self):
        return sum(self._store)             # 总进程数
```

它还负责计算每个进程的 `local_rank`、`local_world_size`，为 colocate（多个模型共享同一组 GPU）提供基础。

### 2.3 WorkerGroup：一组 worker 的“遥控器”

`WorkerGroup` 把一组 worker 封装成一个对象。driver 调用 `wg.some_method(data)` 时，`WorkerGroup` 会：

1. 根据 `some_method` 注册的 **dispatch 模式**，把输入数据 `data` 切分/复制成 `world_size` 份；
2. 通过 Ray RPC 把每一份发给对应的 worker 执行；
3. 根据 **collect 模式**，把所有 worker 的返回值聚合成一个结果返回给 driver。

Ray 后端实现是 `verl/single_controller/ray/`，核心类有 `RayWorkerGroup`、`RayClassWithInitArgs`、`ResourcePoolManager`。

## 3. @register：让方法“可被分发调用”

这是 HybridFlow 最优雅的部分。看 `decorator.py` 中的 `register`：

```python
def register(dispatch_mode=Dispatch.ALL_TO_ALL,
             execute_mode=Execute.ALL,
             blocking=True,
             materialize_futures=True):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)
        # 把分发信息挂到函数上（用一个魔法属性名避免冲突）
        attrs = {"dispatch_mode": dispatch_mode,
                 "execute_mode": execute_mode,
                 "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper
    return decorator
```

它做的事情其实很简单：**给 worker 的方法打上一个“标签”（`MAGIC_ATTR`），记录这个方法应该用哪种 dispatch/collect 策略、是否阻塞**。

一个典型的 worker 方法长这样：

```python
class ActorRolloutRefWorker(Worker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # 这里是在单个 rank 上运行的逻辑
        ...
```

当 `WorkerGroup` 被构造时，它会扫描 worker 类上所有带 `MAGIC_ATTR` 的方法，**动态地在自己身上生成同名方法**。于是 driver 就能直接 `wg.generate_sequences(...)`，而分发/聚合逻辑由标签决定。

- `dispatch_mode`：输入如何切分下发（dispatch）+ 输出如何聚合回收（collect）。
- `execute_mode`：在所有 rank 执行（`Execute.ALL`）还是只在 rank0（`Execute.RANK_ZERO`）。
- `blocking`：是否等待结果（同步）；非阻塞可返回 future，实现 overlap。

## 4. Dispatch：几种关键的分发模式

`Dispatch` 是一个动态枚举（`DynamicEnum`），预注册了这些模式：

```python
Dispatch.register("RANK_ZERO")
Dispatch.register("ONE_TO_ALL")
Dispatch.register("ALL_TO_ALL")
Dispatch.register("DP_COMPUTE")
Dispatch.register("DP_COMPUTE_PROTO")
Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
Dispatch.register("DP_COMPUTE_METRIC")
Dispatch.register("DIRECT_ROLLOUT_METHOD")
```

每种模式对应一对 `(dispatch_fn, collect_fn)`。理解最常用的几个就够了：

### 4.1 ONE_TO_ALL：广播

把同一份参数复制给所有 worker。适合“初始化模型”这类所有 rank 都要执行且输入相同的操作。

```python
def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs
```

### 4.2 ALL_TO_ALL：原样透传

不做切分，原样传给所有 worker，也原样收集。是默认模式，适合参数已经自己处理好分发的情况。

### 4.3 DP_COMPUTE_PROTO：数据并行的核心模式 ★

这是 RL 中最常用的模式。它把一个 `DataProto` 沿 batch 维切成 `world_size` 份，每个 worker 处理一份，最后把结果 `concat` 回来：

```python
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    # 自动 padding 到 world_size 的整数倍，再 chunk
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size, *args, **kwargs)
    return splitted_args, splitted_kwargs

def collect_dp_compute_data_proto(worker_group, output):
    # 把各 worker 的 DataProto 结果按 batch 维拼接
    assert BatchData(output).is_concatable()
    return _concat_data_proto_or_future(output)
```

注意 `_split_args_kwargs_data_proto_with_auto_padding`：当 batch 不能被 `world_size` 整除时会**自动 padding**，保证每个 worker 拿到等长的数据；聚合后再 `unpad` 掉。这解决了“数据量不整除 GPU 数”的常见麻烦，对用户完全透明。

`generate_sequences`、`compute_log_prob`、`compute_values`、`update_actor` 这些方法基本都用 `DP_COMPUTE_PROTO`。

### 4.4 ND_COMPUTE / LAZY：支持更复杂的并行

`dispatch_nd_compute` / `dispatch_lazy_compute_data_proto` 用于 Megatron 这类同时存在 TP/PP/DP 的 N 维并行场景：数据只需按 **DP（数据并行）维度**切分，而同一个 DP 组内的多个 rank（TP/PP）应拿到**相同**的数据分片。它通过 `dp_rank_mapping`（每个全局 rank 对应哪个 DP rank）来完成这种“按 DP 分、组内复制”的分发，并用 `collect_mask` 决定哪些 rank 的输出需要回收（避免 TP/PP 组内重复收集）。

## 5. 把它们串起来：一次调用的完整生命周期

以 `wg.compute_log_prob(batch)` 为例（`DP_COMPUTE_PROTO`）：

```
driver 侧:
  wg.compute_log_prob(batch)             # batch: DataProto, len = 256
        │
        ▼ dispatch_fn（自动 padding 到 world_size 整数倍后 chunk）
  [shard0, shard1, ..., shard7]          # world_size=8，每份 32 条
        │  Ray RPC 并行下发
        ▼
  worker0.compute_log_prob(shard0)  ...  worker7.compute_log_prob(shard7)
        │  各自返回带 log_prob 的 DataProto
        ▼ collect_fn（concat）
  一个完整的 DataProto（len=256，含 old_log_prob）
        │ unpad
        ▼
  返回给 driver
```

driver 只写了一行 `wg.compute_log_prob(batch)`，中间的切分、并行 RPC、聚合、padding/unpadding 全部由 `@register(DP_COMPUTE_PROTO)` 自动完成。**这就是 HybridFlow “几行代码写 RL 数据流”的底层魔法。**

## 6. Colocate：多个模型共享 GPU

RL 里 Actor、Ref、Critic 等多个模型往往要放在同一组 GPU 上（否则卡不够）。verl 用 `create_colocated_worker_cls`（`single_controller/ray/base.py`）把多个 worker 类“融合”进同一组 Ray actor 进程，从而共享物理 GPU，并通过参数/显存的 offload 与唤醒来错峰使用显存。`ResourcePool` 的 `max_colocate_count` 就是为此服务的。

## 7. 小结

- **WorkerGroup = 一组 worker 的遥控器**；driver 调它的方法，就等于对一组 GPU 下发分布式任务。
- **`@register(dispatch_mode=...)` 给 worker 方法贴标签**，WorkerGroup 据此自动生成同名方法并处理分发/聚合。
- **`Dispatch` 定义了切分与聚合策略**，其中 `DP_COMPUTE_PROTO`（数据并行 + 自动 padding + concat）是 RL 中最核心的一种。
- 这套机制让 driver 侧可以用**同步、直观**的代码表达复杂 RL 数据流，而底层的分布式复杂度被完全封装。

要让分发能“切分数据”，数据本身必须是可切分、可拼接的统一容器——这就是下一篇要讲的 **DataProto**。
