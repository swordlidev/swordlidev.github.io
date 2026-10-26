---
title: "slime 学习笔记（二）：从 prompt 到梯度的完整链路"
description: "跟着一条 Sample 的生命周期走一遍 slime 内部：数据契约、rollout 的超采样与动态过滤、Data Buffer 的张量化与 DP 切分、Megatron Actor 的五步训练、advantage 与 loss、权重同步四步协议，以及 colocate 下的 offload。"
pubDate: 2026-08-12
tags: ["slime", "强化学习", "RLHF", "后训练", "Megatron", "SGLang", "源码解读"]
---

> **本文是《slime 学习笔记》系列第 2 篇**。[第 1 篇](/blog/slime-01-getting-started/) 讲了框架有哪些模块、怎么摆 GPU、怎么启动；本文往里钻一层，讲**模块内部怎么运转**。算法参数的深入解读在[第 3 篇](/blog/slime-03-algorithm-design/)，定制接口的用法在[第 4 篇](/blog/slime-04-search-agent-rl/)。
>
> 系列目录：
> 1. [入门与整体架构](/blog/slime-01-getting-started/)
> 2. **从 prompt 到梯度的完整链路**（本文）
> 3. [算法层：参数、Advantage 与 Loss 的设计](/blog/slime-03-algorithm-design/)
> 4. [实战：从零训一个 Search Agent](/blog/slime-04-search-agent-rl/)

## 一、一条 Sample 的生命周期

理解 slime 最好的切入点是跟着一条数据走一遍。整条链路横跨三个 Ray actor 和两次格式转换：

```text
sequenceDiagram
    participant DS as DataSource
    participant RM as RolloutManager
    participant SG as SGLang Router
    participant BB as BatchBuilder
    participant TA as MegatronTrainRayActor

    RM->>DS: get_samples(over_sampling_batch_size)
    DS-->>RM: list[list[Sample]]（按 group 组织）
    loop 每个 group 一个 asyncio task
        RM->>SG: POST /generate
        SG-->>RM: tokens + logprobs
        RM->>RM: reward 计算 / 动态过滤
    end
    RM->>BB: convert(samples)
    BB->>BB: reward 后处理 → 张量化
    BB->>BB: split_by_dp（按训练并行布局切分）
    BB-->>TA: 每个 DP rank 的数据引用
    TA->>TA: log_prob → advantage → forward/backward
    TA->>SG: update_weights（权重回灌）
```

两个关键转换点：**`SGLang Router` 之后**数据从 prompt 变成了带 token 和 reward 的完整轨迹；**`BatchBuilder` 之后**数据从 Python 对象变成了按 DP rank 切好的张量。

## 二、数据契约：Sample

`slime/utils/types.py` 的 `Sample` 是贯穿全局的唯一数据结构。字段很多，按用途分组后其实很清晰：

| 分组 | 字段 | 说明 |
| :--- | :--- | :--- |
| **身份** | `index`、`group_index`、`rollout_id`、`session_id` | `group_index` 标识同一 prompt 的采样组；`rollout_id` 决定 loss 聚合时算几次 rollout |
| **输入** | `prompt`、`tokens`、`label`、`multimodal_inputs`、`metadata` | `prompt` 可以是字符串或 messages 列表 |
| **输出** | `response`、`response_length`、`reward` | `reward` 可以是标量，也可以是 dict（配合 `--reward-key` 取值） |
| **训练信号** | `loss_mask`、`rollout_log_probs`、`weight_versions`、`teacher_log_probs` | `loss_mask` 决定哪些 token 参与 loss |
| **状态** | `status`、`remove_sample` | 五种状态，见下 |
| **定制钩子** | `generate_function_path`、`custom_rm_path` | 允许**单条样本**覆盖全局的生成/奖励函数 |

`Sample.Status` 是**五态**枚举，不是常见的三态：

| 状态 | 含义 |
| :--- | :--- |
| `PENDING` | 尚未生成 |
| `COMPLETED` | 正常结束（命中 stop） |
| `TRUNCATED` | 达到长度上限被截断 |
| `ABORTED` | 被中止（权重即将更新、或整批已凑够） |
| `FAILED` | 可恢复的非致命失败（工具调用失败、外部 API 错误、解析错误）。与 `ABORTED` 不同，`FAILED` 可能仍带部分有效输出 |

写自定义生成函数时，拼接 token 的推荐方式不是手动改 `tokens`，而是调用 `Sample.append_response_tokens`：

```python
sample.append_response_tokens(
    args, tokens=..., log_probs=..., trainable=True,
    meta_info=..., text=..., update_terminal_info=True,
)
```

`trainable` 参数直接决定这段 token 的 `loss_mask` 是 1 还是 0。这在多轮场景里至关重要——模型自己生成的要训练，工具返回的内容不能训练。

## 三、Rollout 侧：数据怎么生出来

### 三段式职责

`slime/ray/rollout.py` 的 `RolloutManager._generate` 只做三件事：调生成函数拿数据 → 记指标 → 交给 `BatchBuilder` 转换。生成逻辑本身在 `slime/rollout/sglang_rollout.py`，默认实现是 `generate_rollout`。

### 超采样循环

`generate_rollout_async` 的主循环是理解 rollout 侧的核心：

```python
while len(data) < rollout_batch_size:
    while state.remaining_batch_size < rollout_batch_size:
        samples = data_source(args.over_sampling_batch_size)   # 多取一些
        state.submit_generate_tasks(samples)                   # 每 group 一个 task
    done, pendings = await asyncio.wait(pendings, return_when=FIRST_COMPLETED)
    for task in done:
        group = task.result()
        if 动态过滤判定该丢:          # 例如整组全对或全错，无梯度信号
            continue
        data.append(group)
aborted = await abort(...)        # 多余的请求中止掉
```

```text
flowchart TD
    A["取 over_sampling_batch_size 个 prompt group"] --> B["提交异步生成任务"]
    B --> C["asyncio.wait<br/>FIRST_COMPLETED"]
    C --> D{"动态过滤<br/>这组有用吗？"}
    D -- "丢弃" --> E["remaining_batch_size -= 1"]
    E --> F{"已凑够 rollout_batch_size？"}
    D -- "保留" --> G["加入 data"]
    G --> F
    F -- "否" --> B
    F -- "是" --> H["abort 剩余请求"]
```

这里的设计要点：

- **`--over-sampling-batch-size` 决定采样粒度**。不设时默认等于 `--rollout-batch-size`，且框架强制要求它 ≥ `rollout_batch_size`。配合 `--dynamic-sampling-filter-path` 使用——过滤会丢掉一部分组，超采样保证还能凑够一个完整 batch。
- **流式收集而非等齐**。用 `FIRST_COMPLETED` 而不是 `gather`，哪个组先完成就先处理，天然缓解长尾。
- **凑够就中止**。`data` 满了之后剩下的请求是浪费，直接 abort。

### 并发控制

`GenerateState` 是个单例（`SingletonMeta`），集中持有整个生成过程的全局状态：tokenizer、processor、`asyncio.Semaphore`（总并发上限 = `sglang_server_concurrency × 引擎数`）、`AsyncPacer`（节流器）、`sampling_params`，以及 `dp_counts` 做 DP rank 均衡——`dp_rank_context()` 每次挑当前负载最低的 rank，避免请求倾斜。

### abort 与 partial rollout

`abort()` 做两件事：取消可取消的 task，然后对还在服务端跑的请求调 `abort_servers_until_idle` 逼停引擎。

如果开了 `--partial-rollout`，被中止的**半成品不会浪费**：已生成的部分会被收回 buffer（`RolloutDataSourceWithBuffer.add_samples`），并在 `metadata["start_rollout_id"]` 里记下它是哪一轮开始的，下一轮接着往后生成。

代价是这条轨迹的前半段是旧策略产出的。是否把这部分从 loss 里剔除，由 `--mask-offpolicy-in-partial-rollout` 控制——开启后只有本轮 on-policy 生成的 token 参与训练。

### Reward 层

奖励计算在 `slime/rollout/rm_hub/`，有两种模式：

- **逐样本**：`async_rm(args, sample)`，每条独立算分。规则类奖励（数学答案匹配等）走这条。
- **整组**：开 `--group-rm` 后走 `batched_async_rm(args, group)`，一次拿到整组再算。适合需要组内对比的奖励（排序、相对打分）。

## 四、Data Buffer：从 Sample 到张量

### 数据源契约

`slime/data/data_source.py` 的 `DataSource` 抽象基类只要求五个方法：

```python
get_samples(num_samples) -> list[list[Sample]]   # 按 group 返回
add_samples(samples)                             # 回填（partial rollout 用）
save(rollout_id) / load(rollout_id)              # 游标持久化
__len__()                                        # 数据集长度
```

内置实现 `RolloutDataSource` 维护 `epoch_id` / `sample_offset` / `sample_index` 三个游标，`get_samples` 按 `--n-samples-per-prompt` 把一个 prompt 深拷贝成一个 group。`RolloutDataSourceWithBuffer` 在它之上加了 buffer，优先从 buffer 取（`pop_first`，或开 `--buffer-sort-by-staleness` 后用 `pop_oldest` 优先消费最旧的）。

有个细节值得学习：`state_dict()` 在快照游标前会**先暂停所有 consumer**。因为后台 worker 可能已经预取了一批 prompt，如果不暂停就快照，恢复时这批样本会被跳过。

### 转换与切分

`slime/data/batch_builder.py` 的 `BatchBuilder` 负责两步：

1. **`convert(samples)`** —— reward 后处理（`_post_process_rewards`，可被 `--custom-reward-post-process-path` 覆盖）→ 把 token / mask / logprob 张量化。整个转换也可以被 `--custom-convert-samples-to-train-data-path` 整体替换。
2. **`split_by_dp(data)`** —— 按训练侧上报的并行布局（`dp_size` / `cp_size` / `vpp_size`）切分，内部调 `slime/utils/dp_schedule.py` 的 `build_dp_schedule` 生成每个 rank 的 microbatch 调度。

切分策略受两个开关影响：

- `--balance-data`：用 Karmarkar-Karp 算法在各 DP rank 间均衡估算的训练 FLOPs，避免某个 rank 拿到一堆长序列成为木桶短板；
- `--use-dynamic-batch-size` + `--max-tokens-per-gpu`：按 token 总量而非固定样本数打包 microbatch。

### 两种传输后端

| 后端 | 开关 | 机制 | 适用 |
| :--- | :--- | :--- | :--- |
| **object-store** | 默认 | 数据放 Ray 对象存储，传引用 | 常规规模 |
| **straw** | `--rollout-data-transport straw` | 数据落共享存储（如 JuiceFS），Ray 只传控制消息与 manifest | 超大规模；支持崩溃后重放已接收的 batch |
| **nixl** | `--rollout-data-transport nixl` | 启用 Ray tensor transport | 特定硬件链路 |

straw 模式下 `BatchBuilder.begin()` 会先算一个 **plan digest**（把选样索引、转换配置、并行布局哈希在一起），用于校验重试的一致性：普通重试必须命中同一个 plan，而 trainer 重放允许改并行度但复用原 plan 的身份。

## 五、训练侧：Megatron Actor 的五步

`slime/ray/actor_group.py` 的 `RayTrainGroup` 只是个广播代理——`async_train` 对每个 rank 调一次 `actor.train.remote()`，拿回一组 Ray ref。真正的逻辑在 `slime/backends/megatron_utils/actor.py` 的 `MegatronTrainRayActor.train_actor`，顺序**严格固定**：

```python
data_iterator = get_data_iterator(rollout_data)

# ① ref 模型前向（仅当需要 KL）
if "ref" in weights_backuper.backup_tags:
    _switch_model("ref"); rollout_data["ref_log_probs"] = compute_log_prob(...)

# ② teacher 模型前向（仅 on-policy distillation）
if "teacher" in weights_backuper.backup_tags:
    _switch_model("teacher"); rollout_data["teacher_log_probs"] = compute_log_prob(...)

# ③ actor 旧策略 log_prob
_switch_model("actor"); rollout_data["log_probs"] = compute_log_prob(...)

# ④ 算 advantage（必须在训练前，因为可能要全局归一化）
compute_advantages_and_returns(args, rollout_data)

# ⑤ 真正训练
train(rollout_id, model, optimizer, scheduler, data_iterator, ...)
```

两个值得专门拿出来讲的优化：

**权重切换省显存。** actor / ref / teacher 三个模型**共用同一份模型结构**。`TensorBackuper` 把各自的权重备份在 CPU，`_switch_model(tag)` 做的是把对应权重搬回 GPU。这意味着加一个 ref 模型只增加 CPU 内存，不增加显存中的模型副本。代价是切换时的 H2D 拷贝。

**跳过冗余前向。** 第 ③ 步的 actor log_prob 前向，在特定条件下可以省掉——`can_reuse_log_probs_in_loss` 判断：单 microbatch、`loss_type == "policy_loss"`、`kl_coef == 0`、未开 `use_rollout_logprobs`、未取 mismatch 指标、无 critic、无 OPD、且 estimator 不是 `gspo`。满足时直接在训练 forward 里把 log_prob 当作旧策略值复用，省一次全量前向。条件这么苛刻，是因为任何需要"训练前就拿到旧 log_prob"的特性都会破坏这个复用。

## 六、Advantage 与 Loss

### 六种 advantage estimator

`slime/backends/megatron_utils/loss.py` 的 `compute_advantages_and_returns` 支持：

| `--advantage-estimator` | 机制 | 是否需要 critic |
| :--- | :--- | :-- |
| `grpo` | 组内相对优势 | 否 |
| `gspo` | 序列级 KL 的 GRPO 变体 | 否 |
| `cispo` | 单侧裁剪的重要性采样 | 否 |
| `ppo` | GAE，`get_advantages_and_returns_batch` | **是**（自动开启） |
| `reinforce_plus_plus` | 带折扣的 return | 否（但必须开 `--normalize-advantages`） |
| `reinforce_plus_plus_baseline` | 带 baseline 的变体 | 否（同上） |

默认 `grpo`。也可以用 `--custom-advantage-function-path` 整体替换。

### 策略损失的构成

`policy_loss_function` 最终的 loss 是多项叠加，每一项都可独立开关：

| 组成项 | 控制参数 |
| :--- | :--- |
| 策略梯度主项（ppo / gspo / cispo / reinforce 四种 `pg_loss_type`） | `--pg-loss-type`、`--eps-clip`、`--eps-clip-high`、`--eps-clip-c` |
| 熵正则 | `--entropy-coef` |
| KL 损失 | `--use-kl-loss`、`--kl-loss-coef`、`--kl-loss-type` |
| TIS / OIS 离策略校正 | `--use-tis`、`--tis-clip`、`--tis-clip-low`，或 `--custom-tis-function-path` |
| score centering | `--use-score-centering`、`--score-centering-top-k` |
| OPSM 掩码 | `--use-opsm` |

另有 `value_loss_function`（critic）和 `sft_loss_function`（SFT 模式），以及 `--custom-loss-function-path` 完全自定义。

### 一个容易错的分布式细节

开 `--normalize-advantages` 时，whitening 必须在 **DP × CP 组**上做（`mpu.get_data_parallel_group(with_context_parallel=True)`），不能用排除 CP 的组。原因是 context parallel 下每个 CP rank 只持有序列的一部分（zigzag 切分），如果各自用自己的均值方差归一化，同一条序列的两半会被施加**不同的仿射变换**。

代码里还有一处注释特别值得留意：这个 collective 必须**无条件执行**。因为某个 CP rank 可能一个 response token 都不持有（prompt 太长，它的两个 chunk 全落在 prompt 里），如果在那个 rank 上跳过 all_reduce 就会死锁。`distributed_masked_whiten` 对空张量的处理是贡献 0 到归约和里。

## 七、权重同步

`slime/backends/megatron_utils/update_weight/` 下有四种实现，由 `--update-weight-mode`（`full`/`delta`）× `--update-weight-transport`（`nccl`/`disk`）组合选择：

| 文件 | 适用场景 |
| :--- | :--- |
| `update_weight_from_tensor.py` | colocate，训练与推理在同一张卡，直接拷张量 |
| `update_weight_from_distributed.py` | 分离部署，NCCL 广播（默认 transport） |
| `update_weight_from_disk.py` | 跨集群 / 外部引擎，写完整 HF checkpoint 后 reload |
| `update_weight_from_disk_delta.py` | 只传相对上一版变化的权重字节 |

同步不是简单写一下内存，而是一套四步协议（见 `RayTrainGroup._reload_rollout_weights_from_disk`）：

```text
pause_generation(mode="abort" | "in_place")   # 冻结生成
  → flush_cache()                             # 必要时清 KV / prefix cache
  → update_weights_from_disk(...)             # 装载新权重
  → continue_generation()                     # 恢复
```

`mode` 的选择取决于是否要刷缓存（由 `--flush-cache-interval` 决定）：要刷就 `abort`（丢弃在途请求），不刷就 `in_place`（原地暂停，请求保留）。

权重格式从 Megatron 的切分布局转到 HF/SGLang 布局的逻辑在 `slime/backends/megatron_utils/megatron_to_hf/`，按模型族分文件。

## 八、offload：显存怎么腾出来

colocate 模式下训练和生成共享显卡，必须有人让位。slime 用 `torch_memory_saver` 实现：

- `--offload-train`：训练侧在生成期间把参数、优化器状态、激活从显存挪走。`MegatronTrainRayActor.sleep()` 做的事是 `clear_memory` → `destroy_process_groups()` → `torch_memory_saver.pause()`；`wake_up()` 反向操作并 `reload_process_groups()`。
- `--offload-rollout`：推理引擎侧分别有 `onload_weights()` / `onload_kv()`，可以只唤回权重而不唤回 KV cache——权重同步只需要前者，粒度更细、峰值更低。

注意 `--colocate` 会**强制**把这两个 offload 都打开。

一个工程细节：`sleep()` 里会销毁 NCCL 进程组，`wake_up()` 重建。PP > 2 时重建后要先 `dist.barrier()` 预热 WORLD 组，因为 Megatron 的 batched P2P 用的是默认组，而 PyTorch 要求该组上的首次 NCCL 操作必须全 rank 参与。

## 小结

把本文压缩成一张流程：

```text
DataSource.get_samples          → list[list[Sample]]（按 group）
generate_rollout_async          → 超采样 + 动态过滤 + 流式收集 + abort
rm_hub                          → reward（逐样本 or 整组）
BatchBuilder.convert            → reward 后处理 + 张量化
BatchBuilder.split_by_dp        → 按 DP/CP/VPP 布局切 microbatch
MegatronTrainRayActor.train_actor → ref/teacher/actor log_prob → advantage → train
update_weight/*                 → 四步协议把新权重推回引擎
```

记住三个设计选择就够了：**超采样 + 动态过滤**保证 batch 质量，**CPU 权重备份 + 模型切换**用内存换显存，**权重同步的暂停协议**保证轨迹不跨策略边界。

下一篇 [《slime 算法层：参数、Advantage 与 Loss 的设计》](/blog/slime-03-algorithm-design/) 把焦点收到算法层：Megatron 与 SGLang 的重点参数各自解决什么问题，以及 advantage 与 loss 到底在对付哪类难题。
