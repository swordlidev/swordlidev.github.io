---
title: "slime 学习笔记（一）：入门与整体架构"
description: "slime 是智谱 THUDM 开源、面向 RL scaling 的 LLM 后训练框架，把 Megatron 训练与 SGLang 生成接到同一套代码里。本系列第 1 篇建立整体认知：它解决什么问题、由哪三块组成、GPU 怎么摆、参数怎么分、以及怎么跑起第一个实验。"
pubDate: 2026-08-05
tags: ["slime", "强化学习", "RLHF", "后训练", "Megatron", "SGLang", "源码解读"]
---

> **本文是《slime 学习笔记》系列第 1 篇**。slime 是智谱 THUDM 团队开源、面向 RL scaling 的 LLM 后训练框架，把 Megatron 训练与 SGLang 生成真正接到同一套代码里，GLM-4.5 到 GLM-5.3 系列的 RL 后训练都跑在它上面。本系列自顶向下拆解它的骨架、内部机制、算法设计与实战落地。
>
> 系列目录：
> 1. **入门与整体架构**（本文）
> 2. [从 prompt 到梯度的完整链路](/blog/slime-02-core-pipeline/)
> 3. [算法层：参数、Advantage 与 Loss 的设计](/blog/slime-03-algorithm-design/)
> 4. [实战：从零训一个 Search Agent](/blog/slime-04-search-agent-rl/)

本篇目标是在 15 分钟内让你搞清楚：slime 要解决什么问题、它由哪几块组成、GPU 怎么分配、以及怎么跑起第一个实验。本文**不深入任何模块内部**——那是[第 2 篇](/blog/slime-02-core-pipeline/)的任务。

## 一、为什么 RL 后训练需要一个专门的框架

预训练和 SFT 的流程很简单：读数据、前向、反向、更新。RL 后训练不一样，它在训练循环里插入了一个**在线生成**环节——模型要先自己产出回答，拿到奖励，才能算梯度。这一下引入三个纯粹的系统问题：

**第一，双引擎割裂。** 训练需要的是 Megatron 这类支持 TP/PP/CP/EP 多维并行、带优化器状态的框架；生成需要的是 SGLang 这类带 continuous batching、PagedAttention、prefix cache 的推理引擎。两者的权重切分格式、内存布局、进程拓扑完全不同。

**第二，权重同步。** 每训一步，新权重必须推给推理引擎，否则生成的数据就是旧策略产出的（off-policy），算法效果打折。而权重动辄几百 GB，同步路径的选择直接决定端到端吞吐。

**第三，GPU 争抢。** 生成和训练都要吃满显卡。是各占一半（分离部署），还是共享同一批卡、靠显存腾挪轮流使用（colocate）？不同规模、不同集群下最优解不同。

很多项目的做法是把 trainer、rollout service、agent framework 拼成三套互不相通的系统。slime 的选择相反：**让训练、生成、数据三条路径收敛到同一套代码里**。

## 二、slime 是什么

slime 是一个面向 RL scaling 的 LLM 后训练框架，提供两个核心能力：

1. **高性能训练**——把 Megatron 和 SGLang 真正接起来，而不是各跑一套再手工搬数据；
2. **灵活的数据生成**——通过「传函数路径」的插拔式接口，支持任意 rollout workflow：多轮对话、工具调用、sandbox 执行、verifier 奖励。

它不是实验室 demo。GLM-4.5、GLM-4.6、GLM-4.7、GLM-5、GLM-5.1、GLM-5.2、GLM-5.3 系列的 RL 后训练都跑在 slime 上。围绕它还长出了一批独立系统：阿里 Accio 的 Dressage（黑盒 agent RL）、RadixArk 的 Miles（企业级特性）、vLLM 官方的 vime（把 rollout 后端换成 vLLM）、RedAI 的 Relax（全模态异步）等。

一句话概括设计哲学：**小到能读懂，但经过 SOTA 级模型完整训练循环的验证。**

## 三、三大模块

| 模块 | 实现 | 职责 |
| :--- | :--- | :--- |
| **training** | Megatron | 主训练流程。从 Data Buffer 读数据，训完把权重同步给 rollout |
| **rollout** | SGLang + router | 生成新数据（含奖励 / verifier 输出），写回 Data Buffer |
| **data buffer** | slime 自己实现 | 桥梁。管理 prompt 初始化、自定义数据、以及 rollout 生成方法 |

```text
flowchart LR
    DB["data buffer<br/>prompt 管理 / 样本转换"]
    RO["rollout<br/>SGLang + router"]
    TR["training<br/>Megatron"]

    DB -- "取 prompt" --> RO
    RO -- "写回带 reward 的 Sample" --> DB
    DB -- "训练 batch（按 DP 切分）" --> TR
    TR -- "权重同步" --> RO
```

值得留意的是 **data buffer 的定位**。它不只是一个队列，而是"生成方法的宿主"——你的 agent loop、工具调用、环境交互，最终都是以某个生成函数的形式挂在这条路径上，而不是另起一个 agent 框架。这是 slime 能同时服务于「前沿模型后训练」和「agentic RL」的关键。

默认的数据传输走 Ray 的 `object-store`。大规模场景可以切到 `--rollout-data-transport straw`，把 prompt 任务、rollout 续跑数据、训练 batch 都落到共享存储上，Ray 只传控制消息和引用（详见 [docs/zh/advanced/straw.md](https://github.com/THUDM/slime/blob/main/docs/zh/advanced/straw.md)）。

## 四、核心节律：一个 rollout_id = 一次完整 RL step

整个框架只有一个入口：仓库根目录的 `train.py`。它的 `train()` 函数就是全部节律所在，不到 60 行。把它理解了，slime 就不神秘了。

```python
# train.py 的 train() —— 伪代码
actor_model, critic_model = create_training_models(...)
actor_model.update_weights()          # 开训前先推一次初始权重，保证同策
rollout_manager.training_ready()

for rollout_id in range(start_rollout_id, num_rollout):
    rollout_data_ref = rollout_manager.generate(rollout_id)   # ① 生成
    offload_rollout_if_needed()                               # ② 让出显存
    actor_model.async_train(rollout_id, rollout_data_ref)     # ③ 训练
    rollout_manager.training_completed(rollout_id)
    save_checkpoint_if_due()                                  # ④ 周期存档
    offload_train_if_needed()
    rollout_manager.pause_rollout_admission()                 # ⑤ 冻结生成
    actor_model.update_weights()                              # ⑥ 权重同步
    rollout_manager.resume_rollout_admission()
    eval_if_due()                                             # ⑦ 周期评测
```

**`rollout_id` 是 slime 的时间轴。** 一个 `rollout_id` 对应一次完整的 RL step：生成一批数据 → 训练一次 → 可能存档 → 把新权重推给推理引擎。日志、checkpoint、数据源游标、权重版本全都以它为锚点。

几个容易忽略但很重要的细节：

- **开训前就要 `update_weights()`**。因为恢复训练时模型权重来自 checkpoint，而推理引擎可能还是旧的；不先推一次，第一批数据就是 off-policy 的。
- **权重同步前要 `pause_rollout_admission()`**。否则正在生成的请求会跨越权重边界，一条轨迹里混进两个策略的 token。
- **失败不等于销毁**。`main()` 用 `Cleanup()` 包裹整个过程，异常时只把 trainer 摘掉，**保留推理引擎和可重放的数据**，供手动重启。这是 slime 容错设计的基础。

如果 `--num-rollout` 没显式指定，slime 会用 `len(data_source) // rollout_batch_size` 算出每 epoch 的 step 数，再乘 `--num-epoch`（见 `slime/ray/rollout.py` 的 `get_num_rollout_per_epoch`）。

## 五、GPU 怎么摆：五种资源布局

这是 slime 最早需要做的决策，代码在 `slime/ray/placement_group.py` 的 `_get_placement_group_layout`。

| 模式 | 开关 | GPU 占用 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **colocate** | `--colocate` | `max(训练卡数, 生成卡数)` | 卡少、中小模型。训练与生成共享同一批卡，靠 offload 轮流驻留显存 |
| **分离部署** | 默认 | `训练卡数 + 生成卡数` | 卡多、大模型。两个独立 placement group，互不干扰 |
| **外部引擎** | `--rollout-external` | 只占训练卡 | SGLang 由训练作业之外的进程托管，甚至跨集群 |
| **只调生成** | `--debug-rollout-only` | 只占生成卡 | 调 rollout 逻辑，不起训练 |
| **只调训练** | `--debug-train-only` | 只占训练卡 | 调训练逻辑，不起推理引擎 |

分离部署下还有一个精巧设计：推理引擎的所有者是一个 **detached Ray actor**（`ServingCluster`，见 `slime/ray/placement_group.py` 的 `_attach_rollout_manager`，用 `lifetime="detached"` + `get_if_exists=True` 创建）。它的生命周期独立于 driver——训练进程崩了重启，引擎和队列状态还活着，新 driver 直接重新挂上去。这让"重启 trainer 不用重启整个集群"成为可能。

## 六、参数三分法

slime 的参数看着多，其实只分三类：

| 类别 | 写法 | 例子 |
| :--- | :--- | :--- |
| **Megatron 参数** | 直接透传，原样使用 | `--tensor-model-parallel-size 2`、`--recompute-granularity full` |
| **SGLang 参数** | 加 `--sglang-` 前缀 | `--mem-fraction-static` → `--sglang-mem-fraction-static` |
| **slime 自有参数** | 定义在 `slime/utils/arguments.py` | `--rollout-batch-size`、`--advantage-estimator` |

这个设计的好处是：你已经掌握的 Megatron 和 SGLang 调参经验可以**零成本迁移**，不需要学一套新的映射关系。

`slime/utils/arguments.py` 大约两千行，不必通读。真正值得一看的是里面的 `slime_validate_args()`——它集中了大量参数互斥与依赖校验，想知道"哪些参数不能一起用"，读它比读文档快。几个例子：

- `--advantage-estimator ppo` 会自动开启 critic（`use_critic = advantage_estimator == "ppo"`），且 critic 的 GPU 数被强制等于 actor；
- `reinforce_plus_plus` 系列的估计器**必须**同时加 `--normalize-advantages`；
- `--kl-coef` 和 `--kl-loss-coef` 互斥，只能设一个；
- `--eps-clip-high` 不设时会自动回落到 `--eps-clip`。

## 七、第一次跑起来

环境准备（镜像、Megatron、数据集、HF → mcore 权重转换）跟着官方 [quick_start.md](https://github.com/THUDM/slime/blob/main/docs/zh/get_started/quick_start.md) 走就行，这里重点看**启动脚本的结构**。以 `scripts/run-qwen3-4B.sh` 为例，它把参数按职责分成几组：

| 参数组 | 管什么 |
| :--- | :--- |
| `MODEL_ARGS` | 模型结构。由 `source scripts/models/qwen3-4B-Instruct-2507.sh` 引入，不用自己写 |
| `CKPT_ARGS` | 权重路径：`--hf-checkpoint`（tokenizer/config 来源）、`--ref-load`（mcore 格式初始权重）、`--load`/`--save`/`--save-interval` |
| `ROLLOUT_ARGS` | 数据与采样：`--prompt-data`、`--input-key`/`--label-key`（parquet/jsonl 字段映射）、`--rollout-batch-size`、`--n-samples-per-prompt`、`--global-batch-size` |
| `EVAL_ARGS` | 评测：`--eval-interval`、`--eval-prompt-data`、`--n-samples-per-eval-prompt` |
| `PERF_ARGS` | 并行与显存：TP/PP/CP/EP 切分、重计算、`--use-dynamic-batch-size`、`--max-tokens-per-gpu` |
| `GRPO_ARGS` | 算法：`--advantage-estimator`、`--pg-loss-type`、KL / entropy 系数、clip 区间、TIS 开关 |
| `OPTIMIZER_ARGS` | 优化器：`--optimizer-cpu-offload`、学习率与 schedule、Adam 系数 |
| `SGLANG_ARGS` | 推理引擎：`--rollout-num-gpus-per-engine`、`--sglang-mem-fraction-static` |
| `MISC_ARGS` | 杂项：dropout 归零、fp32 梯度累加、attention backend |

启动方式是 **Ray job**，不是直接 `python train.py`：

```bash
ray start --head --num-gpus ${NUM_GPUS} --dashboard-port=8265
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py --actor-num-nodes 1 --actor-num-gpus-per-node ${NUM_GPUS} --colocate ...
```

两处容易踩坑：

- **`RUNTIME_ENV_JSON` 里的 `PYTHONPATH` 必须包含 Megatron-LM 的路径**，否则 Ray worker 里 `import megatron` 会失败。后面如果你要用自定义生成函数，自己的代码目录也得加进来（[第 4 篇](/blog/slime-04-search-agent-rl/)会详细讲）。
- 脚本开头那一串 `pkill -9` / `ray stop --force` 是为了**重跑时清理残留**。SGLang 和 Ray 进程不会随 shell 退出而终止，不清理会出现端口占用和显存不释放。

跑通的标志是日志里出现 `Rollout generation` 进度条，以及第一条 `First rollout sample` 打印出的 prompt / response / reward。

## 八、分侧调试：两个最该先学会的开关

RL 训练链路长，一次完整迭代动辄几十分钟。如果每次都端到端跑，调试成本极高。slime 提供了把两侧**切开单独调**的能力，这是实践中最省时间的技巧：

```bash
# 第一步：只跑生成，把数据 dump 下来
python train.py --debug-rollout-only --save-debug-rollout-data /tmp/rollout_{rollout_id}.pt ...

# 第二步：只跑训练，直接吃上一步的数据
python train.py --debug-train-only --load-debug-rollout-data /tmp/rollout_{rollout_id}.pt ...
```

- `--debug-rollout-only` 不起训练进程，只占生成卡。适合调自定义生成函数、奖励函数、prompt 模板。
- `--debug-train-only` 不起推理引擎，只占训练卡。配合 `--load-debug-rollout-data` 用固定数据反复跑，适合调 loss、并行配置、显存问题。

更多技巧见 [docs/zh/developer_guide/debug.md](https://github.com/THUDM/slime/blob/main/docs/zh/developer_guide/debug.md)。

## 小结

本文建立的全局认知可以压缩成四句话：

1. slime 用一条统一路径连接 Megatron 训练与 SGLang 生成，data buffer 是两者之间的桥梁；
2. `train.py` 的 `train()` 主循环是整个框架的节律，一个 `rollout_id` 就是一次完整 RL step；
3. GPU 布局有五种模式，由 `_get_placement_group_layout` 决定，colocate 和分离部署是两种主流选择；
4. 参数分三类——Megatron 透传、SGLang 加前缀、slime 自有。

下一篇 [《slime 进阶：从 prompt 到梯度的完整链路》](/blog/slime-02-core-pipeline/) 会沿着一条 `Sample` 的生命周期往下钻：它从数据源被取出后经历了什么，怎么变成张量，怎么切给各个 DP rank，以及 advantage 和 loss 具体怎么算出来的。
