---
title: "slime 学习笔记（三）：算法层——参数、Advantage 与 Loss 的设计"
description: "把焦点收到算法层：Megatron 与 SGLang 的重点参数各自解决什么问题、RL 的 batch 节律怎么配、六种 advantage estimator 分别在对付什么、策略损失的四种形态，以及 slime 最花力气解决的训推失配问题。"
pubDate: 2026-08-19
tags: ["slime", "强化学习", "RLHF", "后训练", "GRPO", "PPO", "源码解读"]
---

> **本文是《slime 学习笔记》系列第 3 篇**。[第 1 篇](/blog/slime-01-getting-started/) 讲框架骨架，[第 2 篇](/blog/slime-02-core-pipeline/) 讲从 prompt 到梯度的链路。本文把焦点收到**算法层**：Megatron 与 SGLang 的重点参数各自解决什么问题，RL 的 batch 节律怎么配，以及 advantage 与 loss 到底是怎么设计的、分别在对付哪类难题。第 2 篇给过概览表，本文负责讲清楚「**为什么**」和「**怎么调**」。想动手做 agentic RL，接着看[第 4 篇](/blog/slime-04-search-agent-rl/)。
>
> 系列目录：
> 1. [入门与整体架构](/blog/slime-01-getting-started/)
> 2. [从 prompt 到梯度的完整链路](/blog/slime-02-core-pipeline/)
> 3. **算法层：参数、Advantage 与 Loss 的设计**（本文）
> 4. [实战：从零训一个 Search Agent](/blog/slime-04-search-agent-rl/)

## 一、参数三分法的算法视角

[第 1 篇](/blog/slime-01-getting-started/)讲过参数分三类：Megatron 透传、SGLang 加 `--sglang-` 前缀、slime 自有。那是「从哪来」的视角。做算法时更有用的是「管什么」的视角——同一次训练里，真正影响收敛与稳定的参数其实集中在四条线上：

```text
flowchart TD
    A["一次 RL step 的算法配置"] --> B["训练能不能跑起来<br/>Megatron: 并行/重计算/显存"]
    A --> C["生成快不快、准不准<br/>SGLang: tp/显存/并发"]
    A --> D["样本怎么凑<br/>batch 节律: rollout/global/采样数"]
    A --> E["梯度怎么算<br/>advantage + loss + 失配校正"]
```

前两条是「系统正确性」，后两条是「算法正确性」。本文按这个顺序走：先把 B、C 两条讲到够用，再把大半篇幅留给 D、E。

需要强调的一点：**RL 后训练对数值确定性比 SFT 敏感得多**。因为生成和训练分属两套引擎，任何不一致都会以 off-policy 偏差的形式进入梯度。所以你会看到脚本里一堆「归零 / 提精度」的参数——它们不是性能调优，是算法正确性的前提。

## 二、Megatron 侧：既要算得动，也要算得对

以 `scripts/run-qwen3-4B.sh`（dense）和 `scripts/run-qwen3-30B-A3B.sh`（MoE）为对照，`PERF_ARGS` / `OPTIMIZER_ARGS` / `MISC_ARGS` 三组是算法相关的重点。

### 2.1 并行切分：按「模型放不放得下 + 序列多长」来配

| 参数 | 4B | 30B-A3B | 解决的问题 |
| :--- | :-- | :-- | :--- |
| `--tensor-model-parallel-size` (TP) | 2 | 4 | 单层算子切分，降单卡参数/激活显存 |
| `--sequence-parallel` | 开 | 开 | 配合 TP，把 LayerNorm/Dropout 的激活也按序列切开，进一步省激活 |
| `--pipeline-model-parallel-size` (PP) | 1 | 1 | 按层切分，跨卡放不下的大模型才需要 |
| `--context-parallel-size` (CP) | 2 | 2 | **长序列专用**：把一条序列按 token 切到多卡。response 长到 131072 时必须靠它 |
| `--expert-model-parallel-size` (EP) | 1 | 8 | **MoE 专用**：把专家分到不同卡 |
| `--expert-tensor-parallel-size` (ETP) | 1 | 1 | MoE 专家内部的 TP 切分 |

几个要点：

- **`sequence-parallel` 几乎总是跟 TP 一起开**，它是 TP 的免费搭档——不增加通信量却能再省一块激活显存。
- **CP 是长序列 RL 的命门**。agent 任务动辄几万 token，不开 CP 单卡激活直接爆。但 CP 引入了算法上的复杂度：序列被 zigzag 切分后，advantage 归一化、loss 归约都必须在 **DP×CP 组**上做，否则同一条序列的两半会被施加不同的仿射变换（[第 2 篇](/blog/slime-02-core-pipeline/)的分布式细节）。
- **MoE 用 EP 而非 TP 切专家**。30B-A3B 的配法是 `TP4 + EP8`，专家走 EP、非专家走 TP，这是 MoE 训练的标准摆法。

### 2.2 重计算：用算力换显存

```bash
--recompute-granularity full      # 整层重算（相对 selective 更省显存但更慢）
--recompute-method uniform        # 每隔若干层设一个检查点
--recompute-num-layers 1          # 每 1 层重算一次（最省显存、最慢的一端）
```

前向时丢弃中间激活，反向时重新算一遍。`recompute-num-layers 1` 是最激进的省显存设置，几乎把激活显存压到最低，代价是多一次前向。显存够的话调大这个值能换回速度。

### 2.3 批大小：从「固定样本数」切到「固定 token 数」

```bash
# --micro-batch-size 1        # 固定每 microbatch 的样本数（被注释掉）
--use-dynamic-batch-size       # 改为按 token 总量打包
--max-tokens-per-gpu 9216      # 4B；30B-A3B 用 20480
--log-probs-chunk-size 1024    # 算 logprob 时按 1024 token 分块，控制词表维度的峰值显存
```

RL 的 response 长度方差极大（有的几十 token，有的几万）。按固定样本数打 microbatch，会出现「一个 batch 里混进一条超长序列把显存顶爆」的情况。`--use-dynamic-batch-size` 改成按 `--max-tokens-per-gpu` 的 token 预算打包，长序列自成一包、短序列拼在一起，显存利用率稳定得多。这也是为什么脚本默认把 `--micro-batch-size` 注释掉。

`--log-probs-chunk-size` 则是另一层优化：算 logprob/entropy 要在 `[T, V]` 的全词表 logits 上做 softmax，V 动辄十几万，一次性算峰值显存很高。分块后把 FP32 工作区压到 `O(chunk × V)`（见 `slime/utils/ppo_utils.py` 的 `_ChunkedLogProbEntropy`）。

### 2.4 优化器与精度：RL 的保守配置

```bash
--optimizer adam --lr 1e-6 --lr-decay-style constant
--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
--optimizer-cpu-offload              # 优化器状态放 CPU，省显存
--overlap-cpu-optimizer-d2h-h2d      # D2H/H2D 拷贝与计算重叠，抵消 offload 的延迟
--use-precision-aware-optimizer
```

```bash
# MISC_ARGS —— 算法正确性相关
--attention-dropout 0.0 --hidden-dropout 0.0   # dropout 归零
--accumulate-allreduce-grads-in-fp32           # 梯度 all-reduce 用 fp32 累加
--attention-softmax-in-fp32                    # attention softmax 用 fp32
--attention-backend flash
```

- **学习率出奇地小（1e-6）且常数衰减**。RL 后训练是在一个已经不错的模型上做微调，大 lr 极易崩。常数衰减是因为 RL 的「epoch」概念弱，用线性/cosine 衰减反而不好对齐。
- **dropout 必须归零**。Megatron 默认 dropout 0.1，但 RL 里 old_log_prob 和训练时的 log_prob 要可比，dropout 带来的随机性会直接污染 IS ratio。
- **fp32 累加 + fp32 softmax** 是精度保险。生成侧已经有训推失配了（见第八节），训练侧再引入 bf16 累加误差会雪上加霜。

> MoE 脚本还通过 `MODEL_ARGS_ROTARY_BASE=10000000` 覆盖 rotary base，这类属于模型结构参数，由 `scripts/models/*.sh` 引入，不在算法调参范围。

## 三、SGLang 侧：生成引擎的关键旋钮

### 3.1 前缀机制：零成本复用 SGLang 调参经验

slime 不重新定义 SGLang 参数，而是在 `slime/backends/sglang_utils/arguments.py` 的 `add_sglang_arguments` 里，把 SGLang 原生的 `ServerArgs.add_cli_args` 包了一层：每个原生 flag 自动加上 `--sglang-` 前缀，对应的 `dest` 也加 `sglang_` 前缀。

所以 SGLang 文档里的 `--mem-fraction-static`，在 slime 里就是 `--sglang-mem-fraction-static`，语义完全一致。

但有一批参数被 **`skipped_args` 列表拦截、不对外暴露**，因为它们由 slime 接管：

```python
skipped_args = [
    "model_path", "config", "trust_remote_code", "random_seed",
    "enable_memory_saver",                       # offload 由 slime 控制
    "tp_size", "port", "nnodes", "node_rank",    # 拓扑由 slime 分配
    "dist_init_addr", "base_gpu_id", "nccl_port",
    "skip_server_warmup", "enable_return_routed_experts",
]
```

这解释了一个常见困惑：为什么 SGLang 的 `--tp-size` 在 slime 里设不了——因为 TP 大小是从 slime 的 `--rollout-num-gpus-per-engine` 推出来的。

### 3.2 重点参数

| 参数 | 含义与解决的问题 |
| :--- | :--- |
| `--rollout-num-gpus-per-engine` | 每个推理引擎占几张卡。`validate_args` 里 `sglang_tp_size = rollout_num_gpus_per_engine // pp_size`——**它直接决定生成侧的 TP 大小**。4B 用 2，30B-A3B 用 8 |
| `--sglang-mem-fraction-static` | KV cache 的静态显存占比。colocate 下训练和生成抢显存，这里设 0.7 是给训练留出余量；分离部署可以更高 |
| `--sglang-server-concurrency` | 单引擎并发上限，默认 512。与引擎数相乘得到全局并发信号量（[第 2 篇](/blog/slime-02-core-pipeline/) 的 `GenerateState`） |
| `--sglang-enable-dp-attention` + `--sglang-dp-size` | DP attention。`validate_args` 里 `dp_size > 1` 时**强制断言必须开 dp-attention** |
| `--sglang-ep-size` | 生成侧的专家并行度，MoE 推理用 |
| `--sglang-cuda-graph-bs` | CUDA graph 捕获的 batch size 列表。30B-A3B 脚本显式给了 `1 2 4 8 $(seq 16 8 256)`，为 MoE 的典型并发规模定制图捕获，减少 kernel launch 开销 |

一个细节：30B-A3B 的 `--rollout-num-gpus-per-engine 8` 意味着整机 8 卡做一个推理引擎（TP8），而训练侧是 `TP4 + EP8`。**生成和训练的并行布局可以完全不同**——这正是 slime 连接两套引擎的价值，权重同步时由 `megatron_to_hf/` 做布局转换（[第 2 篇](/blog/slime-02-core-pipeline/)）。

## 四、RL 节律参数：一个 batch 是怎么凑出来的

这几个参数定义了「一次 RL step 吃多少数据、怎么采样」，是算法的节拍器。

```bash
--rollout-batch-size 32       # 一个 step 取多少个 prompt
--n-samples-per-prompt 8      # 每个 prompt 采样几条（= GRPO 的组大小）
--global-batch-size 256       # 训练 batch 大小，这里 32 × 8 = 256
--rollout-max-response-len 131072
--rollout-temperature 1       # 采样温度；RL 探索通常用 1.0
--balance-data                # 按估算 FLOPs 在 DP rank 间均衡长序列
```

关系要拎清：

```text
rollout_batch_size (32 prompt)
        × n_samples_per_prompt (8)
        = global_batch_size (256 条轨迹) —— 正好一个训练 batch
```

- **`--n-samples-per-prompt` 是 GRPO 的命根子**。GRPO 用「组内其它采样」当 baseline，组太小（比如 2）优势估计噪声极大。8 是常见起点。
- **`--rollout-temperature` 影响探索与失配**。温度不为 1 时，训练侧算 logprob 要用同样的温度缩放 logits（`slime/backends/megatron_utils/loss.py` 会 `logits.div(rollout_temperature)`），否则 old_log_prob 对不上。
- **超采样 + 动态过滤**（[第 2 篇](/blog/slime-02-core-pipeline/) 详述）：`--over-sampling-batch-size`（≥ rollout-batch-size）多取一些，`--dynamic-sampling-filter-path` 把「整组全对或全错」这类零梯度信号的组丢掉，超采样保证还能凑够一个完整 batch。内置的 `check_reward_nonzero_std` 就是干这个的——组内 reward 方差为 0 的直接扔。

这里藏着一个 GRPO 的冷启动陷阱：**如果 reward 是二值的（0/1），模型在学会任务前所有采样都是 0 分，组内全零 → 优势全零 → 没梯度 → 动态过滤把整批丢光 → 训练停滞**。解法是设计有梯度的分层奖励，这点[第 4 篇](/blog/slime-04-search-agent-rl/)会专门讲。

## 五、Advantage：六种估计器分别解决什么问题

`--advantage-estimator` 的六个选项（默认 `grpo`），实现集中在 `slime/backends/megatron_utils/loss.py` 的 `compute_advantages_and_returns` 与 `slime/utils/ppo_utils.py`。[第 2 篇](/blog/slime-02-core-pipeline/)给过一张「要不要 critic」的速查表，这里讲每一种**在对付什么**。

### 5.1 GRPO 家族：用「组内相对好坏」替代价值网络

PPO 需要一个和策略同样大的 critic 来估计 baseline，显存翻倍。GRPO 的洞见是：同一个 prompt 采样 N 条，**用这 N 条的平均 reward 当 baseline**，就不需要 critic 了。

关键在于这个「减均值」发生在**奖励后处理阶段**，不在 advantage 计算里。看 `slime/data/batch_builder.py` 的 `_post_process_rewards`：

```python
# 对每个 prompt group：
centered = rewards[positions] - rewards[positions].mean()          # 减组均值
if grpo_std_normalization and len(positions) > 1:
    centered = centered / (centered.std() + 1e-6)                  # 可选：再除组标准差
```

然后 `compute_advantages_and_returns` 里 `grpo` 分支只是把这个标量 reward 广播到每个 token（`get_grpo_returns`）：

```python
advantages = [torch.ones_like(kl[i]) * reward[i] for i in range(...)]
```

两个开关直接对应两篇论文：

| 参数 | 默认 | 作用 |
| :--- | :-- | :--- |
| `--disable-grpo-std-normalization` | std 归一化默认**开** | 关掉即 **Dr.GRPO**（[arXiv:2503.20783](https://arxiv.org/pdf/2503.20783)）：只减均值不除 std，避免长度/难度偏置 |
| `--disable-rewards-normalization` | 归一化默认**开** | 彻底关掉组内中心化，退回原始 reward |

注意两者适用范围不同：减均值覆盖 `grpo / gspo / cispo / reinforce_plus_plus_baseline`，而除 std 只覆盖前三个。

### 5.2 GSPO：序列级 IS，专治 MoE 与长序列的高方差

GSPO 的 advantage 和 GRPO 完全一样，差别在 **loss 里用的 KL/ratio 是序列级而非 token 级**（`compute_gspo_kl`）：把整条序列的 KL 平均后，广播回每个 token。

它解决的问题是：token 级重要性采样比值 `π_θ/π_old` 在长序列、MoE 路由抖动的场景下方差爆炸，单个离群 token 就能毁掉整条序列的更新。序列级聚合把方差压下来，代价是粒度变粗。这也是为什么 `can_reuse_log_probs_in_loss` 把 `gspo` 排除在复用优化之外（[第 2 篇](/blog/slime-02-core-pipeline/)）——它需要训练前就拿到完整序列的 old_log_prob 做 all-gather。

### 5.3 CISPO：保住被裁剪 token 的梯度

标准 PPO 的 clip 有个副作用：ratio 一旦超出 `[1-ε, 1+ε]`，该 token 的梯度被直接**清零**。在 RL 里，ratio 大的恰恰往往是「模型最想改变的 token」，把它们的梯度扔掉很浪费。

CISPO（来自 MiniMax-M1，`compute_cispo_loss`）的做法是：

```python
ratio_truncated = clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)
pg_losses = -ratio_truncated.detach() * advantages * log_probs   # 注意 detach
```

把裁剪后的 ratio 放进 `detach()`（stop-gradient），梯度转而从 `log_probs` 流出。这样**被裁剪的 token 依然贡献梯度**，只是权重被截断控制住。canonical CISPO 会关掉下界（`eps_clip >= 1.0`）。

### 5.4 PPO：唯一需要 critic 的经典路线

**一句话原理：GAE 是在「只信 critic 一步」和「看到轨迹结尾」之间调一个旋钮。**

单步 TD 误差 `δ_t = r_t + γ·V(s_{t+1}) - V(s_t)`，衡量这一步比 critic 的预期好多少。GAE 把它按 `(γλ)^k` 加权累加：

```text
A_t = δ_t + γλ·A_{t+1}    即    A_t = Σ_k (γλ)^k · δ_{t+k}
```

λ 就是那个旋钮，它解决的唯一问题是**偏差-方差权衡**：

- `λ=0` → `A_t = δ_t`，只用一步 TD，方差小但 critic 不准就全错（偏差大）；
- `λ=1` → `A_t = G_t - V(s_t)`，退化成蒙特卡洛，无偏但方差大；
- 中间值在两者间插值。

`γ`（`--gamma`）是另一回事——未来回报打折。LLM 任务一般取 1.0，因为奖励本就只在末尾，打折等于凭空惩罚长序列。两者默认都是 1.0。

两个容易混淆的点：

- **token 级 reward 哪来的？** `r_t = -kl_coef·KL_t`，只有最后一个 token 再加上结果分。所以 PPO 路线的 KL 是 **reward shaping**，不是 loss 项（对比第七节）。
- **`returns = advantages + values` 是干什么的？** 由 `A_t = G_t - V(s_t)` 反推得 `G_t = A_t + V(s_t)`，它是**critic 的回归目标**，不参与策略梯度。

实现上 slime 默认走 `chunked_gae`（分块并行扫描）而非 O(T) 串行递推，长序列下差距显著。

### 5.5 REINFORCE++ 两个变体：带折扣的 return

**一句话原理：去掉 critic 就没有 `V(s_t)`，也就没有 δ，只能直接累加 token 级 reward。**

```text
G_t = r_t + γ·G_{t+1}    即    G_t = Σ_k γ^k · r_{t+k}
```

**关键点：`G_t` 是绝对回报，不是优势。** 少了 baseline 会出两个问题：

1. reward 全正时所有 advantage 都为正 → 梯度只「鼓励」从不「抑制」，退化成加权最大似然；
2. 尺度随 reward 量级和序列长度漂移，学习率难稳。

所以框架**强制**要求 `--normalize-advantages`——白化（减均值除标准差）在这里**就是 baseline**。这是硬约束而非建议。

两个变体的分水岭正是「baseline 从哪来」：

| | `reinforce_plus_plus` | `reinforce_plus_plus_baseline` |
| :--- | :--- | :--- |
| advantage 怎么算 | 用 `γ` 做折扣累加 | **不折扣**，标量 reward 直接广播到每个 token |
| baseline 来源 | 全 batch 白化 | 组内减均值（同 GRPO）+ 白化 |

后者其实更接近 GRPO，只是额外逐 token 减了 KL 惩罚。

### 5.6 归一化：全局白化在哪做

`--normalize-advantages` 打开后，`compute_advantages_and_returns` 末尾对所有 advantage 做 masked whitening。关键是它必须在 `mpu.get_data_parallel_group(with_context_parallel=True)` 上做——这个 DP×CP 分布式细节[第 2 篇](/blog/slime-02-core-pipeline/)已经讲透，这里不重复。

一句话记法：**GRPO 系在「reward 后处理」减组均值，`--normalize-advantages` 在「全 batch」白化，两者作用的层级不同，可以叠加。**

## 六、Loss：策略梯度主项的四种形态

策略损失的主项由 `get_pg_loss_type`（`slime/utils/ppo_utils.py`）解析，`--pg-loss-type` 的显式取值只有 `ppo` 和 `reinforce`，但配合 estimator 会推出四种实际形态。解析逻辑：

```python
objective = configured or ("reinforce" if 用了 score_centering
                           else "cispo" if estimator == "cispo"
                           else "ppo")
```

### 6.1 四种主项对照

| 形态 | 触发 | 公式核心 | 裁剪行为 |
| :--- | :--- | :--- | :--- |
| **ppo** | 默认 | `max(-ratio·A, -clip(ratio)·A)`，`ratio = exp(-ppo_kl)` | 双侧 clip，可选 dual-clip |
| **cispo** | estimator=cispo | `-sg(clip(ratio))·A·log_probs` | 裁剪 ratio 但保留梯度 |
| **reinforce** | 用了 score centering | `-A·log_probs` | 无 ratio、无 clip |
| **dual-clip ppo** | 设了 `--eps-clip-c` | 在 adv<0 时再加下界 `-eps_clip_c·A` | 三重裁剪 |

对应的裁剪参数：

| 参数 | 默认 | 说明 |
| :--- | :-- | :--- |
| `--eps-clip` | 0.2 | PPO 下界 `1 - eps_clip` |
| `--eps-clip-high` | None → 回落到 `eps-clip` | 上界 `1 + eps_clip_high`。设得比下界大即**非对称裁剪**，给正向更新更多空间（DAPO 的做法） |
| `--eps-clip-c` | None | dual-clip 下界，须 > 1.0。防止 adv<0 时 ratio 过大导致的梯度爆炸（[arXiv:1912.09729](https://arxiv.org/pdf/1912.09729)） |

### 6.2 完整 loss 的叠加结构

`policy_loss_function` 最终的 loss 是逐项叠加，每项独立开关：

```text
loss = pg_loss                              # 主项（上面四选一）
     - entropy_coef · entropy_loss          # 熵正则（鼓励探索）
     + kl_loss_coef · kl_loss               # KL 损失项（约束不偏离 ref）
     (+ score_centering 修正项)
     (× TIS/MIS 重要性权重，× OPSM mask)     # 失配校正，见第八节
```

把这段和 `policy_loss_function` 的源码对一遍，就能看懂每个 `--xxx-coef` 落在哪一项上。

## 七、稳定性三件套：KL、熵、归一化

### 7.1 两个 KL，别搞混

slime 里有**两处**KL，作用位置完全不同：

| | reward shaping KL | loss KL |
| :--- | :--- | :--- |
| 参数 | `--kl-coef`（默认 0） | `--use-kl-loss` + `--kl-loss-coef`（默认 0） |
| 注入点 | advantage/return 计算里（当 token-level 奖励惩罚） | 最终 loss 上加一项 |
| 适用 | PPO / REINFORCE++ 路线 | GRPO 路线 |

通常**二选一**：GRPO 系用 loss KL，PPO 系用 reward shaping。两个都设容易重复惩罚。

KL 的估计方式由 `--kl-loss-type` 选（`compute_approx_kl`，源自 Schulman 的 KL 近似）：

| 类型 | 特点 |
| :--- | :--- |
| `k1` | `log_ratio`，无偏但方差大 |
| `k2` | `log_ratio²/2` |
| `k3` / `low_var_kl` | `exp(-log_ratio) - 1 + log_ratio`，非负、无偏、低方差。`low_var_kl` 额外 clamp 到 `[-10, 10]` 防数值炸 |

还有一个 `--use-unbiased-kl`：给 KL 乘上 IS 比值做无偏估计（DeepSeek-V3.2 的做法）。

### 7.2 熵正则

`--entropy-coef`（默认 0）控制 `- entropy_coef · entropy_loss`。调大鼓励探索、防过早收敛到单一模式；调太大会让输出变乱。RL 后训练常设 0 或极小值，靠温度和 KL 来控探索。

### 7.3 ref 模型的更新

KL 约束的锚点是 ref 模型。`--ref-update-interval` 控制每隔多少 rollout step 把 ref 更新成当前 actor。不设则 ref 固定为初始权重——约束越来越松 vs 固定靶子，是一个需要按任务权衡的选择。ref 模型本身靠 CPU 权重备份 + 模型切换实现，不额外占显存（[第 2 篇](/blog/slime-02-core-pipeline/)）。

## 八、训推失配：slime 最花力气解决的问题

这是本文的重头。**失配（train-infer mismatch）是 RL 后训练最隐蔽、最致命的问题**，slime 为它提供了一整套工具。

### 8.1 问题的根源

生成用 SGLang（为吞吐做了低精度、融合 kernel、不同的采样实现），训练用 Megatron（高精度、不同的算子）。对**同一条序列的同一个 token**，两边算出的 logprob 并不相等。于是：

- rollout 时记录的 `rollout_log_probs` ≠ 训练时重算的 `log_probs`；
- 这个差异让本应「on-policy」的数据实际带了 off-policy 偏差；
- 偏差随训练累积，表现为 reward 虚高后突然崩塌，极难归因。

诊断入口是 `--get-mismatch-metrics`，它会报 `train_rollout_logprob_abs_diff`（两边 logprob 的平均绝对差）和 `ois`（old importance sampling ratio）。**先量化，再决定要不要校正。**

### 8.2 校正手段谱系

```text
flowchart TD
    A["发现 train/infer 失配"] --> B["轻度：乘一个 IS 权重拉回"]
    A --> C["中度：区间外直接拒绝该 token"]
    A --> D["重度：整条序列级别处理"]
    A --> E["MoE 特有：路由不一致"]
    B --> B1["TIS<br/>--use-tis"]
    C --> C1["MIS / IcePop<br/>--custom-tis-function-path"]
    D --> D1["OPSM<br/>--use-opsm"]
    E --> E1["routing replay<br/>--use-rollout-routing-replay"]
```

**TIS（Truncated Importance Sampling，`--use-tis`）** 是最常用的一档。`vanilla_tis_function` 算 `ratio = exp(old_log_prob - rollout_log_prob)`，clamp 到 `[--tis-clip-low, --tis-clip]`（默认 `[0, 2]`）后乘到 pg_loss 上。本质是用重要性采样把生成分布拉回训练分布，截断控制方差。

**MIS / IcePop（`icepop_function`，用 `--custom-tis-function-path` 指定）** 更激进：用 `importance_weights(mode="mis")`，ratio 落在 `[low, high]`（默认 `[0.5, 5]`）**区间外的 token 直接置 0**——不是缩放而是拒绝采样式丢弃。适合失配严重、宁可丢 token 也不要毒样本的场景。

**OPSM（Off-Policy Sequence Masking，`--use-opsm` + `--opsm-delta`）** 升级到序列级（`compute_opsm_mask`）：如果一条序列的序列级 KL 超过 `opsm_delta`（默认 1e-4）**且** advantage < 0，就把整条序列 mask 掉。逻辑是——一条已经偏离很远、又是负优势（要惩罚）的序列，惩罚它的梯度方向很可能是错的，不如不学。

**routing replay（`--use-routing-replay` / `--use-rollout-routing-replay`）** 是 MoE 专属。MoE 的专家路由在生成和训练时可能选到不同专家，这本身就是一种失配。routing replay 把生成时的路由决策记录下来，在训练时「回放」，强制走相同的专家（分别对应 [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) 和 [arXiv:2510.11370](https://arxiv.org/abs/2510.11370)）。这就是 30B-A3B 脚本开了 `--use-rollout-routing-replay` 而 4B 没开的原因——dense 模型没有路由问题。

**score centering（`--use-score-centering` + `--score-centering-top-k`）** 是另一条路线：改用 REINFORCE 的 score-centering 目标（`get_score_centering_terms`），在 ratio 之外再做一次中心化修正。它强制 `--pg-loss-type reinforce`，`--score-centering-top-k`（默认 128）控制 `rollout-top-p=1` 时保留多少 sampler top logprobs。两个官方脚本都默认开了它，可见这是当前推荐的失配缓解方案。

### 8.3 一个取舍

要开 TIS/MIS 这类校正，前提是**生成时必须记录 `rollout_log_probs`**（SGLang 侧 `return_logprob: True`）。代价是为保 token/logprob 对齐会禁用响应字符串后处理，并增加传输与计算开销。`--use-rollout-logprobs` 则更直接——干脆拿 rollout logprob 当 old_log_prob 用，省掉一次训练前的 actor 前向，但这把失配「接受」进了 IS 比值的分母里，是另一种权衡。

> 相关实现可参考 `examples/train_infer_mismatch_helper/`，以及进阶文档 `docs/zh/advanced/policy-mismatch.md`。

## 九、一个隐形选择：per-token 还是 per-sample

最后一个常被忽略、却实实在在改变梯度的参数：`--calculate-per-token-loss`。

loss 要把一批长短不一的序列归约成一个标量，归一化的分母怎么选，决定了「长序列和短序列谁的话语权大」。`slime/backends/megatron_utils/cp_utils.py` 的 `get_sum_of_sample_mean` 实现了两种口径：

| 口径 | `calculate_per_token_loss` | 分母 | 效果 |
| :--- | :-- | :--- | :--- |
| per-sample mean | False（默认） | 每条序列按自身 `loss_mask.sum()` 归一，再按样本数平均 | 每条序列**等权**，无论长短 |
| per-token mean | True | 按 rollout 组的 token 总数（`rollout_mask_sums`）归一 | 每个 **token 等权**，长序列权重更大 |

两个官方脚本都开了 `--calculate-per-token-loss`。对长序列 agent/推理任务，per-token 更合理——否则一条 10 token 的序列和一条 10000 token 的序列对梯度贡献相同，模型会偏向优化短序列。

实现上还有个为 Megatron 适配的 rescale（`loss_function` 末尾）：per-token 时乘 `cp_size` 抵消 CP 的重复；per-sample 时乘 `num_microbatches / step_global_batch_size × dp_cp_size`，抵消 Megatron 内部的梯度累加除法。这段不用背，知道「两种口径的归一常数不同、都做了分布式修正」即可。

## 小结

把算法层压缩成四句：

1. **Megatron 参数分两半**——并行/重计算/批大小管「算得动」，dropout 归零 + fp32 累加管「算得对」，后者是 RL 特有的正确性前提；
2. **SGLang 靠 `--sglang-` 前缀零成本复用**，但 TP、随机种子、显存管理等被 slime 接管，`--rollout-num-gpus-per-engine` 才是生成侧 TP 的真正入口；
3. **Advantage 的核心分歧是「要不要 critic」**——GRPO 家族用组内相对好坏省掉价值网络，GSPO/CISPO 各自针对高方差和梯度裁剪做改良，PPO/REINFORCE++ 是经典路线；
4. **Loss 的重点不在主项而在校正项**——TIS/MIS/OPSM/routing replay/score centering 这一套都在对付同一个敌人：train-infer 失配。

下一篇 [《slime 实战：从零训一个能用在业务上的 Search Agent》](/blog/slime-04-search-agent-rl/) 把这些旋钮放到一个真实任务里，看它们在 search agent 的配置中如何取值、为什么这样取。
