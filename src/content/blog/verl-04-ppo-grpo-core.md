---
title: "verl 源码解读（四）：PPO / GRPO 主循环与核心算法"
description: "把 WorkerGroup、DataProto 串起来读 verl 的“剧本”：RayPPOTrainer.fit() 主循环，以及 core_algos.py 里 GAE 与 GRPO 组内归一化、PPO 裁剪目标、KL 惩罚的具体实现。"
pubDate: 2026-02-18
tags: ["verl", "PPO", "GRPO", "强化学习", "源码解读"]
---

> **本文是《verl 源码解读》系列第 4 篇**。前三篇分别建立了整体认知、分发机制（[WorkerGroup/Dispatch](/blog/verl-02-hybridflow-single-controller/)）和数据总线（[DataProto](/blog/verl-03-dataproto/)）。本篇把它们串起来，讲 verl 的“剧本”：`RayPPOTrainer.fit()` 主循环，以及 `core_algos.py` 里的 advantage 与 policy loss 的具体实现。
>
> 系列目录：[1 架构](/blog/verl-01-architecture/) · [2 HybridFlow](/blog/verl-02-hybridflow-single-controller/) · [3 DataProto](/blog/verl-03-dataproto/) · **4 PPO/GRPO** · [5 Rollout 混合引擎](/blog/verl-05-rollout-hybrid-engine/)

## 1. 两个关键文件

| 文件 | 职责 |
|------|------|
| `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer`：组织 worker、驱动 `fit()` 主循环（**流程**） |
| `verl/trainer/ppo/core_algos.py` | advantage 估计、policy loss、KL 惩罚等**纯算法**（无分布式依赖） |

设计上刻意做了分离：**`core_algos.py` 全是纯张量函数**，不关心 FSDP/Megatron/Ray，任何分布式策略都能复用；`ray_trainer.py` 只负责编排。

## 2. 角色（Role）与按需初始化

`trainer/ppo/utils.py` 定义了 RL 中的角色：

```python
class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2      # Actor 和 Rollout 合并（最常见）
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7
    TeacherModel = 8
```

并不是每种算法都需要全部角色，trainer 会按配置决定实例化哪些：

- `need_critic(...)`：只有用 GAE（即 PPO）时才需要 Critic；GRPO 等**没有 Critic**。
- `need_reference_policy(...)`：当 `use_kl_in_reward` 或 `use_kl_loss` 时才需要 Ref。
- `need_reward_model(...)`：用模型式奖励时才需要 RM；函数式（可验证）奖励则不需要。

这正是“PPO 比 GRPO 多一个价值网络”在代码层面的体现。

## 3. fit() 主循环逐段拆解

`RayPPOTrainer.fit()` 是整个训练的心脏。精简后的结构：

```python
def fit(self):
    logger = Tracking(...)                 # wandb/swanlab/tensorboard
    self._load_checkpoint()                # 断点续训
    if val_before_train:
        self._validate()                   # 训练前先验证一次

    for epoch in range(total_epochs):
        for batch_dict in self.train_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            batch.meta_info["temperature"] = ...
            batch.non_tensor_batch["uid"] = np.array([uuid4() ...])  # 每条样本一个 group id

            # ① 取生成所需列，并按 n 复制（GRPO 分组采样）
            gen_batch = self._get_gen_batch(batch)
            gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

            with marked_timer("step", timing_raw):
                # ② 生成（Rollout，vLLM/SGLang）
                with marked_timer("gen", timing_raw):
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)

                batch = batch.union(gen_batch_output)      # 回填 responses 等

                # ③ 重算 old_log_prob（训练引擎口径）
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                batch = batch.union(old_log_prob)

                # ④ Ref 的 log_prob（用于 KL）
                if self.use_reference_policy:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

                # ⑤ Critic 估值（仅 PPO）
                if self.use_critic:
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

                # ⑥ 奖励（函数式 / 模型式）
                reward_tensor = self.reward_fn(batch)   # token_level_scores
                batch.batch["token_level_scores"] = reward_tensor

                # ⑦ 计算 advantage（driver 本地）
                batch = compute_advantage(batch, adv_estimator=..., gamma=..., lam=...)

                # ⑧ 更新 Critic（仅 PPO）
                if self.use_critic:
                    self.critic_wg.update_critic(batch)

                # ⑨ 更新 Actor
                if self.global_steps >= critic_warmup:
                    self.actor_rollout_wg.update_actor(batch)

                # ⑩ 权重同步给 Rollout 引擎（见第 5 篇）
                self.checkpoint_manager.update_weights(self.global_steps)

            logger.log(metrics, step=self.global_steps)
            self.global_steps += 1
```

对照前几篇：

- 每个 `xxx_wg.method()` 都是第 2 篇的**分布式分发调用**（`DP_COMPUTE_PROTO`）；
- `batch.union(...)` / `repeat(...)` 是第 3 篇的 **DataProto 操作**；
- 第 ⑦ 步 advantage 在 **driver 本地**算，因为它是轻量张量运算，无需分发。

### 为什么 old_log_prob 要重新算一次？

生成用 vLLM，但 vLLM 输出的概率和训练引擎（FSDP/Megatron）在数值实现上有差异。PPO 的重要性采样比值 `ratio = exp(log_prob - old_log_prob)` 要求分子分母口径一致，因此 `old_log_prob` 必须由**训练引擎**在生成结果上重新 forward 一遍得到。

## 4. Advantage 估计：core_algos 的“插件化”注册

verl 用注册表 + 装饰器把各种 advantage 估计器做成“插件”：

```python
@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(...): ...

@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(...): ...
```

通过 `adv_estimator` 配置选择用哪个。已内置 GAE、GRPO、Dr.GRPO、RLOO、REINFORCE++、ReMax、OPO、GPG 等十余种。下面看两个最典型的。

### 4.1 GAE（PPO 用，需要 Critic）

```python
def compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam):
    with torch.no_grad():
        nextvalues, lastgaelam = 0, 0
        advantages_reversed = []
        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam
            # 只在 response token 上推进（跳过 padding/observation）
            nextvalues  = values[:, t]  * mask[:, t] + (1 - mask[:, t]) * nextvalues
            lastgaelam  = lastgaelam_   * mask[:, t] + (1 - mask[:, t]) * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)   # 标准化
    return advantages, returns
```

要点：从后往前递推 GAE；用 `response_mask` 保证只在真实生成 token 上累积；最后 `returns = advantages + values` 作为 Critic 的回归目标；`masked_whiten` 对 advantage 做标准化以稳定训练。**这里需要 `values`，所以 PPO 必须有 Critic。**

### 4.2 GRPO（无 Critic，靠组内归一化）

GRPO 的核心思想：**没有价值网络，用“同一 prompt 的 n 条回答”的组内均值作为 baseline。**

```python
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, epsilon=1e-6,
                                   norm_adv_by_std_in_grpo=True):
    scores = token_level_rewards.sum(dim=-1)        # 每条回答一个标量 reward

    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])        # 按 group id（uid）分组

    for idx in id2score:                             # 每组算均值、标准差
        id2mean[idx] = mean(group); id2std[idx] = std(group)

    for i in range(bsz):
        if norm_adv_by_std_in_grpo:                  # 原始 GRPO：减均值除标准差
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:                                        # Dr.GRPO：只减均值
            scores[i] = scores[i] - id2mean[index[i]]
    scores = scores.unsqueeze(-1) * response_mask    # 广播到每个 token
    return scores, scores
```

这里的 `index` 就是第 3 篇里 `repeat(n)` 之前给每个 prompt 分配的 `uid`——同一 prompt 的 n 条回答共享同一个 id，从而构成一个 group。`norm_adv_by_std_in_grpo` 开关决定是原始 GRPO（减均值除标准差）还是 Dr.GRPO（只减均值）。

> 对比：GAE 需要逐 token 的 value；GRPO 只用一条回答的**总分**做组内归一化，省掉了整个 Critic 网络，这就是 GRPO 更省资源的原因。GRPO 在图像生成领域的变体（Flow-GRPO 等）可参见[《图像生成与强化学习》](/blog/image-generation-and-rl/)。

## 5. Policy Loss：PPO 的裁剪目标

policy loss 同样是注册表（`@register_policy_loss`），默认 `"vanilla"` 就是标准 PPO 裁剪目标。核心片段：

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, -20.0, 20.0)   # 数值稳定
ratio = torch.exp(negative_approx_kl)                               # 重要性采样比

pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)             # 标准 PPO 裁剪

# dual-clip：对 advantage<0 的情况再加一层下界裁剪，防止过大的负梯度
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)
```

几个细节：

- **非对称裁剪**：`cliprange_low` / `cliprange_high` 可分别设置（DAPO 的 clip-higher 技巧）。
- **dual-clip PPO**：当 advantage<0 时多加一层下界 `clip_ratio_c`，避免梯度爆炸。
- 还内置了 GSPO、CISPO、clip-cov、kl-cov、geo-mean 等多种 loss，换个配置名即可切换。

### loss 聚合模式 agg_loss

逐 token 的 loss 矩阵最终要聚成一个标量，`loss_agg_mode` 控制方式：

- `token-mean`：所有有效 token 求平均（默认，长回答权重更大）；
- `seq-mean-token-mean`：先序列内平均再序列间平均（每条样本等权）；
- 等等。这会影响长短回答的梯度权重，是调参的一个点。

## 6. KL 惩罚：两种加法

KL 用来约束策略别偏离参考模型太远，有两种接入方式：

1. **加在 reward 里**（`use_kl_in_reward`）：把 `-beta * KL(policy || ref)` 作为 token 级惩罚并入 `token_level_rewards`，再去算 advantage。
2. **加在 loss 里**（`use_kl_loss`，GRPO 常用）：直接在 policy loss 上加一项 KL loss。

`kl_penalty(logprob, ref_logprob, kl_penalty_type)` 支持 `kl`、`abs`、`mse`、`low_var_kl`（即 k3 估计）等多种估计方式（参考 John Schulman 的 KL 近似博客）。是否需要 Ref 模型，就取决于这两个开关（见第 2 节 `need_reference_policy`）。

## 7. 一图总结 PPO vs GRPO

```
                 ┌─────────── PPO ───────────┐   ┌────────── GRPO ──────────┐
Critic           有（估 value，GAE 需要）        无
Advantage        GAE（逐 token，需 value）        组内归一化（每 prompt n 条）
Reference        可选（KL）                        常用（KL loss）
采样              每 prompt 可 1 条                每 prompt n 条（repeat + uid 分组）
资源              多一个价值网络                   省掉 Critic，更轻
```

## 8. 小结

- `fit()` 是 single-controller 的“剧本”：取数据 → 生成 → 重算 log_prob → (ref/critic) → reward → advantage → 更新 critic/actor → 同步权重，全程用 DataProto 当数据总线、用 WorkerGroup 做分发。
- `core_algos.py` 用注册表把 advantage 估计器和 policy loss 做成插件，改配置即可切换算法。
- **GAE vs GRPO** 的差别（要不要 Critic、怎么算 baseline）在代码里非常直白，是理解各类 RL 算法的最佳切入点。
- advantage 在 driver 本地算，重计算密集的 forward/backward 在 worker 上分布式执行——职责划分清晰。

剩下还有一个关键工程问题没讲：Actor 训练完后的新权重，怎么同步给用另一套引擎跑的 Rollout？这就是下一篇“训练推理混合引擎”的主题。
