---
title: "verl 源码解读（六）：RL 算法详解 —— PPO / GRPO / Dr.GRPO / GSPO / DAPO / OPD"
description: "用 verl 真实代码把 6 个主流 RLHF/RLVR 算法一次讲透：以“算法 = Advantage 估计 × Policy Loss”的统一框架，逐一拆解 PPO、GRPO、Dr.GRPO、GSPO、DAPO、OPD，以及 RLOO/ReMax/REINFORCE++/OPO/GPG 等 Critic-free 变体。"
pubDate: 2026-04-01
tags: ["verl", "强化学习", "PPO", "GRPO", "GSPO", "DAPO", "源码解读"]
---

> **本文是《verl 源码解读》系列第 6 篇**。前五篇讲清了框架机制（[HybridFlow](/blog/verl-02-hybridflow-single-controller/) / [DataProto](/blog/verl-03-dataproto/) / [fit 主循环](/blog/verl-04-ppo-grpo-core/) / [Rollout 引擎](/blog/verl-05-rollout-hybrid-engine/)），本篇聚焦**算法本身**：用 verl 的真实代码，把 6 个主流 RLHF/RLVR 算法一次讲透。
>
> 系列目录：[1 架构](/blog/verl-01-architecture/) · [2 HybridFlow](/blog/verl-02-hybridflow-single-controller/) · [3 DataProto](/blog/verl-03-dataproto/) · [4 PPO/GRPO](/blog/verl-04-ppo-grpo-core/) · [5 Rollout 混合引擎](/blog/verl-05-rollout-hybrid-engine/) · **6 RL 算法详解**
>
> 核心源码：`verl/trainer/ppo/core_algos.py`（advantage 估计器 + policy loss 注册表）、`verl/trainer/distillation/losses.py`（OPD）、`docs/algo/dapo.md`（DAPO）。

## 0. 先建立统一框架

verl 把一个 RL 算法拆成两个正交的可插拔组件：

> **算法 = Advantage 估计（怎么算“这步好不好”）× Policy Loss（怎么把优势变成梯度）**

```python
# 优势估计器注册表
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(...): ...

# policy loss 注册表
@register_policy_loss("gspo")
def compute_policy_loss_gspo(...): ...
```

所有这些算法共享同一个策略梯度骨架：

$$
\mathcal{L}(\theta) = -\,\mathbb{E}\Big[\min\big(r_t(\theta)\,\hat{A}_t,\ \text{clip}(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon)\,\hat{A}_t\big)\Big]
$$

其中 $r_t(\theta)=\dfrac{\pi_\theta}{\pi_{\theta_{old}}}$ 是重要性采样比，$\hat A_t$ 是优势。**各算法的区别，几乎都落在“$\hat A_t$ 怎么算”和“$r_t$ 怎么定义/裁剪”这两点上。** 带着这个框架往下看，一切都很清晰。

下表先给全局观感（后面逐个展开）：

| 算法 | Advantage | 重要性比 | 需要 Critic | 一句话 |
|------|-----------|----------|:---:|--------|
| **PPO** | GAE（逐 token，用 value） | token 级 | ✅ | 经典，稳但重 |
| **GRPO** | 组内归一化（减均值/除标准差） | token 级 | ❌ | 用“同题多答”当 baseline |
| **Dr.GRPO** | 组内只减均值（不除 std） | token 级 | ❌ | 修 GRPO 的长度/难度偏置 |
| **GSPO** | 同 GRPO | **序列级** | ❌ | 把 ratio 提到序列级，更稳 |
| **DAPO** | GRPO | token 级 + clip-higher | ❌ | GRPO + 4 个工程 trick |
| **OPD** | $-$蒸馏损失 当优势 | token 级 | ❌(有 teacher) | 在线策略蒸馏 |

---

## 1. PPO —— 一切的基线

**配置**：`adv_estimator=gae` + `policy_loss.loss_mode=vanilla`，需要 Critic。

### 1.1 Advantage：GAE（广义优势估计）

```python
# core_algos.py: compute_gae_advantage_return
for t in reversed(range(gen_len)):
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]   # TD 误差
    lastgaelam_ = delta + gamma * lam * lastgaelam                          # GAE 递推
    nextvalues  = values[:, t]  * mask[:, t] + (1 - mask[:, t]) * nextvalues
    lastgaelam  = lastgaelam_   * mask[:, t] + (1 - mask[:, t]) * lastgaelam
    advantages_reversed.append(lastgaelam)
advantages = torch.stack(advantages_reversed[::-1], dim=1)
returns = advantages + values              # Critic 的回归目标
advantages = masked_whiten(advantages, response_mask)   # 标准化
```

GAE 需要**逐 token 的价值估计 `values`**，所以 PPO 必须养一个 Critic 网络（和 Actor 同等规模），这是它最大的成本。

### 1.2 Policy Loss：裁剪 + dual-clip

```python
# core_algos.py: compute_policy_loss_vanilla
ratio = torch.exp(log_prob - old_log_prob)                # 重要性采样比
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)   # 标准 PPO 裁剪

# dual-clip：advantage<0 时再加一层下界，防止负优势下梯度爆炸
pg_losses3 = -advantages * clip_ratio_c                   # clip_ratio_c 默认 3.0
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

> PPO 的裁剪保证了“单步更新不会偏离旧策略太远”，这是它稳定的根本。verl 还支持非对称裁剪（`cliprange_low/high` 可分别设），这为 DAPO 的 clip-higher 留了接口。

---

## 2. GRPO —— 干掉 Critic

**配置**：`adv_estimator=grpo` + `vanilla` loss，无 Critic。

PPO 的痛点是 Critic 太贵。GRPO 的核心洞察是：**对每个 prompt 采样 n 条回答，用这组回答得分的均值当 baseline，就不需要价值网络了。**

```python
# core_algos.py: compute_grpo_outcome_advantage
scores = token_level_rewards.sum(dim=-1)       # 每条回答 → 一个标量总分

id2score = defaultdict(list)
for i in range(bsz):
    id2score[index[i]].append(scores[i])        # 按 group id（uid）分组

for idx in id2score:                             # 每组算均值、标准差
    id2mean[idx] = torch.mean(group)
    id2std[idx]  = torch.std(group)

for i in range(bsz):
    if norm_adv_by_std_in_grpo:                  # ★ 关键开关
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)   # GRPO
    else:
        scores[i] = scores[i] - id2mean[index[i]]                                    # Dr.GRPO
scores = scores.unsqueeze(-1) * response_mask    # 广播到每个 token
return scores, scores
```

几个要点：

- `index` 就是主循环里 `repeat(n)` 前给每个 prompt 分配的 `uid`，同题的 n 条回答共享它，构成一个 group。
- 优势是**整条回答一个标量**（广播到所有 token），因为 reward 是 outcome-level（答对/答错），不是逐 token。
- GRPO 公式（原始版）：$\hat A_i = \dfrac{R_i - \text{mean}(R_{group})}{\text{std}(R_{group})+\epsilon}$

> GRPO 不仅用于文本，在图像生成里也有一系列变体（Flow-GRPO/DanceGRPO 等），详见[《图像生成与强化学习》](/blog/image-generation-and-rl/)。

---

## 3. Dr.GRPO —— 修掉 GRPO 的隐藏偏置

Dr.GRPO（Dr. = “Done Right”）发现 GRPO 的两处系统性偏置，并用**极简的改动**修正。在 verl 里，它**不是独立算法，而是 GRPO 的一个配置开关**：

```python
norm_adv_by_std_in_grpo = False   # 这就是 Dr.GRPO
```

对应上面代码里走 `scores[i] = scores[i] - id2mean[...]`——**只减均值，不除标准差**。

### 为什么要去掉除以 std？

GRPO 除以组内 std 会引入**难度偏置**：

- 对于**简单题**（n 条答案大多对，std 很小），`/std` 会把优势放得很大 → 模型过度关注简单题；
- 对于**难题**（答案五花八门，std 大），`/std` 把优势压小 → 难题学得少。

这相当于给不同难度的题目**隐式地加了不同权重**，并非我们想要的。Dr.GRPO 去掉 std 归一化，让所有组的优势尺度一致（论文还同时修正了 loss 聚合的长度偏置，见第 7 节）。

> 一句话：**GRPO = 减均值 ÷ 标准差；Dr.GRPO = 只减均值。** verl 里切一个 `norm_adv_by_std_in_grpo` 布尔值即可。

---

## 4. GSPO —— 把重要性比从 token 级提到序列级

**配置**：`policy_loss.loss_mode=gspo`（优势仍用 grpo）。论文：Qwen 团队 arXiv:2507.18071。

GRPO/PPO 的重要性比 $r_t$ 是**逐 token** 的。GSPO 指出：对于长序列，token 级 ratio 的方差会累积、不稳定（尤其 MoE 模型）。解法是把 ratio 定义到**整个序列**上。

### 序列级重要性比

$$
s_i(\theta) = \left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{old}}(y_i\mid x)}\right)^{1/|y_i|}
= \exp\!\left(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta}{\pi_{\theta_{old}}}\right)
$$

注意那个 $1/|y_i|$ 指数——**对序列长度做几何平均**，这样长短序列的 ratio 尺度一致。

```python
# core_algos.py: compute_policy_loss_gspo
negative_approx_kl = log_prob - old_log_prob
seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
# 序列级：对数比在序列内求平均
negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

# 关键技巧：组合成 token 级但梯度走序列级（sg = stop-gradient = .detach()）
# s_{i,t} = sg[s_i] · π_θ / sg[π_θ]
log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
seq_importance_ratio = torch.exp(log_seq_importance_ratio)

pg_losses1 = -advantages * seq_importance_ratio
pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)

# ★ GSPO 必须用序列级聚合
pg_loss = agg_loss(..., loss_agg_mode="seq-mean-token-mean", ...)
```

那行 `log_prob - log_prob.detach() + seq_ratio.detach()` 很巧妙：**数值上等于序列级 ratio，但梯度只从当前 `log_prob` 流出**，从而既用了序列级的比值、又保留逐 token 的梯度位置。配合 `seq-mean-token-mean` 聚合，实现真正的序列级优化。

> 直觉：PPO/GRPO 问“这个 **token** 相对旧策略变了多少”；GSPO 问“这**整条回答**相对旧策略变了多少”，后者方差更小、训练更稳。

---

## 5. DAPO —— GRPO + 4 个工程 trick

**配置**：`adv_estimator=grpo` + `vanilla` loss + 一组 trick 开关。论文：字节&清华 arXiv:2503.14476（AIME 2024 达 50 分，超过 DeepSeek-R1-Zero-32B）。

DAPO 不改 advantage 公式，而是叠加 4 个关键工程技巧（来自 `docs/algo/dapo.md`）：

### Trick 1: Clip-Higher（解耦上下裁剪）

```yaml
actor_rollout_ref.actor:
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28        # 上界放宽
```

上界放宽（0.2→0.28），给低概率 token **更大的上升空间**，缓解 GRPO 训练中的**熵坍缩**（模型过早变得确定、丧失探索）。实现就是 vanilla loss 里早已支持的非对称裁剪。

### Trick 2: Dynamic Sampling（动态采样 + 组过滤）

```yaml
algorithm.filter_groups:
  enable: True
  metric: acc
  max_num_gen_batches: 10
```

```python
# 伪代码（dapo.md）：过滤掉“全对”或“全错”的组，不够一个 batch 就继续采
if num_prompt_in_batch < prompt_bsz:
    num_gen_batches += 1
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        continue        # 继续生成更多样本
```

**一个组若 n 条回答全对（acc=1）或全错（acc=0），组内优势全为 0，对训练零贡献。** 动态采样把这些“废组”过滤掉、补采有效组，保证每个 batch 的梯度都有信息量。

### Trick 3: Token-level Loss（token 级损失聚合）

```yaml
actor_rollout_ref.actor.loss_agg_mode: "token-mean"
```

默认就是 `token-mean`：对 mini-batch 内**所有序列的所有 token** 求平均（而非先序列内平均再序列间平均）。这让长回答里的每个 token 都平等贡献梯度，避免长回答被稀释——对长 CoT 推理很关键。

### Trick 4: Overlong Reward Shaping（超长惩罚）

```python
# dapo.md: 对接近长度上限的回答做线性软惩罚
if overlong_buffer_cfg.enable:
    exceed_len = valid_response_length - (max_resp_len - overlong_buffer_len)
    overlong_reward = min(-exceed_len / overlong_buffer_len * penalty_factor, 0)
    reward += overlong_reward
```

在硬截断之前设一个“缓冲区”，长度进入缓冲区就**线性扣分**，温和地引导模型别写过长，而不是硬生生截断（截断会产生噪声梯度）。

> DAPO 的启示：**算法创新 ≠ 改损失函数公式。** 它的 SOTA 来自对 GRPO 一堆“工程细节”的精细打磨——这正是 verl 把这些做成配置开关的价值。

---

## 6. OPD —— 在线策略蒸馏

**代码**：`verl/trainer/distillation/losses.py`。参考 Thinking Machines 的 on-policy distillation 博客和 arXiv:2604.13016。

OPD（On-Policy Distillation）和前面几个不太一样：它有一个 **teacher 模型**，目标是让 student 在**自己生成的轨迹**上，模仿 teacher 的分布。关键创新在于把蒸馏和策略梯度统一起来：

```python
# losses.py: distillation_loss，当 use_policy_gradient=True
if loss_config.use_policy_gradient:
    policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
    distillation_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=-distillation_losses.detach(),   # ★ 用“负蒸馏损失”当优势！
        response_mask=response_mask,
        ...
    )
```

**最精妙的一行**：`advantages = -distillation_losses.detach()`。

- 蒸馏损失（student 与 teacher 的 KL）越小 → 优势越大 → 策略梯度鼓励这类 token；
- 于是蒸馏被包装成了一个 RL 问题：“把‘接近 teacher’当作奖励信号”。

这样 OPD 就能**复用整套 PPO/GRPO 的策略梯度机制**（on-policy、重要性采样、裁剪都能用），而不是传统的离线 SFT 蒸馏。

### 两种蒸馏损失

OPD 的“接近 teacher”有两种度量（都注册在表里）：

```python
# 1. top-k 前向 KL（teacher 提供 top-k log probs）
@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))

# 2. 单样本 KL 估计器（复用 core_algos 的 kl_penalty）
@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True))
```

另外 `use_policy_gradient=False` 时，就退化成**直接把蒸馏损失当监督损失反传**（传统蒸馏，arXiv:2306.13649）。verl 用一个配置把“在线策略蒸馏”和“离线蒸馏”统一在同一套代码里。

> OPD 的价值：**用一个更强的 teacher 在 student 自己的分布上纠偏**，比纯 SFT 蒸馏（teacher 分布）更能避免 exposure bias，比纯 RL 更 sample-efficient。

**名词解释**：**Exposure bias（暴露偏差）** 指训练和推理时模型看到的前缀分布不一致。SFT 蒸馏只在 teacher 写出的“完美前缀”上训练，推理时 student 一旦自己写错一步，就会进入训练中从未见过的状态，不知道如何恢复，导致错误越积越多。OPD 的轨迹由 student 自己生成，所以 teacher 能在这些（包括出错的）状态上逐 token 指导 student 纠正。**Sample-efficient（样本高效）** 指用更少的采样学到同样多的东西。纯 RL 的一条长轨迹最终只得到一个“对/错”标量，很难判断是哪一步出了问题；OPD 由 teacher 给每个 token 打分（$\log\pi_{teacher}-\log\pi_{student}$），同样一次采样提供的有效信号多得多，因此收敛更快。

---

## 7. 一个容易被忽视的关键：loss 聚合模式

前面 DAPO/GSPO/Dr.GRPO 都提到了 `loss_agg_mode`，它看似不起眼，却会显著影响结果：

```python
# core_algos.py: agg_loss
if loss_agg_mode == "token-mean":                 # 所有 token 一起平均
    loss = masked_mean(loss_mat, loss_mask)
elif loss_agg_mode == "seq-mean-token-sum":       # 序列内求和，再序列间平均
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
    loss = torch.mean(seq_losses)
elif loss_agg_mode == "seq-mean-token-mean":      # 序列内平均，再序列间平均
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
    loss = torch.mean(seq_losses)
```

- `token-mean`（DAPO 用）：长回答的 token 多，整体权重大 → 鼓励详细推理。
- `seq-mean-token-mean`（GSPO 用）：每条序列等权，与长度无关。
- **这正是 Dr.GRPO 修正的另一个“长度偏置”点**：聚合方式选不对，会隐式偏向长/短回答。

---

## 7.5 其他 Critic-free 算法（RLOO / ReMax / REINFORCE++ / OPO / GPG）

`examples/` 下还有一批算法，它们**都不需要 Critic**，本质是 REINFORCE 配不同的 baseline / returns 处理。只需记住“baseline 怎么构造”即可，代码都在 `core_algos.py`。

### RLOO（Leave-One-Out）—— `adv_estimator=rloo`

组内留一法：每条回答的 baseline 是**同组其余回答**的均值（无偏估计）。实现上等价于一个缩放形式：

```python
# compute_rloo_outcome_advantage
response_num = len(id2score[index[i]])        # 组内回答数 n
scores[i] = scores[i] * n/(n-1) - id2mean * n/(n-1)   # ≈ R_i - mean(其余 n-1 条)
```

> 和 GRPO 很像，但 baseline 排除自己、且不除 std，理论上更无偏。

### ReMax —— `adv_estimator=remax`

baseline 用**贪心解码（greedy）的那条回答得分**。所以主循环会额外跑一次 greedy 生成当基线：

```python
# compute_remax_outcome_advantage
returns = (token_level_rewards * mask).flip(-1).cumsum(-1).flip(-1)   # 累积奖励
advantages = returns - reward_baselines.unsqueeze(-1) * mask          # 减去 greedy baseline
```

> 只需多采 1 条 greedy，轻量；适合显存紧张场景。

### REINFORCE++ —— `adv_estimator=reinforce_plus_plus`

**不分组**，用带折扣的累积回报 + 全局 whiten（标准化）当优势：

```python
# compute_reinforce_plus_plus_outcome_advantage
for t in reversed(range(T)):                 # 带 gamma 折扣的回报
    running_return = r[:, t] + gamma * running_return
    returns[:, t] = running_return * mask[:, t]
advantages = masked_whiten(returns, mask)    # 全 batch 标准化
```

- **REINFORCE++-baseline**（`reinforce_plus_plus_baseline`）：先按组减均值，再全局 whiten，结合了分组 baseline 和全局标准化。

### OPO —— `adv_estimator=opo`

组内 baseline 用**长度加权**的平均分（偏向长回答的最优基线）：

```python
# compute_opo_outcome_advantage
id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()   # 长度加权均值
scores[i] = scores[i] - id2bsl[index[i]]
```

### GPG —— `adv_estimator=gpg`

组内减均值后，再乘一个全局归一化因子 `f_norm`，对纯策略梯度做了校正（细节见源码）。

### 小抄

| 算法 | baseline 来源 | 是否分组 | 一句话 |
|------|--------------|:---:|--------|
| GRPO | 组内均值 ÷ std | ✅ | 标准组内归一化 |
| RLOO | 组内留一均值 | ✅ | 无偏，排除自己 |
| ReMax | greedy 解码得分 | ❌ | 多采 1 条 greedy |
| REINFORCE++ | 全局 whiten | ❌ | 折扣回报 + 标准化 |
| RF++-baseline | 组内均值 + 全局 whiten | ✅ | 两者结合 |
| OPO | 组内长度加权均值 | ✅ | 偏向长回答 |
| GPG | 组内均值 + f_norm | ✅ | 带校正因子 |

> 它们都接 `vanilla` policy loss，区别**只在优势估计器**——正是第 0 节“算法 = Advantage × Loss”框架的最佳例证。

---

## 8. 总结：一张图看懂演进脉络

```
                PPO（GAE + Critic，稳但重）
                  │  去掉 Critic，用"同题多答"组内 baseline
                  ▼
                GRPO（减均值 ÷ 标准差）
         ┌────────┼─────────────┬──────────────┐
         │修偏置   │ 换 ratio 粒度 │ 叠工程 trick  │ 换"奖励"来源
         ▼         ▼             ▼               ▼
     Dr.GRPO     GSPO          DAPO            OPD
   (只减均值)  (序列级 ratio) (clip-higher+   (−蒸馏损失
              + seq 聚合)     动态采样+        当优势)
                              token-loss+
                              overlong shaping)
```

**一句话收束**：这些算法共享同一个策略梯度骨架，差异集中在 **① 优势怎么算（要不要 Critic、怎么做 baseline、修哪些偏置）** 和 **② 重要性比怎么定义与裁剪（token 级还是序列级、裁剪上下界、聚合方式）**。verl 用两个注册表把它们做成可自由组合的插件——读懂这一篇，再看任何新 RL 算法，都能迅速定位它改的是哪一块。

## 配置速查

| 算法 | 关键配置 |
|------|----------|
| PPO | `adv_estimator=gae`，`policy_loss.loss_mode=vanilla`，启用 critic |
| GRPO | `adv_estimator=grpo`，`norm_adv_by_std_in_grpo=True` |
| Dr.GRPO | `adv_estimator=grpo`，`norm_adv_by_std_in_grpo=False`，`loss_agg_mode=seq-mean-token-sum` |
| GSPO | `policy_loss.loss_mode=gspo`，`loss_agg_mode=seq-mean-token-mean` |
| DAPO | `adv_estimator=grpo` + `clip_ratio_low/high` + `filter_groups.enable` + `loss_agg_mode=token-mean` + `overlong_buffer.enable` |
| OPD | `distillation.enabled=True` + `distillation_loss.use_policy_gradient=True` + teacher 模型 |
