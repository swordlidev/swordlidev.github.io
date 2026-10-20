---
title: "策略梯度与重要性采样：RLVR 的几个基本量"
description: "从策略梯度出发，梳理重要性采样、PPO 裁剪目标，以及 token 级与序列级权重带来的方差差异，作为高效策略优化工作的背景笔记。"
pubDate: 2026-10-18
updatedDate: 2026-10-19
tags: ["RL", "LLM", "笔记"]
---

这篇笔记整理训练大模型做推理时常用的强化学习目标，顺带记录一下 token 级与序列级重要性权重在方差上的差别 —— 这正是我们在 TGPO 里关心的问题之一。

## 策略梯度

记策略为 $\pi_\theta(a\mid s)$，轨迹 $\tau=(s_0,a_0,\dots)$ 的回报为 $R(\tau)$。目标是最大化期望回报：

$$
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\big[R(\tau)\big].
$$

利用对数导数技巧 $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$，可得经典的策略梯度：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\!\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t\mid s_t)\, A_t\right],
$$

其中 $A_t$ 是优势函数（advantage），用于降低方差。

## 重要性采样

在线采样代价很高，常希望用旧策略 $\pi_{\theta_{\text{old}}}$ 采得的数据来更新当前策略。引入重要性权重

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},
$$

则目标改写为 $\mathbb{E}_{\pi_{\theta_{\text{old}}}}[\, r_t(\theta)\, A_t\,]$。当两个分布差异过大时，$r_t$ 的方差会迅速膨胀，这是离线/半离线 RL 不稳定的主要来源。

## PPO 裁剪目标

PPO 通过裁剪把更新限制在信任域内：

$$
\mathcal{L}^{\text{clip}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t A_t,\; \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\, A_t\big)\Big].
$$

直觉上，当 $A_t>0$（动作比平均好）时，更新被限制在 $r_t \le 1+\epsilon$；当 $A_t<0$ 时，限制在 $r_t \ge 1-\epsilon$。这样避免一步走得太远。

## token 级 vs 序列级权重

对一条长度为 $T$ 的回答，token 级重要性权重是逐 token 的 $r_t$；而序列级权重是整条序列的乘积

$$
r_{1:T}(\theta) = \prod_{t=1}^{T} r_t(\theta) = \exp\!\Big(\sum_{t=1}^{T} \log r_t(\theta)\Big).
$$

- **token 级**：信用分配更细，但高方差 —— 单个 token 的比值波动会直接进入梯度。
- **序列级**：更稳定，但把整条序列视为一个整体，credit assignment 偏粗。

一个折中的思路是对 $\log r_t$ 做"门控"或锚定，抑制少数异常 token 的贡献，同时保留序列级的稳定性。写成带权和：

$$
\tilde{r}(\theta) = \exp\!\Big(\sum_{t=1}^{T} g_t \,\log r_t(\theta)\Big),\qquad g_t \in [0,1],
$$

其中 $g_t$ 是信息门控系数。$g_t\equiv 1$ 退化为序列级权重，$g_t$ 为示性函数时近似 token 级。这类设计在 RLVR 中能同时兼顾稳定性与效率。

## 小结

| 方案 | 信用分配 | 方差 | 稳定性 |
| ---- | -------- | ---- | ------ |
| token 级 | 细 | 高 | 低 |
| 序列级 | 粗 | 低 | 高 |
| 门控折中 | 可调 | 中 | 中-高 |

更完整的分析见我们的工作 [TGPO (ICML 2026)](/publications/)。

```python
import torch

def clipped_pg_loss(logp, logp_old, adv, eps=0.2):
    # logp, logp_old, adv: [batch, seq_len]
    ratio = torch.exp(logp - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(unclipped, clipped).mean()
```

下次聊聊如何把"门控系数" $g_t$ 和奖励的稠密化结合起来。
