---
title: "Rubrics：把“好不好”拆成可验证的评分表"
description: "从 HealthBench、TICK 到 RLCF、RaR、OnlineRubrics，梳理 rubric（结构化评分表）如何同时成为 LLM 的评估基准与 RL 奖励信号：为什么标量奖励不够用、rubric 怎么自动生成、怎么用于 DPO/GRPO 训练、静态 rubric 的天花板在哪，以及一套可落地的设计原则与三类应用路径。"
pubDate: 2025-11-12
tags: ["Rubric", "LLM", "评估", "强化学习", "后训练", "对齐"]
---

> 把一个回答压缩成 “7/10”，你丢掉的不只是精度，而是“到底哪里不对”。Rubric 做的事很简单：把这个数字拆回成一组可独立验证的 YES/NO。

## 一句话说清 Rubric

**Rubric = 针对每条指令/问题单独写的结构化评分表**，由一组 YES/NO 检查项（criteria）组成，每项带权重，逐条判断回答是否达标。

```
传统方式:  instruction → response → 🤔 "7/10"
Rubric:    instruction → response → ✅ 事实正确? ✅ 包含剂量? ❌ 安全警告? → 2/3
```

差别在于：传统评估把“好不好”压成一个标量，rubric 把“好”拆成了可独立验证的子目标。

## 为什么标量奖励不够用

| 痛点 | 说明 | Rubric 如何解决 |
|------|------|-----------------|
| **信息瓶颈** | 标量 RM 把多维偏好压成一个数字，信号丢失 | 每条 criterion 独立评分，保留多维信息 |
| **Reward Hacking** | 模型学会“写长文”“加 emoji”骗分 | 评分锚定在具体事实/行为上，难以投机 |
| **不可解释** | 打了 6 分，不知道那 4 分扣在哪 | 每条 YES/NO 自带解释，可 debug |

这三点在开放式长文任务上尤其致命——没有标准答案可比对，RLVR 那套“脚本判对错”直接失效，而偏好模型给出的单一分数既噪声大又无法归因。

---

## 五篇论文，两条主线

这个方向目前有两条清晰的主线：**用 rubric 做评估**（HealthBench、TICK），和**用 rubric 做训练信号**（RLCF、RaR、OnlineRubrics）。

### HealthBench：rubric 评估的标杆

| 项目 | 内容 |
|------|------|
| 来源 | OpenAI，[arXiv:2505.08775](https://arxiv.org/abs/2505.08775)，2025-05 |
| 定位 | 医疗领域 LLM 评估基准 |
| 规模 | 5000+ 多轮对话 × 48,562 条 rubric criteria，262 位医生参与 |

核心设计：每条对话配一份**医生手写的 rubric**，criteria 分值从 -10 到 +10（负分直接惩罚有害内容）；覆盖 7 个主题（急诊、全球健康、临床数据等）× 5 个评分轴（准确性、完整性、沟通等）；由模型打分器逐条判定是否满足，加权求和得总分。

几个值得注意的结果：

- o3 得分 60%，GPT-4o 仅 32%——benchmark 远未饱和；
- GPT-4.1 nano 超过了 GPT-4o，且便宜 25 倍；
- **模型打分与医生打分的一致性 ≈ 医生之间的一致性**，这是 rubric 自动化评分可信的关键证据；
- 另发布 HealthBench Hard（最强模型仅 32%）与 HealthBench Consensus（34 条医生共识 criteria）。

> 启示：在医疗、法律、金融这类高风险领域，rubric 的价值不只是打分更准，而是能定位“到底错在哪”。

### TICK：让 LLM 自己生成 checklist

| 项目 | 内容 |
|------|------|
| 来源 | Oxford + Cohere，[arXiv:2410.03608](https://arxiv.org/abs/2410.03608)，2024-10 |
| 思想 | 用 LLM 自动为每条指令生成 YES/NO checklist，替代人工评分 |

```
指令 → LLM 生成 checklist（5-15 条 YES/NO 问题）
     → LLM 逐条回答 → 汇总 Pass Rate
```

| 指标 | 直接打分 | TICK |
|------|----------|------|
| LLM-人类偏好一致率 | 46.4% | **52.2%** |
| 人类标注者间一致性 (κ) | 0.194 | **0.256** |

进阶用法 **STICK（Self-TICK）**：让模型拿 checklist 评自己的回答再定向修正，推理任务 +7.8%；用 checklist 做 Best-of-N 选择，WildBench +6.3%。对照组很说明问题——无结构的 Self-Refine 在推理任务上反而退化，而有 checklist 约束的自我修正是稳定提升的。

> 启示：checklist 不必人写。LLM 生成的质量已接近训练有素的标注员，而且即时生成、零成本扩展。

### RLCF：把 checklist 当成 RL 奖励

| 项目 | 内容 |
|------|------|
| 来源 | CMU + Apple，[arXiv:2507.18624](https://arxiv.org/abs/2507.18624)，2025-07，NeurIPS 2025 |
| 思想 | 从指令中提取 checklist → 逐项打分 → 作为 DPO 的奖励 |

```
指令 → 候选回答对比法生成 checklist
     → AI 打分 + 程序化验证（如“包含 3 个逗号”用代码检查）
     → 加权求和 → DPO 训练
```

在 Qwen2.5-7B-Instruct 基线上的对比：

| 方法 | FollowBench | InFoBench | Arena-Hard | 全部提升？ |
|------|-------------|-----------|------------|--------------|
| Reward Model (Skywork) | +1.2 | -0.8 | +2.1 | ❌ |
| AI Judge (Likert) | +0.5 | +1.3 | -1.0 | ❌ |
| **RLCF (Checklist)** | **+4.0** | **+6.0** | **+3.0** | **✅** |

RLCF 是唯一在全部 5 个 benchmark 上都提升的方法。三个关键细节：

- **候选回答对比法 > 直接提取法**：先生成不同质量的回答，再让 LLM 总结“差在哪”，这样得到的 checklist 更客观、更全面；
- **程序化验证**：字数、格式、是否包含特定词这类可精确检查的条件，生成验证代码执行，比 LLM 判断准得多；
- **通用规则正则化**：所有 checklist 附加“不跑题”“语气匹配”两条通用规则，堵住 reward hacking 的口子。

### RaR：rubric 驱动的 on-policy RL

| 项目 | 内容 |
|------|------|
| 来源 | Scale AI，[arXiv:2507.17746](https://arxiv.org/abs/2507.17746)，2025-07 |
| 思想 | 把 rubric 直接作为 GRPO on-policy RL 的奖励函数 |

为每条 prompt 生成 7–20 条 criteria，按重要性分类加权：

| 类别 | 权重 | 含义 |
|------|------|------|
| Essential | 1.0 | 核心要求，必须满足 |
| Important | 0.7 | 重要但非致命 |
| Optional | 0.3 | 锦上添花 |
| Pitfall | 0.9 | 常见错误，避免则加分 |

聚合方式有两种：**Explicit**（逐条独立评分再加权求和，透明可控）和 **Implicit**（把全部 criteria 一次性给 judge，让它出一个总分，更简单）。

| 模型 (Qwen2.5-7B) | HealthBench | GPQA-Diamond |
|-------------------|-------------|--------------|
| Direct-Likert（传统打分） | baseline | baseline |
| **RaR-Implicit** | **+31%** | **+7%** |

一个容易被忽略但很实用的发现：**rubric 对小 judge 更友好**。传统 Likert 打分对 judge 模型大小非常敏感，小模型打分方差极大；而 rubric 把评估拆成简单的 YES/NO 判定后，小 judge 也能稳定打分，跨 judge 的评分方差显著下降。

> 启示：judge 模型受成本约束时，rubric 能明显弥补能力差距——这对线上评估系统的成本优化很直接。

### OnlineRubrics：让 rubric 跟着训练一起演化

| 项目 | 内容 |
|------|------|
| 来源 | Scale AI，[arXiv:2510.07284](https://arxiv.org/abs/2510.07284)，2025-10 |
| 问题 | 静态 rubric 在训练中会被 hack，新出现的好/坏模式无法覆盖 |

```
每个训练步骤:
  1. 当前策略 + 参考策略 各生成一批回答
  2. 配对比较 → LLM 提取两者差异 → 转化为新 criteria
  3. 去重 → 合并进原有 rubric → 用于本步奖励计算
```

为什么是配对比较而不是逐条审视？Point-wise 只看一个回答容易遗漏问题，pairwise 看两个回答的差异更容易暴露问题——这和人类偏好标注的逻辑一致：**比较比绝对判断更容易做对**。

| 方法 | AlpacaEval | GPQA | ArenaHard |
|------|------------|------|-----------|
| 静态 Rubric only | baseline | baseline | baseline |
| **+ OnlineRubrics** | **+5~8%** | **+4~6%** | **+3~5%** |

有意思的是自动发现的 criteria 主题高度集中在几类：透明性（不自吹自擂）、实用性（给具体操作步骤）、结构化（分点清晰）、推理深度（不浮于表面）——基本就是模型在训练中逐渐学会的几种“投机姿势”的反面。

---

## 全景对比

| 论文 | 用途 | Rubric 生成方式 | 用于训练 | 核心贡献 |
|------|------|----------------|-------------|---------|
| **HealthBench** | 评估 | 医生手写 | ❌ 仅评估 | 定义 rubric 评估的黄金标准 |
| **TICK** | 评估 + 推理提升 | LLM 自动生成 | ❌ 推理时用 | 证明 LLM 生成的 checklist ≈ 人类 |
| **RLCF** | RL 训练 (DPO) | LLM + 候选对比法 | ✅ Off-policy | Checklist 奖励是唯一全面提升的方法 |
| **RaR** | RL 训练 (GRPO) | LLM 从参考答案生成 | ✅ On-policy | Rubric 对小 judge 更友好 |
| **OnlineRubrics** | RL 训练 (GRPO) | 训练中动态生成 | ✅ On-policy | 解决静态 rubric 被 hack 的问题 |

---

## 设计 Rubric 的四条原则

来自 RaR，实际写过几轮 rubric 之后会发现这四条基本是必须项：

**1. 专家锚定**：criteria 应基于参考答案或领域专家知识。没有专家就用强模型（GPT-4o、o3）配合参考答案生成，不要凭空写。

**2. 全面覆盖**：不只检查“对不对”，还要覆盖完整性、安全性、逻辑性、风格。特别是要**包含 Pitfall 类 criteria**——常见错误应该被显式检测并扣分。

**3. 重要性分级**：Essential > Important > Optional。医疗场景里事实准确性 >> 表述优美，这个差距必须用权重或分类标签体现出来，否则模型会去优化最容易拿分的那一维。

**4. 自包含可判定**：每条 criterion 要能独立判断，不依赖外部知识。判断标准是：一个人（或 LLM）只看这一条，能不能给出确定的 YES/NO。做不到就说明这条写得太主观，需要拆开或改写。

---

## 三类应用路径

### 场景一：评估现有模型（不训练）

走 TICK 的路子，零人工成本、可解释、比直接打分更准：

```python
for instruction in eval_set:
    checklist = llm.generate_checklist(instruction)   # 5-15 条 YES/NO
    response = model.generate(instruction)
    scores = [llm.judge(q, response) for q in checklist]
    pass_rate = sum(scores) / len(scores)
```

### 场景二：RL 后训练对齐

走 RLCF 的候选对比法：

1. 准备指令集（如 WildChat 过滤后的数据）；
2. 用不同大小的模型生成候选回答；
3. 让强 LLM 对比候选 → 生成 checklist + 权重；
4. 训练时采样回答对 → checklist 打分 → DPO。

关键配置：checklist 生成器用 Qwen2.5-72B 或 GPT-4o；格式/数量约束走程序化验证；附加“不跑题 + 语气匹配”两条通用规则防 hack；偏好对只保留差异最大的 40%。

### 场景三：持续训练

走 OnlineRubrics：在 GRPO 训练中每 N 步用配对比较法补充新 criteria，重点盯模型新长出来的投机行为（冗长前言、自我吹捧等），同时靠去重保证 criteria 集合不膨胀。

---

## 小结

1. **Rubric/Checklist 是目前最有前景的结构化奖励方案**，在可解释性和稳定性上全面优于标量奖励。
2. **不需要人写 rubric**。LLM 生成的质量已接近人类专家，关键在生成方法——候选对比法明显优于直接提取法。
3. **Rubric 不只是评估工具，更是训练信号**。从 TICK（推理时用）到 RLCF/RaR（训练时用），rubric 正在重塑后训练范式。
4. **静态 rubric 有天花板**。训练中动态演化 criteria 能额外带来 5–8% 提升，尤其在防 reward hacking 上。
5. **小 judge 也能用好 rubric**。把评估拆成简单 YES/NO，降低了对 judge 能力的要求，意味着线上评估可以用更小更便宜的模型。

把这套方法具体落到深度研究类 Agent 的数据链路上，可以继续看[《Deep Research 后训练数据工程》](/blog/deep-research-rl-data-engineering/)。

---

## 附录：论文与资源

| 论文 | 链接 |
|------|------|
| RaR (Rubrics as Rewards) | [arXiv:2507.17746](https://arxiv.org/abs/2507.17746) |
| HealthBench | [arXiv:2505.08775](https://arxiv.org/abs/2505.08775) |
| TICK | [arXiv:2410.03608](https://arxiv.org/abs/2410.03608) |
| RLCF | [arXiv:2507.18624](https://arxiv.org/abs/2507.18624) |
| OnlineRubrics | [arXiv:2510.07284](https://arxiv.org/abs/2510.07284) |

| 开源资源 | 链接 |
|---------|------|
| HealthBench 数据 + 代码 | [github.com/openai/simple-evals](https://github.com/openai/simple-evals) |
| RaR-Medicine 数据集 | [HuggingFace](https://huggingface.co/datasets/anisha2102/RaR-Medicine) |
| RaR-Science 数据集 | [HuggingFace](https://huggingface.co/datasets/anisha2102/RaR-Science) |
| RLCF 代码 + 数据 | [github.com/viswavi/RLCF](https://github.com/viswavi/RLCF) |
