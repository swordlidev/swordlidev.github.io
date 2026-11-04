---
title: "Jev 与 System One 决策模型：原理、校准与工程落地"
description: "Jev 不生成文本，只对预定义选项输出带校准概率的结构化判断。本文拆解它的三种原语、confidence 的真实公式、为什么快一到两个数量级、架构假说与 RLCD，以及校准测量、阈值选择、置换去偏等可直接落地的工程方法。"
pubDate: 2026-10-31
tags: ["Jev", "System One", "决策模型", "概率校准", "Agent", "LLM 工程"]
---

> **本文要解决的问题**：2026 年 9 月 Jev 发布后，"不会写字的模型"成了 Agent 工程里的高频词。它是什么、凭什么快两个数量级、概率能不能信、怎么接进生产、开源能不能平替？
>
> 说明：Jev 本体闭源，涉及架构与训练的部分会标注"官方披露 / 第三方实测 / 推测"，示意代码以看清机制为目的。

## 一、Jev 是什么

**Jev** 是 TypeSafe AI（旧金山，创始人 Diogo Almeida 为前 OpenAI 研究员、RLHF 共同发明人之一，4000 万美元种子轮）于 **2026-09-15** 发布的首个 **System One 模型**，当前版本 `jev-1.13.0`。

**一句话**：不生成文本，只对预定义选项输出带校准概率的结构化判断。

名字来自卡尼曼《思考，快与慢》：GPT、Claude 是逐 token 推演的 System 2，而软件里大量判断（工单归谁、回复合不合格、命令危不危险）只需要 System 1。官方的心智模型是 **"智能 if 语句"**：代码能算的条件照常写，需要主观判断的条件交给 Jev。

| 维度 | Jev | 传统 LLM（含 JSON mode） |
|------|-----|---------|
| 输出 | 预声明集合内的类型化答案 + 概率分布 | 自由文本 / 受约束解码的 JSON |
| 延迟 | 70–500ms（官方，美西测试） | 数秒–数十秒 |
| 计费 | 输入 $0.042 / M tokens，**输出免费** | 输入输出都计费 |
| 概率 | 原生返回，训练目标就是校准 | logprobs 只能近似，通常过度自信 |
| "幻觉" | 不会输出集合外的值，但**会选错** | 会编造 |

"零幻觉"的准确含义是**格式永远合法 ≠ 判断永远正确**。真正的差异在于：原生概率分布、以校准为训练目标、所有问题并行评估、数量级更低的延迟与成本。

| 时间 | 事件 |
|------|------|
| 2026-09-15 | Jev 发布，有限早期访问 |
| 2026-09-18 | TechCrunch 报道 Jev **完全用合成数据训练**；Vercel 披露 24h 内 **~13% 付费团队**调用 |
| 2026-09-20 | Raschka 发文《It's Easy to Dismiss Jev as Just a Classifier》 |
| 2026-10-01 | Cloudflare 发布开源 Clef，给出同名 "RLCD" 的定性描述 |
| 2026-10-13 | 学术工作 Jeva（arXiv:2610.18773）提交 |

## 二、三种原语与 API 契约

设计原则：**Atomic questions, composed in code**。一个请求 = 一个 `state`（字符串、JSON 或文本数组）+ 若干类型化问题；各问题**独立**评估同一个 state，一次并行返回。

| 原语 | 问什么 | 主要返回值 | 代码里怎么用 |
|------|--------|-----------|-------------|
| **Choice** | 属于哪个预定义类别 | `choice`、`probabilities`、`confidence` | `switch` |
| **Score** | 落在有序刻度的哪一档 | `score`、`probabilities`、`confidence` | 与阈值比较 |
| **Noul** | 某命题是否为真 | `noul`（0–1 概率） | 阈值 + `if` |

> 字段名按 2026-09 官方文档整理，接入时以 SDK 类型定义为准。

### 2.1 Choice：多选一

```json
{
  "state": "用户说：我的账号已经三天登不上了，每次都报 502",
  "questions": [{
    "type": "choice",
    "question": "这个工单应该分派给哪个团队？",
    "options": ["billing", "infrastructure", "account-security", "product-bug"]
  }]
}
```

```json
{
  "answers": [{
    "type": "choice",
    "choice": "infrastructure",
    "probabilities": { "billing": 0.02, "infrastructure": 0.89, "account-security": 0.06, "product-bug": 0.03 },
    "confidence": 0.853
  }]
}
```

注意 `confidence` 是 0.853 而不是 0.89（见 2.4）。契约细节：

- **最多 255 个选项**，模型同时读选项名和描述——选项名本身就是 prompt，必要时附描述。
- **封闭集合**：概率和为 1，**必然**选一个。可能出现列表外输入时，加 `other` 或另加 Noul 判断"都不合适"。
- **只支持单选**。多标签用多个 Noul 并行；超过 255 个用分层 Choice。

### 2.2 Score 与 Noul

**Score** 返回的 `score` 不是档位下标，而是**按概率加权的期望档位**（从 0 编号），可能是非整数：

$$
\text{score} = \sum_{k=0}^{K-1} k \cdot p_k
$$

例如 5 档概率 `[0.01, 0.04, 0.15, 0.60, 0.20]`，score = **2.94**。用法是 `score >= 2.5` 这类阈值判断，而不是 `== 3`。

**Noul** 名字取自 Bernoulli 分布，返回命题为真的概率。**Noul 0.5 表示"真假难分"，不表示"中等程度"**，衡量程度要用 Score。

### 2.3 confidence 不是准确率

这是官方文档里最容易被忽略、也最影响阈值设计的一条。Choice 的 confidence 是**相对均匀分布的归一化 margin**：

$$
\text{confidence} = \frac{p_{\max} - 1/K}{1 - 1/K}
$$

上例 $K=4$、$p_{\max}=0.89$，得 $(0.89-0.25)/0.75 \approx 0.853$。同样 $p_{\max}=0.6$，K 不同差异很大：

| K | 2 | 3（官方示例） | 4 | 10 |
|:---:|:---:|:---:|:---:|:---:|
| confidence | 0.200 | 0.400 | 0.467 | 0.556 |

工程推论：

1. **阈值和 K 绑定**：加减选项后，基于 confidence 的阈值必须重新标定。
2. **想控制错误率，看 $p_{\max}$**：模型校准时，$p_{\max}$ 才是"这次选对"的概率估计。
3. **跨类型阈值不能互通**：同一语义用 Noul 和 Choice 问，概率可能不同。

### 2.4 问题之间相互独立

同一请求里的问题共享 state 但**互不知情**，逻辑互补的问题不保证一致——官方示例中一个命题与其否定的概率之和为 **1.19**。校准是一批预测上的统计性质，逻辑约束要写在代码里（见 5.4）。

## 三、调用示例

```python
import os
import requests


def decide(state: str, questions: list[dict], model: str = "jev-1.13.0") -> dict:
    resp = requests.post(
        "https://api.typesafe.ai/v1/systemone",
        headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
        json={"model": model, "state": state, "questions": questions},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()


data = decide(
    state="Customer says: I was charged twice for the same order #12345.",
    questions=[
        {"type": "choice", "question": "What is the primary issue category?",
         "options": ["billing", "shipping", "product-quality", "account", "other"]},
        {"type": "score", "question": "How urgent is this ticket?",
         "scale": ["low", "medium", "high", "critical"]},
        {"type": "noul", "statement": "This ticket requires immediate refund processing"},
    ],
)
```

两处有意的改动：**固定版本号**而不用 `jev-latest`（社区已观测到 1.12 → 1.13 的输出漂移，阈值是按版本调的）；**加 `other` 选项**避免强行归类。

SDK 写法更简洁（TypeScript SDK 会把 `choice` 推断成字面量联合类型，`switch` 漏分支直接编译报错——"TypeSafe"名字的由来）：

```python
from typesafe import TypeSafe

client = TypeSafe()  # 从环境变量 TYPESAFE_API_KEY 读取密钥
result = client.decide(
    state="The deployment pipeline failed at the integration test stage.",
    questions=[
        client.choice("What should the on-call engineer do first?",
                      options=["rollback", "retry", "escalate", "investigate-logs"]),
        client.noul("This is a P0 incident requiring all-hands"),
    ],
)
print(result.answers[0].choice, result.answers[0].probabilities, result.answers[1].noul)
```

本地部署可用社区基于 Qwen3.5 的复刻 **Kev**，接口兼容（服务只绑定回环地址）：

```bash
pip install vllm typesafe
vllm serve kev-community/kev-8b --host 127.0.0.1 --port 8000
# 客户端：TypeSafe(base_url="http://127.0.0.1:8000/v1/systemone")
```

## 四、技术原理

### 4.1 为什么快：一笔延迟账

$$
T_{\text{LLM}} \approx T_{\text{prefill}}(L_{\text{in}}) + N_{\text{out}} \cdot t_{\text{decode}}, \qquad
T_{\text{S1}} \approx T_{\text{prefill}}(L_{\text{in}} + L_{\text{opts}}) + T_{\text{head}}
$$

- **prefill 计算密集、高度并行**，几千 token 几十到一两百毫秒。
- **decode 访存密集、串行**，每个 token 都要读一遍权重和 KV Cache。以 40 token/s 计，300 个输出 token 就是 7.5s，和官方演示里 LLM 基线的 **8.566s** 同一量级。
- 去掉 $N_{\text{out}}$ 就是 1–2 个数量级的差距；**输出免费**也是因为根本没有输出 token；多个问题共享一次 prefill，加问题几乎不加延迟。

成本：2,000 token 的 state 单次约 $\$0.000084$，与官方演示的 **$0.000081 / 次** 吻合，每天 100 万次约 84 美元。

### 4.2 架构：两种假说

官方只说了"新架构、新采样器、新训练算法"。社区有两种猜测：

| | 双向编码器 + 决策头 | Decoder LLM 骨干 + 读出头 |
|---|---|---|
| 代表 | Raschka 推测的 (Modern)BERT 式编码器；Laya | NanoJev、Kev、Clef；第三方称 Jev 为 "parallel causal readout head" |
| 优势 | 小、快（CPU 几十 ms），天然双向 | 继承 LLM 知识与指令理解，零样本泛化强 |
| 劣势 | 知识与长上下文弱；开箱易过度自信 | 参数更大；选项拼进上下文会带来位置偏置 |

HuggingFace 上 129+ 个开源决策模型中 **>65% 基于 Qwen**，社区实际押注的是第二条路线。

和传统分类器的本质区别在于**选项在输入端编码、而不是写死在输出层**：传统分类器的类别数固定在 `nn.Linear(d, K)` 里，换类别就要重训；决策模型的"类别"是输入，同一个模型能服务邮件分类、玩 Doom、股票交易等完全不同的任务。

### 4.3 一个最小可读实现

参考 NanoJev 的结构（Choice 用共享打分头 + 集合注意力 + softmax，Noul 用 sigmoid，Score 求期望档位）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SystemOneHeads(nn.Module):
    """骨干之上的三类决策头（示意实现，省略 batch 与 mask 细节）"""

    def __init__(self, d: int, n_heads: int = 8):
        super().__init__()
        self.set_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.scorer = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.noul_head = nn.Linear(d, 1)
        self.coverage = nn.Linear(d, 1)  # "答案是否在候选中"，对应 Jeva 的空值感知

    def choice(self, h_q: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        # h_q: (d,) 为 state+问题的池化表示；h_opts: (K, d) 为每个选项的表示
        o, _ = self.set_attn(h_opts[None], h_opts[None], h_opts[None])  # 无位置编码 → 置换等变
        o = o[0]
        feats = torch.cat([h_q.expand_as(o), o, h_q * o], dim=-1)
        return F.softmax(self.scorer(feats).squeeze(-1), dim=-1)  # 共享打分 → 支持任意 K

    def noul(self, h_q: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.noul_head(h_q)).squeeze(-1)

    def score(self, h_q: torch.Tensor, h_levels: torch.Tensor):
        p = self.choice(h_q, h_levels)
        k = torch.arange(len(p), dtype=p.dtype, device=p.device)
        return p, (p * k).sum()  # 档位分布 + 期望档位
```

- **集合注意力不加位置编码**：选项之间可以互相比较，输出对顺序置换等变。
- **共享打分器**：K 可以是 2 也可以是 255。
- **Score 头偷懒复用了 Choice**，没利用档位顺序。更讲究的做法是有序回归（CORAL：$P(y>k)=\sigma(f(x)-b_k)$，$b_k$ 单调），或给相邻档位部分分（Clef 的做法）。

训练目标是各头损失之和：

$$
\mathcal{L} = \underbrace{-\log p_{y}}_{\text{Choice / Score}} + \lambda_1\, \underbrace{\mathrm{BCE}(q,\, o)}_{\text{Noul}} + \lambda_2\, \underbrace{\mathrm{BCE}(c,\, \mathbb{1}[\text{答案在候选中}])}_{\text{覆盖度}}
$$

NanoJev 的一个实践值得抄：**骨干与决策头分开设学习率**（骨干 1e-5、头 1e-4），避免随机初始化的头把预训练骨干带歪。

### 4.4 Jeva：决策模型的四项结构性要求

Jeva（arXiv:2610.18773）批评社区"把选项序列化成文本拼进上下文"的做法不尊重决策问题的结构，提出四项要求：

| 要求 | 含义 | 对应机制 |
|------|------|---------|
| **空值感知** | 区分"候选里没有答案"和"在候选间犹豫" | 独立的覆盖度估计 |
| **置换不变** | 打乱选项顺序不改变决策 | 选项独立编码 + 对称聚合 |
| **选项可扩展** | 候选集变大不需重定义输出空间 | 共享打分 |
| **可缓存** | 反复出现的选项可跨请求复用计算 | 选项表示与请求无关，作为不可变的选项记忆 |

摘要报告：空值检测近乎完美、所有置换下**决策翻转为零**、可扩展到数千候选（未给具体数字）。

这几条为什么重要？对照第三方对 Jev 1.13 的实测：**颠倒选项顺序**，同一输入的概率从 0.84–0.89 变成 0.93–0.96；参考文档**放开头** 16 题对 12，**放末尾** 16 题全对。这是"选项拼进因果上下文"路线的典型症状（推测 Jev 1.13 属于这一路线，未经官方确认），工程上需要自己去偏（见 5.3）。

### 4.5 RLCD：公开了什么，没公开什么

**已公开**：目标是"返回决策和校准后的概率"；校准定义为"被赋予 0.8 概率的结果约 80% 发生"，**描述的是一批预测而非单个答案**；数据完全合成；动机是规避 RLHF 的**谄媚**与**模式丢失**——"人类偏好与机器可信度是不同的优化目标"。

**未公开**：奖励函数、损失、训练阶段、数据集、代码、校准指标（官方 evals 只报准确率），也没有论文。唯一有细节的同名方法来自 **Cloudflare Clef**（应视为另一种技术）：作为第二训练阶段，对有序相邻选项给部分分，并施加参考惩罚限制漂移（类似 RLHF 的 KL 项）。

**可能的形式：严格适当评分规则**。Raschka 推测做法类似 RLCR（[arXiv:2507.16806](https://arxiv.org/abs/2507.16806)），奖励 = 正确性 + Brier：

$$
R(\hat y, q) = \mathbb{1}[\hat y = y] - \big(q - \mathbb{1}[\hat y = y]\big)^2
$$

Brier 为什么能逼出诚实的概率？设真实概率 $p$，报告 $q$，$o \sim \mathrm{Bernoulli}(p)$：

$$
\mathbb{E}\big[-(q - o)^2\big] = -\big[(q - p)^2 + p(1-p)\big]
$$

第二项与 $q$ 无关，期望奖励在 $q = p$ 时唯一最大——**报真实信念是唯一最优策略**。

**交叉熵本身也是严格适当的，为什么还要 RL？**（以下为我的理解）

1. **bandit 反馈**：Agent 选工具、游戏选动作时只知道"选的那个成没成"，没有完整标签，只能用 RL。
2. **不可导的结构化奖励**：相邻部分分、输出精确性、跨问题一致性，写成奖励更自然。
3. **纠偏**：RLHF 过的骨干和 one-hot 硬标签 SFT 都会把模型推向过度自信，需要显式拉回。

**零成本的校准基线是温度缩放**，而且**对 API 返回的概率同样适用**：$\log p_i$ 与 logit 只差一个常数，对 $\log p_i / T$ 重新 softmax 等价于对 logit 做温度缩放：

```python
import numpy as np
from scipy.optimize import minimize_scalar


def _tempered(probs: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, 1.0)) / T
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """probs: (N, K) 为 API 返回的概率；y: (N,) 为人工标注的真实类别下标"""
    nll = lambda T: -np.log(_tempered(probs, T)[np.arange(len(y)), y] + 1e-12).mean()
    return minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x
```

$T>1$ 说明过度自信，$T<1$ 说明过于保守。Noul 同理，对 $\mathrm{logit}(q)/T$ 做 sigmoid。

## 五、置信度工程：把概率变成分支

Jev 的价值不在准确率（前沿 LLM 也能做到），而在**可以直接用于工程决策的概率**——前提是在自己的数据上验证过。

### 5.1 测量校准

校准不是全局属性，**要按问题类型和输入结构分别测**：记录概率和事后真实结果 → 分桶 → 比较每桶平均概率与实际命中率。

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n}\, \big|\, \text{acc}(B_m) - \text{conf}(B_m) \,\big|
$$

```python
import numpy as np


def ece(p_max: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """p_max: 被选中答案的概率；correct: 0/1"""
    idx = np.clip(np.digitize(p_max, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    return sum(
        (idx == b).mean() * abs(correct[idx == b].mean() - p_max[idx == b].mean())
        for b in range(n_bins) if (idx == b).any()
    )
```

这里用 $p_{\max}$ 而不是 `confidence`（见 2.3）。参照值：社区在 150 类意图数据集上测得 Jev 校准误差 **0.0204**。中文场景必须自测——Jev 主要训练语言是英语，官方承认 CJK 存在性能差异。

### 5.2 选阈值：错误率约束下最大化自动化率

不存在通用阈值，它取决于误执行的代价，应当作约束优化来解：

```python
def pick_threshold(p_max: np.ndarray, correct: np.ndarray, max_err: float = 0.02):
    best = None
    for t in np.arange(0.50, 1.00, 0.01):
        auto = p_max >= t
        if not auto.any():
            continue  # 自动处理件数为 0 时错误率"不可计算"，不能记为 0%
        err, rate = 1 - correct[auto].mean(), auto.mean()
        if err <= max_err and (best is None or rate > best[1]):
            best = (round(float(t), 2), float(rate), float(err))
    return best  # (阈值, 自动化率, 自动处理错误率)
```

拿到阈值后就是三档路由——$p \ge T_{\text{auto}}$ 自动执行，$p \ge T_{\text{review}}$ 转人工，否则升级到 LLM。两条纪律：**阈值按问题分别存**，不要全局共用一个 0.9；**高置信度不代表可以跳过权限检查**，鉴权和业务规则必须独立于模型实现。

### 5.3 选项置换去偏

针对 4.4 的顺序敏感：换着顺序问、再平均。利用"同一请求多问题共享 state、并行评估"，**把 m 种排列放进同一个请求**，延迟几乎不增加：

```python
import random


def choice_debiased(client, state: str, question: str, options: list[str], m: int = 4, seed: int = 0):
    rng = random.Random(seed)
    perms = [rng.sample(options, len(options)) for _ in range(m)]
    r = client.decide(state=state, questions=[client.choice(question, options=p) for p in perms])
    avg = {o: sum(a.probabilities[o] for a in r.answers) / m for o in options}
    flip_rate = 1 - max(sum(a.choice == o for a in r.answers) for o in options) / m
    return max(avg, key=avg.get), avg, flip_rate
```

`flip_rate` 本身是很好的额外不确定性信号：翻转率高的样本即使平均概率很高也应复核。另一个零成本技巧：参考文档、检索结果**放在 state 末尾**。

### 5.4 否定一致性

```python
r = client.decide(state=s, questions=[client.noul(stmt), client.noul(neg_stmt)])
p_pos, p_neg = r.answers[0].noul, r.answers[1].noul
p = (p_pos + 1 - p_neg) / 2          # 对称化后的估计
incoherence = abs(p_pos + p_neg - 1)  # 官方示例中 p_pos + p_neg = 1.19
```

`neg_stmt` 要写成直接的反向陈述，不要用双重否定。`incoherence` 大说明模型对该命题理解不稳定，同样可作为路由信号。

### 5.5 已知失败模式与对策

| 失败模式 | 对策 |
|---------|------|
| 按字面理解措辞 | 把隐含条件写进问题 |
| 计算、计数、日期比较 | 交给代码 |
| 双重否定、多步推理 | 拆成直接的单步问题 |
| 输入含大量无关内容 / 诱导性措辞 | 只传必要字段；评测集纳入诱导样本 |
| 问题之间缺乏数学一致性 | 在代码里管理逻辑关系（5.4） |
| 选项顺序 / 文档位置敏感 | 置换平均、文档放末尾（5.3） |
| 分布外输入仍过度自信 | 监控输入分布，OOD 样本强制复核 |
| 版本间漂移 | 固定版本号，升级前回归 |

一个很有启发的国际象棋实验：直接输入原始棋局，Jev 与随机无异；用代码提取结构特征后明显好于随机；再给出显式战术事实，准确率达 **78%**。结论是——**System One 不做推理，推理先在代码或 LLM 里做掉，把结论喂给它做判断**。

## 六、典型应用场景

**Agent 工具路由**：让 LLM 生成 JSON 选工具慢、贵、解析脆弱，换成 Choice 即可。工具描述写进选项描述；一定要有 `ask_user` / `none` 兜底，否则封闭集合会强行选一个工具。

```python
ans = client.decide(
    state=f"User request: {user_msg}\nConversation summary: {summary}",
    questions=[client.choice("Which tool should be called next?", options=tool_names + ["ask_user", "none"])],
).answers[0]
```

**模型路由**：生产流量 70–80% 是简单查询。用一个 Choice 把请求分成 easy / medium / hard，分发到不同档位的模型（~350ms、~$0.00035/次），LangChain 就用 Jev 为每个请求挑"能胜任的最便宜模型"。

**Jev-as-a-Judge**：目前最有说服力的场景，优势**不在速度，在方差**。LangChain 的映射是 Noul 验证事实依据、Score 做质量评级、Choice 做失败归因（`unnecessary_tool_call` / `premature_abort` …）。第三方对同一条 trace 重复评分 500 次，方差仅 **0.0000149**，比主流 LLM judge 低 **92–913 倍**；评 1,000 条 trace 约 $0.35，LLM judge 约 $30–50。短板是没有解释，混合方案是 Jev 做门禁、不通过时再调 LLM 生成原因。打分标准的设计可参考[《Rubrics：把"好不好"拆成可验证的评分表》](/blog/rubrics-for-llm-eval-and-rl/)。

**护栏**：用 Noul 检测 PII、提示词注入、危险命令，用 Score 给整体风险定级。LangChain 的 "Auto Mode" 护栏即基于 Jev。

**落地数据**：

| 场景 | 效果 | 来源 |
|------|------|------|
| 官方演示工作流 | 0.114s vs 8.566s；$0.000081 vs $0.013880（快 193.6×、便宜 444.6×） | 厂商自报 |
| 候选人筛选（Jack & Jill） | 成本 −88%，速度 2×，质量持平；10 天上线 | 官方案例 |
| 形参绑定决策 | 23s → 1.7s | 社区 |
| GUI Agent 控件选择 | 准确率与前沿 LLM 持平，延迟降 90% | 社区 |

厂商数字要打折：对照基准是其他大模型**预测的平均值**而非真实标注，且大多在美西笔记本上测试。独立测试的中位延迟是 350–440ms。

## 七、开源生态（2026.10）

| 模型 | 来源 | 骨干 | 要点 |
|------|------|------|------|
| **Jev 1.13** | TypeSafe AI（闭源） | 未公开 | 标杆，API 最成熟 |
| **Kev** | 开源社区 | Qwen3.5 | 兼容 Jev 接口，本地部署首选 |
| **Laya** | Convai Innovations | ModernBERT（322M–421M） | CPU ~33ms，100+ 语言；基础 ckpt 过度自信需微调 |
| **Clef** | Cloudflare | Qwen | 边缘部署；公开了同名 RLCD 的定性描述 |
| **NanoJev** | TianyuCodings | Qwen3-0.6B | 完整训练管线开源（MIT） |
| **Decision-on-1** | Microsoft | Qwen3.5-9B | Decision Index 高于 Jev |
| **WeMM-Jev** | 腾讯 | 多模态 | 文本 + 多图，登顶视觉决策榜 |
| **Span-02** | — | — | Decision Index 高于 Jev，更便宜 |
| **Jeva** | 学术论文 | 决策原生架构 | 满足四项结构性要求（4.4） |

HuggingFace 上已有 **129+** 开源决策模型，**>65% 基于 Qwen**。警惕"蹭命名"：大量带 RLCD 字样的模型创建于 9 月 16–19 日，有的实际跑的是未微调的原版 Qwen。

**NanoJev** 最适合拿来学习：Qwen3-0.6B 骨干 + 共享决策头（Choice 2–255 候选，Score 2–10 档）；多任务混合 SFT + 完整候选集交叉熵；数据划分为 train / dev / **calibration** / test / **OOD**，专门留出校准集和分布外集的做法值得照搬。留出测试集结果：

| 模型 | Maze | Snake | ViZDoom Basic | Predict Position |
|------|:---:|:---:|:---:|:---:|
| NanoJev | 4/10 | 8/8 | **128/128** | **27/128** |
| Jev | **7/10** | 8/8 | 56/128 | 11/128 |
| 未微调 Qwen3-0.6B | 2/10 | 0/8 | 56/128 | 11/128 |

读表要小心：Jev 与未微调 Qwen 在后两项上分数完全相同，可能主要由共用的 ε-greedy 控制器决定；Maze、Snake 样本量只有 10 和 8。它证明的是"0.6B 小模型在窄任务上 SFT 后可以超过通用 Jev"，而非全面超越。

## 八、局限与争议

1. **不是通用模型**：不能对话、推理、生成，是分类与路由的极致特化。
2. **闭源 + 自测**：架构、权重、RLCD 均未公开，基准由厂商自建，影响采购与合规评估；没有解释轨迹，在 EU AI Act 约束的高风险场景中也是障碍。
3. **"就是个分类器？"**：Raschka 认为真相介于吹捧和否定之间——RL 用于编码器式模型并非首创，重要的是它**确实泛化得好**；秘诀可能更多在数据和 API 设计，就像 ChatGPT 之于 InstructGPT。
4. **校准只在总体层面成立**：单个预测仍可能静默出错，分布外输入时尤甚。
5. **语言偏差**：中文场景要用真实用户输入（含省略句、间接表达）重新评估，不能只用翻译文本。

## 九、关键 Takeaways

1. **范式判断**：判断类任务不需要自回归，去掉 decode 就是 1–2 个数量级的延迟和成本，这是"按职能拆分模型"的第一步落地。
2. **技术核心不新，组合是新的**：Transformer 骨干 + 选项在输入端编码的共享打分头 + 以校准为目标的训练 + 极简 API。
3. **校准才是产品**：可信的概率让"自动执行 / 人工复核 / 升级 LLM"的分级路由成为可能。但 confidence ≠ 准确率，阈值要用 $p_{\max}$ 在自己的数据上按问题标定。
4. **结构性缺陷有明确解法**：顺序敏感、空值、否定不一致是"选项拼进上下文"路线的症状，Jeva 给出了架构层面的答案；短期用置换平均、`other` 选项和对称化在工程层兜底。
5. **与 LLM 不是替代关系**：Jev 做高频判断，LLM 做低频复杂生成和解释，按概率路由。

## 参考资料

- TypeSafe AI 官方：[typesafe.ai](https://typesafe.ai)（Docs：Choice / Score / Noul / Confidence / Models / Jev 1.13 jaggedness）
- Raschka (2026)：[It's Easy to Dismiss Jev as Just a Classifier](https://sebastianraschka.com/blog/2026/jev-classification-generalization.html)
- Li & Wang (2026)：[Jeva: System One Models from First Principles](https://arxiv.org/abs/2610.18773)
- Damani et al. (2025)：[Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty](https://arxiv.org/abs/2507.16806)
- RLCD 公开信息梳理：[systemonemodels.org/guides/rlcd-explained](https://systemonemodels.org/guides/rlcd-explained/)
- NanoJev：[github.com/TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)
- Kev 本地部署：[gitcode.com/gh_mirrors/kev2/kev](https://gitcode.com/gh_mirrors/kev2/kev)
- [Injoys: Jev Decision Model: Branching Design and Limitations](https://injoys.com/en/site/ai-dev/jev-decision-model-branching-design-and-limitations)
- [Regolo: Jev and System One Models](https://regolo.ai/jev-and-system-one-models-benchmarks-open-source-alternatives-and-when-to-use-them)
- LangChain (2026)：Can Jev Be a Better Agent Evaluator?
- Guo et al. (2017)：On Calibration of Modern Neural Networks
