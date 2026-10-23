---
title: "图像生成与强化学习：从 DDPO 到 Flow-DPPO 的技术演进"
description: "以“优化机制”为主轴梳理用 RL 对齐扩散/Flow 生成模型的技术谱系：扩散与 Flow Matching 的数学基础、为何 ODE 要转 SDE、GRPO 家族（Flow-GRPO/DanceGRPO/G²RPO/GRPO-Guard/Flow-DPPO/Flow-OPD）、DPO 与可微反传，以及 Flow-GRPO 与 LLM-GRPO 的本质区别。"
pubDate: 2026-09-18
tags: ["扩散模型", "强化学习", "GRPO", "Flow Matching", "生成模型", "综述"]
---

> 关键词：Diffusion Model, Flow Matching, Reinforcement Learning, GRPO, DPO, Reward Modeling, Human Preference Alignment

## 导语

扩散模型从数据里学到的是「世界是什么样」，而我们真正想要的，是「图像应该是什么样」——从「是」到「应该」之间，并不存在一条可微的路。

强化学习就是在这道断裂处架桥：先生成，再被评判，然后让评判回过头来重塑生成者本身。

> 本文较长，建议在宽屏下阅读（右侧会自动生成目录）。若你是从 LLM RLHF 转过来的，建议先跳到第 5 节《Flow-GRPO 与 LLM-GRPO 的区别》。

## 1. 引言

近年来，扩散模型（Diffusion Models）和流匹配模型（Flow Matching Models）在文本到图像生成（Text-to-Image, T2I）领域取得了巨大成功，以 Stable Diffusion、SD3/FLUX、DALL-E 3、Midjourney 等为代表。然而，预训练生成模型往往只学习了数据分布，并不能很好地对齐人类偏好——比如生成的图像是否美观、是否符合语义、是否安全合规。

这催生了一个新兴且快速发展的研究方向：**用强化学习（Reinforcement Learning, RL）方法微调扩散/Flow 生成模型**，使其更好地对齐人类偏好。这一方向从 2023 年的 DDPO 首次把去噪过程建模为马尔可夫决策过程开始，到 2025–2026 年 GRPO 家族在流匹配模型上的爆发式演进，已形成一个系统化的技术谱系。

本文以「**优化机制**」为主轴，对该领域的代表性工作进行梳理，并尽量给出各方法的关键实验数据，方便横向比较。

## 2. 背景知识

### 2.1 扩散模型与 Flow Matching

> **记号约定**（重要，两类模型的文献惯例正好相反）：
> - **扩散模型**：$t$ 从 0 增到 $T$ 是**加噪**方向，$x_0$ 是干净数据、$x_T$ 是纯噪声；采样沿 $t$ 递减进行，一步转移记作 $x_t \to x_{t-1}$。
> - **Flow Matching**：$t$ 从 0 增到 1 是**去噪**方向，$x_0$ 是噪声、$x_1$ 是数据；采样沿 $t$ 递增进行，一步转移记作 $x_t \to x_{t+\Delta t}$。
>
> 后文涉及 flow 模型的策略与比率时统一使用后一种惯例。

#### 扩散模型（DDPM / Score-based）

通过前向加噪 + 逆向去噪的马尔可夫链，从噪声中逐步生成图像。**前向过程**把干净数据 $x_0$ 逐步破坏为高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)
$$

利用 $\alpha_t = 1-\beta_t,\ \bar\alpha_t = \prod_{s=1}^{t}\alpha_s$ 可一步闭式采样：

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I)
$$

**反向过程**由网络参数化，条件 $c$ 为文本 prompt：

$$
p_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}\!\left(x_{t-1};\ \mu_\theta(x_t, t, c),\ \sigma_t^2 I\right)
$$

训练目标是简化后的噪声预测 MSE：

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0,\,\epsilon,\,t}
\left[\left\| \epsilon - \epsilon_\theta(x_t, t, c) \right\|^2\right]
$$

噪声预测与数据分布的 score 函数直接相关，这是 score-based 视角的桥梁：

$$
\nabla_{x_t} \log q(x_t) \approx -\frac{\epsilon_\theta(x_t, t, c)}{\sqrt{1-\bar\alpha_t}}
$$

由于 $p_\theta(x_{t-1}\mid x_t, c)$ 是一个显式的高斯分布，整条去噪链 $x_T \to \cdots \to x_0$ 的轨迹概率可分解为

$$
p_\theta(x_{0:T} \mid c) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t, c)
$$

因此可自然地视为一个 $T$ 步 MDP：状态 $s_t=(x_t,c)$、动作 $a_t = x_{t-1}$、策略 $\pi_\theta(a_t\mid s_t)=p_\theta(x_{t-1}\mid x_t,c)$，奖励只在 $t=0$ 给出。这正是 DDPO / DPOK 等方法的出发点（详见 §4.1）。

#### Flow Matching（Rectified Flow）

将生成建模为从噪声到数据的常微分方程（ODE）流，训练一个速度场网络。以最简单的线性插值路径（Rectified Flow）为例，取 $x_0 \sim \mathcal{N}(0,I)$、$x_1 \sim p_{\text{data}}$：

$$
x_t = (1-t)\,x_0 + t\,x_1, \qquad t \in [0, 1]
$$

其对应的真实速度场为常量差值 $\dot{x}_t = x_1 - x_0$，训练目标（Conditional Flow Matching）即回归该速度：

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\,x_0,\,x_1}
\left[\left\| v_\theta(x_t, t, c) - (x_1 - x_0) \right\|^2\right]
$$

采样时求解**确定性** ODE，常用 Euler 离散（步长 $\Delta t$）：

$$
\frac{\mathrm{d}x_t}{\mathrm{d}t} = v_\theta(x_t, t, c)
\qquad\Longrightarrow\qquad
x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c)
$$

#### 二者的关键差异：为何 RL 适配不同

扩散采样每一步都带随机性，$\log \pi_\theta(a_t\mid s_t)$ 有闭式表达，可直接做 policy gradient。而 Flow Matching 的 ODE 采样是**确定性映射**，给定 $x_0$ 后轨迹唯一，策略退化为 $\delta$ 分布，既无探索也无法定义 log-probability。

Flow-GRPO 的做法是将 ODE 转为等价的 SDE（ODE-to-SDE），在保持边际分布 $p_t(x)$ 不变的前提下重新注入随机性：

$$
\mathrm{d}x_t = \left[v_\theta(x_t,t,c) + \frac{\sigma_t^2}{2}\nabla_{x_t}\log p_t(x_t)\right]\mathrm{d}t + \sigma_t\,\mathrm{d}w_t
$$

离散化后每一步变为高斯分布，从而恢复了可计算的策略概率：

$$
\pi_\theta(x_{t+\Delta t} \mid x_t, c) = \mathcal{N}\!\left(x_t + \Delta t\cdot \tilde{v}_\theta(x_t,t,c),\ \ \sigma_t^2 \Delta t\, I\right)
$$

其中 $\sigma_t$ 控制探索强度：$\sigma_t \to 0$ 退回原始确定性 ODE，$\sigma_t$ 过大则偏离预训练分布。这一"噪声注入"是把 GRPO 类 on-policy 方法搬到 Flow Matching 上的核心前提。

### 2.2 为什么需要 RL？

传统训练目标（去噪 MSE / 对数似然）只保证「生成真实图像」，但真正关心的下游目标——美学质量、文本-图像对齐、物体计数与空间关系、文字渲染、安全合规——往往**不可微，也无法用似然表达**。RL 提供了直接优化这类「黑箱奖励」的统一框架：

$$
\max_\theta\ \mathbb{E}_{c \sim p(c),\ x_0 \sim \pi_\theta(\cdot \mid c)}
\big[\, r(x_0, c) \,\big]
\quad \text{s.t.} \quad
D_{\mathrm{KL}}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right) \le \epsilon
$$

其中 $r$ 为奖励模型（ImageReward / HPSv2 / OCR 准确率 / 安全分数等），KL 约束防止模型偏离预训练分布过远而导致画质崩塌。

### 2.3 两大顽疾

MDP 抽象带来可优化性的同时，也暴露了本方向所有后续工作都要对抗的两个问题：

1. **奖励稀疏（sparse reward）**：信号只在终局 $t=0$ 出现，如何把它分配到中间的每一步？
2. **奖励劫持（reward hacking）**：模型学会「讨好」奖励模型而非真正提升质量。

后文可以看到，整条技术谱系本质上就是围绕这两个问题不断给出新答案。

## 3. 相关工作全景图

### 3.0 分类体系：五个分析轴

IIIT Hyderabad 的综述（arXiv:2505.17352）提出用五个分析轴组织全部文献，这也是理解本方向的一张好地图：

| 分析轴 | 取值范围 |
|------|------|
| ① 反馈来源 | 人类比较 / 人类评分 / AI 反馈（RLAIF, VLM）/ oracle |
| ② 信号形式 | 显式奖励标量 / pairwise 偏好 / binary utility（点赞点踩）/ 可微 surrogate / 安全分数 |
| ③ 优化机制 | KL 正则策略优化 / DPO 直接优化 / 可微奖励反传 / 群体相对（GRPO）/ 区域感知微调 |
| ④ 分布漂移 & 过优化处理 | KL / proximal 约束 · early stopping · 时序感知 · critic 重置 · 组相对 baseline |
| ⑤ 安全约束 | 作为显式硬约束 vs 作为泛化偏好；是否覆盖对抗 / OOD prompt |

本文以第 ③ 轴「优化机制」为主线，把代表性工作归入五大家族：**策略梯度/RLHF**（DDPO、DPOK）、**GRPO 家族**（Flow-GRPO、DanceGRPO、G²RPO、GRPO-Guard、Flow-DPPO、Flow-OPD）、**直接偏好优化**（Diffusion-DPO）、**可微奖励反传**（ReFL、DRaFT、AlignProp、SRPO），以及作为燃料的**奖励模型与评测**（HPSv2、UnifiedReward）。

下面按技术路线和时间线梳理该领域的核心工作。

### 3.1 奖励模型（Reward Model）基础

奖励模型是所有 RL 方法的「燃料」——没有可靠的偏好信号，再精巧的优化机制也只会 hack 到错误目标。

| 论文 | 年份 | 要点 |
|------|------|------|
| **ImageReward** (Xu et al.) | 2023 | 首个通用 T2I 人类偏好奖励模型（137k 比较），超越 CLIP 等指标 |
| **HPS / HPSv2** (Wu et al.) | 2023 | HPD v2：79.8 万条二元偏好 / 43.4 万对图像，来自 9 个 T2I 模型 + COCO 真实图；用 ChatGPT 清洗 prompt 去除风格偏置词 |
| **PickScore** (Kirstain et al.) | 2023 | 基于 Pick-a-Pic 数据集的偏好评分 |
| **UnifiedReward** (Wang et al.) | 2026 | 基于 LLaVA-OneVision-7B，23.6 万统一偏好数据覆盖「图像/视频 × 理解/生成」四象限，同时支持 pairwise ranking + pointwise scoring |

**偏好预测准确率对比（HPD v2 测试集）**：CLIP 65.1 → ImageReward 74.0 → PickScore 79.8 → **HPSv2 83.3**（已超过人类标注一致性 78.1）。UnifiedReward 在 GenAI-Bench Image 上 tau/diff 达 54.8/70.9，作为 GRPO 奖励用于 FLUX.1-dev 时把 ImageReward 从 1.27 提到 1.38，优于 HPSv2/PickScore；但跨任务数据不均衡会损害欠表示任务（如视频生成），需平衡采样。

### 3.2 策略梯度与 GRPO 家族（On-Policy）

| 论文 | 年份 | 会议 | 核心思想 |
|------|------|------|----------|
| **DDPO** (Black et al.) | 2023 | ICLR 2024 | 首次将去噪过程严格建模为多步 MDP，利用每步高斯策略的**精确**对数似然做策略梯度 |
| **DPOK** (Fan et al.) | 2023 | NeurIPS 2023 | 在线 RL + 对预训练模型的 **KL 正则**；因边缘似然不可解，改用 per-step 条件 KL 的上界近似 |
| **Flow-GRPO** (Liu et al.) | 2025 | NeurIPS 2025 | 🔥 **ODE→SDE 转换** + GRPO 首次进入 Flow Matching；SD3.5-M 在 GenEval 上 63% → 95% |
| **DanceGRPO** (Xue et al.) | 2025 | arXiv | 统一 diffusion + rectified flow 到 SDE，证明二者是「一枚硬币的两面」 |
| **G²RPO** (Zhou et al.) | 2025 | arXiv | **单步随机采样** + 多粒度优势融合（MGAI），解决每步注噪导致的信用分配失效 |
| **GRPO-Guard** (Wang et al.) | 2025 | arXiv | 诊断出重要性比率裁剪**系统性失效**，用 RatioNorm + 梯度重加权修复（治标） |
| **Flow-DPPO** (Ping et al.) | 2026 | arXiv | 用流模型高斯策略的**精确闭式 KL** 做散度掩码替代比率裁剪（治本） |
| **Flow-OPD** (Fang et al.) | 2026 | arXiv | 换范式：多教师**在线蒸馏**替代稀疏标量奖励，绕开多目标梯度干扰 |

### 3.3 偏好优化 / Offline 方法（DPO 系列）

| 论文 | 年份 | 会议 | 核心思想 |
|------|------|------|----------|
| **Diffusion-DPO** (Wallace et al.) | 2023 | CVPR 2024 | 将 DPO 从 LLM 迁移到扩散模型，直接在人类偏好对比数据上优化，无需显式奖励模型 |
| **D3PO** (Yang et al.) | 2024 | — | Denoising Diffusion Direct Preference Optimization |
| **D-Fusion** | 2025 | arXiv | 构建 DPO 可训练的视觉一致性偏好对，解决图像对比数据质量问题 |
| **SPO** (Liang et al.) | 2024 | — | Step-aware Preference Optimization，每步去噪都有偏好信号 |
| **VideoDPO** | 2024 | — | 将 DPO 方法扩展到视频扩散模型 |

### 3.4 可微奖励反向传播方法

| 论文 | 年份 | 会议 | 核心思想 |
|------|------|------|----------|
| **ReFL** (Xu et al.) | 2023 | — | 直接通过可微奖励模型反传梯度至扩散模型 |
| **DRaFT** (Clark et al.) | 2024 | — | 截断反传的奖励微调，更高效 |
| **AlignProp** (Prabhudesai et al.) | 2023 | ECCV 2024 | 全时间步反向传播，对齐扩散模型到下游奖励函数 |
| **SRPO** (Tencent HunYuan) | 2025 | — | **Direct-Align** + **语义相对偏好优化**；约 10 分钟收敛 |

### 3.5 其他 RL-based 方法

| 论文 | 年份 | 核心思想 |
|------|------|----------|
| **REBEL** | 2024 | Reward-weighted regression 应用于扩散模型 |
| **Diffusion-KTO** (Li et al.) | 2024 | 用二元 like/dislike 反馈做效用最大化，无需 pairwise 比较也无需奖励模型 |
| **O2O** | 2025 | Online-to-Offline RL 微调扩散模型 |
| **Towards Better Alignment** (Hu et al.) | 2025 | CVPR 2025，针对稀疏奖励问题提出新 RL 框架 |
| **UniRL** (Tencent HunYuan) | 2026 | 统一多模态 RL **训练框架**（非单一模型），详见 §7.2 |

## 4. 重点方法详解

### 4.1 DDPO：开山之作

**核心 Insight**：将扩散模型的 T 步去噪过程视为一个 T 步 MDP：

- **状态** $s_t = (x_t, c)$：当前带噪图像 + 文本条件
- **动作** $a_t = x_{t-1}$：模型输出的去噪预测
- **奖励**：仅在最终步给出 $r(x_0, c)$（如 ImageReward 评分）

由于每步策略是各向同性高斯，**每步对数似然可精确计算**，这正是策略梯度得以严格套用的关键前提。DDPO 给出两种估计器：

$$
\nabla_\theta J_{\text{SF}} = \mathbb{E}\left[\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\cdot r(x_0, c)\right]
$$

$$
\nabla_\theta J_{\text{IS}} = \mathbb{E}\left[\sum_{t=1}^{T}
\nabla_\theta \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}\cdot r(x_0,c)\right],
\quad \text{ratio clipped to } 1\pm\epsilon
$$

DDPO_SF 即 REINFORCE；DDPO_IS 用重要性采样 + PPO 式裁剪，实践中更优——注意其 **clip range 仅 $10^{-4}$**，远小于 LLM 常用的 0.2，这一细节在后文 GRPO-Guard 处会成为关键线索。

**关键数据**：美学奖励下 base 5.95 → DDPO_IS 6.63（20k queries）；推理 4 秒/图 vs Universal Guidance 近 2 分钟；全面超过 DPOK 且未使用 KL；在 45 类动物上训练可泛化到未见物体与新活动。此外它用 VLM（LLaVA + BERTScore）自动生成奖励来改进 prompt 对齐，是 RLAIF 的雏形。

**局限**：梯度信号稀疏（仅终态奖励）；未系统研究 over-optimization（会退化为高频噪声、对 LLaVA 的「排版攻击」）；计算量大（需完整采样链）。

### 4.2 DPOK：加上 KL 正则

**动机**：已有的 supervised fine-tuning 在固定数据集上训练常导致画质退化（过饱和、非真实）。

**核心做法**：同样是 T-horizon MDP + policy gradient，但目标中加入对预训练模型的 KL 正则。由于边缘似然不可解，实际使用 per-step 条件 KL 的上界：

$$
\max_\theta\ \mathbb{E}\big[r(x_0,c)\big] - \alpha \sum_{t=1}^{T}
\mathbb{E}\Big[D_{\mathrm{KL}}\big(p_\theta(x_{t-1}\mid x_t,c)\,\|\,p_{\text{pre}}(x_{t-1}\mid x_t,c)\big)\Big]
$$

**关键数据**：MS-COCO（104 prompt）ImageReward 0.22 → 0.55；DrawBench（183 prompt）0.13 → 0.58。一个典型的去偏例子："Four roses" 原本生成同名威士忌（−0.52），微调后生成花（1.12）。去掉 KL 项则出现过饱和与形状不自然。

**与 DDPO 的分工**：两篇同期工作，DPOK 侧重 KL 正则的理论与稳定性（多为单 prompt），DDPO 侧重多 prompt、多奖励与泛化。

### 4.3 Flow-GRPO：将 GRPO 引入 Flow Matching

**动机**：Flow Matching 模型（如 SD3）使用确定性 ODE 而非随机 SDE，DDPO 那套「精确 log-prob + 随机探索」直接失效；同时多步去噪的 rollout 代价很高。

**核心贡献**：

1. **ODE→SDE 转换**：构造与原 ODE 在所有时刻边缘分布完全一致的等价 SDE（公式见 §2.1），注入随机性且 KL 有闭式解。噪声水平取 $\sigma_t = a\sqrt{t/(1-t)}$，实验最优 $a \approx 0.7$。
2. **GRPO 适配**：同一 prompt 采样一组 $G$ 张图，组内归一化得优势，终点奖励 broadcast 到各步：

$$
A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)},
\qquad A_{i,t} = A_i \ \ \forall t
$$

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T}\sum_{t=1}^{T}
\min\Big(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i\Big)\right]
+ \beta D_{\mathrm{KL}}
$$

其中 $\rho_{i,t} = \pi_\theta(x_{i,\,t+\Delta t}\mid x_{i,t},c) \big/ \pi_{\theta_{\text{old}}}(x_{i,\,t+\Delta t}\mid x_{i,t},c)$（flow 侧沿 $t$ 递增去噪，见 §2.1 记号约定）。

3. **Denoising Reduction**：训练仅用 10 步（推理仍 40 步），带来 >4× 加速。
4. **KL 正则**抑制 hacking。

**效果**：SD3.5-M 在 GenEval 上 63% → 95%（超过 GPT-4o 的 84%），其中 Position 子项 0.24 → 0.99；文字渲染 OCR 59% → 92%；在 2–4 物体上训练可泛化到 12 物体。

**局限**：KL 有效但需更长训练，个别 prompt 仍出现 hacking；组大小敏感（$G=24$ 稳定，$G \le 12$ 崩溃）；仅验证 T2I，未做视频。

### 4.4 DanceGRPO：统一的视觉生成 RL 框架

**核心 Insight**：把 diffusion 与 rectified flow 统一到 SDE 视角下，证明二者是「一枚硬币的两面」，从而用同一套 GRPO 覆盖全部基座。

**与 Flow-GRPO 的两处相反选择**（很有意思，值得注意）：

- **默认省略 KL 项**（Flow-GRPO 依赖 KL）
- **同一 prompt 组内共享初始噪声**（Flow-GRPO 用不同噪声）——作者发现不同噪声会导致视频 hacking

其他设计：多奖励时聚合 **advantage** 而非直接聚合奖励；40% timestep dropout。

**关键数据**：HunyuanVideo 文生视频 motion quality **+181%**、visual +56%；SD1.4 HPS-v2.1 0.239 → 0.365；FLUX HPS 0.304 → 0.372；I2V motion +118%。一个重要观察：**只用 HPS 会让图像变「油腻」，需要 CLIP 一起约束才能保持自然**。

**定位**：广度优先，覆盖 2 范式 × 3 任务（T2I/T2V/I2V）× 4 基座 × 5 奖励模型，是第一个统一的视觉生成 RL 框架。

### 4.5 G²RPO：把稀疏奖励变稠密

**问题诊断**：Flow-GRPO / DanceGRPO 在**每一步**都注入 SDE 随机性，终点的单一奖励无法对应到某一步的具体方向，信用分配实际失效；且单一去噪粒度下的评估不鲁棒。

**核心方法**：

1. **单步随机采样**：只在单个选定步 $k$ 施加 SDE，其余步保持确定性 ODE。这样组内奖励方差**完全由该步扰动决定**，从而得到稠密而精确的 step-aware 优势。
2. **多粒度优势融合（MGAI）**：对同一方向用多种去噪粒度 $\Lambda = \{1,2,3\}$ 分别计算优势后求和。

**关键数据**：基座 Flux.1-dev，仅用 HPS 训练相比 DanceGRPO 相对提升 6.52%；HPS+CLIP 联合时全面超越；UniGenBench++ Overall 61.59 → 69.21；对 10/20 步少步推理更鲁棒。

**代价**：多粒度采样使训练开销 +45.7%（一次性，不影响推理）。

### 4.6 GRPO-Guard：比率裁剪失效的诊断与修复

这篇的价值在**诊断**：它指出把 LLM 的离散 token 概率机制照搬到高斯连续策略上，会让 PPO 的比率裁剪**系统性失效**。

**问题**：重要性比率分布的均值 $<1$（整体左移），且方差随 timestep 剧烈变化。后果是正优势样本很难进入上裁剪区间 $(1+\epsilon)$，**正梯度事实上不受约束** → 隐式过优化：proxy 奖励一路上升而图像质量崩塌。论文推导出比率天然负偏：

$$
\mathbb{E}\big[\log \rho_t\big] < 0
$$

**解法**：

1. **RatioNorm**：对 $\log \rho_t$ 做标准化，把均值归零并消除跨步方差差异，恢复裁剪对正样本的约束能力。
2. **梯度重加权**：引入 $1/\mathrm{d}t$ 因子均衡低噪步异常大的梯度（跨步差异从 20× 降到 2.5×）。

注意它**不依赖重 KL**。

**关键数据**：在相当的 proxy 分数下显著提升 gold 分——SD3.5-M GenEval 归一化 Average 0.84 → 0.89、OCR 0.88 → 0.99；用在 DanceGRPO 上 GenEval 0.88 → 1.02；人工评测图像质量胜率 82–88%。

**局限**：无法消除源于奖励模型自身缺陷的 hacking，那需要 scaling reward model 来解决。

### 4.7 Flow-DPPO：用精确 KL 替代比率裁剪

**动机**（对 GRPO-Guard 的进一步批判）：比率裁剪对流模型在结构上就不合适——每个 $|\rho_t - 1|$ 只是真实策略散度的**单样本蒙特卡洛估计**，在高维连续隐空间中噪声被严重放大，导致过约束与欠约束并存。GRPO-Guard 只是把比率分布重新居中（治标），比率本身仍然含噪（未治本）。

**关键洞察**：流模型每步都是高斯策略，其 KL 有**精确闭式解**，且只需两次已有的前向传播、零额外成本——这与 LLM 必须在十万量级词表上做近似的处境完全不同：

$$
D_t = D_{\mathrm{KL}}\big(\pi_{\theta_{\text{old}}}(\cdot\mid x_t) \,\|\, \pi_\theta(\cdot\mid x_t)\big)
= \frac{\|\mu_\theta(x_t,t) - \mu_{\theta_{\text{old}}}(x_t,t)\|^2}{2\sigma_t^2 \Delta t}
$$

**散度掩码替代裁剪**：仅当更新「远离旧策略**且** $D_t > \delta$」时才阻断梯度；朝向旧策略的更新**永不阻断**——保留了 PPO 的非对称结构，使模型能快速从过冲中恢复。论文同时给出流模型下的策略改进界理论保证。

**关键数据**：FLUX2-9B GenEval2 46.8（Flow-GRPO）→ 57.7；SD3.5 47.8（GRPO-Guard）→ 51.6；KL 散度最低（$0.77 \to 0.17 \times 10^{-3}$，抗遗忘）；OOD 退化显著更小；支持稳定的多 epoch 训练。

**局限**：需要设定散度阈值 $\delta$；去掉非对称性会导致训练崩溃。

### 4.8 Flow-OPD：换范式——用在线蒸馏绕开稀疏奖励

**问题诊断**：多任务对齐下 Flow-GRPO 有两大瓶颈——标量奖励太稀疏；联合优化异质目标时梯度互相干扰（梯度内积 $<0$），造成指标「跷跷板效应」和普遍 hacking（加 OCR 奖励使 GenEval 掉 5%，再加 PickScore 又掉 7%）。

**核心方法**：不修 GRPO，改用**多教师在线蒸馏**：

1. 先用单一奖励的 Flow-GRPO 训出各领域专家 teacher；
2. 冷启动（SFT 或模型合并）；
3. student 在线采样 + **硬路由**把每个 prompt 指派给唯一领域 teacher，由 teacher 提供**稠密速度场监督**。

关键推导：流模型下 OPD 的 reverse-KL **坍缩为速度场之间的时间加权 MSE**，数学上等价于 LLM 的 policy-gradient OPD，但**梯度方差降为 0**：

$$
\mathcal{L}_{\text{OPD}} = \mathbb{E}_{t,\, x_t \sim \pi_\theta}
\left[\, w(t)\,\big\| v_\theta(x_t,t,c) - v_{\text{teacher}}(x_t,t,c) \big\|^2 \right]
$$

另用任务无关的美学 teacher 做流形锚定（MAR）。

**关键数据**：SD3.5-M GenEval 63 → 92、OCR 59 → 94，综合 Avg 0.717 → 0.902（比 GRPO-Mix 的 0.817 高约 10 分）。更有意思的是出现了 **Teacher-Surpassing**：统一的 student 在边缘案例上超过了所有单领域教师。

**局限**：性能受 teacher 天花板限制，教师的错误会通过稠密监督传播；要求师生架构同构。

### 4.9 Diffusion-DPO：无需奖励模型的偏好对齐

**思路**：类比 LLM 中的 DPO，用 Bradley-Terry 模型建模人类偏好，直接在偏好对数据上优化。由于扩散模型的边缘似然不可解，改用 **ELBO 在整条扩散链上定义奖励**，经 Jensen 不等式化简后得到一个简单的 per-timestep 去噪误差差分损失：

$$
\mathcal{L}_{\text{D-DPO}} = -\mathbb{E}\,\log \sigma\!\left(
-\beta T \Big[
\big(\|\epsilon^w - \epsilon_\theta(x_t^w)\|^2 - \|\epsilon^w - \epsilon_{\text{ref}}(x_t^w)\|^2\big)
- \big(\|\epsilon^l - \epsilon_\theta(x_t^l)\|^2 - \|\epsilon^l - \epsilon_{\text{ref}}(x_t^l)\|^2\big)
\Big]\right)
$$

直观解释：让模型对 winning 样本 $w$ 比对 losing 样本 $l$ **去噪得更好**。训练数据为 Pick-a-Pic v2 的 85 万偏好对。

**优势**：不需要训练奖励模型，离线训练，也无需从当前模型采样；训练更稳定、数据效率高；天然支持开放词表 + 等推理成本 + 分布约束。

**关键数据**：DPO-SDXL vs SDXL-base，PartiPrompts 偏好胜率 70.0%；仅用 base（3.5B）就胜过完整的 base+refiner（6.6B）；HPSv2 打分 28.16 登顶当时榜单；用 AI feedback 打伪标签也有效（胜率 59.8% → 63.3%）。

**局限**：偏好数据收集成本，且偏好并非普适；Offline 方法无法持续探索。

### 4.10 SRPO：可微反传的工业级落地

**动机**：可微奖励直接对齐有两个痛点——多步去噪 + 反传打分昂贵，实际只能优化后期少数步，因而容易 hacking（HPSv2 偏红、ImageReward 偏过曝）；且缺少在线调整奖励的机制。

**核心方法**：

1. **Direct-Align**：先注入预定义的高斯噪声，再利用闭式解从**任意步（含早期高噪步）单步恢复干净图**，避免多步梯度爆炸。这使得优化可以作用在早期步，并配合折扣因子降低后期 hacking。
2. **SRPO（语义相对偏好优化）**：把奖励视为文本条件信号，用正/负 control words 在线调节偏好方向，并用**相对差分**抵消奖励模型的通用偏置。
3. inversion 方向传播负梯度作为正则。基座为 FLUX.1-dev。

**关键数据**：收敛仅约 **10 分钟**；人工评测真实感 ≈3.7 倍、美学 ≈3.1 倍（Realism excellent 8.2% → 38.9%）；GPU hours 5.3 vs DanceGRPO 480（≈75× 效率）；未观察到 reward hacking。

**局限 / 定位**：control token 有一部分落在奖励模型的域外；相比 GRPO 它用直接反传而非策略优化，收敛更快（<30 步 vs >100 步），且无需 KL 与 reference model。可以看作连接「可微反传」与「GRPO」两条路线的桥梁。

## 5. Flow-GRPO 与 LLM-GRPO：同一个外壳，不同的策略空间

GRPO 原生于大语言模型（DeepSeekMath / R1）。把它搬到 Flow Matching 上时，「组内相对优势 + 无需 value function」这层外壳几乎原封不动，但**壳里每一个量的数学含义都换了**。这一节把两者逐项对照——理解了这张表，才能理解为什么 §4.6–4.8 那些「修补性」工作是必要的。

### 5.1 核心差异对照表

| 维度 | LLM-GRPO | Flow-GRPO |
|------|------|------|
| **策略分布** | Categorical（词表上的 softmax） | Gaussian（连续 latent 上的高斯） |
| **log-prob 来源** | `log_softmax(logits)[token]` | 高斯闭式 $-\dfrac{(x-\mu)^2}{2\sigma^2}-\log\sigma-\dfrac{1}{2}\log 2\pi$ |
| **每步 log-prob 形状** | 标量（1 个 token 对应 1 个） | 对 latent 张量 `[N,C,h,w]` 求 **mean** 聚合成标量 |
| **随机性来源** | 词表采样（multinomial / top-p） | SDE 注入的高斯噪声 $\sigma_t\,\mathrm{d}w_t$ |
| **哪些步参与** | 全部生成的 token | 只有 SDE 窗口内的几步（其余 ODE 步不计入） |
| **ratio 维度** | `[B, T_tokens]` | `[G, S']`（$S'$ = 窗口步数，如 10） |
| **动作空间** | 有限离散（$\lvert V\rvert \approx 10^5$） | 连续无限（$\mathbb{R}^D$，$D \sim 10^4$–$10^5$） |
| **轨迹长度** | 可变，遇 EOS 终止 | 固定 $T$ 步，无提前终止 |
| **KL 计算** | 词表上求和或 k3 近似估计 | **精确闭式解**（两次前向即可） |
| **优势广播** | sequence-level 优势广播到每个 token | 终点奖励广播到每个 timestep |
| **奖励来源** | 规则 verifier / RM，常可精确判对错 | 神经奖励模型（HPSv2、ImageReward…），必然含偏置 |

### 5.2 逐项展开

**① 策略分布：离散 vs 连续**

LLM 每步是词表上的分类分布，支撑集有限、概率有上界 1：

$$
\pi_\theta(a_t \mid s_t) = \operatorname{softmax}(z_t)[a_t], \qquad a_t \in V
$$

Flow-GRPO 经 ODE→SDE 转换后，每步是连续高斯：

$$
\pi_\theta(x_{t+\Delta t}\mid x_t, c) = \mathcal{N}\!\left(x_{t+\Delta t};\ \mu_\theta(x_t,t,c),\ \sigma_t^2 \Delta t\, I_D\right)
$$

这里是**概率密度**而非概率质量，取值可以大于 1，且随 $\sigma_t$ 变化量级剧烈漂移。PPO 那套「比率在 1 附近、用 $\epsilon=0.2$ 裁剪」的直觉，是建立在离散概率上的。

**② log-prob 的聚合：mean 而不是 sum**

LLM 里一个 token 天然对应一个标量 log-prob。而 flow 模型一步的「动作」是整个 latent 张量，逐元素 log-prob 需要聚合成标量。实现上（DDPO / Flow-GRPO 的代码）用的是 **mean 而非 sum**：

```python
# 概念示意
log_prob = -((x_prev - mu) ** 2) / (2 * sigma ** 2) - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))   # [N,C,h,w] -> [N]
```

这个选择影响巨大：真正的联合 log-prob 应该是 **sum**，量级在 $10^4$ 以上，$\exp(\Delta \log \pi)$ 会直接数值爆炸。改用 mean 等价于把整个散度除以维度 $D$：

$$
\log \rho_t^{\text{mean}} = \frac{1}{D}\log \rho_t^{\text{sum}}
\quad\Longrightarrow\quad
\rho_t^{\text{mean}} = \left(\rho_t^{\text{sum}}\right)^{1/D}
$$

也就是一次**隐式的温度缩放**。它让训练数值上可行，但同时让裁剪阈值 $\epsilon$ 彻底失去了「策略变化幅度」的原本含义——这正是 DDPO 需要把 clip range 设到 $10^{-4}$（而非 LLM 的 0.2）的原因，也是 GRPO-Guard 的问题起点。

**③ 随机性来源：采样温度 vs 注入噪声**

LLM 的探索来自词表上的随机采样，天然存在、无需设计。Flow 的 ODE 采样是确定性的，探索必须**人为注入**：$\sigma_t$ 成了一个必须调的超参数（Flow-GRPO 取 $\sigma_t = a\sqrt{t/(1-t)}$，$a\approx0.7$）。$\sigma_t$ 太小则组内奖励方差趋零、优势全为 0 无梯度；太大则偏离预训练分布、画质崩坏。**探索强度在 LLM 里是免费的，在 flow 里是需要权衡的设计变量。**

**④ 参与更新的步：全部 vs 窗口**

LLM 里生成的每个 token 都进入损失。Flow-GRPO 由于 Denoising Reduction（训练只走 10 步、推理仍 40 步），以及只在 SDE 窗口内的步才有随机性，实际只有 $S'$ 步产生有效的 log-prob 与梯度——**训练时的 MDP 和推理时的采样过程并不是同一个东西**，这是 flow 侧独有的 train/inference 不一致问题。

**⑤ ratio 的形状与裁剪的失效**

两侧的比率张量形状不同：LLM 是 `[B, T_tokens]`，flow 是 `[G, S']`。更关键的是分布性质。两者都满足 $\mathbb{E}_{\text{old}}[\log\rho] = -D_{\mathrm{KL}} \le 0$，但：

- LLM 中 per-token KL 极小（$\sim10^{-3}$），比率几乎紧贴 1，裁剪按预期工作；
- Flow 中 $\sigma_t$ 随 timestep 变化数量级，导致各步 KL 量级差异巨大（GRPO-Guard 测得跨步梯度差达 **20×**）。单一固定的 $\epsilon$ 无法同时适配高噪步与低噪步，结果是比率分布均值左移、正优势样本难以进入上裁剪区，**正梯度事实上不受约束** → 隐式过优化。

**⑥ KL 的代价：近似 vs 闭式**

这是一处 flow 侧反而**占优**的差异。LLM 要算精确 KL 需在十万量级词表上求和，实践中只能用 k3 等无偏估计器。而 flow 模型每步都是各向同性高斯，KL 有精确闭式解，且只需两次已有的前向传播、零额外成本：

$$
D_{\mathrm{KL}}\big(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta\big)
= \frac{\big\|\mu_\theta(x_t,t) - \mu_{\theta_{\text{old}}}(x_t,t)\big\|^2}{2\sigma_t^2 \Delta t}
$$

Flow-DPPO 正是抓住这一点：既然能算精确散度，就不必再用含噪的单样本比率去代理它。

### 5.3 三处「水土不服」与对应的修补工作

把上面的差异串起来，就能看清 2025 下半年那批工作在修什么：

| 症状 | 根因（差异来源） | 对应工作 |
|------|------|------|
| proxy 奖励升、gold 质量崩（隐式过优化） | mean 聚合 + 跨步 $\sigma_t$ 差异 → 裁剪对正样本失效 | **GRPO-Guard**：RatioNorm 归零均值、消跨步方差 + $1/\mathrm{d}t$ 梯度重加权 |
| 过约束与欠约束并存、OOD 退化 | $\lvert\rho_t-1\rvert$ 只是高维连续散度的**单样本 MC 估计**，噪声被维度放大 | **Flow-DPPO**：改用精确闭式 KL 做非对称散度掩码 |
| 信用分配失效、奖励稀疏 | 每步都注噪 → 终点单一奖励无法归因到某一步 | **G²RPO**：只在单步注噪，使组内奖励方差完全由该步决定 |
| 多目标梯度互相干扰（跷跷板） | 标量奖励信息量太低（LLM 侧有 verifier，图像侧只有 RM 打分） | **Flow-OPD**：改用教师速度场的**稠密**监督 |

一句话总结：**GRPO 的「组内相对优势」可以无损迁移，但 PPO 的「比率裁剪」不能**。前者只依赖奖励的序关系，后者依赖概率比率的统计性质——而后者在从「离散 token 概率」换到「高维连续高斯密度」时被破坏了。

## 6. Survey 论文推荐

好消息是，这个方向已经有多篇高质量 survey/tutorial：

**Survey 1：Alignment and Safety of Diffusion Models via RL and Reward Modeling: A Survey**（Lamba et al., IIIT Hyderabad）。arXiv [2505.17352](https://arxiv.org/abs/2505.17352)，2025.05（2026.05 更新 v2，27 页）。最全面的 survey，从反馈来源、信号形式、优化机制、分布偏移与 reward hacking 处理、安全约束五个维度组织文献。⚠️ 该 survey 无实验复现，对比表的「效率」列多为定性描述。**强烈推荐作为入门与全景参考。**

**Survey 2：Understanding RL-Based Fine-Tuning of Diffusion Models: A Tutorial and Review**（Uehara et al.）。arXiv [2407.13734](https://arxiv.org/abs/2407.13734)，2024.07。教程式 survey，系统讲解 PPO、differentiable optimization、reward-weighted MLE 等算法，有配套代码。

**Survey 3：Preference Alignment on Diffusion Models: A Comprehensive Survey**，Computer Science Review (ScienceDirect), 2026。专注偏好对齐。

**相关资源**：[Awesome-Visual-Generation-Alignment-Survey](https://github.com/XueZeyue/Awesome-Visual-Generation-Alignment-Survey)（DanceGRPO 团队维护的 paper list）。

## 7. 开源实现与复现路径

论文之外，2026 年出现了两个成熟的**多模态生成 RL 训练框架**，前文提到的方法大多在其中有可跑的 recipe。

### 7.1 VeRL-Omni（verl-project / 字节）

`verl` 的官方下游扩展包，专注多模态生成模型的 RL 后训练；以 git commit pin 的方式硬依赖 verl，纯文本 LLM 的 PPO/GRPO 直接转发给 verl 原生 trainer。

- **覆盖模型**：Qwen-Image / Qwen-Image-Edit、SD3.5、FLUX.1、Wan2.2、LTX2.3、MiniMax-H3、BAGEL、Qwen3-Omni
- **算法**：FlowGRPO、DanceGRPO、MixGRPO、**GRPO-Guard**、**Flow-DPPO**、**DiffusionOPD（含多教师 MOPD，即 §4.8 的 Flow-OPD 路线）**、DiffusionNFT、Diffusion-DPO、GSPO
- **奖励**：HPSv2/v3、PickScore、ImageReward、GenRM-OCR、UnifiedReward、CLAP、ImageBind、JPEG 可压缩性，支持 HTTP 外部打分服务；自研 **DiNa-LRM** 直接给 clean latent 打分、**跳过 VAE decode**
- **工程**：Ray + Hydra；vLLM-Omni rollout；VeOmni / FSDP2 + USP/TP/DP 组合并行；异步多奖励与 rollout 重叠；支持昇腾 NPU。官方称在 Qwen-Image FlowGRPO 设置下端到端吞吐比 diffusers 版 `flow_grpo` 高约 **25%**

### 7.2 UniRL（腾讯混元 × Sea AI Lab）

口号是「A RL Framework for Unified Multimodal Models」，但**统一的是框架维度**——扩散 / AR / 统一模型 / Prompt-Enhancer / Agentic 五条路径共用同一个 `rollout → reward → advantage → train → weight sync` 循环，而非某个统一基座。

- **最大价值：它是 Flow-DPPO 的官方实现**（`unirl/algorithms/flowdppo.py` + `FlowDPPO/` 教程）。另有两篇团队自研的 LLM 侧算法：**DRPO**（arXiv:2606.09821）与 **CPPO**（arXiv:2606.10968）
- **算法**：GRPO、GSPO、PPO、DPPO、DRPO、CPPO、FlowGRPO、Flow-DPPO、DiffusionNFT、DiffusionOPD、SFT
- **奖励**（26 个打分器）：aesthetic、CLIP、HPSv2/v3/v3pp、ImageReward、PickScore、OCR、GenEval2、VideoAlign、CLAP、LLM-judge，另含数学/选择题 verifier
- **基座**：SD3/3.5、Qwen-Image、FLUX.2-Klein、Z-Image、WAN 2.1/2.2、HunyuanVideo、LTX-Video、Qwen3 / Qwen-VL / Qwen3-Omni，以及统一 AR+扩散的 HunyuanImage3、Bagel、SenseNova-U1.5

两个值得注意的实现细节，恰好印证前文 §5 的判断：

1. **DanceGRPO / MixGRPO 不是独立算法类**，只是 `flowgrpo.py` 基类下**不同 SDE 注噪窗口调度**的 recipe——差异确实在噪声注入与裁剪机制，而非 GRPO 骨架。
2. UniRL 让 rollout forward 与 train forward 跑在**同一个** transformer 实例上，确保 `new_logp` 与 `old_logp` 落在同一份权重上——这是 §5.2 讨论的 log-prob 一致性问题在工程上的处理。

### 7.3 怎么选

| 需求 | 推荐 |
|------|------|
| 复现 **Flow-DPPO**（§4.7） | UniRL（官方实现 + 教程） |
| 复现 **GRPO-Guard**（§4.6）/ **Flow-OPD**（§4.8） | VeRL-Omni（`grpoguard_trainer` / `diffusionopd_trainer`） |
| 已有 verl 技术栈、追求吞吐与 NPU 支持 | VeRL-Omni |
| 需要 Agentic / Prompt-Enhancer / 统一模型（Bagel、HunyuanImage3） | UniRL |
| 官方原始复现（SD3.5 + GenEval 63→95） | 作者原版 [`yifan123/flow_grpo`](https://github.com/yifan123/flow_grpo) |

## 8. 技术对比与演进趋势

```
时间线演进：
2023.05  DDPO / DPOK         → RL 用于扩散模型的开创，去噪 = MDP
2023.06  HPSv2               → 奖励模型基础设施成型（去偏置、多来源）
2023.11  Diffusion-DPO       → Offline 偏好优化路线开辟
2024     AlignProp / DRaFT   → 可微奖励反传的效率优化
2025.05  Flow-GRPO           → ODE→SDE 打开流模型 RL 大门 (NeurIPS 2025)
2025.05  DanceGRPO           → 统一图/视频生成的 GRPO 框架，视频 motion +181%
2025.09  SRPO (HunYuan)      → 工业级落地，10 分钟收敛、真实感提升 3 倍
2025.10  G²RPO               → 单步随机 + 多粒度优势，修奖励稀疏
2025.10  GRPO-Guard          → 诊断比率裁剪失效，RatioNorm（治标）
2026.05  Flow-OPD            → 多教师在线蒸馏，换范式绕开稀疏奖励
2026.06  Flow-DPPO           → 精确 KL 散度掩码替代裁剪（治本）
2026     UnifiedReward       → 图/视频 × 理解/生成四象限统一奖励
2026     VeRL-Omni / UniRL   → 开源训练框架成熟，上述算法有了统一的工程实现
```

**三条主线判断**：

1. **基座迁移：Diffusion (SDE) → Flow Matching (ODE)**。新一代生成模型（SD3、FLUX）采用确定性 ODE 采样，无法直接套用为随机探索而生的 RL，催生了「ODE→SDE 转换」这一关键技术，并由此打开 GRPO 家族的整个战场。
2. **算法迁移：PPO → GRPO**。无需价值函数、依赖组内相对优势的 GRPO 更稳定省资源，成为 2025 年后的绝对主流；但它从 LLM 照搬到高斯连续策略上，暴露出比率裁剪失效、奖励稀疏、隐式过优化等一系列缺陷，被后续工作逐一修补。
3. **范式并存：Online RL / Offline DPO / 可微反传 / 在线蒸馏 四线并进**。没有单一赢家——DPO 稳定省心但无法持续探索；可微反传（SRPO）10 分钟收敛但受奖励模型偏置制约；在线蒸馏（Flow-OPD）用稠密监督绕开稀疏标量奖励但受教师天花板限制。

### 方法对比表

| 方法 | 类型 | 需奖励模型 | 需偏好数据 | Online/Offline | 适用架构 | 一句话定位 |
|------|------|:---:|:---:|:---:|------|------|
| DDPO | Policy Gradient | ✅ | ❌ | Online | Diffusion (SDE) | MDP + 策略梯度开山之作 |
| DPOK | Policy Gradient | ✅ | ❌ | Online | Diffusion (SDE) | 加 KL 正则的稳定版 |
| Flow-GRPO | GRPO | ✅ | ❌ | Online | Flow Matching | ODE→SDE，GRPO 入流匹配奠基 |
| DanceGRPO | GRPO | ✅ | ❌ | Online | Diffusion + Flow | 统一图/视频的广度框架 |
| G²RPO | GRPO | ✅ | ❌ | Online | Flow Matching | 单步随机 + 多粒度优势 |
| GRPO-Guard | GRPO | ✅ | ❌ | Online | Flow Matching | 修比率裁剪（治标） |
| Flow-DPPO | GRPO | ✅ | ❌ | Online | Flow Matching | 精确 KL 散度掩码（治本） |
| Flow-OPD | 在线蒸馏 | ✅(教师) | ❌ | Online | Flow Matching | 多教师蒸馏替代稀疏奖励 |
| Diffusion-DPO | Preference Opt | ❌ | ✅ | Offline | Diffusion / Flow | 离线偏好优化基石 |
| AlignProp / DRaFT | Reward Gradient | ✅(可微) | ❌ | Online | Diffusion | 全时间步 / 截断反传 |
| SRPO | Reward Gradient | ✅(可微) | ❌ | Online | Flow Matching | 10 分钟收敛，工业级 |
| HPSv2 | 奖励/评测 | — | ✅ | — | — | 公共偏好基准与打分器 |
| UnifiedReward | 奖励/评测 | — | ✅ | — | — | 统一多模态奖励模型 |

## 9. 开放挑战与未来方向

1. **多目标对齐**：美学、语义、安全、多样性常互相冲突。Flow-OPD 实测出「跷跷板效应」——加 OCR 奖励使 GenEval 掉 5%，再加 PickScore 又掉 7%，梯度内积为负。如何逼近 Pareto 前沿仍未解。
2. **Reward Hacking 抑制**：KL（DPOK/Flow-GRPO）→ RatioNorm（GRPO-Guard）→ 散度掩码（Flow-DPPO）层层修补，但根因是**奖励模型本身的缺陷**，最终需要 scaling reward model 来解决。
3. **高效训练**：Denoising Reduction、单步随机（G²RPO）、可微反传（SRPO 5.3 GPU·h vs DanceGRPO 480）、在线蒸馏都在压采样成本，但视频等长轨迹场景的 rollout 代价仍是瓶颈。
4. **视频与长轨迹 RL**：从图像到视频，奖励更稀疏、轨迹更长、显存翻倍。DanceGRPO 已迈出第一步，但 text-video alignment 维度仍不稳定。
5. **对抗鲁棒的安全对齐**：把安全作为**显式硬约束**而非泛化偏好、覆盖对抗/OOD prompt，仍是最薄弱的环节。代表性尝试如 **Focus-N-Fix**（用 attention mask 只微调问题区域）和 **AlignGuard**（把安全专家做成 LoRA），但两者都不针对对抗 prompt。
6. **可解释奖励建模**：能否把标量分数分解为「太暗 / 缺物体 / 可能冒犯」等可审计因子？
7. **持续对齐**：随社会规范漂移做增量更新而不遗忘旧偏好。

## 10. 总结

图像生成 × 强化学习是一个快速发展的交叉领域。从 2023 年 DDPO 的开创，到 Flow-GRPO 在 NeurIPS 2025 的发表，再到工业界（如腾讯混元 SRPO）的落地，这个方向在三年内已形成完整的方法论体系，并在 2025 下半年进入「深水区修补」阶段——从「能用」走向「用好」。

核心趋势是：

- **从 Diffusion SDE → Flow Matching ODE**：确定性采样破坏了 RL 的探索前提，ODE→SDE 转换成为新一代基座的必经之路
- **从 PPO → GRPO**：无需 value function 的组相对优化更稳定；但连续高斯策略下裁剪机制本身需要重新设计（GRPO-Guard / Flow-DPPO）
- **从稀疏标量奖励 → 稠密监督**：单步随机（G²RPO）、教师速度场蒸馏（Flow-OPD）都在解决信用分配
- **从单一奖励 → 多维度对齐**：美学 + 安全 + 文本忠实 + 结构正确，且必须处理目标间的梯度干扰

如果你想入门这个方向，建议的阅读顺序是：

1. **DDPO** → 理解 MDP + 策略梯度的基本框架
2. **Diffusion-DPO** → 理解离线偏好路线
3. **Flow-GRPO** → 理解 ODE→SDE 与当前 SOTA 范式
4. **GRPO-Guard / Flow-DPPO** → 理解 GRPO 的缺陷与修补
5. **Survey [2505.17352]** → 建立系统全景

若关注工业落地，重点看 **SRPO**；若关注多任务对齐，看 **Flow-OPD**；**若你是从 LLM RLHF 转过来的，建议先读本文 §5**——GRPO 的外壳不变，但离散 token 概率换成高维连续高斯密度后，PPO 的比率裁剪机制会失效，这是后续一系列工作的共同起点。

## 参考文献

1. Black et al. "Training Diffusion Models with Reinforcement Learning" (DDPO), ICLR 2024. [arXiv:2305.13301](https://arxiv.org/abs/2305.13301)
2. Fan et al. "DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models", NeurIPS 2023. [arXiv:2305.16381](https://arxiv.org/abs/2305.16381)
3. Liu, Liu et al. "Flow-GRPO: Training Flow Matching Models via Online RL", NeurIPS 2025. [arXiv:2505.05470](https://arxiv.org/abs/2505.05470)
4. Xue, Wu et al. "DanceGRPO: Unleashing GRPO on Visual Generation", 2025. [arXiv:2505.07818](https://arxiv.org/abs/2505.07818)
5. Zhou, Ling et al. "Fine-Grained (Granular) GRPO for Precise Preference Alignment in Flow Models" (G²RPO), 2025. [arXiv:2510.01982](https://arxiv.org/abs/2510.01982)
6. Wang, Liang et al. "GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping", 2025. [arXiv:2510.22319](https://arxiv.org/abs/2510.22319)
7. Ping, Zhou, Pang et al. "Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models", 2026. [arXiv:2606.11025](https://arxiv.org/abs/2606.11025)
8. Fang, Huang et al. "Flow-OPD: On-Policy Distillation for Flow Matching Models", 2026. [arXiv:2605.08063](https://arxiv.org/abs/2605.08063)
9. Wallace et al. "Diffusion Model Alignment Using Direct Preference Optimization", CVPR 2024. [arXiv:2311.12908](https://arxiv.org/abs/2311.12908)
10. Shen, Li et al. "Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference" (SRPO), Tencent HunYuan, 2025. [arXiv:2509.06942](https://arxiv.org/abs/2509.06942)
11. Wu, Hao et al. "Human Preference Score v2" (HPSv2), 2023. [arXiv:2306.09341](https://arxiv.org/abs/2306.09341)
12. Wang, Zang et al. "Unified Reward Model for Multimodal Understanding and Generation" (UnifiedReward), 2026. [arXiv:2503.05236](https://arxiv.org/abs/2503.05236)
13. Lamba, Ravish, Kushwaha, Kumar. "Alignment and Safety of Diffusion Models via RL and Reward Modeling: A Survey", 2026. [arXiv:2505.17352](https://arxiv.org/abs/2505.17352)
14. Prabhudesai et al. "AlignProp: Aligning Text-to-Image Diffusion Models with Reward Backpropagation", ECCV 2024. [arXiv:2310.03739](https://arxiv.org/abs/2310.03739)
15. Clark et al. "DRaFT: Directly Fine-Tuning Diffusion Models on Differentiable Rewards", ICLR 2024.
16. Xu et al. "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation", NeurIPS 2023. [arXiv:2304.05977](https://arxiv.org/abs/2304.05977)
17. Uehara et al. "Understanding RL-Based Fine-Tuning of Diffusion Models: A Tutorial and Review", 2024. [arXiv:2407.13734](https://arxiv.org/abs/2407.13734)
18. "Preference Alignment on Diffusion Models: A Comprehensive Survey", Computer Science Review, 2026.
19. Hu et al. "Towards Better Alignment: Training Diffusion Models with RL Against Sparse Rewards", CVPR 2025.
20. VeRL-Omni: RL training framework for diffusion and omni-modality models, verl-project, 2026. [GitHub](https://github.com/verl-project/verl-omni)
21. UniRL: A Reinforcement Learning Framework for Unified Multimodal Models, Tencent-Hunyuan, 2026. [GitHub](https://github.com/Tencent-Hunyuan/UniRL)
22. Yao et al. "Rethinking the Divergence Regularization in LLM RL" (DRPO), 2026. [arXiv:2606.09821](https://arxiv.org/abs/2606.09821)
23. "Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning" (CPPO), 2026. [arXiv:2606.10968](https://arxiv.org/abs/2606.10968)
