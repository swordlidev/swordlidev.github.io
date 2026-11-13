---
title: "Kimi K3：把开源推回第一条 scaling 轴"
description: "2.8T 总参 / 104B 激活 / 原生多模态 / 1M 上下文的开源模型：KDA 混合注意力、Attention Residuals、Stable LatentMoE 与 Quantile Balancing、3 域 × 3 effort 的九专家 RL + 多教师 on-policy 蒸馏，以及支撑百万 token agentic RL 的全套基础设施。本文逐层拆解，并照搬它承认的差距。"
pubDate: 2026-08-28
tags: ["Kimi K3", "MoE", "线性注意力", "Agentic RL", "长上下文", "推理优化", "开源模型"]
---

> **本文解决一个问题**：LLM 有两条 scaling 轴——**预训练算力**与**测试时计算**。推理模型兴起后第二轴被充分开发，但开源生态**只在第二轴上跑**，近期模型基本停在 1T 参数级；越来越精巧的 RL 方法被反复施加到**同一量级的基座**上，结果是开源内部趋同、与闭源差距扩大。
>
> K3 的立场是**两条轴一起推**：基座做到 3T 级（首个开源 3T 模型），同时在 1M 上下文上扩 RL、扩 reasoning effort、扩长程交互。本文按架构、预训练、后训练、环境、基础设施、评测六段拆解，重点放在三件可迁移的事：**用算法改动去打开硬件路径**、**负载均衡从 sign 更新换成分位数**、以及**百万 token agentic RL 需要什么样的沙箱与缓存**。
>
> 论文：*Kimi K3: Open Frontier Intelligence*（Moonshot AI，arXiv:2607.24563v2）。规格：2.8T 总参 / 104B 激活 / 原生多模态 / 1M 上下文 / 全量权重开源。定位写得诚实：**整体落后 Claude Fable 5 与 GPT-5.6 Sol，但稳定领先其余所有开源与闭源模型**。

## 一、架构：沿三个维度扩展信息流

**序列长度（token mixing）、网络深度（layer mixing）、模型宽度（channel mixing）**，外加输入端原生视觉通路。

```text
每个 block = 3× KDA + 1× Gated MLA，每个注意力层后接一个 Stable LatentMoE
AttnRes 用可学习 pseudo-query 在 embedding 与前序 block 输出上做 attention
backbone 末尾补一个 Gated MLA，确保最后一层一定是全局注意力
```

### 1.1 Hybrid Attention（KDA : MLA = 3 : 1）

KDA 是带**通道级遗忘门**的 delta rule 递归：

$$S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top$$

相对 Kimi Linear 的三处改动：

| 改动 | 内容 | 动机 |
|---|---|---|
| **下界化衰减** | log-decay 从无界 `−e^A·Softplus(z)` 换成 `g_min·Sigmoid(e^A z)`，`g_min = −5` | 分块形式用 `1/Γ` 重标定 key，无界衰减会溢出 |
| **全秩输出门** | 低秩门 → 输入相关的全秩投影 | 让每个 token 调制从注意力读出的通道 |
| **FP32 注意力输出** | 训练保持 FP32，输出 tile 与 KV staging buffer 重叠而非与 query tile 重叠 | 修正 flash attention 的有偏舍入，换来更深 KV 流水 |

> **下界化是用算法改动换硬件路径**：`g_min = −5` ⇒ 16-token tile 上累积 log-decay 落在 `(−80, 0)`，倒数重标定因子 `< e⁸⁰`，**留在 BF16 动态范围内**。于是对角 tile 也能走稠密 Tensor Core 矩乘，**消掉 Kimi Linear 那条按位置对逐个算的对角路径**——intra-chunk 的主瓶颈。

**Gated MLA**：沿用 K2 的 MLA，但**所有 MLA 层改 NoPE**——位置敏感性交给中间的 KDA 层，MLA 只做无约束全局内容交互。收益：**扩上下文无需调 RoPE base 或上 YaRN**。

### 1.2 Attention Residuals

标准残差把前序信息压进单一状态——**深度维度上的 RNN 瓶颈**。AttnRes 让每层用可学习 pseudo-query 在 `{embedding, 各前序层输出}` 上做 softmax 注意力（key 上套 RMSNorm，防大幅值层主导）。

深度不大（L<100），`O(L²d)` 算力可接受；真正的开销是 `O(Ld)` 的**显存与跨 PP stage 通信**。**Block AttnRes** 把 L 层切成 N 块、块内求和归约、跨块只在 N 个块表示上做全注意力：开销降到 `O(Nd)`，同时**界定推理期状态**。K3 取 **8 块 × 12 层**。

### 1.3 Stable LatentMoE（896 专家 / 16 激活 / 稀疏度 56）

LatentMoE 解耦**模型全宽**与**路由专家宽度**：shared expert 走全宽，routed expert 在 `ℓ = 0.5d` 的潜空间算。极端稀疏放大两个失效模式：

**① 激活爆炸**（routed 路径是近四次连乘的病态结构）：

- **聚合后、`W↑` 前插 RMSNorm**——除稳定性外，validation loss 与下游 benchmark 都一致变好；
- **SiTU-GLU**：对 Swish 门的线性因子与 up 分支**分别**施加软顶 `β·tanh(x/β)`（`β₁=4` / `β₂=25`）。原点处与 SwiGLU **一阶吻合**，`β→∞` 退化为 SwiGLU，输出有硬界 `≤ β₁β₂ = 100`。相比 hard clamp，平滑软顶**在饱和边界外保留非零梯度**。

**② 近 10³ 专家的负载均衡**超出 auxiliary-loss-free 偏置更新的适用区间 → **Quantile Balancing (QB)**：

| | 原方法 | QB |
|---|---|---|
| 更新式 | `b ← b + γ·sign(目标 − 实际载荷)` | `b ← (1−k/n) 分位数(margin)` |
| 超参 | `γ` 要在「适应慢」与「振荡」间权衡 | **无学习率类超参** |
| 本质 | 对同一对偶目标做 SignSGD | **直接跳到该对偶目标的精确坐标最小点** |

实现巧思：路由把 Top-k 换成**对带偏置分数的 Top-(k+1)**，第 (k+1) 项即 cutoff，省掉单独算 token 侧分位数。

> **百万 token 全局 batch 上怎么求分位数**：不求精确值，**按专家维护分箱直方图**。各 rank 本地 scatter-add，步末**一次整数 all-reduce**（`nB` 值/层/步，与 token 数无关，<1% 于逐 micro-batch 交换原始 margin）。三条性质：误差被箱宽界住（B=1000 时几个 10⁻³，实测无残余不均衡）；通信便宜；**因计数可加，全局直方图对 token 分片方式严格不变**——估的是池化全局 batch 的分位数，而非「各 rank 分位数的平均」。

### 1.4 原生视觉：MoonViT-V2 从零训

与 K2.5 的关键分歧：**视觉编码器完全从零用 next-token prediction 训，不从 SigLIP 初始化**。动机首先是稳定性（SigLIP 初始化的 MoonViT-3D 梯度范数持续更高且频繁尖峰），其次是表示由语言建模目标而非偏好全局语义的对比损失塑形。

> 结论：**MoonViT-V2 在各项视觉评测上与 SigLIP 初始化基线持平**——「规模化多模态 LM 上，对比预训练作为初始化并非必需。」

27 层 / 0.4B，RMSNorm、**线性与注意力投影全部去 bias**；图像与视频共享参数，注意力拆帧内空间 + 帧间时序；投影前 2×2 pixel-shuffle 把视觉 token 砍到 1/4，使 3584×3584 输入在 1M 上下文内可负担。

### 1.5 Per-Head Muon

注意力投影**沿 head 维切动量矩阵、逐 head 做 Newton–Schulz 正交化**。全矩阵正交化会把所有 head 当成单一耦合块，大尺度 head 主导共享更新方向。附带好处：瘦高块上跑 Newton–Schulz 更便宜。

> 与 [GLM-5 的 Muon Split](/blog/glm-5-agentic-engineering/)是同一思路的两次独立命中：**正交化的粒度要和注意力头的语义粒度对齐**。

### 1.6 K2 → K3 diff

| | K2 | K3 | Δ |
|---|---|---|---|
| 层数 | 61 | 93 | ↑52% |
| 总参 / 激活 | 1.04T / 32.6B | 2.78T / 104.2B | ↑167% / ↑220% |
| Hidden | 7,168 | 7,168 | = |
| 路由专家 / 每 token 激活 | 384 / 8 | 896 / 16 | ↑133% / ↑100% |
| 共享专家 | 1 | 2 | ↑100% |
| 注意力头 | 64 | 96 | ↑50% |
| 训练上下文 | 128K | 1M | ×8 |
| 注意力机制 | 61 MLA | 69 KDA + 24 MLA | – |
| 激活函数 | SwiGLU | SiTU-GLU | – |
| ViT | – | 401M / 27 层 | – |

**整体 scaling 效率相对 K2 提升约 2.5×。**

## 二、预训练

**数据**：四个文本域 + 大规模视觉语料，域内规则启发式 + 分类器质量打分 + 去重，**采样率由小模型消融决定**。视觉侧坐标监督**同时给绝对与归一化两种格式**；大幅扩**程序化多模态数据**——代码片段与其渲染结果配对（SVG / 3D 资产 / 网页 / 游戏 / CAD）。

> **Scaling law 的方法论观察**：cosine 一致优于 WSD，**但前提是各自独立搜超参**。两种调度在同样规模与 token 预算下，最优峰值 lr 与 batch size 差异很大；共享一套超参比较，只是在测「谁更贴合这组超参」。

**长上下文**：四段式扩窗——**预训练内 8K → 64K，cooldown 阶段 256K → 1M**，把昂贵的长序列计算压在总预算一小部分里。NoPE 让模型**直接外推到 1M**。数据侧两点：天然长文档低质内容多，需专门清洗 + **上采样真正长且连贯的样本**；更关键的是——**长度本身不赋予长程能力**，因此合成长上下文数据时精心置换拼接多模态文档与子任务，使任务**只有靠关注散布在整个 1M 窗口的信息才能解**，防止注意力退化成局部模式。

## 三、后训练：SFT → 9 个专家 → 统一

**SFT 冷启动 → 按域 × effort 训专家 → Multi-Teacher On-Policy Distillation (MOPD) 合回单模型**。

**SFT**：用前代**域专精模型**合成轨迹，经多阶段验证与 human-in-the-loop 标注，全部用 XTML template 序列化；**从 SFT 起就开 QAT**（MXFP4 权重 / MXFP8 激活）。

**RL：3 域 × 3 effort = 9 个专家**。三域为 general（通用体验 / 视觉 / 推理 / faithfulness / 搜索 / 知识工作）、general agents（长程助理 / deep research / 段落级写作）、coding agents（SWE / 编码体验 / kernel / Web 开发）。趋势：**随 RL FLOPs 增长，tool-call 步数一致上升，并伴随能力全面提升**。

算法层三点：

1. **Partial rollout**：每轮采 `N×K` 轨迹，**只要 λ 比例完成就暂停生成**，不被执行长尾卡住；暂停的 rollout 入队、下轮优先恢复。长程轨迹因此跨多轮迭代产生 staleness——**靠 per-token 正则把更新约束在局部邻域**，在极端 off-policy 下仍稳定。
2. **Reasoning Effort RL**：冷启动模型估出每题预算 `b₀(x)`，总量超过 `τ·b₀(x)` 的轨迹**奖励直接覆写为 −1**（general 任务数思考 token，agentic 任务数累计输出 token 含工具调用参数）。课程按 `τ` 分段：**先训较大 τ 的 max 变体（仍封顶以抑制过度思考），再退火 τ 得到 high / low 专家**，`τ` 按域配置、human-in-the-loop。各档专家的轨迹**统一回收**用于 SFT 与 MOPD。
3. **Agentic GRM**：非可验证任务用锦标赛式分组 + 二元比较，judge **强制走固定协议**（读产物 → 生成 rubric → 按 rubric 打分 → 记入 scorepad）。防冗长 hacking 用**预算式控制**——与 effort 控制同构：输出超 `σ·ℓ₀` 的候选**自动判负**。

**MOPD**：给定域 d 与采样 effort e，由对应教师引导，per-token 奖励是**截断的对数似然比**：

$$r_{\text{opd}} = \mathrm{clip}\!\left(\mathrm{sg}\!\left[\log \frac{\pi_{\text{teacher}}^{(d,e)}(y_t)}{\pi_\theta(y_t)}\right], -R_{\max}, R_{\max}\right)$$

**稠密奖励**，可无缝接入现有 RL 框架（partial rollout 等优化照样适用）。更细粒度的 top-k 蒸馏目标**在收敛速度与最终性能上都没看到优势**。

**部署感知**：

- **MXFP4 QAT 贯穿 SFT 与 RL**，只量化 MoE 专家权重，注意力/latent 投影/共享专家/路由器保持高精度。关键在于 **RL 期间 rollout 与 training 共用同一量化方案，训练—推理失配被消除**。
- **Draft model**：K3 自带结构与 backbone block 一致的 MTP 层，正好匹配 EAGLE-3 的单解码层 draft，于是微调成 draft（冻结目标、unroll 7 步），输入融合第 1/4/末 AttnRes 块输出，融合矩阵初始化为 `[0 0 I]` ⇒ **初始时刻恰等于 MTP 层预训练时的输入**。
  > 目标函数值得记：加速由**逐 token 接受率** `Σ min(p,q)` 决定，而最小化 KL 代理**并不保证**在容量受限 draft 上最大化它 → **直接优化接受率的负对数** `L_LK`，温度 1，不加交叉熵辅助项。

## 四、RL 任务合成与 agentic 环境

**统一白盒环境**：固定 harness 会让模型过拟合特定工具 schema、system prompt、上下文管理机制或协议。K3 把 **harness 抽象成可配置、可组合的模块集合**（工具接口 / system prompt / 上下文管理 / skills / memories / subagents），配置即可实例化 Kimi Code / Claude Code / Codex / OpenClaw / Hermes 乃至全新 harness，**训练时为不同任务组动态换配置**。

**知识图谱引导的任务合成**：任务质量与多样性主要由源材料决定。构建**自演化分层知识图谱（DAG）**——种子节点派 agent 做 web search 调查，**新增节点前先探索已有图谱**以复用、减重复，边一律从粗指向细，直到 agent 判定概念足够原子。用图时按目标分布采样节点，关键词 + **祖先节点上下文**合成 web query，检索到的真实材料交给合成 agent 产出任务。

**五类环境的奖励设计**：

| 环境 | 设计要点 |
|---|---|
| **可验证 agentic** | 多步信息搜索；投行/数据分析/法律等真实工作（sandbox 内操作域工具，几十到几百步交付）；**视觉 in-the-loop**——Python 沙箱里迭代写码裁剪/缩放图像、校验中间结果，**执行输出（含生成图像）作为新 observation** |
| **Kernel 优化** | 覆盖 CUDA / Triton / CuTe DSL / Gluon / ThunderKittens / TileLang 与 BF16/FP8/FP4。**同时评正确性与性能**：超数值误差阈值得 0，**追平专家实现 0.5，逼近 roofline 趋向 1**。另建 **hacking 检测**惩罚 CUDA graph replay、输入缓存、降精度 |
| **个人助理** | Gmail / Notion / Slack / Canvas 的**真实感 mock**（保留核心语义，可复现、无限流）。agent 在**持续演化的持久环境**里跨多个模拟日工作，遭遇跨应用的数十个相互依赖事件；**单条 rollout 可达数千次工具调用与数百万上下文 token** |
| **AET（自主执行）** | 给初始状态、受约束目标、动作空间、执行预算、**独立 verifier**；agent **只看目标与验证接口**，无参考轨迹与预定流程。奖励**锚在 verifier 对最终环境状态的评估，而非 agent 自报完成**。防 hack 三招：**agent 与 verifier 隔离**、**诊断性公开 verifier + held-out 隐藏 verifier 配对**、**有限提交预算下的惩罚式奖励** |
| **Web 开发** | 产物涵盖网站 / 游戏 / 3D-WebGL / 可视化 / SVG / 全栈应用，**在多种 scaffold 下 rollout** 促进泛化。奖励 = 确定性检查（功能测试 + 结构与像素级相似度）+ 模型评判（审源码或真的交互产物）。**构建失败、运行报错、「假装实现」直接归零** |

> 「不要固定 harness」这条与[《Harness Engineering 综述》](/blog/harness-engineering-survey-2026/)里的观察互为正反面：harness 的每个可配置维度，既是工程自由度，也是模型可能过拟合的对象。

## 五、基础设施

K3 把三件很少同时出现的事叠在一起：**混合 KDA 注意力、3T 级稀疏多模态训练与推理、百万 token agentic 负载**。

### 5.1 KDA 的算法—系统协同

定长递归状态换掉增长的 KV cache：代价是串行更新难并行，收益是状态定长、传输与复用便宜。三个执行区间各配方案：

- **训练 / prefill**：**FlashKDA**（CUTLASS）把 **intra-chunk 计算与跨 chunk 状态传播重叠**（否则串行传播期间 SM 全闲），已作为 FLA 后端自动分派。
- **超长 prefill 的设备内 CP**：TP 只切 head、**从不缩短递归**。关键观察是**每段状态转移可独立于入段状态求值、事后精确复合**，于是在**单 rank 的 SM 之间**切序列并行求转移再合并，**零跨设备通信**。
- **跨设备 KCP**：
  > vanilla 线性注意力状态是加性的，各 rank 从 `S=0` 算完求和即可。但 KDA 的 delta rule **先把 `M_t` 作用到入段状态上**，局部段的效果**无法只由 `S=0` 的结果决定**。KCP 于是把每段效果分解成**两个本地可算的量**——累积转移 `M^{t←1}` 与从零起算的 `S̃^t`，得 `S^t_{[i+1]} = S̃^t_{[i+1]} + M^{t←1}_{[i+1]} S^{T_i}_{[i]}`。rank 级更新**满足结合律**，入段状态由 **prefix scan** 恢复，通信只需**一次定长 all-gather**。

### 5.2 3T 级预训练

并行策略：PP + virtual stage、EP、ZeRO-1 DP、Pipeline ZeRO-2 梯度分片、CP。三个问题逐一对应：

**① EP token 负载不均** → **MoonEP**（开源）。要求**每 rank 精确收到 `S×K` 个 token**，用动态冗余专家实现完美均衡。核心是一个**存在性证明**：**每 rank 至多 `E/R` 个冗余专家即保证平衡计划存在，且该界本质上紧** ⇒ 预留 `E/R` 槽位即**保证规划永有可行解、训练永不中断**；对比 ECHO / UltraEP 预设冗余数或设 per-rank 上限——**无可行解时必须停训**，上限还得手调且仍留残余不均衡。配套：ILP 离线解作参照的**近最优 GPU 规划 kernel**；**零拷贝通信**（最坏不均衡下 DeepEP 需 `S·K·R` 缓冲，MoonEP 只要固定 `S·K`）；完美均衡让形状**静态可知**，**消除 per-layer 的 MoE host–device 同步**；rank 内仍偏斜则用**负载感知的 Expert-GEMM 调度器**，shared expert 丢独立 stream 重叠。

**② 显存超预算** → 关键是一层抽象：**统一 activation manager**——每个反向保存的张量绑定可插拔存储后端，**重计算 / 量化 / offload 都只是该抽象下的存储策略**，张量粒度自由组合、注解声明、与模型代码解耦（K3 实配：多数激活走 block-wise FP8 + offload）。另四项针对性优化：MoE 的 `permuted_probs` 梯度经变换**从依赖前向 `output` 改写成只依赖中间激活**、group GEMM 反向**重算 dispatch**；AttnRes 块表示边界层生成一次全层共享且整体 checkpointing 包裹 ⇒ **每层保存的激活与标准残差完全相同**，PP 下只增量传输新块，**达到显存理论下界**；Mooncake Transfer Engine **把激活远程 offload 到其他 PP rank** 平衡 1F1B 偏斜；**P2P Muon 正交化**只从 owner rank 取自己的分片，消掉全参数 buffer。

**③ 视觉编码器的高可变计算在关键路径上** → 大图沿 patch 维跨设备切分 + sub-CP 组间负载均衡；更关键的是**把 ViT 算进 PP 气泡**——interleaved 1F1B 下首批 micro-batch 的文本前向全挤在开头、末批反向只在末尾完成，于是首批 ViT 前向提前执行、**其余调度进气泡**，**视觉编码器的有效开销基本消失**。

> 这一节与 [Megatron-LM 的并行与显存系列](/blog/megatron-lm-06-moe-ep-cp/)可对照阅读：同样的 EP / CP / 激活策略，在 3T 级稀疏模型上被推到了另一个量级。

### 5.3 1M agentic RL

co-located 训练把每个 1M 上下文实验控制在**几百张 GPU**内，partial rollout 压长尾。

- **外部 KV cache 池**：1M 下 prefix miss 极贵，而 partial rollout 在每轮开头把**上轮大量未完成的长 prefill 同时放进来**、投机解码又抬高 **prefix block churn**，都会触发抢占。方案是 **write-back 解耦 prefix 保留与 GPU 驻留**——活跃解码块留 GPU，**空闲 prefix 只在被逐出时才写回 CPU DRAM**，复用前预取；**KDA 状态与对应 MLA KV 块生命周期对齐**。为腾 DRAM，训练迭代结束后把权重与优化器态 **offload 到 NVMe**。
- **Rollout 自动限流**：上下文随轨迹增长，固定并发既难估又早期保守、设高则后期抢占 ⇒ 用活跃数 / 排队数 / KV 利用率**动态控制发往引擎的请求量**。
- **梯度 buffer 复用**：reference model 权重留 CPU 按需物化，**参数张量直接后端到策略模型的 FP32 梯度 buffer**——不额外分配、不碎片，且安全（稍后会被真实梯度覆写）；按 chunk 流式载入，一槽做前向、另一槽预取。

### 5.4 Sandbox：AgentENV

容器不够用的理由很具体：**agent 越强、探索越激进，甚至尝试 reward hacking**——早期容器 runtime 下观察到**多次由非预期 agent 操作引发的 kernel panic 与死锁**；而复杂任务又需要接近真实的环境（能挂磁盘、跑容器、甚至启虚拟机）。于是基于 **Firecracker microVM**，三组能力：

- **增量 checkpoint**（只存变脏页）：**checkpoint 133ms / resume 49ms**；
- 三个高层操作——**Pause/Resume**（暂停不占内存 CPU，而 agent 等推理结果**可占 sandbox 生命周期 98%**）、**Fork**（精确状态复制且原件继续运行，用于**无副作用评奖**）、**Snapshot**（错误恢复）；
- **高密度**：数万个各带独特镜像的 sandbox 需在数秒内创建 ⇒ OverlayBD + 自研 ublk 驱动 + 存储层共享 + P2P 传输做到**亚秒启动**，CoW 内存与 page cache 优化把内存**超售比做到 6.5×**。

> 全程共创建 **51,219,741 个 sandbox，跨 1,505,678 个镜像**。

### 5.5 推理与在线服务

**① KDA-aware prefix cache**。两种缓存大小与生命周期完全不同，但**一个 prefix 只有在两者能在同一边界一起恢复时才可复用**。

- **统一 paged 布局**：KDA 状态塞进与 MLA KV **同一 paged block 池**、页大小统一，分配/引用计数/逐出共用一套实现；页内各 head 连续存放使**每个 head 字节流自包含**，TP 度不同的 prefill/decode 分离时 **re-layout 在传输路径上完成，GPU 侧零重排**。
- **粒度解耦**（最实用的一点）：块哈希缓存要求所有层共享块大小、且命中边界必须已持久化 KDA 状态；而 KDA 每序列只有一个大状态、快照只在稀疏边界划得来 ⇒ 共享块大小被迫定在 **1024–6144 token**，**缓存近乎无用**（短于一块的请求永不可复用）。
  解法是**反向对齐**：**前缀哈希跑在 MLA 页内的 512-token 细哈希块上，物理块仍是粗分配单元；KDA checkpoint 只存在哈希端点的稀疏子集上**，且**对话轮边界上的 checkpoint 保留供跨请求复用**、缓存项是**永不原地改的只读快照**。查找取**同时满足两阶段的最长边界**——总是哈希块整数倍、**但从不要求是物理块整数倍**：6144 物理块内，前 2800 token 匹配的请求命中于 `B = 2560`，`[0,B)` 零重算。
  三条并发一致性规则：**命中块在任何分配前于所有组 pin 住**、**本调度步内刚分配/注册的块排除出匹配**、**逐出某组 checkpoint 原子失效其兄弟**。结果是**混合模型的前缀缓存达到与全注意力同等的通用性**。

**② 专用 kernel**

- **KDA 解码**瓶颈从「榨并行」转为「管理原地更新的递归状态」。**MTP 投机解码**下拒掉部分 draft token 时**状态已前进且无法回滚**，而逐位存快照会成倍放大状态流量（大 batch 下的主导成本）。关键观察：**已接受前缀之后的状态完全由 draft token 的投影输入决定，而它们远小于状态本身** ⇒ **只缓存投影输入、片上重建、只写回已验证与 bonus token**，重放与下一 draft 窗口**共享单个融合 kernel 内的同一条递归循环**，验证延迟**次线性**增长。
- **Block AttnRes**：prefill 把 TP all-reduce 拆成 reduce-scatter + all-gather、**intra-block kernel 插在两者之间**，使块表示**恰好只在一个 rank 物化**；decode 把 inter-block 放**侧流**重叠，intra-block 的合并与 RMSNorm **融进前面的 all-reduce**，取消专用 kernel。
- **Stable LatentMoE**：latent 下投影与 router **融成单个 GEMM**，latent 权重跨 rank 分片并把 all-gather **融进 GEMM epilogue**。小 batch 下 group GEMM 退化为**访存受限的权重流式读取**，tile-centric kernel 水土不服 ⇒ 改用 **WarpDecode 的 token-centric 设计**（每 warp 负责一个输出神经元）+ lane team 细分，权重布局**离线置换**降低运行时反量化开销。

**③ Fleet 级调度**：挑战从「单请求效率」变成「**可预测性**」。

- **Cache-aware affinity**：1M 下典型编码输入带 **400K prefix 但只需 4K prefill 增量** ⇒ 路由到持有其 cache 的集群。这会把 session 绑死，于是**一致性哈希钉到两个集群**；次集群无 cache、failover 需重 prefill，而**一致性哈希把次集群分配均匀打散，重 prefill 被分摊到许多集群而非集中到一个**。
- **Budget-based admission control**：流量混合 2K 与 1M 请求，**单请求成本跨约三个数量级**，基于「平均请求」的容量规划、排队模型、限流配额**全部失效**（典型失效：一波长请求打满算力，随后短请求无法及时调度，**全局 TTFT 一起恶化**）⇒ 给不同请求类**各自分配预算**。

## 六、评测

### 6.1 公开 benchmark（max / xhigh effort，温度 1.0）

| | **Kimi K3** | Fable 5 | GPT-5.6 Sol | Opus 4.8 | GPT-5.5 | GLM-5.2 |
|---|---|---|---|---|---|---|
| GPQA Diamond | 93.5 | 92.6 | **94.1** | 91.0 | 93.5 | 91.2 |
| CritPt | 23.4 | 28.6 | **32.3** | 20.9 | 27.1 | 20.9 |
| HLE-Full（无/有工具） | 43.5 / 56.0 | **53.3 / 63.0** | 44.5 / 58.0 | 49.8 / 57.9 | 41.4 / 52.2 | – |
| ProgramBench | **77.8** | 76.8 | 77.6 | 71.9 | 70.8 | 63.7 |
| Terminal-Bench 2.1 | 88.3 | 88.0 | **88.8** | 84.6 | 83.4 | 82.7 |
| FrontierSWE | 81.2 | **86.6** | 71.3 | 66.7 | 64.9 | 67.3 |
| SWE-Marathon | **42.0** | 35.0 | 39.0 | 40.0 | 14.0 | 13.0 |
| BrowseComp | **91.2** | 88.0 | 90.4 | 84.3 | 84.4 | – |
| MCPMark-Verified | **94.5** | 87.4 | 92.9 | 76.4 | 92.9 | – |
| GDPval-AA v2 (Elo) | 1686 | **1747** | 1736 | 1593 | 1491 | 1510 |
| OSWorld-Verified / 2.0 | 84.8 / 58.3 | **85.0 / 66.1** | 83.0 / 62.6 | 83.4 / 55.7 | 79.0 / 49.5 | – |
| Math-Vision（无/有 Python） | 94.3 / 97.8 | 94.8 / **98.6** | **95.8** / 97.8 | 86.7 / 97.1 | 92.2 / 96.8 | – |
| ZeroBench-main (pass@5) | 23.0 / 41.0 | 23.0 / **46.0** | 17.0 / 35.0 | 17.0 / 34.0 | 22.0 / 41.0 | – |

- **推理与知识**：研究生级已贴前沿，但**研究级任务仍有差距**（HLE-Full、CritPt），作者点名为重点方向。
- **编码**：ProgramBench 第一；**SWE-Marathon 领先 Fable 5 整 7 分**；Terminal-Bench 2.1 几乎追平 GPT-5.6 Sol；长程的 FrontierSWE 居第二且远超其余。
- **Agentic**：大片 SOTA（另有 DeepSearchQA 95.0、ResearchRubrics 76.2、τ3-Banking 33.4、Harvey Lab-AA 94.6 等），例外集中在 **Elo 制知识工作套件**与更难的 computer-use。
- **视觉**：**Python 工具显著放大能力**——Math-Vision 94.3→97.8，ZeroBench 23.0→41.0。

> 协议细节：BrowseComp 的 91.2 用的是**300K token 触发的上下文压缩**；改为**用满 1M 窗口、完全不做上下文管理**则为 90.4。

### 6.2 内部评测：优劣分得比公开榜更清楚

- **最强是编排型与研究型 agency**：Swarm Bench 76.3、Deep Research Bench 90.0 明显领先。
- **编码的「体验分」高于「任务分」**：Kimi CodeBench 2.0 仅次 Fable 5，而 **Coding Experience 最佳**——沟通质量、行为得体性、指令遵循稳定性**领先于裸任务分数**。Kimi Webdev Bench 盲评对比 Opus 4.8：整体 **Win−Lose +31.0%**，**3D/WebGL/Shader 达 +59.1%**。
- **弱项**：AgentBehavior Bench、MIRA Bench、24/7 ClawBench 2.0、Agentic Vision Bench、KWV Bench。其中 AgentBehavior Bench 把评估从**结果正确性**扩到**过程质量**（工具行为、效率、纪律性），K3 65.0 vs Fable 5 75.5 / GPT-5.6 Sol 76.4——**差距最大的一项恰是过程质量**。

### 6.3 成本效率与第三方

**四个套件上都位于或接近成本—效率前沿**：Kimi CodeBench 2.0 比 Fable 5 低 4.0 分但**成本只有 38%**（**high effort 已追平 Opus 4.8 的 max 分数，成本约 1/3**）；BrowseComp **最高分 91.2 且 \$2.03/任务**，为 GPT-5.6 Sol 的一半、比 max effort 的 Claude 系列便宜一个数量级；GDPval-AA v2 与 GPT-5.6 Sol 差 50 Elo 内而成本低 13%，**比 Fable 5 便宜 2.6×**。

第三方：Artificial Analysis Intelligence Index v4.1 **57.1（580 个模型中第 4）**；Vals Index **74.7（39 个中第 2）**；**WebDev Arena 1,678 Elo 第 1——首个登顶该榜的开源模型**；Text Arena 第 8；Agent Arena 第 4。

### 6.4 网络安全（写得异常克制）

**Anthropic 与 OpenAI 的前沿模型拒绝 cyber 任务，无法可比评测，故排除出该套件。**

**Tier 1（漏洞发现 + PoC，偏防御）**：跨数十个广泛部署系统发现数百个候选漏洞，**人工复核中约 70% 确认为真**，含**跨 6 个项目的 16 个此前未知漏洞**。两个 Linux 内核案例都指向同一类成因——**上游修复本身引入的缺陷**：一个由不完整修复导致、影响其后所有版本的**可远程触发堆越界写**；一个是 RDMA 子系统的 **Dirty-COW 类漏洞**，早先修复**无意删掉一处权限检查**。

**Tier 2（端到端 exploit）**：36 任务自建套件（16 用户态 + 20 内核），**每题均经人类专家验证可解**，整套约需 **540 专家小时**。K3 解出 **14/36 (38.9%)** vs GLM-5.2 的 8/36——但**14 个中 10 个来自用户态；内核轨道两个模型都有 3/4 未解**。四类复发失败模式：① **已拿到原语却完不成 exploit 链最后一段**；② **缓解措施下策略选择差**；③ **陷入无产出的调试循环**；④ **提交前验证不足**。独立评估（UK AISI + NIST CAISI）结论一致：优于 GLM-5.2（ExploitBench 32% vs 24%），但**端到端完成度落后前沿 cyber-capable 模型，41 个任务上取得任意代码执行 0 次**。作者明确把自评视为**能力下界**。

## 七、Chat template：XTML

三个目标：**可扩展性**（新能力靠向后兼容的消息格式引入而非改模板）、**低对齐税**（极少监督数据即可学会，支持「轻微调直接进 RL」）、**解码友好**。于是把 XML 的尖括号换成保留 token `[open] [sep] [close]`（加 `[end_of_msg]`），**每个结构边界都是显式特殊 token**，消除元素边界的分词歧义。

- **按作用域安排 option message 的位置，目的是护 KV cache**：全局选项（工具声明、effort）放**输入消息之前**（反正改了就要失效缓存）；一次性选项（`tool_choice`、`response_format`）**追加在之后**，使**逐请求变化不动历史 KV cache**；**input option message** 交织在中间，这正是**动态加载工具**的机制——中途用一条 `tool-declare` 宣告，**工具集随即扩张而无需重建上下文**。
- **Channel**（`think` / `response` / `tools`）：**两种生成模式纯由生成前缀选择**而非两套模板；K3 **只支持 preserved thinking**——`think` channel **内容为空也保留在历史中**，使跨轮消息结构一致。
- **工具调用**：`tool` + `index` 给并行调用编号，tool-result 重复同一对。**参数带类型**，字符串以**原始文本**出现 ⇒ **代码这类自由文本是一等公民，而非被转义的 JSON 字符串**；纯 JSON 兜底块**只出现在输入 token，训练时 loss 被 mask**。
- **Reasoning effort 是一条自然语言 option message**，**不改生成前缀、不暴露 token 预算**；schema 预留四级，K3 支持其子集（RL 实训 low/high/max）。
  > 这是所有 option message 的统一实现：都翻译成上下文里的一句自然语言指令而非专用语法。因为预训练模型本就很会遵循这类指令，**新选项几乎不需额外训练——「低对齐税」的直接体现**。

## 八、Case studies

| 案例 | 结果 |
|---|---|
| **GPU kernel 优化** | 同配置 sandbox、每题 24 小时预算。**AttnRes 283.6ms → 114.4ms**，DSA / KDA 运行时**降 55.1% / 73.6%**。相对 FLA Triton 基线：K3 **+59.7%** / Fable 5 +57.1% / GPT-5.5 +30.8% / GPT-5.6 Sol +17.3%。**开发后期一个早期 checkpoint 已在承担团队大部分 kernel 优化工作** |
| **GPU 编译器** | 自建 **MiniTriton**（开源）：tile 级 Python 前端 + warp 级 MLIR 层 + PTX codegen + 含 autograd 与 NCCL 原语的双模 tensor 库。L20 上几何平均**优于 torch eager 与 torch.compile**，从零写的 tensor core matmul **达实测机器上限约 90%**，端到端训 GPT 时**全模型梯度与 torch autograd 差异不超过 torch 自身 fp32 舍入误差（10⁻⁴）** |
| **芯片设计** | 为同架构 nano 模型设计推理芯片原型。**单次 48 小时自主运行**（开源 EDA + Nangate45），**4mm² 内 100MHz 收敛时序**，RTL 仿真解码 **>8,700 tokens/s** |
| **科研编码** | 复现 I–Love–Q 普适关系：交叉验证 **20+ 篇论文**、评估 **300+ 状态方程**、**发现已发表公式中的不一致**、3,000+ 行 Python + 交互式 dashboard——**约 2 小时，对照研究者通常 1–2 周** |
| **知识工作** | 覆盖 **42 年 AI ASIC 行业**的研究网站：120+ 轮迭代，**87 份季报 + 99 份 PDF（11,000+ 页）**，**2,800+ 次搜索 + 1,100+ 次终端查询**。另一例：**20+ 并发 subagent** 分析 GWTC-5 的 **391 个引力波事件** |
| **视频** | 3Blue1Brown 风格的**自身架构**动效讲解；从 **56 个源片段**剪出 teaser（含帧级卡点、多轮修订），熟练剪辑师通常需 1–2 天 |

## 九、可迁移的做法

1. **用算法改动去打开硬件路径**：KDA 把 log-decay 下界化，表面是换了个参数化，实质是让倒数重标定留在 BF16 内，从而**整条按位置对计算的对角路径被稠密 Tensor Core 矩乘取代**。
2. **把位置编码的责任交给局部机制**：全局层一律 NoPE ⇒ **扩上下文不动位置编码参数**。
3. **负载均衡别用带步长的 sign 更新**：QB 直接跳到同一对偶目标的精确坐标最小点，**无学习率类超参**；大规模下**用 per-expert 直方图 + 一次整数 all-reduce 读分位数**，误差被箱宽界住且**对分片方式严格不变**。
4. **冗余资源要证上界而非预设数量**：证明「每 rank 至多 `E/R`」后预留该数，**规划永有可行解、训练永不中断**。
5. **完美负载均衡的第二重收益是静态形状**：消除 per-layer host–device 同步，通信缓冲从 `S·K·R` 降到 `S·K`。
6. **把 recompute / 量化 / offload 统一成「存储策略」**，张量级注解声明、与模型代码解耦，三者可自由组合而非三套互斥实现。
7. **对比预训练不是多模态初始化的必需品**：从零用 NTP 训视觉塔，梯度更稳且评测持平。
8. **RL 不要固定 harness**：否则模型过拟合的是工具 schema 与协议惯例，而非能力。
9. **防 reward hacking 三件套**：**agent 与 verifier 隔离**、**诊断性公开 verifier + held-out 隐藏 verifier 配对**、**有限提交预算下的惩罚式奖励**；性能类奖励还需**专门的 hacking 检测**。
10. **冗长 hacking 用「预算 + 直接判负」而非扣分**（超 `σ·ℓ₀` 自动输掉比较；effort 控制同构）。
11. **QAT 要覆盖 RL 而不只是 SFT**：rollout 与 training 共用同一量化方案，训练—推理失配直接消失。
12. **投机解码直接优化接受率**：KL 代理在容量受限 draft 上不保证最大化 `Σ min(p,q)`。
13. **混合缓存的粒度要反向对齐**：细粒度哈希跑在逐 token 的那一侧，定长大状态的 checkpoint 只落在**哈希端点的稀疏子集**上——否则块大小被迫粗到缓存失效。
14. **投机解码 + 递归状态**：别存每个 draft 位的状态快照，**只缓存投影输入、片上重放**。
15. **成本跨数量级的流量必须按类分预算**：基于「平均请求」的所有容量模型在 2K–1M 跨度下失效。
16. **长上下文能力不来自长度本身**：要合成「只有跨整个窗口取信息才能解」的任务。
17. **调度器之间的公平比较必须各自独立搜超参**。
18. **Agentic RL 的 sandbox 要能 pause/fork**：等推理可占生命周期 98%（pause 直接省掉），fork 让「无副作用评奖」成为原语。

## 十、作者承认的差距

- **研究级推理**（HLE-Full、CritPt）明显落后，被点名为关键方向；
- **Elo 制知识工作**与更难的 computer-use 仍由闭源领先；
- **过程质量弱于结果质量**——内部 AgentBehavior Bench 是差距最明显的一项；
- **Cyber**：面对加固目标**完成完整 exploit 链仍是瓶颈**，独立评估下端到端 ACE 为 0/41；评测被明确视为**能力下界**。

## 小结

把 K3 压成一句话：**第一条 scaling 轴重新被推动之后，剩下的全部工作是让系统层跟得上。**

三条最该记住的：

- **架构改动的真正判据是它打开了哪条硬件路径**：KDA 下界化、QB 分位数均衡、AttnRes 分块，三者都是「算法为系统让路」的样本。
- **长上下文与长时序是同一个问题的两面**：1M 窗口只有配上 partial rollout、外部 KV 池、可 pause/fork 的沙箱，才能真正用于 RL。
- **诚实的报告比榜单更有价值**：过程质量落后 10 分、端到端 exploit 0/41、研究级推理有差距——这些才是下一轮工作的坐标。

延伸阅读：同期的开源路线对照见[《GLM-5：从 Vibe Coding 到 Agentic Engineering》](/blog/glm-5-agentic-engineering/)；并行与显存工程的基础见[《Megatron-LM（六）：MoE 与 EP/CP》](/blog/megatron-lm-06-moe-ep-cp/)；agent 侧 harness 的可配置维度见[《Harness Engineering 综述》](/blog/harness-engineering-survey-2026/)。
