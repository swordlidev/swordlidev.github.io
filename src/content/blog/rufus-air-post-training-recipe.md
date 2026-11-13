---
title: "Rufus-Air：一份把话说完的开源后训练配方"
description: "Amazon 把一条完整后训练流水线的数据、奖励、阶段顺序、工程细节和逐阶段结果全部写了出来：SFT + 七个 RL 阶段串行，零新增人工标注，RL 只用 8–32 个 8×H200 节点。本文拆解它的八阶段配方、难度过滤与奖励可靠性排序逻辑、Agentic RL 基础设施，以及 11 条可直接迁移的做法。"
pubDate: 2026-11-01
tags: ["后训练", "强化学习", "RLVR", "Agent", "开源模型", "LLM"]
---

> **本文解决一个问题**：开源基座已经把后训练的门槛降下来了，但「配方」本身仍然写得很薄——多数技术报告读起来像 system card 而不是可复现的 recipe，真正有用的细节恰恰披露最少。
>
> Rufus-Air 的贡献不是模型，而是**把一条完整流水线的数据构成、奖励设计、阶段顺序、工程细节、逐阶段结果全部写下来**。本文按它的八个阶段逐段拆解，重点放在三件可迁移的事：难度过滤怎么当课程用、阶段顺序凭什么这么排、Agentic RL 的基础设施到底要做哪几件事。
>
> 论文：*Rufus-Air: An Open LLM Post-Training Recipe*（Amazon，arXiv:2609.29421v2）。基座：GLM-4.5-Air-Base（MoE，106B 总参 / 12B 激活）。

## 一、这份报告为什么值得读

它满足三个「可复现」条件，这在同类报告里并不常见：

- **全栈基于开源组件**：Slime / Megatron-LM / SGLang / Strands Agents / E2B；
- **算力足够小**：RL 阶段跑在 **8–32 个 8×H200 节点**上，非前沿实验室可负担；
- **零新增人工标注、零自建蒸馏教师**：SFT 数据全部来自 17 个公开数据集，按发布原样使用。

换句话说，它不是「我们花了多少卡训出一个强模型」，而是「这条链路每一环长什么样，你照着搭能跑起来」。

## 二、四条核心结论

| # | 结论 | 含义 |
|---|---|---|
| 1 | **SFT 是能力构建阶段，不是热身** | 强 SFT checkpoint 奠定能力下限，RL 只负责打磨而非从零建立 |
| 2 | **难度过滤把 prompt 锁在「可学习带」** | 丢掉已会的、（多数阶段）丢掉永远不会的，过滤器本身就是自动课程 |
| 3 | **奖励可靠性决定阶段顺序** | 硬验证器在前、软判官在后，缩短可被 hack 的奖励暴露在优化压力下的时间 |
| 4 | **基础设施是配方的一部分** | token-in/token-out、chat template 一致性、长时 sandbox、大 batch + R3，缺一不可 |

第 3 条需要一点补充，因为阶段顺序**并非严格按「奖励硬度」**排：IF RL 用的是 rubric LLM judge，却排在 General / Coding Agent（规则断言 + 执行测试）之前。

> 排的是「**被 hack 的暴露度**」而不是奖励格式。指令遵循离模型已有行为很近，judge 没什么可被钻的空子；而 RLHF 的偏好奖励是开放目标上的 judge，风险真实存在，所以放在最后。

## 三、八阶段流水线

```text
GLM-4.5-Air-Base (106B-A12B)
  → ① SFT              9M 样本 / 27B 训练 token      token 监督
  → ② Reasoning RL     数学 & STEM 推理              规则验证器
  → ③ Coding RL        竞赛编程                      执行验证器
  → ④ IF RL            指令遵循                      混合验证（规则 + LLM judge）
  → ⑤ General Agent    工具使用与规划                规则验证器
  → ⑥ Coding Agent     软件工程                      执行测试
  → ⑦ Search Agent     开放网络检索                  LLM judge
  → ⑧ RLHF             偏好对齐                      奖励模型
  → Rufus-Air
```

每阶段在**上一阶段产出的 checkpoint** 上继续训练，并**只与自己的起点对比**（SFT 例外，对比公开 GLM-4.5-Air）。

### 1. SFT：先把地板垫高

**数据**：9.01M 样本 / 44.5B 原始 token / 66.7M 轮次（23.5M 受监督）。mask 掉 system、user、tool observation 后，**27.0B token（60.8%）进入 loss**。

| 类别 | 样本占比 | 训练 token 占比 |
|---|---|---|
| General Agent | 39.6% | 14.2% |
| General Chat | 17.7% | 12.2% |
| STEM | 15.1% | 8.8% |
| Math | 12.7% | **29.4%** |
| Code | 8.8% | 15.7% |
| Coding Agent | 6.1% | **19.7%** |

两个分布差异巨大：Math + Coding Agent 占 18.8% 样本却贡献 49.1% 训练 token。

> **样本数描述来源覆盖度，loss token 才描述真实监督量**——两个视角都必须报告。

**预处理三步**：格式归一化（role-aware 多轮 schema + per-message loss mask）→ 交错思考监督（一条完整 think–action–observation 链为一个训练目标，历史轮的 assistant rationale 剥离并 mask）→ 校验与长度过滤（`<think>` 配平、role 转移合法、按生产 chat template 计长，上限 120K）。

**去污染**：词级 8-gram 重叠（每个 benchmark item 采 ~20 条 8 词短语，命中 ≥50% 即丢弃），链式跑 10 个 benchmark，删 3,529 条；数学/知识类额外加穷举 8-gram + Llama-NV-Embed 稠密检索，确认 68 对污染（58 条唯一样本，以 AIME 24 原文/轻改写为主），连带边界样本共删 658 条，**合计移除 4,187 条**。同一套流程对每个 RL prompt 集重跑。

**训练**：64 节点 × 8 H200 = 512 GPU，3 epoch，batch 4096 序列，AdamW(0.9/0.95)，wd 0.1，lr 100 步 warmup 到 5e-5 后 cosine 到 5e-6，128K packing。单 epoch ≈ 2380 步 / 105 小时，三轮 ≈ 13 天 / **832 node-days，是全流程最大单项开销**。

**关键观察**：loss 从 0.84 降到 0.34，但**held-out 分数在第一个 epoch 内就平掉了**——后续 epoch 只买到更低的 loss，买不到更好的评测。因此取**平台期内的 checkpoint 3799**（而非最终步）进入 RL。

| | IFEval | IFBench | GPQA | AIME 25 | AIME 26 |
|---|---|---|---|---|---|
| GLM-4.5-Air（含 RL 的官方版） | 83.00 | 33.60 | 73.90 | 84.20 | 86.50 |
| Rufus-Air SFT (3799)，**零 RL** | 88.33 | 57.75 | 68.18 | 90.83 | 90.00 |
| Δ | **+5.33** | **+24.15** | −5.72 | **+6.63** | **+3.50** |

纯 SFT 已在单轮指令遵循和两届 AIME 上超过官方后训练版，只在 GPQA 落后——**而 GPQA 和多轮指令遵循正是后续 RL 阶段的靶子**。

### 2. Reasoning RL：收益由数据策展决定，不由目标函数决定

**数据**：121,161 条可验证单轮 prompt / 25.07M prompt token。

| 任务族 | 来源 | 验证器 | Prompt 数 | Token 占比 |
|---|---|---|---|---|
| Math | HF 爬取 + 合成 | Math-Verify 规范化匹配 | 57,736 (47.7%) | 25.3% |
| Science | 多学科爬取 | 模糊字符串匹配 | 43,199 (35.7%) | 19.2% |
| Puzzles | Enigmata / ReasoningGym | 生成的 Python checker | 20,226 (16.7%) | **55.5%** |

**两道入口过滤器是本阶段最重要的选择**：

- **正确性过滤**：强教师（GPT-OSS-120B）拿不到正奖励的 prompt 直接剔除，排除题目错误/实际不可解，并证明保留的题至少存在一条有效解路径；
- **可学习性过滤**：通过率 >0.8 丢（太易），0 成功丢（当前不可学）。

**在线版本**（DAPO 式动态采样）：4× 超采样（1,024 prompt），只保留组均值 $\bar{r} \in (0, 0.8]$。窗口固定而策略在进步，**死 prompt 不断重新进入有效带，难度前沿自己往前推，无需人工分段**。

**训练**：GSPO（序列级重要性比与裁剪，在长 MoE rollout 上更稳，配合 R3 吸收 FP8 rollout / BF16 training 的失配）。ε=1e-3、ε_high=2e-3，KL 与熵系数为 0，lr 1e-6 常数。256 prompt × n=16，30K 响应预算，两次优化步（global batch 2048）。8 节点，FP8 SGLang 引擎与 BF16 训练共卡、引擎 offload 交替。

**长度惩罚**：相对组内最短正确 rollout 的线性惩罚。既是行为偏好（RL 推理模型普遍无谓变长），也是训练经济学（冗长吃 rollout 吞吐、抬高截断率——截断率从 ~3% 涨到 40 个 rollout 后的 15–20%）。

**结果**：GPQA 68.2 → 73.5（**+5.3**），AIME 两届 −2.6/−2.8。作者明确把 AIME 当护栏而非目标：每届仅 30 题，<3 分不足一题（同条件重跑 AIME 25 在 SFT checkpoint 上波动即达 4.6 分）。

### 3. Coding RL：把截断去掉，奖励就继续涨

**数据**：29,405 题（全 Python），两种验证模态——functional（assert 单测）与 stdin/stdout（输入输出对）。

| 来源 | 模态 | 题数 | 占比 |
|---|---|---|---|
| EvolveCoder（合成，对抗演化测试用例） | functional | 14,743 | 50.1% |
| Nemotron（真实竞赛，Codeforces 为主） | stdin/stdout | 9,393 | 31.9% |
| Dolci | 混合 | 3,109 | 10.6% |
| ADR（技术标签算法题） | stdin/stdout | 2,160 | 7.3% |

去泄漏对 LiveCodeBench v6（1,055 题）做 400 字符前缀 hash 精确匹配 + 字符级 60-gram 重叠：**零命中**。

**奖励**：取响应里最后一个 `python` 围栏代码块在沙箱执行，二值 0/1。每题测试用 **SHA-256 确定性种子**下采样至 ≤50 条，保证同题跨 rollout 测试集一致，奖励不随采样随机性漂移。格式合规在 iteration 0 已达 98–100%，故不加格式惩罚。

**本阶段最有用的工程发现**：64K 预算下 11–28% 样本触顶被 loss-mask，reward 卡在 0.34；**第 23 步把预算扩到 128K**，截断率掉到 <0.1%，reward 继续爬到 0.42，LCB v6 在第 33 步见顶 75.9。

| | LCB v6 Pass@1 | Pass@8 |
|---|---|---|
| Reasoning RL | 68.6 | 85.1 |
| + Coding RL | 75.9 | 87.4 |
| Δ | **+7.3** | +2.3 |

### 4. IF RL：rubric 只写「必须满足」，回答反而变短

沿两个轴构造数据：

- **多约束**（单轮）：14K 条，人工单约束池 → Qwen3-235B-A22B 自动扩写为 2–6 个原子约束的复合指令，并为每条约束生成 Python 验证函数。难度过滤（>80% 通过丢）+ 质量过滤（教师 4 次做不出就丢）。
- **多轮**（13K 条，平均 6.1 轮 / 2.7 条 rubric）：教师 LLM 扮演用户**对抗式生成**——分层加指令、改口推翻先前要求、插入干扰项；assistant 角色由多个开源模型轮换以避免风格偏置。部分带 system prompt，用作**指令层级**训练（用户挑战 system prompt 时必须继续遵守）。三道过滤：sanity（去过长对话、下采样「应当拒绝」样本以防过度拒答）→ 质量 → 难度。

**奖励**：rubric 二值——**全部 rubric 满足才给正奖励**。可程序化的用 Python 判定，其余交 LLM judge。**刻意排除「可选 rubric」**。

**训练**：GRPO，256 prompt × 16，16K 预算，lr 1.5e-6，90 步，取末步。

| | IFEval | IFBench | Multi-challenge | AdvancedIF | GPQA | AIME 25 |
|---|---|---|---|---|---|---|
| Coding RL | 90.5 | 63.8 | 31.1 | 45.5 | 67.9 | 87.6 |
| + IF RL | 94.5 | 77.8 | 55.8 | 62.9 | 71.1 | 86.4 |
| Δ | +4.0 | **+14.0** | **+24.7** | **+17.4** | +3.2 | −1.2 |

三条值得记的观察：

1. **IF RL 不以推理能力为代价**（GPQA +3.2，AIME −1.2 在噪声内）；
2. **先推理是 IF 的天然课程**：在 Reasoning RL checkpoint 上训比直接在 SFT 上训收益更大，AdvancedIF 尤其明显——满足复杂指令需要先对约束做推理；
3. **rubric 限定为必要条件会让最终回答变短**（5.4K → 3.6K token）。反向实验：给可选内容也挂 rubric（每条样本约十条），策略立刻产生显著长度偏置，输出冗长以求覆盖尽可能多的 rubric。

### 5. General Agent：先学通用工具使用

**数据**：AgentWorldModel（AWM）10K 个单 server MCP 任务 / 1K 合成环境；经正确性 + 难度双过滤后剩 **~2K**。每个任务绑定一个环境：场景专属数据库 + 读写该库的 MCP server（AWM 中约 35 个工具/环境）。工具调用改变环境状态，verifier 在 episode 结束后检查状态。

**奖励**：只用 AWM 的代码验证函数，终局二值。**主动砍掉 AWM 的两项奖励**——LLM judge（其职能由正确性过滤器承担）和格式奖励（不再用 `list_tools`/`call_tool` 两个 meta-tool，而是把所有工具 schema 直接注册进 system prompt，无需发现步骤可强制）。

**训练**：GRPO + DAPO 动态采样 + R3。每步 3,072 条 rollout，对有状态 MCP server 的 rollout 是 CPU 密集型（server I/O、DB 变更、MCP 传输），必须走分布式 rollout。

| | MCP-Atlas | Tau2-Retail | AIME 26 | IFEval | GPQA | LCB |
|---|---|---|---|---|---|---|
| IF RL | 34.95 | 74.00 | 87.08 | 94.73 | 73.36 | 71.21 |
| + General Agent | 42.75 | 83.80 | 87.60 | 96.16 | 76.39 | 73.36 |
| Δ | **+7.80** | **+9.80** | +0.52 | +1.43 | +3.03 | +2.15 |

**合成 MCP 环境能提供有用的通用工具编排先验**——这是本阶段的核心主张。

### 6. Coding Agent：跑到算力耗尽，不是跑到收敛

**数据**：Endless-Terminal（程序化生成容器化终端任务）、SETA-Env（4,500+ Harbor 格式环境，源自 Ask Ubuntu / Stack Overflow，难度可控演化）、Scale-SWE（预构建 Docker + fail-to-pass 单测）。结构过滤 + 可学习性过滤后 **~4K** 任务。

**环境**：统一 Harbor 格式（`instruction.md` + Dockerfile + `tests/test.sh`）。工具面**只有一个 `execute_command`**——读文件、改代码、构建、跑测试全部表达为 shell 命令。每个任务镜像**预先烘焙成固定规格的 sandbox template**，rollout 时直接启动已固化模板，无在线构建，保证 RL 级并发下的供给速度。

**工程要点**：高并发下部分 rollout 因基础设施原因失败（sandbox 启动失败、verifier 超时、连接断开）。这类样本被标为 **aborted**，与「测试失败 reward=0」区分，并**在计算 advantage 前从采样组中移除**，避免污染组基线。

| | Terminal-Bench 2.1 | SWE-bench Verified | AIME 26 | IFEval | LCB |
|---|---|---|---|---|---|
| General Agent | 38.76 | 65.60 | 87.60 | 96.16 | 73.36 |
| + Coding Agent | 40.17 | 67.80 | 87.40 | 95.68 | 75.64 |
| Δ | +1.41 | +2.20 | −0.20 | −0.48 | +2.28 |

**作者自陈的 caveat**：本阶段按可用算力而非收敛训练——每个奖励都要启动沙箱、在几十到几百次工具调用后跑完整测试套件，单位梯度信号成本远高于 Reasoning / Coding RL。这些数字说明「环境—奖励—rollout 通路端到端可用且能推动目标 benchmark」，而非该阶段的收敛值。SWE-bench 的增益到最终 checkpoint 没保住（回到 65.6），Terminal-Bench 2.1 反而继续涨到 42.7。

### 7. Search Agent：奖励噪声如何改写优化器选择

**数据**：从 MiroVerse 抽可验证 QA 对，剔除需视觉的、代理策略几乎不解的子集 → 36,614 候选。因 rollout 昂贵，两道激进过滤：

1. **关工具** rollout 8 次，任何一次答对就剔除——这些题靠参数化知识就能答，训练它们奖励的是「绕过工具」而非研究行为（去掉约 1/4）；
2. **开工具** rollout 8 次、用训练同款 judge 打分，按正确数 $c$ 保留 $0 < c < 4$。两端极值占比极高（15.5% 全错、31.4% 全对，组相对优势恒为 0 无梯度）。最终 **~2.2K** 题。

**环境**：Web Search（排序 URL + snippet）、Web Scrape（HTML 正文抽取；超长时由摘要器压成结构化证据，人名/日期/数字逐字保留）、Python（无网络无文件系统的沙箱解释器）。训练 100 次工具调用上限 / 评测 200 次，整条轨迹 128K token，最终答案必须 `\boxed{}`。

**奖励设计（防捷径是重点）**：

- LLM judge 连续 [0,1] 打分，列表/描述类答案给部分分；
- **只有显式给出最终答案的轨迹才送审**，且**正奖励要求至少 2 次成功工具调用**；
- 按畸形工具调用比例**最多缩放到 0.5**；
- 耗尽预算未给答案 **−0.1**；
- **没有任何无条件正项**——早期的「产出 boxed 答案」格式奖励被移除，因为策略学会了用无工具瞎猜来领取它；
- 训练 judge 用 Qwen3-32B（每步要打 512 条轨迹），评测用更强的独立 judge。

**优化器的关键改动**是 **mean-only advantage**（Dr. GRPO 式，G=16），但**保留 GRPO 的按序列长度归一化**（30–100K token 轨迹上，常数归一化会让最长轨迹主导 batch 梯度）。

> 去掉标准差归一化的理由值得记：除以组标准差会**抹掉奖励差距的大小**。$G$ 个 rollout 中 $k$ 个成功、与失败相隔 gap $\Delta$，归一化后成功样本的优势为 $\sqrt{(G-k)(G-1)/(kG)}$，**与 $\Delta$ 无关**。$G=16$ 时，「唯一一次满分成功」和「唯一一次 0.1 部分分成功」都拿 +3.75（约为均衡组中成功样本优势的 3.9 倍）。而可学习性带恰恰大量保留 1/8 成功率的题——归一化会把打分噪声放大到真实解题的量级。配对消融中 mean-only 在每个匹配评测点都领先（55.8→64.8 vs 49.7→58.7）。

另：保留对初始化的小 KL 锚（系数 1e-3）——没有它策略熵从 0.72 漂到 1.12；有它熵随验证分数一起缓升到 0.94。

**训练动态里的两段现象**：工具迭代数先降后升。

- **降**：是**全面压缩而非回避工具**——赢的轨迹（reward ≥0.9）中位数少 4 次调用 / 7K token，输的轨迹少 7 次 / 16K token，而 held-out 通过率持续上升；
- **升**：是**新出现的「有效长搜索」尾部**——赢家中位数仍约 25 次调用，但赢家 90 分位从 40 升到 54.5，>45 次调用的轨迹从 49 条增到 133 条且其中赢家占比从 24% 升到 43%，同时赢家文本中位数从 42K 缩到 33K token。**长搜索从失败特征变成了可行策略。**

| | BrowseComp | Seal-0 | HLE-Verified |
|---|---|---|---|
| Coding Agent | 34.2 | 48.6 | 47.7 |
| + Search Agent | 37.2 | 54.0 | 51.1 |
| Δ | +3.0 | +5.4 | +3.4 |

评测用的检索后端是**策略训练中从未用过的**（Serper + Jina），增益因此指向「通用搜索行为变好」而非「吃住了训练后端」。

### 8. RLHF：放在最后，因为它最容易被 hack

此处 RLHF 指**对学习到的奖励模型做 on-policy RL**，而非直接用人类标注——与全程不新增标注的立场一致。用 on-policy 而非 DPO，是为了让奖励作用在策略自己的 rollout 上。奖励模型用开源的 Skywork-Reward-V2-Qwen3-8B。

**三个候选 prompt 集与三条负面发现**：

| 数据集 | 原始 / 有效 | 平均 chosen 分 | 平均 rejected 分 | chosen tokens | rejected tokens |
|---|---|---|---|---|---|
| Arena Human Preference | 84,402 / 53,502 | 17.06 | 12.08 | 1055.1 | 623.7 |
| HelpSteer3 | 38,459 / 29,506 | 9.60 | 2.95 | 439.1 | 373.2 |
| HH-RLHF | 112,052 / 75,815 | −4.23 | −8.41 | 78.8 | 72.9 |

1. 未过滤的 HH-RLHF 会明显损伤其他能力（编码与指令遵循最明显）→ 催生有效性过滤；
2. HelpSteer3 任务更多样，但其 prompt 让奖励模型难以稳定打分：reward 涨得慢，科学推理与指令遵循反而退化；
3. **指令遵循类和 agentic prompt 不适合本阶段**——奖励模型无法可靠评估复杂约束满足度。

最终用过滤后的 **75,815 条 HH-RLHF prompt**（只用 prompt，偏好标签仅用于上表的离线分析），GRPO + **线性长度惩罚**（只在原始 reward 为正时削减超长回答，负 reward 不变，因为 Bradley–Terry 分数无界且有噪声）。取 iteration 29。

| | Arena-Hard v2 (HP) | Arena-Hard v2 (CW) |
|---|---|---|
| Search Agent | 83.06 | 38.56 |
| + RLHF | 89.05 | 52.97 |
| Δ | **+5.99** | **+14.41** |

监控 benchmark 在本阶段移动不超过 1.3 分。

## 四、基础设施：报告里最少被写、实际最关键的部分

两个自研组件，都建在开源之上：**Rufus-Slime**（Slime 的改造版，Megatron-LM 后端 + SGLang rollout 引擎，跑 SFT 和全部 RL）、**Rufus-Gym**（agent 环境与编排层，Strands Agents + SGLang）。

**通用 RL 的四项稳定性选择**：

- **大 rollout batch**：组相对方法的 advantage 是用同 prompt 的其他回答估的，每步采 4,096–8,192 条响应（Reasoning/IF RL：256×16；Coding RL 扩预算前：128×64）。
- **Rollout Routing Replay (R3)**：MoE 上即使精度相同，rollout 与 training 也可能把同一 token 路由到不同专家，叠加在精度失配之上造成 log-prob gap。R3 缓存 rollout 期的路由决策并在训练前向中重放，消除这一分量。
- **FP8 rollout + BF16 训练**：前 3 层 BF16、其余 FP8 做 rollout，参数更新全程 BF16。
- **截断率监控**：截断样本被 mask 掉 loss，截断率上升 = 每个 batch 贡献梯度的比例缩水。Coding RL 正是据此把预算从 64K 扩到 128K。

**Agentic RL 的四项要点**：

1. **环境接口**：Environment 是中心抽象——吃一个 Task（prompt + context），吐一个 RolloutResult（**token 轨迹 + 奖励 + 带类型的终止信号**）。不论底层是代码执行、网页搜索还是 MCP server，契约不变。Agent Loop 是 ReAct 式循环，建在 Strands Agents 上，用 hook 订阅生命周期事件（如 `BeforeModelCallEvent`）保持环境无关，而不是 fork 整个 loop。

2. **Token 保真（token-in/token-out）**：on-policy RL 需要**采样时的那批 token 本身**。环境在每轮产生时直接追加 token，而不是把对话重渲染成文本再重新分词。SGLang 后端直接收 prompt token id、返回生成 token id，连同 per-token log prob、loss mask、路由专家分配原样传给 trainer。这一条堵死三个失败模式：**retokenization drift**、**对增长历史反复套 chat template 导致早期轮次被悄悄改写**、**对轨迹外被修补过的 tool call 做训练**。文本 transcript 仅供检查，**token 轨迹才是权威训练产物**。

3. **Sandbox 后端**：每条活跃轨迹独占一个沙箱，**单位是 Firecracker microVM 而非容器**——guest kernel 与进程状态都在 VM 内，可对完全初始化好的环境做快照并按轨迹恢复，同时把策略写出的代码挡在硬件虚拟化之后，而不是挡在与宿主及同机轨迹共享的内核之后。自托管开源 E2B runtime，跑在裸金属 EC2（Firecracker 需要 KVM），控制面与不可信负载分机部署。另两个细节：**agent 命令不幂等**，只有在能证明「执行从未开始」的失败上才重试，一旦可能已开始就上报可恢复错误；**工具结果按固定 token 预算截断**后才回灌对话，避免冗长构建/测试日志撑爆上下文。

4. **分布式 rollout**：Ray 把环境 actor 铺到多节点，用跨节点通信换真并行——CPU 密集的工具执行、负载传输、奖励计算跑在专用进程里，减少 CPU 争用。

**全流程训练配置一览**：

| 阶段 | LR | Batch | 每 prompt 采样 | 优化步/rollout | 响应预算 | 节点 |
|---|---|---|---|---|---|---|
| SFT | 5e-5→5e-6 | 4096 seq | – | – | 128K ctx | 64 |
| Reasoning RL | 1e-6 | 256 prompt | 16 | 2 | 30K | 8 |
| Coding RL | 1e-6 | 128→64 prompt | 64 | 8 | 64K→128K | 32 |
| IF RL | 1.5e-6 | 256 prompt | 16 | 1 | 16K | 8 |
| General Agent | 1e-6 | 48 prompt | 64 | 1 | 32K | 16 |
| Coding Agent | 1e-6 | 32 prompt | 32 | 1 | 60K | 16 |
| Search Agent | 3e-6 | 32 prompt | 16 | 2 | 32K/轮，128K/轨迹 | 16 |
| RLHF | 1e-6 | 768 prompt | 5 | 2 | 30K | 8 (+1 RM) |

## 五、结果

### 与同量级开源模型对比

前四列为同一 harness 自测：

| | Rufus-Air | GLM-4.5-Air | INTELLECT-3 | Nemotron-3-Super |
|---|---|---|---|---|
| 总参 / 激活 | 106B / 12B | 106B / 12B | 106B / 12B | 120B / 12B |
| IFBench | **76.9** | 33.6 | 29.3 | 68.6 |
| IFEval | **95.4** | 83.0 | 79.5 | 91.3 |
| Multi-challenge | **65.8** | 36.0 | 36.5 | 50.7 |
| Arena-Hard v2 (HP) | **89.1** | 55.0 | 54.6 | 74.5 |
| Arena-Hard v2 (CW) | 53.0 | 60.3 | 45.4 | **61.5** |
| AIME 25 | 88.3 | 84.2 | 83.8 | **89.1** |
| AIME 26 | **86.7** | 86.5 | 82.6 | 86.0 |
| LiveCodeBench v6 | **76.4** | 59.6 | 70.1 | 71.6 |
| GPQA | 75.6 | 73.9 | 72.4 | **75.9** |
| Tau2-Retail / Airline / Telecom | 85.3 / 84.0 / **93.0** | 80.7 / 77.0 / 32.7 | 75.7 / 76.0 / 40.8 | **86.4 / 86.0** / 72.4 |
| MCP-Atlas | **45.3** | 35.9 | 26.0 | 40.4 |
| BrowseComp | **37.1** | 22.7 | 12.3 | 35.2 |
| Seal-0 | **51.4** | 33.3 | 43.2 | 44.1 |
| HLE-Verified Gold | **51.1** | 20.2 | 27.2 | 46.9 |
| Terminal-Bench 2.1 | **42.7** | 24.7 | 25.8 | **42.7** |
| SWE-bench Verified | **65.6** | 50.6 | 46.0 | 59.2 |

- 对**同基座**的 GLM-4.5-Air 和 INTELLECT-3：**除 Arena-Hard v2 Creative Writing 外全线领先**。
- 对 Nemotron-3-Super：在指令遵循、LCB、Tau2-Telecom、MCP-Atlas、三项 Search、SWE-bench 上领先；AIME / GPQA / FrontierScience / Tau2-Retail / TB2.1 在两分之内；CW 和 Tau2-Airline 落后。
- **差距集中在指令遵循与 agentic 行为上——也正是配方花了最多阶段的地方**；数学与科学上几乎持平（只有 Reasoning RL 一个阶段在做）。配方在哪儿花训练信号，就在哪儿移动。

### 逐阶段累积

| Checkpoint | GPQA | AIME25 | AIME26 | LCBv6 | IFEval | IFBench | Multi-ch | MCP-A | Tau2-Re | TB2.1 | SWE-V | BrowseC | Seal-0 | HLE-V | AH(HP) | AH(CW) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLM-4.5-Air | 73.9 | 84.2 | 86.5 | 59.6 | 83.0 | 33.6 | 36.0 | 35.9 | 80.7 | 24.7 | 50.6 | 22.7 | 33.3 | 20.2 | 55.0 | 60.3 |
| SFT (3799) | 68.2 | 90.8 | 90.0 | | | | | | | | | | | | | |
| + Reasoning RL | **73.5** | 88.0 | 87.4 | | | | | | | | | | | | | |
| + Coding RL | | | | **75.9** | | | | | | | | | | | | |
| + IF RL | | | | | **94.5** | **77.8** | **55.8** | | | | | | | | | |
| + General Agent | | | | | | | | **42.8** | **83.8** | | | | | | | |
| + Coding Agent | | | | | | | | | | **40.2** | **67.8** | | | | | |
| + Search Agent | | | | | | | | | | | | **37.2** | **54.0** | **51.1** | | |
| + RLHF | | | | | | | | | | | | | | | **89.1** | **53.0** |
| **Rufus-Air** | 75.6 | 88.3 | 86.7 | 76.4 | 95.4 | 76.9 | 65.8 | 45.3 | 85.3 | 42.7 | 65.6 | 37.1 | 51.4 | 51.1 | 89.1 | 53.0 |

**每个阶段都推动了自己的目标。** 单阶段最大增幅：Multi-challenge 在 IF RL（+24.7）、Arena-Hard CW 在 RLHF（+14.4）、Tau2-Retail 在 General Agent（+9.8）；最小是算力耗尽的 Coding Agent（+1.4 / +2.2）。

**阶段顺序确实保住了早期收益**：Reasoning RL 之后的两个阶段把 GPQA 和 AIME 26 留在一分以内（73.5→73.36，87.4→87.08），同时叠加了各自的目标技能。

## 六、容易被忽略但很实用的几点

**Agentic RL 的 GPU 之外成本**：

- Coding Agent 每条轨迹全程持有沙箱并在其中跑测试套件，沙箱服务按**约一万个并发沙箱**规模配置，**维护成本约 1 万美元/月**；
- Search Agent 按外部调用付费：约 100 次调用/题，BrowseComp 1,266 题单遍 ≈ **10⁵ 次搜索与抽取调用，按 list price 数百美元**（Serper 2 美元/1K 次，Jina 0.05 美元/1M token）；训练时改用更便宜的内部检索后端，且每条训练 rollout 还要过一次 LLM judge。
- **结论：规划 8–32 节点的 RL 时，这两个阶段要单独立预算，不能按「每阶段均值」摊。**

**Search Agent 的上下文管理**（评测侧，但结论通用）：比较了 4 个家族 12 种策略，**「外部记忆 + scratchpad」最好**——每轮研究后把关键发现、假设、待解问题写入 scratchpad，活跃上下文只保留当前步。**关掉它 BrowseComp 掉 10.7 分，是本阶段所有单项变动里影响最大的一个。**

**深度不是越多越好**（BrowseComp）：25 次迭代内完成的轨迹准确率 88%，25–50 次为 64%，>50 次低于 30%；错误答案的轨迹深度是正确答案的两倍多（92 vs 43 次）。**超过约 50 次迭代，更多深度是「卡住」的信号而非进展的信号。**

**评测协议本身就是变量**：Tau2-Bench 的用户模拟器**单独一项就能移动分数达 23.7 分**（GLM-4.5-Air retail：Sonnet 4 下 57.0，Sonnet 5 下 80.7）。跨论文比较必须对齐模拟器设置。同理，同一 checkpoint 在不同来源下分差可达数分（Qwen3.5-122B 的 AIME 25 在三处分别是 92.92 / 90.36 / 89.48）。

## 七、作者自己划出的边界

- **阶段顺序是「一个跑通的顺序」，不是被证明最优的顺序**——它受早期实验和算力预算塑形；也没有尝试「训练领域专家再用多教师 on-policy 蒸馏合并」这条替代路线。
- **Coding Agent 没训到收敛**，只训到算力用完。
- **没有一套 benchmark 贯穿所有阶段**，因此某个 benchmark 在两个报告它的阶段之间的变化**无法归因到任何单一阶段**（例如 TB2.1 40.2→42.7、SWE-V 67.8→65.6、Seal-0 54.0→51.4 跨越了中间所有阶段）。
- 去污染用的是 n-gram + embedding 过滤，**不能排除全部泄漏**，尤其是改写或语义等价的题目。

## 八、可直接迁移的做法清单

1. **SFT 按「loss token 占比」而非样本数来设计配比**，并同时报告两个视角。
2. **SFT 取平台期 checkpoint 而非最终步**——held-out 分数常在第一个 epoch 内就饱和。
3. **每个 RLVR 阶段配两道过滤器**：强教师正确性过滤（证明题目可解）+ 可学习性过滤（通过率 >0.8 丢；视阶段决定是否丢 0 通过率）。有条件的加在线版（组均奖励落在 (0, 0.8] 才保留）。
4. **测试用例做确定性种子下采样**，保证同题跨 rollout 奖励不随机漂移。
5. **监控截断率**：截断样本被 mask = 有效 batch 缩水。涨上去就扩响应预算。
6. **rubric 只写必要条件**，严禁给「可选内容」打分——否则必然诱发长度偏置。
7. **奖励里不要有无条件正项**，并对「绕过工具的捷径」显式设门（如正奖励要求 ≥2 次成功工具调用）。
8. **奖励噪声大时去掉组标准差归一化**（保留长度归一化），避免把打分噪声放大到真实解题的量级。
9. **多轮 on-policy RL 必须 token-in/token-out**，杜绝 retokenization drift 与 chat template 反复套用。
10. **把基础设施故障与任务失败区分开**（aborted vs reward=0），计算 advantage 前剔除，别污染组基线。
11. **按「奖励被 hack 的暴露度」而非奖励格式来排阶段顺序**，把开放目标上的 judge 放最后。

## 小结

把这份报告压成一句话：**后训练的收益主要由数据策展和基础设施决定，目标函数只是其中一个可调项。**

三条最该记住的：

- **SFT 决定上限的位置，RL 决定上限的形状**。零 RL 的 SFT checkpoint 已经能在 IFBench 上打出 +24.15，后面七个 RL 阶段各自只负责把自己那一格推上去。
- **过滤器就是课程表**。正确性过滤保证题目可解，可学习性过滤保证题目有梯度，在线动态采样让难度前沿自己往前走——三者加起来比任何手工分段的 curriculum 都省事。
- **奖励的可靠性排序 > 奖励的格式**。谁容易被 hack，谁就往后排；越靠后，被优化压力撬动的时间越短。

延伸阅读：RL 框架实现侧可对照[《slime 学习笔记（三）：算法设计》](/blog/slime-03-algorithm-design/)与[《verl 源码剖析（四）：PPO/GRPO 核心》](/blog/verl-04-ppo-grpo-core/)；rubric 作为奖励的设计细节见[《Rubrics 用于 LLM 评测与 RL》](/blog/rubrics-for-llm-eval-and-rl/)；Search Agent 的工程落地见[《slime 学习笔记（四）》](/blog/slime-04-search-agent-rl/)。
