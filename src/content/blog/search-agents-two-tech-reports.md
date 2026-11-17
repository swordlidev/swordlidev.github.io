---
title: "搜索智能体的两条路：把交互做深，还是把数据与评估做诚实"
description: "对读 MiroThinker v1.0 与 Iris 两篇技术报告：前者把交互深度论证成第三条 scaling 轴，后者押注数据可验证性、SFT–RL 交替攀登与把上下文管理当成被测变量。附共识 / 分歧对照与 11 条可落地的工程清单。"
pubDate: 2026-10-24
tags: ["搜索智能体", "MiroThinker", "Iris", "Deep Research", "RL", "SFT", "上下文管理", "技术报告解读", "Agent 评测"]
---

读两篇技术报告：**MiroThinker v1.0**（MiroMind，arXiv:2511.11793）与 **Iris: Climbing to the Search Frontier**（AllSpark，arXiv:2609.04304）。两者都是 ReAct 单智能体 + 合成多跳数据 + SFT→RL，但主张的"胜负手"完全不同：MiroThinker 押注**交互深度作为第三条 scaling 轴**；Iris 押注**数据可验证性、SFT-RL 交替攀登，以及把上下文管理当成被测对象而非默认福利**。

---

## 0. 一分钟速览

| 维度 | MiroThinker v1.0 | Iris |
|---|---|---|
| 规模 | 8B / 30B / 72B（Qwen2.5/Qwen3 初始化） | mini 35B-A3B、pro 397B-A17B（MoE，256K） |
| 核心主张 | 模型规模 × 上下文长度 × **交互深度** | **数据构造 + SFT–RL climbing + 评估协议** |
| 工具 | 搜索、抓取、Linux 沙箱、文件上下传 | 仅 `SEARCH` + `SCRAPE` |
| 上下文管理 | 近期保留（retention K=5）+ 结果截断 | 对照组：无 CM vs `discard-all`（可选 `retry`） |
| 交互预算 | 256K 内 **最多 600 次工具调用** | 深度下限约束 + 回灌**最短**成功轨迹 |
| 训练 | SFT → DPO(+SFT 辅助损失) → GRPO | 双层过滤 SFT ↔ 组相对 RL，交替若干轮 |
| 旗舰成绩 | GAIA 81.9 / HLE 37.7 / BC 47.1 / BC-ZH 55.6 | BC 88.6 / BC-ZH 85.1 / DeepSearchQA 92.9 / HLE 56.4 |

> **注意**：两份报告**不在同一时间轴**。Iris 标注 2026-09，对比的是 MiroThinker-1.7/H1，初始化模型为 Qwen3.5/3.6，榜单里出现 Kimi-K3、GPT-5.6；本文所读 PDF 中的 MiroThinker 是 v1.0。**跨文数字不可直接相减。**

---

## 1. MiroThinker：把"交互次数"当成一条 scaling 轴

### 1.1 核心命题

测试时 scaling（更长 CoT）是**孤立自洽**的，链越长越容易自我强化错误；而交互 scaling 每一步都有**环境反馈**可用于纠错与补证据。于是：交互的**频次与深度**与模型规模、上下文长度并列，构成第三维。

形式化仍是标准 ReAct：$H_t=\{(T_i,A_i,O_i)\}$，思考 $T_t=f_\theta(q,H_t)$ → 动作 $A_t=\pi_\theta(H_t,T_t)$ → 观测 $O_t=\mathrm{Tool}(A_t)$，直至 $A_t=\varnothing$ 后进入 summary 阶段产出答案。

### 1.2 三条支撑性工程

- **工具接口**：沙箱（`create_sandbox` / `run_command` / `run_python_code`）、双向文件传输、`google_search` + `scrape_and_extract_info`。抓取工具内嵌轻量 LLM（如 Qwen3-14B）做**条件化信息抽取**，天然是一层上下文压缩。
- **近期保留（Recency-Based Retention）**：只保留最近 $K$ 条工具返回，**完整保留全部 thought 与 action**：
  $$\widehat{O}_i=O_i\ (i\ge t-K),\quad \varnothing\ \text{otherwise}$$
  关键经验结论：这种"朴素"策略**不掉点**，却把省下的上下文全部兑换成更多交互轮次。评测中 $K=5$。
- **结果截断**：`run_command` / `run_python_code` 超长输出截断并打 `[Result truncated]`。

二者叠加，才有"256K 装 600 轮"（此前开源体系普遍 <100 轮）。

### 1.3 数据：MultiDocQA + 轨迹合成

QA 合成链路：**语料构建**（Wikipedia / Common Crawl，清洗但保留超链接）→ **类目均衡采样 + 沿内链递归扩子图** → **文档整合**（转 markdown，裁掉指向子图外的链接）→ **事实抽取**（只留需跨文档才能验证的陈述）→ **约束混淆**（时空信息泛化：`March 15, 2023 → in the spring of the 2020s`、`Paris → a European capital`；实体改用属性/语境指代）→ **问题生成**（组合多条混淆约束）。

轨迹合成刻意追求多样性：ReAct 单智能体 **+** MiroFlow 多智能体；Function Calling **+** MCP；GPT-OSS / DeepSeek-V3.1 等多教师。再叠加 MuSiQue、HotpotQA、WebShaper、Toucan-1.5M 等开源 QA（只取 QA 对，自行重合成轨迹），并混入通用后训练语料防退化。

### 1.4 三阶段训练

1. **SFT**：工具返回视作 user turn，只对 $(T_t,A_t)$ 计损。强调即使用顶级 LLM 合成，原始轨迹仍充满响应内重复、跨响应重复、非法工具调用，必须**严格过滤 + 数据修复**。
2. **DPO**：偏好仅由**最终答案正确性**决定，**明确拒绝**手工启发式（规划长度、步数、推理结构）——这类约束引入系统性偏差、损害跨域可扩展性。质量控制上要求 chosen 推理连贯且含显式规划，rejected 也必须产出合法答案（否则对比无意义）。目标为 $L_{DPO}+\lambda L^{(+)}_{SFT}$。
3. **GRPO**（全在线，rollout 只更新一次）：
   - 环境支持数千并发 rollout（多源搜索、抓取摘要、Python、Linux VM）+ 低延迟 LLM 判分。
   - **流式 rollout**：轨迹耗时长尾在数百轮交互下被放大；worker 从任务队列流式取题，凑够一个 batch 即停，未完成任务回推队列。
   - 奖励 $R=\alpha_c R_{correct}-\alpha_f R_{format}$。
   - **轨迹清洗双向做**：正确轨迹里剔除连续 >5 次网络异常、同动作冗余重试、超时（环境噪声≠解题能力）；错误轨迹里剔除纯格式失败、动作死循环、未探索即早停（无学习价值）。

### 1.5 结果与"交互扩展"证据

72B：GAIA 81.9（超 MiniMax-M2 6.2）、HLE 37.7（同工具集下超 GPT-5-high 2.5）、BrowseComp 47.1、BrowseComp-ZH 55.6、xbench-DeepSearch 77.8、SEAL-0 51.0。8B/30B 同样是各自尺寸段最优。

最有价值的是 30B 的 **SFT vs RL 轮数分布对比**：RL 后分布显著右移、拖尾更长，准确率同步 +8~10 点（BC 32.2→41.2，BC-ZH 37.6→47.8，HLE 24.1→33.4，GAIA 65.4→73.5）。这是"交互深度可预测地换性能"的直接证据。

### 1.6 作者自陈的四个坑

工具调用质量下降（RL 后调用更频繁但部分冗余）、CoT 过长重复、非英文输入语言混杂、沙箱能力弱（用 `run_python_code` 去读网页/PDF、忘记先建沙箱）。

---

## 2. Iris：难度与可解性必须同时被证明

### 2.1 数据管线：从超链接结构**反向**构造问题

把语料建成有向图 $G=(V,E)$，边为超链接。注意一个实务细节：**reader 服务渲染页面时会剥掉内联 anchor**，因此 Iris 用"结构化语义镜像（RDF 三元组）+ 渲染后 markup"双源合并来恢复真实出链集，最大化 link recall。

1. **种子与子图**：answer-anchored 采样——先定答案实体，再取描述它的页面作种子 $v_0$，沿出链展开为 $G_{sub}$。
2. **实体图蒸馏**：$G_e=(V_e,R_e)=f_{ext}(G_{sub})$，只留位于"通往种子主题的多跳路径上"的实体与带类型关系，得到稠密关系骨架。
3. **多跳出题**：$(q_0,P)=f_{gen}(G_e,y)$，硬约束 $|P|\ge N$ 且 $y\notin q_0$，路径遍历**唯一**导向 $y$。
4. **Anchor 抽象**（关键）：对每个非答案实体施加算子 $A$，要求 $\mathrm{name}(e),\mathrm{alias}(e)\notin A(e)$ 且 $A(e)$ 唯一指向 $e$。于是题目只能靠**推理消歧**，不能靠字符串检索抄近道。
5. **双准则验证**：
   $$c_{diff}=\mathbb{I}[M_{ref}(\tilde q)\ne y],\quad c_{solv}=\mathbb{I}[M_{ref}(\tilde q\mid G_e)=y],\quad D=\{(\tilde q,y)\mid c_{diff}\cdot c_{solv}=1\}$$
   闭卷答不出 ⇒ 够难；给证据能答对 ⇒ 答案唯一且正确。**两条必须同时成立**，这把"难"和"题没出错"分开检验了。

### 2.2 SFT：粗筛看轨迹，细筛看回合

**粗筛（trajectory-level）**

- 正确性：$c_{corr}=\mathbb{I}[\text{success}]\cdot\mathbb{I}[J(q,\hat y_\tau,y^*)=\text{CORRECT}]$
- 退化检测：主检测器是**滑窗 zlib 压缩比** $\rho_{cr}(w)=|w|/|\mathrm{zlib}(w)|$，$O(n)$ 复杂度。重复文本压缩率远高于流畅文本，因此任一窗口触阈即判死循环，**与周期长度和起始位置无关**——比写死的 n-gram 规则鲁棒得多。辅助检测器补：周期性重复行、长单字符串、参数字节相同的连续工具调用。
- 深度：$T_{tool}(\tau)\ge K$，砍掉"一次查询就命中"的浅样本。
- 全消息序列精确去重（保留同题的不同 rollout）。

**细筛（turn-level）**：教师在能力边界附近时，**正确轨迹里仍可能藏坏回合**（冗余搜索、幻觉工具名、推理与实际动作不一致）。难点是"孤立看像浪费的一步，往往是合法探索"。Iris 的做法是**不手写 rubric**：先采样轨迹让 judge 自由批评，再把反复出现的失败模式归纳成显式 rubric 注入最终判题 prompt。judge 对每个 assistant turn 输出 KEEP/MASK，且**单条轨迹最多 mask 10%**。被 mask 的回合**留在上下文里但不计 loss**——既净化信号，又不破坏交互历史。

损失：$L_{SFT}=-\mathbb{E}\sum_t m_t\log\pi_\theta(u_t\mid C_{<t})$，其中 $C_{<t}$ 由 harness 的 replay 算子从 append-only 会话重建，保证**与推理时字节一致**（训推对齐的硬要求）。

### 2.3 RL：请求级部分 rollout + 集群内自给自足

- **请求级 partial rollout**：长尾会话不丢弃、不走全异步。一步凑够已完成轨迹后，**中断**在飞的超采样会话，下一步从其**已提交前缀**续跑。rollout 组织成森林，节点为 message state，缓存 token / loss-mask / logprob / 权重版本差；续跑轨迹因此拼接了不同权重下的前缀，用**截断重要性采样**纠偏。代价是约 2× 超采样余量，收益是同步共置调度下 rollout GPU 不空转。
- **集群内 GenRM + 摘要器**：在训练集群里起若干 FP8 引擎（Qwen3.5-397B-A17B），既当二值奖励模型 $R=\mathbb{I}[\mathrm{GenRM}=A]$（不加格式项——空答案/截断答案抽取为 $\varnothing$ 自然得 0，连 judge 都不必调），又当页面摘要器产出观测 $o_t$。**摘要是 rollout 中唯一的上下文削减机制**，无历史裁剪、无滑动窗口 ⇒ 策略条件的上下文 == 训练的上下文。顺带消除训练环对外部 API 的依赖，一份算力覆盖训练/rollout/奖励。

### 2.4 SFT–RL Climbing：让罕见成功被直接强化

每轮 RL 后挑一小批高质量 rollout 蒸回策略，再继续 RL，一个循环叫一次 **climb**。互补性在于：RL 通过组相对奖励改进"当前策略已能采到"的行为；SFT 则能**直接强化罕见但成功**的轨迹，不必指望它在组内梯度中挣到权重。

选样规则：按组内通过率取 $0<\bar R(q)\le 1/2$（会做但不稳的题），在成功 rollout 中要求 $T_{tool}\ge K_{rft}$，然后取**最短**：

$$\tau^\star_q=\arg\min_{\tau\in S_q}T_{tool}(\tau)$$

深度约束滤掉运气解，最短约束抑制无谓搜索。因为难度带由**当前策略的通过率**定义，它会随策略变强自动右移——**自带课程表，也自带停止信号**（候选池枯竭）。

### 2.5 评估：把 CM 从"实现细节"提到"被测变量"

- pass@1，LLM judge + 各 benchmark 官方 prompt；DeepSearchQA 用 F1，其余用准确率；工具集、上下文上限、最大轮次全程固定。
- **防泄漏三点拦截** `huggingface.co/datasets` 与 `/spaces`：搜索结果中剔除、scrape 时拒绝、tool manager 事后兜底（防模型凭记忆直接给 URL）。
- 拒绝为不同 benchmark 定制不同 CM 策略（"实践意义很小"），主结果统一用 `discard-all`，并**同时报告无 CM 的裸分**。

**主结果（CM 开启）**

| | BrowseComp | BC-ZH | DeepSearchQA | HLE |
|---|---|---|---|---|
| XYZ-Aquila-mini (35B) | 78.8 | 82.9 | **89.5** | 51.1 |
| **Iris-mini (35B)** | **82.2** | **84.8** | 86.9 | **52.3** |
| XYZ-Aquila-pro (397B) | 84.8 | 85.1 | 92.5 | 53.3 |
| **Iris-pro (397B)** | **88.6** | **85.1** | **92.9** | **56.4** |

Iris-mini 在 BrowseComp 上已接近 1T 级模型；Iris-pro 在**标准 ReAct**下即可追平/超过 MiroThinker-H1、Apodex-1.0-H 的**重算力配置**。

### 2.6 CM 消融：最值得抄走的一张表

| Iris-mini (35B) | BC | BC-ZH | DSQA | HLE |
|---|---|---|---|---|
| 无 CM | 64.7 | 72.3 | 81.0 | 43.2 |
| retry | – | 83.0 (+10.7) | 89.1 (+8.1) | 52.0 (+8.8) |
| discard-all | 82.2 (**+17.5**) | 84.8 (+12.5) | 86.9 (+5.9) | 52.3 (+9.1) |
| discard-all + retry | 85.9 (**+21.2**) | 85.1 (+12.8) | 89.9 (+8.9) | 52.4 (+9.2) |

四条结论，含三个反直觉点：

1. **裸分也领先**：无 CM 下 Iris-mini 的 BC/BC-ZH 为 64.7/72.3，高于 FORT-Searcher(55.9/62.1)、OpenSeeker-v2(46.0/58.1)、REDSearcher(42.1/49.8) ⇒ 增益不是推理期包装换来的。
2. **小模型吃 CM 更多**（mini +17.5 vs pro +16.0，且各榜普遍更高）。原因不是预算大小，而是**消耗预算的速度**：小模型解同一组约束要走更多步，更常撞墙，CM 可挽回的也更多。**CM 最值钱的场景 = 模型已学会有效搜索行为，但上下文先耗尽。**
3. **增益强 benchmark 依赖，且与 headroom 无关**：mini 上 BC 最多 +21.2，BC-ZH ≤12.8，HLE ≤9.2，DSQA ≤8.9。HLE 裸分最低（43.2）却涨得少——因为它考专家知识与推理，检索是补充而非主要来源，**它的短板不是上下文问题**。真正决定增益的是"会话多久撞一次上下文墙"。
4. **retry 的账要算算力**：`discard-all + retry` 几乎处处最优（Iris-pro BC 冲到 90.3），但每次 retry 是一整次重搜。Iris 因此只把它当上界探索，主表一律报 `discard-all`——与 DeepSeek-V3.2 "评测 CM 必须计入实际算力成本"的立场一致。

另一个信号：BC-ZH 上三种配置**同为 85.1**（289 题中 246 对），说明剩余误差已不受模型容量约束。附录给了 Q85 的 ground-truth 争议：按剧情，Sansa 第二次正式婚姻是 Ramsay Bolton（Season 5），官方答案却是 Lannister（第一次婚姻 Tyrion）——**榜单标注本身有错**。

### 2.7 结论外溢：search 也许是原子能力

合成的搜索数据与搜索特化模型（作为 on-policy 蒸馏教师）**正向迁移**到未针对的域：通用工具使用（BFCL、τ-bench）与 Cowork（OfficeQA、APEX）。推论：搜索不该被当作垂直特化，而是一种**在信息不完备下行动**的原子能力 ⇒ 搜索数据与搜索教师应贯穿整个训练流程，而非隔离在一个特化阶段。

---

## 3. 对照：共识、分歧与一处真实张力

### 共识（已基本成为标配）

| 做法 | 两篇的具体形态 |
|---|---|
| 从超链接图反向合成多跳题 | MultiDocQA 子图 / 实体图 + $\vert P\vert\ge N$ |
| 破坏字符串捷径 | 约束混淆（时空泛化+指代） / anchor abstraction（形式化约束） |
| ReAct 单智能体做主体 | 均刻意不用 sub-agent 展示裸能力（Iris 还额外声明无 test-time verification） |
| 轻量 LLM 压缩观测 | Qwen3-14B 条件抽取 / 集群内 GenRM 兼摘要器 |
| RL 长尾 rollout 工程 | 任务级流式回队 / **请求级**中断 + 前缀续跑 + 截断重要性采样 |
| 双向清洗 RL 轨迹 | 剔"环境噪声导致的正确"与"格式导致的错误" / zlib 压缩比 + 10% turn mask |
| 屏蔽 HuggingFace 防泄漏 | 工具层禁用 / 三点拦截（含事后兜底） |

### 分歧

| 议题 | MiroThinker | Iris |
|---|---|---|
| 性能来源叙事 | 第三条 scaling 轴：交互深度 | 数据可验证性 + 训练循环 + 诚实评估 |
| CM 形态 | **保留全部 thought/action**，只丢旧观测（K=5） | **全清重启** `discard-all`（可叠 retry 累积"已排除"先验） |
| CM 的定位 | 为交互扩展腾空间的使能手段 | 被测变量，必须给出 w/o 基线 |
| 偏好阶段 | 有 DPO（+SFT 辅助损失） | 无，直接 SFT ↔ RL 交替 |
| 监督粒度 | 轨迹级过滤 + 修复 | 轨迹级 **+ 回合级 mask**（rubric 自数据归纳） |
| 奖励 | 正确性 − 格式惩罚 | 纯二值正确性（空答案天然 0 分） |
| 工具面 | 宽（含沙箱/文件），但自承沙箱用得差 | 窄（仅搜索/抓取），稳 |
| 数据多样性策略 | 多教师 + 多范式（含多智能体）+ FC/MCP | 单强教师 + 严过滤 + RL 自产数据回灌 |

### 一处真实张力

MiroThinker 的证据是：**RL 让轮数分布右移 ⇒ 涨点**，所以要奖励更深更频的交互（上限 600 轮）。Iris 的 climbing 则在成功轨迹中**专挑最短的**回灌 SFT，理由是"抑制无谓搜索"。

两者并不必然矛盾，但指向同一问题的不同侧面：MiroThinker 自己也承认"RL 后调用更频繁，但部分调用贡献边际甚至冗余"。更可能的综合解是：**先把探索上限打开（深度可用），再在成功样本上做效率蒸馏（深度不滥用）**——深度约束保证不是运气，最短约束保证不是蛮力。Iris 的 $T_{tool}\ge K_{rft}$ 与 $\arg\min T_{tool}$ 同时出现，正是这个双边界的写法。

---

## 4. 可直接落地的工程清单

1. **合成题必须过"闭卷失败 + 给证据成功"双闸**，前者保证难度，后者保证题没出错；只做难度筛选会把脏题当硬题训进去。
2. **抽象所有非答案实体**，并显式要求"不含名称与别名、且唯一可识别"——否则 RL 会学会检索字符串而不是推理。
3. **退化检测用滑窗压缩比**（$O(n)$，与循环周期/起点无关），再用周期行、单字符长串、参数字节相同的连续调用补盲区。
4. **坏回合用 mask 而不是丢轨迹**：留在上下文、不计 loss，并设 mask 上限（如 10%）防 judge 过度激进。
5. **训练上下文必须与推理上下文字节一致**：用 replay 算子从 append-only 会话重建历史；若 rollout 里塞了只在训练侧存在的裁剪，训推就已经脱钩。
6. **长尾 rollout 用请求级中断 + 前缀续跑**（配截断重要性采样），比丢弃整条任务省得多；准备约 2× 超采样余量。
7. **把奖励模型与摘要器搬进训练集群**：去掉外部 API 依赖，一份算力覆盖三个角色。
8. **RL 回灌选"0 < 通过率 ≤ 1/2、深度达标、长度最短"**：难度带随策略自动上移，自然形成课程与停机条件。
9. **报告性能必须给 w/o CM 基线**，并说明 CM 策略与其算力成本；只报 CM 后的分数无法区分策略能力与 harness 收益。
10. **用三点拦截防榜单泄漏**（搜索结果过滤、scrape 拒绝、事后兜底），仅在工具层禁用不足以拦住凭记忆直给 URL。
11. **先查 benchmark 标注**再去修模型：BC-ZH 上多配置同分 + ground truth 争议案例，说明部分残余误差不在模型侧。

---

## 5. 结语

MiroThinker 回答的是"**还能往哪个方向加**"——交互深度被证明具备与模型规模、上下文长度同类的 scaling 行为，而让它成立的前提是朴素但有效的上下文管理与能撑住数百轮的 RL 基础设施。

Iris 回答的是"**报出来的分有多少是模型的**"——用双准则验证保证数据不脏，用回合级 mask 保证监督不脏，用请求级 partial rollout 和集群内 GenRM 保证训练闭环不依赖外部，最后用"有/无 CM 双报告"把 harness 收益从策略能力里剥出来。它最有价值的结论恰恰不是榜单第一，而是：**CM 的增益大小由"多久撞一次上下文墙"决定，而非由 headroom 决定**；以及**搜索可能是一种可迁移的原子能力，而非一个垂直方向**。

两条路放在一起看，下一代搜索智能体的配方已经比较清楚：**数据端证明难度与可解性，监督端细到回合，RL 端把长尾与奖励自给自足地工程化，推理端把深度打开但对成功样本做效率蒸馏，评估端把 harness 的贡献明码标价。**

---

**参考**

- MiroMind Team. *MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents via Model, Context, and Interactive Scaling.* arXiv:2511.11793
- AllSpark Team. *Iris: Climbing to the Search Frontier.* arXiv:2609.04304
