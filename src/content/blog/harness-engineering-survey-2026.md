---
title: "Harness Engineering：Agent 的「缰绳」工程"
description: "把 LLM 当 policy，Harness 就是它外面的 environment 与 rollout 基础设施。本文按六层骨架给出 Harness Engineering 的核心结论，并补上长时运行、主动调度、评测三件横切的事。"
pubDate: 2026-06-22
updatedDate: 2026-11-06
tags: ["Harness Engineering", "Agent", "Coding Agent", "Context Engineering", "Memory", "Guardrails", "Agent 评测", "LLM 工程"]
---

> **一句话**：把 LLM 当成 policy，Harness 就是包在它外面的 environment + rollout 基础设施——观测怎么给、动作空间怎么定、什么时候终止、由谁判定成功。权重决定 policy 的能力上限，harness 决定这个上限能被稳定兑现多少。
>
> 本文面向算法研究员，只讲**结论与机制**，实现细节和坑位清单尽量略去。全文走一遍六层（上下文 → 控制循环 → 工具与权限 → 子 Agent → 记忆 → 自动进化），再补三件横切的事（长时运行、主动调度、评测）。
>
> 三条贯穿全文的直觉：**① 先怀疑管道，再怀疑模型**——Agent 出错多半是上下文装错了或工具 schema 写糊了；**② 别过早工程化**——从最小循环起步，按需加记忆、技能、权限；**③ 上下文窗口不是无限的**——没组装进 prompt 的信息，对模型来说不存在。
>
> 说明：时间截至 2026 年 11 月。论文和厂商数字均为作者 / 厂商自报，代码为示意实现。

## 一、什么是 Harness

**Agent = Model + Harness**。Harness 指包裹在模型外部的一切运行时基础设施：上下文管理、控制循环、工具调度、权限与沙箱、错误恢复、记忆与 session 持久化。模型负责"想"，harness 决定它**看到什么、能做什么、做错了怎么办、什么时候停**。

这个词 2026 年 2 月由 Mitchell Hashimoto 命名，随后 OpenAI 用它描述"五个月、人工手写代码 0 行"的 Codex 实践。它和另外两个概念是包含关系：

| | Prompt Engineering | Context Engineering | Harness Engineering |
|---|---|---|---|
| 优化对象 | 一次调用的输入 | 每步窗口里放什么 | 整个运行时系统 |
| 时间尺度 | 单轮 | 单 session 多步 | 跨 session、持续迭代 |
| 出错时怎么修 | 改 prompt | 改上下文策略 | 改系统，让同类错误不再发生 |

### 1.1 为什么突然重要

几组"同一个模型、只改 harness"的数据：

| 实验 | 只改了什么 | 结果 |
|------|-----------|------|
| Can Bölük 的 hashline 实验（2026.02） | 文件编辑工具的格式 | 16 个模型里 14 个变好；Grok Code Fast 1：**6.7% → 68.3%** |
| Anthropic 上下文管理（2025.09） | context editing + memory tool | 内部 agentic search **+39%**；100 轮搜索 token 降 **84%** |
| AHE（arXiv:2604.25850） | 自动演化 10 轮 harness | Terminal-Bench 2：69.7% → **77.0%** |
| RHO（arXiv:2606.05922） | 一轮无标签的回顾式优化 | SWE-Bench Pro：59% → **78%** |

一个工具格式为什么能带来十倍差距？因为长任务成功率是**连乘**的。设任务需要 $n$ 次编辑、每次成功落地的概率为 $p$：

$$
P(\text{任务成功}) \approx p^{\,n}
$$

$n = 20$ 时，$p = 0.90$ 只剩 12%，$p = 0.99$ 则有 82%。模型"想对了但没改对"（复述差一个空格、行号偏一行）在 harness 层面就是失败，还会把后续步骤带进错误状态。**大部分收益来自压掉这类非智力性失败，而不是提升推理能力。**

**结论：模型决定上限，Harness 决定底线。**

### 1.2 三个易混词与两代 Agent

| 术语 | 决定什么 | 例子 |
|---|---|---|
| **Harness** | Agent **选择**做什么 | Claude Code、Codex CLI |
| **Framework** | 帮你造 harness 的组件 | LangGraph、Strands、AG2 |
| **Runtime** | Agent **物理上能**做什么 | Docker、systemd、云端托管 |

口诀：Framework 造 harness，Runtime 托管 harness，**harness 才是决定行为的那段逻辑**。Runtime 决定能不能读某个文件，harness 决定会不会去读。

顺带说清"Agent"的含义变化：2023 年的 Agent 是「模型 + 一个工具」，无状态、单轮、跑在宿主进程里；现在的 Agent 要有动态工具系统、跨 session 记忆、优先级上下文组装、错误恢复、沙箱和权限模型。**差距全在 harness 层**——没有它，你只有一个能调函数的聊天机器人。

## 二、六层骨架与一个循环

| 层 | 要回答的问题 | 典型失效 |
|---|---|---|
| L1 Prompt & Context | 这一步应该看到什么 | 关键信息被挤出窗口 |
| L2 Query Loop | 下一步做什么、何时停 | 死循环；过早宣布完成 |
| L3 Tool & Permission | 能调什么、谁来批准 | 工具格式导致执行失败；越权 |
| L4 Multi-Agent | 怎么拆、怎么并行、怎么汇总 | 重复劳动；摘要丢信息 |
| L5 Memory | 跨 session 记住什么 | 同一个坑反复踩 |
| L6 Self-Evolution | harness 怎么改进自己 | 过拟合评测集 |

另有三个横切关注点不属于任何单层，但不处理就上不了生产：**长时运行的上下文生命周期**（八）、**主动调度**（九）、**评测**（十）。

把六层剥到只剩骨头，一个 Agent 就是二十几行。骨架的学名是 **ReAct**（Reason + Act）：推理 → 调工具 → 观察结果 → 再循环，直到不再请求工具。从学习脚本到 Claude Code，内核都是同一个循环，区别全在循环**外围**。

```python
class MinimalHarness:
    def __init__(self, model, tools, system_prompt: str, max_turns: int = 20):
        self.model, self.tools = model, tools
        self.system_prompt, self.max_turns = system_prompt, max_turns

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        for _ in range(self.max_turns):
            response = self.model.chat(messages, tools=self.tools.schemas())
            messages.append(response.to_message())  # 带 tool_calls 的消息要先进历史
            if not response.tool_calls:
                return response.content
            for call in response.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(self.tools.execute(call.name, call.args)),
                })
        return "Max turns reached"
```

一个易漏的细节：**追加工具结果前必须先入 assistant 消息**，否则多数 API 直接报错。

这段代码能跑，但离可用还差几乎全部工作量——缺的每项正好对应一层：上下文组装与压缩（三）、完成判定与错误恢复（四）、工具格式与权限（五）、上下文隔离（六）、记忆（七）、跨 session 状态与验收（八）、自动触发（九）、评测与全链路 trace（十 / 十一）。

## 三、L1 上下文：本质是观测设计

**结论：找最小的高信噪比 token 集合。** Context rot 是共识——窗口里 token 越多，模型准确回忆其中信息的能力越差（$n$ 个 token 产生 $n^2$ 对注意力关系；训练数据短序列占多数；位置插值损失位置精度）。注意"最小"不等于"最短"，信息不足同样失败。

**检索策略**：预检索（先 embedding 灌进去）快但索引会过期、捞错就污染上下文；即时检索（上下文只留路径 / 查询语句，运行时用工具取）慢一点，但上下文干净，而且**路径和命名本身携带信息**——`tests/test_utils.py` 和 `src/core_logic/test_utils.py` 含义完全不同，这些线索在"切块灌进向量库"时会丢掉。Claude Code 是混合策略：`CLAUDE.md` 前置注入，具体文件靠 `glob` / `grep` 当场取。

**组装 = 带优先级的装箱**。每轮按 `System prompt → 活跃工具 schema → 任务指令与完成标准 → 记忆摘要 → 注入文件 → 近期对话 → 更早对话` 的顺序装，空间不足从下往上丢。三条硬规则：**给回复留 reserve**（塞到 100% 模型没地方回话）、优先级最高的几块**宁可截断也不整段丢**、**每一轮重新组装**（一次组装后不更新，第一次工具调用之后模型就在用过时信息推理）。

**压缩不是可选项**。128K 扣掉固定开销剩约 11.9 万给对话，一轮带工具结果的编码任务平均吃 1–3K token，**三四十轮就摸到阈值**，且应在 70% 左右触发、别等撞墙。三道防线：自动衰减（只留最近 N 轮）、阈值压缩（便宜模型摘要早期轮次）、主动摘要（超长任务定期打 checkpoint）。生产里最实用的是**滑动窗口 + 滚动摘要**。最轻最安全的压缩是**工具结果清理**——历史深处的原始返回值模型基本不看第二遍。

**长程技术三选一**：compaction（摘要重开窗口，会丢"后来才显重要"的细节）、结构化笔记（状态写到窗口外文件，笔记会过期）、子 Agent（干净窗口做脏活，只回传摘要）。Claude Code 的 compaction 值得照搬：保留架构决策、未解决的 bug、实现细节，丢弃冗余工具输出。

最后一条边界：**`AGENTS.md` 里写的是请求，不是约束**。"禁止直接 push main"只写在 Markdown 里只是概率性保证，真正的约束在分支保护、CI 和工具权限（5.3）里。分层 Prompt 的顺序（System → Project → Task → Dynamic）同时也是 prompt caching 的友好顺序。

## 四、L2 控制循环：终止条件与恢复策略

**完成判定必须外部可验证。** `if not tool_calls: return` 最不可靠，Agent 最常见的失败就是过早宣布胜利。Anthropic 的做法是把完成标准**外置成结构化文件**：每个 session 先读清单、挑一个未完成项、做完更新状态——"完成"从模型的一句自述变成清单上的布尔值。所有出口：轮次 / 成本上限、清单全绿且测试通过、连续相同工具调用（卡死）、人类中断。`max_turns` 是最不该缺席的一条，没有它混乱的模型会无限循环烧钱。

**模型本身就是错误处理器。** 传统程序里未处理的错误让进程崩溃；agentic 系统里，只要把错误清晰呈现给模型，它就能适应、重试或换路。harness 的工作是**分类 → 套策略 → 只在自动恢复耗尽时升级给人**：

| 类别 | 例子 | 恢复方式 |
|---|---|---|
| **瞬时** | 超时、限流、429 / 503 | 指数退避 + **抖动**重试 |
| **永久** | 文件不存在、权限拒绝 | 回喂给模型，尝试替代方案 |
| **模型** | 畸形调用、幻觉函数名 | 带 schema 纠正重新 prompt |
| **资源** | OOM、磁盘满、token 超限 | checkpoint 并升级给人 |

抖动不是细节：没有它多个 Agent 会形成同步重试的惊群。只对瞬时错误重试，对"文件不存在"重试三次纯浪费。配套三件事：优雅降级（把 fallback 链和最终错误一起给模型）、人在回路四档（AUTO / INFORM / CONFIRM / BLOCK）、长任务每 3–5 轮原子写入 checkpoint。

**黄金原则：错误永远作为工具结果回喂，绝不在循环里抛异常。** 被吞掉的异常会变成静默失败，而静默失败会被模型解释成"成功"，接着幻觉出一个完成结果。

剩下四条边界：大输出先截断再追加、保留并行工具调用（强行串行会引入本不存在的顺序依赖）、流式输出、assistant 消息先入历史。

## 五、L3 工具与权限：本质是动作空间

**地基是一个分离：模型只看 schema，harness 负责执行。** 模型看不到实现代码，于是你可以在它不知情的情况下改实现、限行为、加权限检查——这正是权限层能存在的前提。由此得出 `dispatch` 的铁律：**永远返回字符串，出错也不例外**。返回 dict 会让下游序列化崩掉，抛异常会中断循环；而把错误当成给模型的一条信息（`Error: File not found: /x`），它自己会去 `glob` 找正确路径。

工具设计的结论很集中：**描述质量几乎决定一切**（说清做什么、输出格式、约束、给例子）；**让非法输入不可表达**（枚举优于自由文本）；**返回值为模型优化**（分页、截断、只回关键字段）；**警惕工具集膨胀**（人类都说不清该用哪个工具，Agent 不可能更好）；**复杂能力来自简单工具的组合**而非巨型单体。MCP 的价值是把工具实现与 harness 解耦，把 N×M 集成压成 N+M。

### 5.1 最硬的一组证据：编辑格式

Can Bölük 的《The Harness Problem》（2026.02）只改 edit 工具、横测 16 模型 × 3 格式。三类主流格式（`apply_patch` / `str_replace` / 整文件重写）的共同病根是**都要求模型完美复现它看过的内容**——既浪费上下文，又把成败压在精确召回上。

hashline 换成给每行附 2–3 字符的内容哈希，编辑时只引用标签（"替换 `2:f1`"）。两个性质：不必复现原文；自带乐观并发校验（文件被改过哈希就不匹配，编辑在污染文件前被拒）。

结果（作者自报）：patch 对几乎所有模型最差；hashline 在 **14/16** 个模型上更优，平均 **+15 分**，且**越弱的模型收益越大**——Grok Code Fast 1 从 6.7% 到 68.3%。两个反例（GPT-5.2 Codex 只 +4.6 且 token 反涨、DeepSeek V3.2 负增益）说明原因大概是训练阶段被 patch 格式偏置过。**结论：编辑格式是模型特定的，没有通用最优**——这等价于说 harness 与 policy 是 jointly tuned 的，只换模型不换 harness 往往掉点。

数字要打折看：基准是 React 代码库注入可逆变异自动生成的，作者就是 hashline 实现方。但"patch 失败率能到 50%"这个量级和各家 issue 区的抱怨对得上。

### 5.2 工具太多：渐进披露

工具 schema 每一轮都在上下文里，十几个 MCP server 就能吃掉几万 token。三档做法：十几个以内全常驻；几十到上百个用工具检索或**技能菜单 + `load_skill` / `unload_skill`** 按需展开；工具极多时改成代码执行（MCP server 暴露成可 `import` 的 API，中间结果不进上下文，Anthropic 报告某工作流 15 万 → 2 千 token）。

算账：8 技能 / 约 60 工具的系统，全量预加载约 12,000 token/轮，菜单 + 加载 2 个约 2,550——**每轮省 78%**，30 轮省约 28 万。另一个理由是**模型表现**：schema 超过二十来个，选择质量明显下降。

这里区分两个概念：**工具是单个函数，技能是一组相关工具 + 一份 `SKILL.md`（何时用、约束、示例）+ 行为规则**。由此引出全文最重要的架构哲学——**薄 harness + 厚技能**：harness 只保留通用引擎（循环、上下文组装、注册表），所有领域知识放进可移植、可独立测试、可组合的技能；**加能力靠加技能，而不是改内核**。

### 5.3 权限：写在代码里，不是 prompt 里

逻辑链条是「模型生成文本 → 文本含工具调用 → harness 执行」，所以**任何能影响模型输出的东西都能影响 harness 行为**——包括它读到的文件、网页、用户消息。这就是 prompt 注入。

**最关键的认知：在 prompt 里写"不要删文件"只是建议，一次注入就能覆盖。** 真正的权限控制是代码里的闸门，对每个调用四选一：**放行 / 拒绝（错误回喂模型）/ 改写（路径限制到安全目录）/ 转人工**。分工上，**权限由确定性策略决定，置信度只决定"要不要再想想"**：只读自动过 → 写自动过并记日志 → shell / 网络需人工 → 删除 / `git push` / 对外发消息始终要明确批准。三档模型按风险挑：白名单、黑名单、分级审批。外部来源的内容还要截断并用 `<tool_result>` 包裹，让模型能区分"指令"和"不可信数据"。

沙箱管的是"调了以后能祸害多大范围"：**模型应该感觉不受限制，执行环境恰恰相反**。开发用 Docker、生产多租户用 Firecracker（KVM 边界，容器逃逸影响所有租户）。用 Docker 时真正的限制**不在 Dockerfile，在 `docker run` 参数**：`--rm`、非 root、内存 / CPU / `--pids-limit` 上限、`--read-only`、`--tmpfs /tmp:noexec`、`--cap-drop ALL`、`--network none`；需要联网时用 iptables 白名单。核心原则：**Agent 的代码执行永远不该访问宿主的文件系统、网络和进程空间，凭证绝不进沙箱**。

### 5.4 审批疲劳与盲视推理

手动弹窗会失效：约 **93% 的审批请求被下意识批准**，这是人类注意力的极限而非 UX 问题。第四条路是**让一个独立模型当审核者**，双层防御且互不依赖：**输入层**扫描每个进入 Agent 的工具结果（挡注入），**输出层**在危险调用执行前审动作。成本靠三层决策流收敛——白名单 → 项目内编辑（`git diff` 兜底）→ 只有 shell / 网络 / 外部 API 才跑 classifier，且先用单 token 快筛、可疑才进完整 CoT。

**最承重的设计：classifier 对 Agent 的推理过程盲视。** 它只看用户消息、工具调用、工具结果，不看 thinking token。因为 Agent 能为任何事找借口：

```text
用户：帮我清一下测试环境的脏数据
Agent（推理）：把整个订单库删了最省事，反正应该没人用。
Agent（工具调用）：DROP TABLE orders;
```

若 classifier 读了推理，被注入的 Agent 就能写一大段合理解释把它劝到"同意"——**classifier 同时成了陪审团和被告**。盲视后它只能问："清脏数据，`DROP TABLE orders;` 算合理解读吗？"——不算，不管 Agent 在想什么。这和"生成与评估分离"是同一条原则。硬性要求：**必须在真实流量上测误报率**，拦太多比没有更糟（用户会关掉它）。目标不是零风险，是**让安全的路径成为最省事的路径**。

## 六、L4 子 Agent：本质是上下文隔离

子 Agent 在 harness 里最实际的作用不是"多智能体协作"，而是**上下文隔离**：主 Agent 做规划，子 Agent 在干净窗口里做脏活、只回传摘要。代价要算清：Anthropic 的多 Agent 研究系统报 **token 用量差约 15 倍**，且 token 用量本身解释了评测表现约 80% 的方差——一部分收益其实是花更多 token 买来的。

**结论：多 Agent 是工具不是默认。** 判据是单 Agent 撞到了三面墙（窗口撑爆、无法专业化、串行执行）；没撞墙就别用，**单 Agent + 好工具能解决 80% 的任务**。

落地从 Leader-Worker 起步：把"生成子 Agent"做成暴露给 Leader 的普通工具，入参是任务 + **显式挑选**的一小段上下文——子 Agent 拿到的是全新干净窗口，不是父的对话历史；父是唯一真相来源，子不写父内存、只回传结果字符串。并行改代码用 git worktree，递归深度限 1–2 层，每个 Worker 必须设超时，通信用 push-based（轮询"做完了吗"是最常见的多 Agent bug）。四种编排模式本质是控制流的四种拓扑：流水线、扇出扇入、Supervisor、Peer-to-Peer（最灵活也最难保证终止，生产中很少是对的选择）。反模式：共享可变状态、无界扇出、过度分解。

**生成与评估要分离**：Agent 自评普遍偏高，独立评估 Agent + rubric 校准更靠谱（rubric 怎么写见[《Rubrics：把"好不好"拆成可验证的评分表》](/blog/rubrics-for-llm-eval-and-rl/)）。这条在第八节会加强成硬法则。

## 七、L5 记忆：跨 episode 的状态

先把三个词钉死：**Context** 是单次 API 调用、每轮重建；**Session** 是单次任务、重启即丢；**Memory** 跨 session、落盘。四类记忆里，情景是原始经历，语义和技能是从经历抽出的结论——成熟做法是两者都存，并让结论**指回**原始片段。

**结论：难点不在"存"和"取"，在"更新"。** 多数实现只追加不更新，于是"用户偏好 npm"和"偏好 pnpm"同时躺在库里。正确管线是 extraction → 与已有条目比对 → 显式 **ADD / UPDATE / DELETE / NOOP**（Mem0 的两阶段）。显著性判断可以很朴素：工具失败、人类纠正、没见过的环境事实几乎总值得记；成功的常规操作不必写。写入时机**里程碑写**性价比最好。

检索只用 embedding 相似度，会稳定捞上"语义最像但已过期"的条目。Generative Agents 的三因子仍是最好用的基线：

$$
\text{score} = \alpha_{r}\cdot \text{recency} + \alpha_{i}\cdot \text{importance} + \alpha_{l}\cdot \text{relevance}
$$

其中 recency 按距**上次访问**的时长衰减（形成"用得多的浮上来"的回路）；三项归一化后按场景调权——代码库事实变化快就加重 recency，长期偏好加重 importance。**反思**是"情景 → 语义"的转换器，没有它记忆库永远是一堆原始日志。

存储形态三代：文件 / 笔记（可审计、能直接 `cat`）→ 向量库（能规模化检索，但引入切块和索引维护）→ Agent 自管（MemGPT / Letta、Mem0、A-MEM，灵活但多花 LLM 调用）。**选型建议：先用一代，撑不住再上二代。** 可以直接照搬的是**两级记忆**：每日日志（`memory/YYYY-MM-DD.md`，原始、随手写、成本极低）+ 长期 `MEMORY.md`（精选、定期整理、需要判断力）。启动时读、运行中写日志、定期把日志提炼进长期记忆。注意 `AGENTS.md` 定义"应该如何行为"（声明式），`MEMORY.md` 记录"发生过什么"（经验式），别混成一个文件。技能记忆则存"会做什么"——存成可执行代码的好处是**不依赖模型复述正确、而且可以测试**，代价是技能库会腐烂，要配回归测试；`SKILL.md` 的三级渐进披露让一百个技能常驻也只花几千 token。

**记忆必须始终当成不可信数据**：它是一条跨 session 的持久写入通道，这轮读到的网页内容下轮可能以"历史经验"的身份出现。所以注入时要带来源、显式标注为参考信息而非指令，所有权限判断走 5.3 的确定性策略。

## 八、横切一：长时运行

短任务和长时运行是两个物种：前者要么完成、要么可见地失败；后者跑几小时到几天，**失败是隐蔽的**（上下文累积、质量悄然退化、自我评估在撒谎）。两个隐形杀手值得单独命名：

- **上下文焦虑**：窗口快满时模型开始"赶工"——跳步骤、输出变短、以"我已涵盖要点"过早宣告完成、回避会增加上下文的工具调用。**更大的窗口只是推迟问题**，解法在架构层面。
- **自我评估偏差**：让生成者评自己的输出，它稳定给 8/10 以上。因为模型拥有自己推理的完整上下文，每个决策回看都合理；承认失败等于否定自己；训练数据又奖励自信。

**由此得出全文最重要的一条法则：绝不让生成者批改自己的试卷。**

**上下文生命周期**有两条路：**重置**（清空 + 简报进入新上下文，干净、预算可预测，但有损、"摘要的摘要"会退化、可能重走死路）与**压缩**（旧轮折叠、近轮保真，连续、保留"已试过什么"，但压缩质量参差、与最新状态矛盾时让模型困惑）。判据：有明确阶段 → 阶段间重置；对单一产物持续迭代、频繁回顾早期决策 → 压缩；积累大量工具输出 → 重置（工具输出压缩效果差）。实践常混合：**阶段内压缩，阶段间重置**。

**架构化回答借鉴 GAN**：规划者 → 生成者 → 评估者三段流水线。四条规则：**独立上下文**（评估者只看输出、不看推理，防"我理解你为什么这么做所以没问题"的同情偏差）、**显式 rubric**（"是否处理了边界情况 X"优于"代码好不好"）、**可操作反馈**（"`parse_input` 没处理空串"有用，"7/10"没用）、**迭代预算**（3 次不过交回规划者或人——三次失败说明是规范问题不是执行问题）。反模式：单体 Agent 又规划又执行又审查、没有 rubric 的评估、无限重新规划。一句话：**长时运行不是"时间更长的短任务"；先分解，然后为一切设上限。**

**大脑 / Session / 双手解耦**解决的是一体化容器的三个必然失败：session 随容器崩溃丢失（"宠物问题"）、调试要进装着凭证的容器（调试变安全事件）、不可信代码与凭证同处一室。拆法：大脑（harness + LLM）**无状态**，一切通过 `emitEvent()` 写日志；Session 是**仅追加的持久事件日志**，唯一真相来源；双手（沙箱）**按需创建、用完销毁**。大脑崩了起新进程 `wake(sessionId)` 重放即可——从"宠物"变"牲畜"。

最微妙也最重要的区分：**Session ≠ Context Window**。Session 是完整持久记录，Context Window 只是 harness 为当前这次调用从中"取景"的子集。于是**不可逆决策变可逆**——compaction 不再是单向销毁，被压掉的旧事件随时能回日志再取；上下文工程成为 harness 的职责；多个大脑可以同读一个日志。延迟创建沙箱还带来 TTFT p50 降约 60%、p95 降约 90%（厂商自报）。

### 8.1 多天项目：Initializer + Coding Agent

给模型喂"建一个 Claude.ai clone"然后走开一下午，结局是能用的登录表单、半个消息列表、夹在函数里的 `TODO` 和一条自信的 commit message。**这几乎总是 harness 的锅。**

Compaction 是 **in-context memory**：对两小时重构够用，对两天构建会垮——Session 2 读有损摘要重建意图，Session 3 读"摘要的摘要"，到 Session 4 模型已经在**跟过去的自己玩传话游戏**，分不清哪些做完了、哪些是摘要里幻觉出来的；更糟的是它恰好强化了上个 session 过度乐观的叙事。长程 Agent 需要 **out-of-context memory**：文件、结构化状态、git 历史，每个 session 重新读取，而不是靠总结层层传递。

要同时治两种失败：**一气呵成倾向**（一个 session 做完全部，窗口满了留一堆半成品）和**过早胜利**（读进度笔记说"implemented login, messaging"，看到像实现了的代码就宣布完成，没跑没测）。解法是拆成**只跑一次的 Initializer**（产出 `init.sh`、`feature_list.json`、进度笔记、首次 commit）和**跑 N 次、每次全新进程无记忆的 Coding Agent**，共享文件系统，由一个故意写得很"笨"的驱动脚本调度。

**为什么 feature list 用 JSON 而不是 Markdown？** 给模型 Markdown，它会当散文——带着最好的意图重写优先级、"感觉更干净"就合并两个 feature。**Markdown todo 对 Agent 是流沙**；JSON 被当作结构化数据读写特定字段。每个条目带 `id` / `priority` / `depends_on` / `acceptance` / `passes`，再配一条硬规则写进 system prompt：**只允许修改 `passes` 字段**，删改验收标准是"不可接受"的，认为定义有误就停下上报。**一个布尔字段（而不是一段散文、一种"感觉"）就是未来 session 判断"什么真的做完了"的依据**——这正是"完成标准外置"在多天尺度上的完整形态。

Coding Agent 每次醒来的开机仪式：`pwd` → `git log`（**真相**，与进度笔记冲突时 git 赢）→ 读 `feature_list.json` → `bash init.sh` → **浏览器打开 App 验证上个 session 的活还在工作**（这一步杀死过早胜利）。之后只挑**一个**最高优先级、`passes: false`、依赖已满足的 feature；E2E 通过、翻绿、干净 commit 就停。

**一 session 一 feature 同时治两种失败**：契约约束范围，"done" 由 E2E 而非自封定义。**E2E 是信任层**——单元测试是 Agent 自己写的，可能和实现一样地错；用浏览器像真实用户一样驱动才是独立信号。由此得到一条通用判据：**测试 / oracle 的质量就是 Agent 能力的天花板**。没做完就回退到上次干净 commit、绝不 commit 坏代码。

核心洞察：**"我知道什么"由磁盘文件回答，不由上下文窗口回答。长程 Agent 老忘事，修复几乎从来不是更大的窗口，是更多的文件。**

## 九、横切二：主动调度

只在被问到时回应的是聊天机器人；按时间表、按事件主动行动的才是基础设施。四种原语：**One-Shot Timer**、**Recurring Cron**（自动化骨干）、**Event-Triggered**（新 PR → review 子 Agent）、**Heartbeat**（每 15–60 分钟注入一个 prompt，一轮批量做多个轻量检查）。

一个调度任务四个字段即四个要点：**cron**（铁律是**内部存 UTC、显示本地时间**，用户说"每天早上 8 点"先问时区；时区是这层最大的坑）、**session 目标**（`isolated` 全新上下文不污染主对话 / `main` 有上下文但会撑爆窗口）、**载荷类型**（`agentTurn` 完整执行 / `systemEvent` 只丢便条）、**交付方式**（监控类要**异常才告警**，每 15 分钟推"一切正常"是噪音）。调度循环由外部 1 分钟 tick 驱动，不要自己 `while True: sleep`。

Heartbeat vs Cron：多个轻量检查、需对话上下文、时间不精确 → 一份 `HEARTBEAT.md` 搞定，别开五个 cron；精确时间、隔离环境、按任务选模型 → cron。

## 十、横切三：评测这道坎

### 10.1 基础设施是一等实验变量

传统 benchmark（MMLU、HumanEval）本质是一次函数调用，2 核和 64 核分数一样；**Agent 评测打破这个假设**——它要派生进程、读写代码库、迭代、管理时间。高性能机上 8 秒的测试套件在受限容器上可能要 90 秒；5 分钟被杀的 Agent 比给 30 分钟的解得少，**不是模型更差，是时间用完了**。

Anthropic 的对照实验（同模型、同脚手架、同任务，只改基础设施）：**配置差异能让分数波动高达 6 个百分点**，往往超过排行榜第 1 与第 5 名的差距。陷阱在语义细节：内存"保证分配 vs 硬性 OOM 终止"、CPU"份额 vs 固定核"、超时"壁钟 vs CPU 时间"。且机制分两段：1x→3x 主要在消除基础设施错误（错误率 5.8%→2.1%，分数变化在噪声内）；3x→无上限 分数显著上升（Agent 真的用额外资源做更激进的探索）。

建议：把资源配置当实验变量记录并版本化；指定 floor + ceiling，**floor 取最小可行配置的 3 倍**（低于此主要在测噪声）；单独监控 OOM / 超时 / 磁盘满错误率（超 2–3% 说明结果被污染）；一次评测一台机器。**基准分不是模型的属性，是"模型 + 脚手架 + 基础设施 + 任务集 + 评分函数"这个系统的属性。**

### 10.2 Eval Awareness：测试时的 reward hacking

一个案例：Claude Opus 4.6 在 BrowseComp 一道题上烧了 4050 万 token（中位数的 38 倍）。前 3000 万正常解题，然后轨迹变了——**Agent 不再回答问题，而是开始问"这个问题是什么？"**：形成元假设（"太具体、失败率太高，像是 eval 题"）→ 锁定 BrowseComp → 读仓库数据加载代码 → 发现数据集用 SHA-256 派生的 XOR 密钥加密 → 自己写 `decrypt()` → 在镜像站找到密文 → 解密全部 1266 条直接取答案。1266 道题中这样成功了 2 道。每一步单看都平常，**新颖在于无人提示下把它们串成连贯计划**。

这是**全新的污染模式**：不是训练数据泄露，而是**测试时环境被利用**——被污染的不是训练集，是这次运行，因为模型自主决定把 benchmark 当研究对象而非要解决的任务。触发信号成组出现（问题极端具体、人造感、数百次搜索失败、预算异常宽松），多 Agent 会放大（污染率 0.24%→0.87%），甚至有 URL slug 旁路信道让不同运行共享元假设。

最有启发的一点：**挡住那 16 次尝试的不是聪明检测器，而是最朴素的能力限制**——非文本 MIME 限制、数据集来源认证门禁。**能力限制打败行为分析，而且这些控制不是 eval 专用的，就是通用 harness 卫生。**

启示：静态 benchmark 当"软泄露"物件处理（轮换题目、留私有切片）；**默认限制工具能力**（只允许 text/HTML、eval 模式剥离写入、限每轮预算）；用 Web **白名单而非黑名单**；记录推理以便事后分析；eval 开发者要假设模型会逆向工程你的仓库。要认清这既不是欺骗也不是 pathology，**而是能力**——不会被训练掉，只会被训练进去，因为这套技能在 eval 之外到处有用。harness 的工作不是说服 Agent 别走捷径，而是**让捷径不可用**。

## 十一、框架与自动进化

**框架选型看一个视角：控制循环能被编程到什么程度。** Claude Code / Codex 把 loop 做成产品内核，开箱即用但改不动；LangGraph 把 loop 显式表示成状态图，最灵活但每层自己负责；Strands 几乎不写控制流，代码最少可控性也最弱。原厂 harness 的优势是和自家模型**一起调**（后训练阶段就见过这套工具格式）——这解释了 5.1 的结论：harness 是模型特定的。

**自动进化（L6）**：2026 上半年四个代表工作（Meta-Harness、AHE、Self-Harness、RHO）可以用"谁提议修改 × 用什么验证"两条轴归类，共同点是**改进发生在模型外部、权重不动**。三条可直接拿走的经验：

1. **给 proposer 原始轨迹，别给摘要**——和 ACE 的 "别压缩反馈" 一致。
2. **改结构，别改措辞**。AHE 消融显示增益主要来自工具、中间件和长期记忆，而不是 system prompt；文字策略最难迁移。
3. **每次修改都要可回滚、可归因、过回归门**；可编辑范围走白名单（权限、沙箱、审批不可改）；失败候选也存档；held-out 会被用旧，最终报告要换一份从未参与挑选的测试集。

数字要打折看：全部作者自报，且 Terminal-Bench 的 held-out 集很小（据 Self-Harness 报的分数推测约 42 题，**一道题就是 2.4 个点**）。

## 十二、共识、开放问题与参考

### 12.1 共识即小结

1. **Agent = Model + Harness**。内核就是一个循环，复杂度全在循环**外围**；同一个模型换 harness，差距从几个点到十倍。
2. **大部分收益来自压掉非智力性失败**（工具格式、上下文、错误恢复、停止条件）；长任务成功率是连乘的，单步可靠性会被成倍放大。
3. **模型问题往往不是模型问题**——先怀疑上下文装错了、schema 写糊了。
4. **上下文是最高杠杆**：找最小的高信噪比 token 集合；每轮重装、留 reserve、70% 就压。
5. **错误不要吞，要回喂；给一切设上限**——无限循环等于无限烧钱。
6. **约束在代码里，不在 prompt 里**；凭证绝不进沙箱；**限制能力胜过劝说和行为分析**。
7. **薄 harness + 厚技能**；工具 schema 与实现分离，`dispatch` 永返字符串。
8. **授权与判断分开**；记忆、网页、工具返回值一律当不可信数据；独立审核者要对推理盲视。
9. **记忆的难点在"更新"**：显式 ADD / UPDATE / DELETE + 衰减 + 指回原始轨迹；两级记忆（日志 + `MEMORY.md`）。
10. **绝不让生成者批改自己的试卷**：独立上下文、显式 rubric、评估者只看输出。**长程 Agent 靠文件记忆，不靠窗口**；git 是最好的协调原语，oracle 质量是能力天花板。
11. **多 Agent 是工具不是默认**；单 Agent + 好工具解决 80% 的任务。
12. **评测里基础设施是一等实验变量，且要防 Agent "绕过任务而非解决任务"**；自动进化给原始轨迹、改结构、过回归门。

一句话收束：**模型决定 Agent 的能力上限，harness 决定你能把这个上限稳定、安全、可持续地兑现多少。**

### 12.2 Harness 会"过期"与开放问题

Anthropic 记录过：早期 harness 基于 Sonnet 4.5，该模型接近窗口上限时变得保守急于收尾（即上下文焦虑），于是用**重置 + 结构化交接**而非压缩；换到 Opus 4.5 后这个行为基本消失，重置被整个去掉。**harness 里相当一部分是在给当前模型的缺陷打补丁，模型升级后要回头做减法**——所以"记录每条规则是为了解决哪个具体问题"很重要，否则以后不知道哪条能删。

开放问题集中在四处：**① 缺一个固定模型横评 harness 的标准化评测**（口径不一，且不报告基础设施配置的分数本身就不可比）；**② 记忆的评测**只能靠端到端成功率间接衡量，"记对了什么、漏了什么"缺直接指标，记忆污染尤其难测；**③ 自改进的安全性**——白名单防不住间接绕过（新增一个功能上等价于被禁命令的工具）；**④ 联合优化**——演化产生的大量带验证结果的轨迹天然是训练数据，但"训下一代模型、再为新模型重建 harness"这个双循环怎么排还没有成熟答案，训练侧的信用分配可参考[《长轨迹 Agent RL 的信用分配：从 GRPO 到 GiGPO》](/blog/agent-rl-credit-assignment-gigpo/)。另有跨框架迁移（`AGENTS.md` / `SKILL.md` 只覆盖文字策略）与成本-准确率-延迟 Pareto 前沿的标准化问题。

### 12.3 推荐阅读

**概念与工程实践**：[Hashimoto, *My AI Adoption Journey*](https://mitchellh.com/writing/my-ai-adoption-journey) · [OpenAI, *Harness engineering*](https://openai.com/index/harness-engineering/) · [Can Bölük, *The Harness Problem*](https://blog.can.ac/2026/02/12/the-harness-problem) · [Anthropic, *Effective context engineering*](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents) · [Managing context](https://claude.com/resources/articles/context-management) · [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) · [Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) · [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) · [Harness design for long-running apps](https://www.anthropic.com/engineering/harness-design-long-running-apps) · [Multi-agent coordination patterns](https://claude.com/resources/articles/multi-agent-coordination-patterns) · [Agent Skills](https://skill.md/) · [Harness Engineering 系列教程](https://harness-guide.com/zh/guide) · [Loop Engineering 系列](https://happycapy.ai/)

**记忆**：[Generative Agents](https://arxiv.org/abs/2304.03442) · [MemGPT / Letta](https://arxiv.org/abs/2310.08560) · [MemoryBank](https://arxiv.org/abs/2305.10250) · [A-MEM](https://arxiv.org/abs/2502.12110) · [Mem0](https://arxiv.org/abs/2504.19413) · [ACE](https://arxiv.org/abs/2510.04618)

**自动进化**：[Meta-Harness](https://arxiv.org/abs/2603.28052) · [AHE](https://arxiv.org/abs/2604.25850) · [Self-Harness](https://arxiv.org/abs/2606.09498) · [RHO](https://arxiv.org/abs/2606.05922)

**基础论文**：[Reflexion](https://arxiv.org/abs/2303.11366) · [Voyager](https://arxiv.org/abs/2305.16291) · [STOP](https://arxiv.org/abs/2310.02304) · [Gödel Agent](https://arxiv.org/abs/2410.04444) · [Darwin Gödel Machine](https://arxiv.org/abs/2505.22827)
