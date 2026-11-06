---
title: "OpenClaw 源码解析（一）：整体架构与核心数据流"
description: "OpenClaw 把 182 个渠道与能力扩展、多家模型 provider 和一套可插拔的 Agent 运行时收进同一个 Gateway。本系列第 1 篇建立全局认知：代码怎么分层、一条消息从 Discord 进来到回复出去要经过哪七站、状态与安全怎么设计，以及算法研究员该从哪里读起。"
pubDate: 2026-06-08
tags: ["OpenClaw", "Agent", "源码解读", "系统架构", "TypeScript", "LLM 工程"]
---

> **本文是《OpenClaw 源码解析》系列第 1 篇**。OpenClaw（`openclaw/openclaw`）是一个用 TypeScript 写的多渠道 Agent 网关：用户可以从 Discord、Telegram、Slack、WhatsApp、Signal 等渠道，或者 Android / iOS / macOS / Linux / TUI 客户端，和同一个 Agent 对话；Agent 背后可以挂任意 LLM provider、MCP 工具和技能。本系列面向算法研究员，目标是用最短路径看懂它的骨架，并找到适合做实验的切入点。
>
> 系列目录：
> 1. **整体架构与核心数据流**（本文）
> 2. [Agent Loop 与工具执行引擎](/blog/openclaw-02-agent-loop/)
> 3. [插件系统与扩展机制](/blog/openclaw-03-plugin-system/)

做 Agent RL 的人迟早会碰到一个问题：**训练时的 rollout 环境和线上真实运行的 harness 长得不一样**。线上会做上下文压缩、会拦截死循环、会因为权限不足拒绝工具调用，而训练环境往往只是一个干净的 `while has_tool_call` 循环。模型在训练里没见过的上下文形态，上线后就是分布外。

读一个生产级 Agent 网关的源码，价值就在这里——它告诉你"真实世界里 Agent 的一轮对话"到底包含哪些环节。本篇不深入任何模块内部，只回答三个问题：代码怎么组织、一条消息怎么流转、状态和安全怎么兜底。

## 一、先看规模：别被 49,000 个文件吓到

| 指标 | 数据 |
| :--- | :--- |
| TypeScript 文件数 | ~49,000 |
| 核心源码（`src/`）子模块 | 79 个 |
| 扩展 / 插件（`extensions/`） | 182 个 |
| 内置技能（`skills/`） | 52 个 |
| 包（`packages/`） | 25 个 |
| 客户端（`apps/`） | Android / iOS / macOS / Linux / TUI |

文件数看着吓人，但大头在 `extensions/`——182 个扩展，每个都是一个自包含的插件（一个渠道、一个模型 provider，或者一种能力）。它们结构高度同质，读懂一个就读懂了一类。

真正承载"算法"的代码集中在 `packages/agent-core/`，规模远小于整个仓库。所以一个合理的阅读顺序是：

1. 先读 `packages/agent-core/`——Agent Loop、工具执行、Compaction，全是纯逻辑；
2. 再读 `src/gateway/agent-turn/`——看纯逻辑是怎么被接上 IO、权限和调度的；
3. 最后按需挑一两个扩展看（推荐 `extensions/anthropic/`）。

## 二、高层架构：四层结构

```text
                    ════════════════ OpenClaw Gateway ════════════════

 [Trunk]    Channel Ingress ─▶ Gateway Router ─▶ Session Controller ─▶ Agent Worker
              (182 ext)          (routing/)         (sessions/)          (worker/)
                                                                            │
 [Core]                     Context Engine ◀────▶ Agent Core ◀──────────────┘
                           (context-engine/)    (Loop + Tools)
                                                      │
                                                  LLM Core
                                             (Providers + Stream)
                                                      │
 [Egress]   Channel Egress ◀────── Gateway ◀──────────┘

 [Ability]  Config Store · Plugin Registry · Skills Runtime · Cron Scheduler
            (config/)      (plugins/)        (skills/)        (cron/)

 [Infra]    State (SQLite) · Security (Audit + Sandbox) · MCP (Bridge + Server) · Memory
            (state/)         (security/)                  (mcp/)                  (memory/)
```

可以把整个 Gateway 拆成四层来理解：

- **消息主干**：Channel Ingress → Gateway Router → Session Controller → Agent Worker。这条链路只负责"把一条外部消息送到正确的 Agent 进程里"，不碰任何模型逻辑。
- **执行核心**：Agent Core（主循环 + 工具）、Context Engine（上下文准备与压缩）、LLM Core（provider 抽象与流式传输）。这是 Agent "思考"发生的地方。
- **能力扩展**：配置、插件注册表、技能运行时、定时任务。它们决定 Agent "能做什么"以及"什么时候被唤醒"——注意 Cron 也是一种入口，Agent 不一定由用户消息触发。
- **基础设施**：SQLite 持久化、安全审计与沙箱、MCP 桥接、记忆系统。

## 三、目录结构：src 编排、packages 算法、extensions 能力

### 3.1 核心源码 `src/`

```text
src/
├── gateway/          # HTTP/WebSocket 网关，请求入口
│   └── agent-turn/   # ★ Agent 执行的完整生命周期
├── agents/           # Agent 运行时：harness、subagent、MCP bundle
├── sessions/         # Session 管理：生命周期、权限、消息路由
├── worker/           # Worker 进程：推理流、嵌入式 Agent、浏览器
├── context-engine/   # 上下文引擎：注册表、隔离、compaction 委托
├── channels/         # 182 个渠道的统一抽象层
├── routing/          # 消息路由：account → session → agent
├── plugins/          # 插件系统：清单、激活、API 生命周期
├── skills/           # 技能系统：发现、加载、运行时快照
├── config/           # 配置：schema 校验、热更新、channel 能力
├── cron/             # 定时任务：调度、执行、结果投递
├── llm/              # LLM 传输层：provider 注册、限流、流式
├── mcp/              # MCP 协议：工具注册、stdio/HTTP 桥接
├── state/            # SQLite 持久化：数据库启动、迁移、清理
├── security/         # 安全审计：exec 沙箱、渠道权限、代码安全
├── memory/           # 记忆系统：MEMORY.md 管理
├── model-catalog/    # 模型目录：模型元数据、能力声明
├── model-picker/     # 模型选择：fallback 策略、cost 优化
└── ...               # 40+ 其他模块
```

### 3.2 包 `packages/`：可独立使用的核心库

```text
packages/
├── agent-core/         # ★ Agent Loop 核心算法（纯逻辑，无 IO）
├── llm-core/           # LLM 请求/响应类型 + 流处理
├── gateway-protocol/   # Gateway WebSocket 协议定义（Schema）
├── plugin-sdk/         # 插件开发 SDK
├── ai/                 # AI SDK 抽象层
├── tool-call-repair/   # 工具调用修复（JSON 解析容错）
├── model-catalog-core/ # 模型元数据
├── normalization-core/ # 字符串/编码/CJK 标准化
└── ...
```

### 3.3 扩展 `extensions/`：182 个插件

```text
extensions/
├── anthropic/        # Claude 系列（API + CLI Backend）
├── openai/           # GPT / o-series
├── google/           # Gemini
├── discord/          # Discord 渠道
├── telegram/         # Telegram 渠道
├── slack/            # Slack 渠道
├── whatsapp/         # WhatsApp 渠道
├── signal/           # Signal 渠道
├── browser/          # Playwright 浏览器自动化
├── mcp-relay/        # MCP 中继
└── ...               # 170+ 其他扩展
```

三个顶层目录的分工非常清晰：

| 目录 | 定位 | 是否有 IO |
| :--- | :--- | :--- |
| `packages/` | 可独立发布的库，尤其是 `agent-core` 承载核心算法 | 基本没有，靠依赖注入 |
| `src/` | 编排层：网关、路由、会话、权限、持久化 | 大量 IO |
| `extensions/` | 能力层：每个扩展一个渠道 / provider / 工具 | 各自封装 |

两个小包值得单独点名：`tool-call-repair/` 专门做工具调用的 JSON 容错——模型输出的参数 JSON 缺括号、多逗号是线上常态，这一层修复能显著降低"格式错误导致的工具失败"；`normalization-core/` 里有专门的 CJK 标准化，说明中日韩文本场景是被认真对待过的。

## 四、核心数据流：一条消息的完整旅程

```text
用户消息 (Discord / Telegram / API / CLI / ...)
    │
    ▼
① Channel Ingress
    extensions/<channel>/  → 标准化为内部消息格式
    │
    ▼
② Gateway Router (src/routing/)
    account-lookup → session-key → agent-id 三级路由
    │
    ▼
③ Agent Turn Admission (src/gateway/agent-turn/)
    ├── agent-request-preflight.ts     # 预检：权限、限流、去重
    ├── agent-run-admission-phase.ts   # 准入：模型选择、配额检查
    ├── agent-run-dispatch.ts          # ★ 调度核心：启动 Worker
    └── agent-run-execution-phase.ts   # 执行：流式输出、超时控制
    │
    ▼
④ Worker Launch (src/worker/)
    launch-descriptor.ts → 构建 Worker 启动参数
    ├── inference-stream.runtime.ts    # 在线推理流
    ├── embedded-agent-live.runtime.ts # 嵌入式 Agent
    └── browser-runtime.ts             # 浏览器 Agent
    │
    ▼
⑤ Agent Loop (packages/agent-core/)
    agent-loop.ts → 核心循环
    ├── Stream LLM Response (StreamFn)
    ├── Parse Tool Calls
    ├── Execute Tools (parallel / sequential)
    ├── Emit Results
    └── Loop until no more tool calls
    │
    ▼
⑥ Context Engine (src/context-engine/)
    ├── Compaction（token 超限时自动压缩历史）
    ├── Memory injection（MEMORY.md / skills / plugins）
    └── Transcript management
    │
    ▼
⑦ Channel Egress
    Gateway → Channel → 用户
```

逐站解释一下：

**① Channel Ingress**：每个渠道扩展负责把平台特有的消息格式翻译成统一的内部消息。之后的所有环节都不知道消息来自 Discord 还是 Telegram——渠道差异在第一站就被抹平了。

**② Gateway Router**：三级寻址，决定"这条消息属于哪个账号、哪个会话、交给哪个 Agent"。下一节细看。

**③ Agent Turn**：这是整个网关的中枢，分四个阶段——**预检**（权限、限流、去重）、**准入**（选模型、查配额）、**调度**（拉起 Worker）、**执行**（流式输出、超时控制）。一次 Agent 运行在这里被完整地"包"起来，所有横切关注点都在这层处理，下游的 Agent Loop 不用关心。

**④ Worker Launch**：Agent 不在网关进程里跑，而是在 Worker 里跑。Worker 有三种形态：普通在线推理流、嵌入式 Agent、浏览器 Agent。进程隔离的好处很直接——单个 Agent 崩溃、卡死、内存泄漏都不会拖垮网关。

**⑤ Agent Loop**：经典的"调 LLM → 解析工具调用 → 执行工具 → 回填结果 → 再调 LLM"循环，直到模型不再调用工具。这是[第 2 篇](/blog/openclaw-02-agent-loop/)的主角。

**⑥ Context Engine**：管理上下文——token 超限时压缩历史、注入记忆（`MEMORY.md`）和技能 / 插件提示、维护 transcript。严格来说它包裹在 Agent Loop 的前后（`prepareContext` / `processResult` 两个钩子，见[第 3 篇](/blog/openclaw-03-plugin-system/)），这里为了叙述按顺序列出。

**⑦ Channel Egress**：Agent Loop 发出的事件流经 Gateway 回到渠道扩展，再翻译成平台消息发给用户。

## 五、关键代码走读

### 5.1 消息路由：三级寻址

```typescript
// src/routing/resolve-route.ts (简化)
export async function resolveRoute(params: {
  channel: string;
  chatId: string;
  senderId: string;
  config: OpenClawConfig;
}): Promise<ResolvedRoute> {
  // Level 1: Account 查找
  const account = resolveAccountId(params.channel, params.config);

  // Level 2: Session Key 生成
  // 格式: "agent:<agentId>:<channel>:<chatId>"
  const sessionKey = resolveSessionKey({
    account,
    chatType: params.chatType,
    chatId: params.chatId,
  });

  // Level 3: Agent 绑定
  const agentId = resolveAgentForRoute(account, params.config);

  return { account, sessionKey, agentId };
}
```

重点在 session key 的格式：`agent:<agentId>:<channel>:<chatId>`。这个四段式 key 隐含了几条会话语义：

- **会话归属于"Agent × 渠道 × 聊天"的组合**。同一个 Telegram 群里，换一个 Agent 就是另一段独立对话；同一个 Agent 在 Discord 和 Slack 上也互不串台。
- **`senderId` 不在 key 里**。也就是说，群聊里多个人和 Agent 对话时共享同一个会话上下文——这和人类在群里聊天的直觉一致。
- **`chatType` 参与生成 key**，为私聊、群聊等不同会话类型采用不同归属策略留出了空间。

### 5.2 Agent Turn 调度入口

```typescript
// src/gateway/agent-turn/agent-run-dispatch.ts (核心流程简化)
export function dispatchAgentRunFromGateway(params: {
  runId: string;
  ingressOpts: AgentIngressOptions;
  io: AgentTurnIo;
  context: AgentTurnContext;
}) {
  // 1. 创建诊断上下文
  const diagnostics = createAgentRunDiagnostics(...);

  // 2. 注册去重 (防并发重复执行)
  setGatewayDedupeEntries({ ... });

  // 3. 启动实际执行
  const command = agentCommandFromGatewayIngress(params.ingressOpts);
  // → 内部调用 Worker 进程 → Agent Loop

  // 4. 处理终端结果
  const outcome = buildAgentRunTerminalOutcome(result);
  // → 投递到 Channel Egress
}
```

第 2 步的去重值得多说一句。多渠道场景下重复请求是常态：渠道 webhook 超时重投、客户端断线重连后重发、用户手快连点两次……如果不去重，同一条消息会触发两个并发的 Agent 运行，轻则回复两遍，重则两个 Agent 同时改同一个文件。去重放在调度入口，是成本最低的位置。

第 4 步的 `buildAgentRunTerminalOutcome` 则把各种结束方式（正常完成、超时、被中断、工具死循环被终止……）统一成一个"终局结果"，再交给渠道投递。

### 5.3 Worker 启动描述符

```typescript
// src/worker/launch-descriptor.ts (关键类型)
export type WorkerLaunchPlan = {
  version: 4;
  admission: {
    sessionId: string;
    sessionKey: string;
    agentId: string;
    // ...权限、配额
  };
  assignment: {
    toolAuthority: WorkerToolAuthority;        // 工具白名单 + exec 权限
    browser?: WorkerBrowserLaunchDescriptor;   // 浏览器配置
    computer?: WorkerComputerLaunchDescriptor; // Computer Use
    github?: WorkerGitHubLaunchBinding;        // GitHub 集成
    permissionMode: SessionPermissionMode;
    workerContainmentRoot: string;             // 沙箱根路径
  };
};
```

这个类型把一次 Worker 启动拆成两半：

- **`admission`**：回答"你是谁、能不能跑"——会话身份、Agent 身份、权限与配额；
- **`assignment`**：回答"给你什么资源、划多大的圈"——工具权限、浏览器 / Computer Use / GitHub 等可选能力、权限模式，以及文件系统沙箱根路径。

`version: 4` 这个字面量类型也很说明问题：Gateway 和 Worker 之间是一条有协议版本的进程边界，启动描述符已经迭代到第四版。把权限和资源以**声明式描述符**的形式一次性交给 Worker，而不是让 Worker 运行时回头来问，是这个设计的关键——Worker 拿到的就是它能用的全部，没有更多。

## 六、状态管理：每个 Agent 一个 SQLite

`src/state/` 的设计是 **SQLite per Agent**——每个 Agent 一个独立数据库：

| 表 | 内容 |
| :--- | :--- |
| `sessions` | 会话元数据 |
| `transcript` | 对话历史 |
| `compaction` | 压缩摘要 |
| `cron_jobs` | 定时任务 |
| `approvals` | 审批记录 |
| `plugin_state` | 插件状态 |

启动时还会做一组完整性检查：

- **Journal bounds check**：校验 WAL 日志边界完整；
- **Snapshot capacity verification**：快照容量校验；
- **Migration runner**：按版本执行 schema 迁移。

为什么每个 Agent 一个库，而不是一个大库加 `agent_id` 列？几点好处：天然避免多租户写竞争（SQLite 的写锁是库级的）；隔离彻底，一个 Agent 的数据损坏不会波及别人；备份、迁移、删除都以 Agent 为单位，操作简单。

另一个细节是 `transcript` 和 `compaction` 分表存放。结合[第 2 篇](/blog/openclaw-02-agent-loop/)里压缩结果用 `firstKeptEntryId` 标记"从哪条消息开始保留"的设计，可以推断：压缩更像是在原始 transcript 之上叠加一层"摘要 + 指针"的视图，而不是物理删除历史。这对调试和事后分析很友好——原始轨迹始终可以完整回放。

## 七、安全模型：四层洋葱

```text
L1: Channel Auth
    └── 每个渠道独立鉴权（OAuth / Token / Webhook Signature）

L2: Gateway Admission
    ├── agent-runtime-approval-authority.ts  # 审批授权
    ├── agent-runtime-identity-token.ts      # 身份令牌
    └── allowlist-match.ts                   # 白名单匹配

L3: Worker Containment
    ├── WorkerContainmentRoot                # 文件系统沙箱
    ├── ToolAuthority.exec.security          # exec 执行权限
    │   (deny | allowlist | ask | auto | full)
    └── ToolAuthority.allowedToolNames       # 工具白名单

L4: Audit System (src/security/)
    ├── Deep code safety analysis
    ├── Cross-agent session access audit
    └── Exec sandbox verification
```

由外到内四层，每层解决不同的问题：

- **L1 渠道鉴权**：确认消息真的来自那个平台（Webhook 签名防伪造）。
- **L2 网关准入**：确认发消息的人有权让这个 Agent 干活——身份令牌、审批授权、白名单。
- **L3 Worker 隔离**：即便前两层都放行了，Agent 能碰到的东西也是有边界的。文件系统被限制在 `WorkerContainmentRoot` 之下；工具有白名单；命令执行权限分成 `deny`、`allowlist`、`ask`、`auto`、`full` 五档，从最严到最松。
- **L4 审计**：事后与持续性的检查——代码安全分析、跨 Agent 的会话访问审计、exec 沙箱校验。

对一个会执行 shell 命令、会操作浏览器、会读写文件的 Agent 来说，L3 是最关键的一层。"ask" 档位意味着高危操作可以停下来等人审批，这正是 `approvals` 表存在的原因。

## 八、配置系统：Schema 驱动的热更新

配置的查找链（优先级从高到低）：

1. CLI 参数
2. 环境变量（`OPENCLAW_*`）
3. `.openclaw/config.yaml`（workspace 级）
4. `~/.openclaw/config.yaml`（用户级）
5. 默认值（由 schema 定义）

所有配置项都有 schema 定义，这带来两个能力：运行时校验，以及**可被程序查询**。Gateway 工具支持以下操作：

| 操作 | 作用 |
| :--- | :--- |
| `config.schema.lookup` | 查询某个 dot path 的 schema |
| `config.get` | 读取当前配置 |
| `config.patch` | 部分更新（热更新） |
| `config.apply` | 全量替换 |
| `restart` | 需要重启时触发 |

哪些能热更新、哪些必须重启，边界划得很有逻辑：

- **热更新**：模型切换、thinking level、工具 allow/deny——都是"每轮运行开始时读一次"的参数，改了下一轮自然生效；
- **需重启**：渠道配置、插件启用 / 禁用、端口变更——都涉及长连接、进程拓扑或插件激活，运行中改不动。

有了 `config.schema.lookup`，Agent 自己就能先查清某个配置项的类型和取值范围，再用 `config.patch` 去改，而不是靠猜。Schema 在这里既是校验器，也是给 Agent 看的"说明书"。

## 小结

本篇建立的全局认知可以压缩成六句话：

1. **Gateway 是中枢**：所有消息都经过 `src/gateway/agent-turn/` 的预检—准入—调度—执行四阶段，横切关注点全在这层解决；
2. **Agent Core 是纯逻辑**：`packages/agent-core/` 不依赖 IO，可独立测试——这是算法研究的最佳切入点；
3. **182 个扩展 = 182 种能力**：每个扩展是一个独立插件，结构统一，读懂一个就读懂一类；
4. **每个 Agent 一个 SQLite**：避免多租户竞争，transcript 与压缩摘要分表，原始轨迹可回放；
5. **安全是四层洋葱**：Channel Auth → Gateway Admission → Worker Containment → Audit；
6. **配置是 Schema 驱动的**：类型可查、可校验、可热更新，并且 Agent 自己也能读懂。

下一篇 [《Agent Loop 与工具执行引擎》](/blog/openclaw-02-agent-loop/) 会钻进 `packages/agent-core/`：主循环怎么写、工具调用经历哪六个阶段、怎么防止 Agent 陷入死循环、上下文压缩为什么必须在 turn 边界截断——以及这些设计对 Agent RL 的 rollout 意味着什么。
