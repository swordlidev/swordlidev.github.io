---
title: "OpenClaw 源码解析（三）：插件系统与扩展机制"
description: "182 个扩展、52 个内置技能、可替换的 Context Engine、双向 MCP 桥——OpenClaw 如何让这么多能力按需加载又互不拖累。系列收尾篇拆解激活规划器、带生命周期的插件 API、SKILL.md 快照、引擎隔离、工具结果中间件与子 Agent 的权限收缩。"
pubDate: 2026-09-22
tags: ["OpenClaw", "Agent", "插件系统", "MCP", "Skills", "源码解读"]
---

> **本文是《OpenClaw 源码解析》系列第 3 篇，也是收尾篇**。[第 1 篇](/blog/openclaw-01-architecture/)看了网关骨架，[第 2 篇](/blog/openclaw-02-agent-loop/)拆了 Agent Loop。本篇回答剩下的问题：OpenClaw 那 182 个扩展、52 个技能是怎么挂进系统的？怎么做到按需加载、出了问题互不影响？
>
> 系列目录：
> 1. [整体架构与核心数据流](/blog/openclaw-01-architecture/)
> 2. [Agent Loop 与工具执行引擎](/blog/openclaw-02-agent-loop/)
> 3. **插件系统与扩展机制**（本文）

一个 Agent 网关的能力边界，几乎完全由它的扩展机制决定。OpenClaw 的扩展机制有三个设计目标贯穿始终：**按需加载**（182 个扩展不能全部启动）、**生命周期受控**（插件不能在不该动手的时候动手）、**故障隔离**（第三方代码出 bug 不能拖垮系统）。

不过在进入源码之前，先把几个高频出现、又特别容易混的概念对齐一下。

## 零、先对齐概念：Tool、MCP、Skill、Extension

| 概念 | 是什么 | 谁定义的 | 模型怎么感知到它 |
| :--- | :--- | :--- | :--- |
| **Tool**（工具） | 一个可被模型调用的函数：有名字、参数 schema、返回值 | 通用 LLM 能力（function calling） | 作为 tools 列表随请求发给模型，模型输出 `tool_use` 来调用 |
| **MCP** | Model Context Protocol，标准化"外部服务如何把工具暴露给 Agent"的协议，走 stdio 或 HTTP | 行业协议（Anthropic 于 2024 年提出并开源） | MCP server 的工具被转换成普通 Tool，和内置工具一起进 tools 列表 |
| **Skill**（技能） | 一个目录 + 一份 `SKILL.md`，用自然语言写清"做什么、何时用、怎么做" | OpenClaw（同类产品也有类似设计） | 名字 + 简介进系统提示词；模型按说明去驱动已有工具 |
| **Extension**（扩展） | 一个 TypeScript 包，可注册渠道、模型 provider、工具、钩子 | OpenClaw | 取决于它注册了什么 |
| **Context Engine** | 上下文的加工器：轮次前准备、轮次后处理、决定何时压缩 | OpenClaw | 不直接可见，但它决定模型"看到什么" |

几句话说清它们的关系：

- **Tool 是能力的最小单位**。模型唯一真正能"调用"的东西就是 Tool，其余概念最终都要落到 Tool 或落到 prompt 上。
- **MCP 是工具的进货渠道**。它解决的是"工具从哪来"：有了 MCP，接入一个外部服务不必为它写专门的扩展代码，协议对上就行。
- **Skill 不新增任何工具，它新增的是知识**——怎么用好已有的 shell、浏览器、文件读写去完成某类任务。
- **Extension 是代码层面的容器**，上面几者的落地形态都可能是某个扩展。

Skill 这一层为什么值得单独存在？因为**工具数量是有成本的**：每个工具的 schema 都要占上下文，工具越多模型选错的概率越高，而带进程的工具还要管生命周期和权限。如果某件事靠"把步骤写清楚 + 用已有工具"就能做成，那它就该是一份 Markdown，而不是一个新工具。Skill 把这类能力从代码下移到了文档。

顺着这个思路，各路能力最终只有**两个汇入点**：要么成为 tools 列表里的一项，要么成为 prompt 里的一段文字。

```text
built-in tools         --+
extension tools        --+--> tools list --> model emits tool_use --> loop runs it
mcp server tools       --+

skill index (SKILL.md) --+
memory (MEMORY.md)     --+--> prompt assembled by Context Engine
transcript + summary   --+
```

上半部分进 tools 列表，由[第 2 篇](/blog/openclaw-02-agent-loop/)讲的工具执行引擎负责跑；下半部分进 prompt，由 Context Engine 负责组装。本篇讲的就是这两条线各自的管理机制。

## 一、插件系统全景

回到代码层面。Extensions（182 个）、Skills（52 个内置）、Context Engines（可插拔）这三类扩展虽然形态差异很大，却汇入同一个插件注册表，再通过统一的 SDK 对外暴露：

```text
  Extensions (182)            Skills (52)          Context Engines
  channel / provider /      SKILL.md files      context management /
  tool / capability                              compaction strategy
          │                        │                      │
          └────────────────────────┼──────────────────────┘
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │          Plugin Registry (src/plugins/)          │
          │      Manifest Registry · Activation Planner      │
          │                  API Lifecycle                   │
          └────────────────────────┬─────────────────────────┘
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │        Plugin SDK (packages/plugin-sdk/)         │
          │  OpenClawPluginApi · Tool Registration · Events  │
          └──────────────────────────────────────────────────┘
```

注册表内部三个组件分工明确：**Manifest Registry** 管"有哪些插件"，**Activation Planner** 管"现在该加载哪些"，**API Lifecycle** 管"加载后能做什么、什么时候能做"。

## 二、Extension 解剖：以 Anthropic 为例

### 2.1 目录结构

```text
extensions/anthropic/
├── api.ts                    # ★ 公开 API 导出
├── register.runtime.ts       # 运行时注册逻辑
├── cli-backend.ts            # CLI 后端实现
├── cli-constants.ts          # 常量定义
├── cli-output.ts             # 输出解析
├── cli-shared.ts             # 共享工具函数
├── cli-tool-naming.ts        # 工具命名规范
├── stream-wrappers.ts        # 流式包装器
├── claude-model-refs.ts      # 模型引用定义
├── auth.runtime.ts           # 认证运行时
└── *.test.ts                 # 测试文件
```

### 2.2 注册方式

```typescript
// extensions/anthropic/api.ts — 公共接口
export { CLAUDE_CLI_BACKEND_ID } from "./cli-constants.js";
export { isClaudeCliProvider } from "./cli-shared.js";
export { buildAnthropicCliBackend } from "./cli-backend.js";
export { buildAnthropicProvider } from "./register.runtime.js";
export {
  createAnthropicBetaHeadersWrapper,
  createAnthropicFastModeWrapper,
  wrapAnthropicProviderStream,
} from "./stream-wrappers.js";

// extensions/anthropic/register.runtime.ts — Provider 注册
export function buildAnthropicProvider(config: ProviderConfig) {
  return {
    id: "anthropic",
    name: "Anthropic",
    models: [
      { id: "claude-sonnet-4-20250514", ... },
      { id: "claude-opus-4-20250918", ... },
    ],
    // 创建推理流
    createStream: (params) => wrapAnthropicProviderStream(
      createAnthropicStream(params),
      config
    ),
  };
}
```

所有扩展都遵循同一个约定：**`api.ts` 做门面，只 re-export；真正的实现分散在各个文件里**。外部代码只依赖 `api.ts`，扩展内部怎么重构都不影响调用方。文件名里的 `.runtime.ts` 后缀也有讲究——从命名看，它标识的是运行时才需要的部分（注册、认证），这和下一节的惰性激活是配套的。

`stream-wrappers.ts` 展示了另一个常用手法：**在 provider 原始流外面一层层套包装器**。Beta 请求头、Fast Mode 这类 provider 特有的开关，不改核心流逻辑，而是以装饰器的形式叠加上去。

## 三、Plugin Registry：插件的大脑

### 3.1 激活规划器：按需加载

182 个扩展如果在启动时全部加载，启动慢、内存大、攻击面也大。`activation-planner.ts` 的思路是：**每个插件在清单里声明自己关心的触发条件，只有条件满足时才激活**。

```typescript
// src/plugins/activation-planner.ts
// 根据触发条件动态决定加载哪些插件

type PluginActivationPlannerTrigger =
  | { kind: "command"; command: string }      // 用户执行命令
  | { kind: "provider"; provider: string }     // 使用某个模型
  | { kind: "agentHarness"; runtime: string }  // Agent 启动
  | { kind: "channel"; channel: string }       // 消息来自某渠道
  | { kind: "route"; route: string }           // 路由匹配
  | { kind: "capability"; capability: string } // 需要某能力

// 激活流程:
// 1. 加载所有 manifest (清单文件)
// 2. 匹配 trigger 条件
// 3. 检查 trust policy (信任策略)
// 4. 返回需要激活的 plugin id 列表

export function resolveManifestActivationPluginIds(
  params: ResolveManifestActivationPlanParams,
): string[] {
  const plugins = loadPluginManifestRegistry(params);
  const matched: string[] = [];

  for (const plugin of plugins) {
    // 检查 manifest 是否声明了匹配的触发条件
    if (matchesTrigger(plugin.manifest, params.trigger)) {
      // 检查信任策略
      if (passesManifestOwnerBasePolicy(plugin)) {
        matched.push(plugin.id);
      }
    }
  }
  return matched;
}
```

六种触发条件覆盖了插件被需要的所有时机：用户敲了某个命令、选了某个模型、某种 Agent 运行时启动、消息来自某个渠道、命中某条路由、需要某种能力。举个例子，只有当一条消息真的来自 Telegram 时，Telegram 扩展才会被激活；只有用户选了 Claude，Anthropic 扩展才会被加载。

注意这里的两段式判断：**清单匹配**之后还有一道**信任策略**（`passesManifestOwnerBasePolicy`）。即便插件声明了匹配的触发条件，不满足信任策略也不会被激活。清单只是插件的"自我声明"，系统不会无条件相信。

### 3.2 Plugin API 生命周期：注册期结束就收权

插件拿到的 `OpenClawPluginApi` 不是一直全功能可用的。`api-lifecycle.ts` 用一个 `Proxy` 实现了"注册期 / 运行期"的分界：

```typescript
// src/plugins/api-lifecycle.ts
// 插件 API 有严格的生命周期管控

function createGuardedPluginRegistrationApi(api: OpenClawPluginApi) {
  let closed = false;

  // 注册期结束后仍可调用的方法
  const LATE_CALLABLE_METHODS = new Set([
    "clearRunContext",
    "emitAgentEvent",
    "enqueueNextTurnInjection",
    "steerActiveRun",         // ← 可在运行时注入转向信号
    "getRunContext",
    "inlineView",
    "resolvePath",
    "sendSessionAttachment",
    "scheduleSessionTurn",
    "setRunContext",
  ]);

  return {
    api: new Proxy(api, {
      get(target, prop) {
        if (closed && !LATE_CALLABLE_METHODS.has(prop)) {
          return undefined; // 注册期结束后，非白名单方法不可用
        }
        return target[prop];
      },
    }),
    close: () => { closed = true; },
  };
}
```

插件在注册期可以调用全部 API（注册工具、挂钩子、声明能力……）；注册期一结束，`close()` 被调用，之后只有白名单里的方法还能用。

白名单里的方法大致分三类：

| 类别 | 方法 | 用途 |
| :--- | :--- | :--- |
| 运行上下文 | `getRunContext` / `setRunContext` / `clearRunContext` | 读写当前运行的插件状态 |
| 干预运行 | `steerActiveRun` / `enqueueNextTurnInjection` / `scheduleSessionTurn` | 转向当前运行、给下一轮注入内容、调度新的一轮 |
| 输出 | `emitAgentEvent` / `inlineView` / `sendSessionAttachment` / `resolvePath` | 发事件、渲染内联视图、发附件、解析路径 |

为什么要这样设计？因为"注册"类操作（比如新增工具）如果在运行中途发生，会让系统状态不一致——Agent 这一轮的工具列表已经交给 LLM 了，中途多出一个工具，模型并不知道。把注册类操作锁死在注册期，既避免了这类状态漂移，也防止插件在运行期不断注册新资源造成泄漏。

`steerActiveRun` 就是[第 2 篇](/blog/openclaw-02-agent-loop/)里 Steering 机制的插件侧入口：插件可以在运行中途向 Agent Loop 注入 `continue` / `skip` / `abort` 信号。

## 四、Skill 系统：用 Markdown 扩展能力

### 4.1 发现与加载

```text
skills/                              # 52 个内置技能
├── browser-operation/SKILL.md       # 浏览器自动化
├── diagram-maker/SKILL.md           # 图表生成
├── edge-browser/SKILL.md            # Edge 浏览器搜索
├── healthcheck/SKILL.md             # 健康检查
├── meme-maker/SKILL.md              # Meme 生成
├── node-connect/SKILL.md            # 节点配对
├── server-deploy/SKILL.md           # 服务部署
├── skill-creator/SKILL.md           # Skill 自创建
└── spike/SKILL.md                   # 快速原型验证

.openclaw/skills/                    # 用户安装的技能
├── weather/SKILL.md
├── westock-data/SKILL.md
└── ...
```

Skill 和 Extension 的根本区别：**Extension 是代码，Skill 是文档**。一个 Skill 就是一个目录加一份 `SKILL.md`，用 frontmatter 声明元信息，正文写清楚"这个技能做什么、什么时候用、怎么用"。Agent 读到这些说明后，用已有的工具（shell、浏览器、文件读写）去完成任务。

这意味着扩展 Agent 能力的门槛从"会写 TypeScript 插件"降到了"会写 Markdown"。内置的 `skill-creator` 甚至允许 Agent 自己写新的 Skill——能力扩展本身也成了 Agent 可以完成的任务。

### 4.2 运行时快照

```typescript
// src/skills/runtime/session-snapshot.ts
// 每次 Agent 启动时，扫描并构建 Skill 快照

export type SkillSnapshot = {
  prompt: string;           // 注入到系统提示词的技能描述
  skills: SkillEntry[];     // 技能列表
  skillFilter?: string[];   // 过滤条件
  version: number;          // 快照版本（缓存刷新用）
};
```

快照的构建流程：

1. 扫描 workspace 的 `skills/` 和 `.openclaw/skills/`；
2. 读取每个 `SKILL.md` 的 frontmatter；
3. 根据 Agent 配置里的 `skillFilter` 过滤；
4. 生成统一的 prompt 片段，注入系统提示词；
5. 缓存快照，避免每次都扫描文件系统。

注意第 2 步只读 frontmatter，不读正文。系统提示词里放的是技能的**索引**（名字 + 简介），而不是全部内容——50 多个技能的完整说明全塞进系统提示词，光这一项就能吃掉大量上下文。`skillFilter` 则让不同 Agent 看到不同的技能子集：写代码的 Agent 不需要知道 meme 生成器的存在。

### 4.3 Cron 场景的快照刷新

```typescript
// src/skills/runtime/cron-snapshot.ts
// 定时任务需要独立的 skill 快照

export async function resolveCronSkillsSnapshot(params: {
  workspaceDir: string;
  config: OpenClawConfig;
  agentId: string;
  existingSnapshot?: SkillSnapshot;
}): Promise<SkillSnapshot> {
  const skillFilter = resolveEffectiveAgentSkillFilter(params.config, params.agentId);

  // 关键: 如果 filter 变了，必须重新构建快照
  if (filterChanged(params.existingSnapshot?.skillFilter, skillFilter)) {
    return refreshSnapshot(params);
  }

  return params.existingSnapshot ?? { prompt: "", skills: [] };
}
```

定时任务为什么要单独处理？因为它是在没有用户交互的情况下被唤醒的，距离上一次构建快照可能已经过去很久，期间 Agent 的配置可能已被修改。如果直接复用旧快照，定时任务就会用过期的技能集运行——比如管理员已经禁用了某个技能，定时任务却还能调用它。所以每次 Cron 触发都要核对 `skillFilter` 是否变化，变了就重建。

## 五、Context Engine：可插拔的上下文管理

### 5.1 注册表

```typescript
// src/context-engine/registry.ts
// Context Engine 是插件化的，可以替换默认实现

export function registerContextEngineForOwner(
  owner: string,
  engine: ContextEngine,
): void;

export function resolveContextEngine(
  config: OpenClawConfig,
): ContextEngine;

// 查询当前轮次应使用的 Context Engine
export function resolveLogicalTurnContextEngines(
  config: OpenClawConfig,
): ContextEngine[];
```

注意 `resolveLogicalTurnContextEngines` 返回的是一个数组——一个逻辑轮次可以有多个 Context Engine 参与。

### 5.2 接口：三个钩子

```typescript
// src/context-engine/types.ts
export interface ContextEngine {
  info: { id: string; name: string };

  // 在 Agent 轮次开始前准备上下文
  prepareContext?: (params: {
    messages: AgentMessage[];
    systemPrompt: string;
    config: OpenClawConfig;
  }) => Promise<PreparedContext>;

  // 在 Agent 轮次结束后处理结果
  processResult?: (params: {
    messages: AgentMessage[];
    newMessages: AgentMessage[];
  }) => Promise<void>;

  // 决定是否需要 compaction
  shouldCompact?: (params: {
    messages: AgentMessage[];
    tokenBudget: number;
  }) => boolean;
}
```

三个钩子全是可选的，对应 Agent 运行的三个时机：

- **`prepareContext`**（轮次前）：拿到历史消息和系统提示，返回加工后的上下文——检索增强、记忆注入、历史重排都在这里做；
- **`processResult`**（轮次后）：拿到本轮新增消息——更新长期记忆、写外部索引、统计分析；
- **`shouldCompact`**（判断时机）：自定义压缩触发策略，替代默认的"超预算就压"。

**对做记忆、上下文管理、RAG 方向的研究员来说，这是整个系统里最干净的实验插入点**。不用改 Agent Loop，不用改网关，实现这三个钩子中的任意几个、注册进去即可。

### 5.3 隔离与检疫

```typescript
// src/context-engine/registry-quarantine.ts
// 有问题的 Context Engine 会被隔离

// 如果某个 engine 在运行时出错:
// 1. 记录错误到 quarantine 列表
// 2. 回退到 LegacyContextEngine（默认实现）
// 3. 不影响其他 session 的运行
```

这一层保证了：即便第三方 Context Engine 有 bug，系统也不会崩，而是悄悄回退到默认实现继续服务。配合上面说的"实验插入点"，意味着你可以把一个实验性的 engine 挂到线上，最坏情况也只是退化为默认行为——实验成本被压得很低。

## 六、MCP 集成：双向桥

### 6.1 桥接架构

```text
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  Agent   │────▶│  MCP Bridge  │────▶│  MCP Server  │
│ (Tool    │     │  (src/mcp/)  │     │ (stdio/HTTP) │
│  Call)   │◀────│              │◀────│              │
└──────────┘     └──────────────┘     └──────────────┘
```

```text
src/mcp/
├── channel-bridge.ts            # 渠道 → MCP 桥接
├── channel-server.ts            # OpenClaw 作为 MCP Server
├── openclaw-tools-serve.ts      # OpenClaw 工具暴露为 MCP
├── plugin-tools-handlers.ts     # 插件工具 → MCP 工具
├── plugin-tools-registration.ts # MCP 工具注册
└── plugin-tools-serve.ts        # 工具服务化
```

### 6.2 既是 Client，也是 Server

```typescript
// 作为 MCP Client: 调用外部 MCP Server 的工具
// src/mcp/plugin-tools-registration.ts
export async function registerMcpTools(
  serverUri: string,
  transport: "stdio" | "http",
): Promise<Tool[]> {
  const client = createMcpClient(serverUri, transport);
  const tools = await client.listTools();
  return tools.map(mcpToolToOpenClawTool);
}

// 作为 MCP Server: 将 OpenClaw 工具暴露给外部
// src/mcp/openclaw-tools-serve.ts
export function serveOpenClawToolsAsMcp(config: {
  tools: Tool[];
  port: number;
}): McpServer {
  return createMcpServer({
    tools: config.tools.map(openclawToolToMcpTool),
    port: config.port,
  });
}
```

两个方向各有价值：

- **作为 Client**：任何 MCP Server 的工具都能通过 `mcpToolToOpenClawTool` 转换后直接给 Agent 用，OpenClaw 不需要为每个外部工具写专门的扩展；
- **作为 Server**：OpenClaw 自己的工具（包括插件注册的工具）通过 `openclawToolToMcpTool` 转换后暴露出去，IDE、其他 Agent 框架等任何 MCP 客户端都能调用。

双向桥让 OpenClaw 不是一个封闭生态，而是 MCP 工具网络中的一个节点。

## 七、工具结果中间件

```typescript
// src/plugins/agent-tool-result-middleware.ts
// 工具结果在返回给 Agent Loop 之前，可以被中间件处理

export interface AgentToolResultMiddleware {
  // 在工具结果返回给 LLM 之前调用
  transform(params: {
    toolName: string;
    toolResult: ToolResultMessage;
    context: MiddlewareContext;
  }): Promise<ToolResultMessage>;
}
```

中间件夹在"工具执行完"和"结果交给 LLM"之间，典型用途有四种：

1. **敏感信息脱敏**（tool-result-redaction）：工具读到的 API key、密码、个人信息，在进入模型上下文前抹掉；
2. **结果截断**：一个 `cat` 吐出几万行日志，不截断就会直接撑爆上下文；
3. **格式标准化**：不同工具的返回格式统一成模型更容易消费的形式；
4. **缓存注入**：命中缓存时直接返回缓存结果。

和第 2 篇的 `beforeToolCall` 钩子放在一起看：`beforeToolCall` 管**进**（能不能调、要不要拦），中间件管**出**（结果怎么加工）。工具调用的两端都有了统一的扩展点。

## 八、Agent Harness：从配置到运行

### 8.1 MCP Bundle 管理

```typescript
// src/agents/agent-bundle-mcp-harness.ts
// 每个 Agent 可以绑定多个 MCP Server

export class AgentMcpBundle {
  private servers: Map<string, McpServerInstance>;

  // 启动 Agent 时，初始化所有配置的 MCP Server
  async initialize(config: AgentConfig): Promise<void> {
    for (const serverConfig of config.mcpServers) {
      const server = await this.launchServer(serverConfig);
      this.servers.set(serverConfig.name, server);
    }
  }

  // 获取所有可用工具
  async listTools(): Promise<Tool[]> {
    const allTools: Tool[] = [];
    for (const server of this.servers.values()) {
      const tools = await server.listTools();
      allTools.push(...tools);
    }
    return allTools;
  }

  // 清理: Agent 会话结束时关闭所有 Server
  async cleanup(): Promise<void> {
    for (const server of this.servers.values()) {
      await server.shutdown();
    }
  }
}
```

MCP Server 的生命周期和 Agent 会话绑定：会话开始时拉起、结束时关闭。stdio 型的 MCP Server 本质上是子进程，如果不跟着会话一起清理，长期运行的网关上会堆积大量僵尸进程。

### 8.2 子 Agent 的身份与权限

```typescript
// src/agents/agent-command-execution-identity-spawn.ts
// 子 Agent 生成时的身份继承

// 关键设计: 子 Agent 的权限 ≤ 父 Agent 的权限
// 通过 execution lineage 追踪完整调用链:
//   main agent → sub-agent → sub-sub-agent
// 每一级都有独立的:
//   - session key
//   - tool authority
//   - containment root
//   - cron creator authority
```

核心原则一句话：**子 Agent 的权限只能收缩，不能扩张**。主 Agent 拉起子 Agent，子 Agent 再拉起孙 Agent，每一级都有自己独立的会话、工具权限、沙箱根路径和定时任务创建权限，并且每一级都不超过上一级。整条调用链通过 execution lineage 完整追踪，审计时可以还原"谁让谁做了什么"。

列表里 `cron creator authority` 值得留意。定时任务会在未来某个时刻、在没有父 Agent 盯着的情况下执行。如果不限制子 Agent 创建定时任务的权限，就可能出现"权限延时逃逸"——子 Agent 自己受限，却留下一个以更高权限运行的定时任务。把它单独列为一项需要收缩的权限，堵的正是这个口子。

## 九、Provider 系统：模型抽象层

### 9.1 统一接口

```typescript
// src/llm/providers/ (每个模型提供商一个模块)

// Provider 实现统一接口:
interface LlmProvider {
  id: string;
  name: string;
  models: ModelRef[];

  createStream(params: {
    model: Model;
    messages: Message[];
    tools?: Tool[];
    signal?: AbortSignal;
  }): AssistantMessageEventStream;
}

// 已支持的 Provider (部分):
// anthropic, openai, google, azure, amazon-bedrock,
// deepseek, cerebras, cohere, cloudflare,
// ollama, lmstudio, vllm, ...
```

所有 provider 只需要实现一个 `createStream`，返回统一的 `AssistantMessageEventStream`。这和第 2 篇里 Agent Loop 接收的 `streamFn` 正好对上——provider 层把各家 API 的差异抹平，Agent Loop 只看到一种流。

列表里同时有商业 API（Anthropic、OpenAI、Google）、云平台（Azure、Bedrock、Cloudflare）和本地推理（Ollama、LM Studio、vLLM）。对研究员来说，vLLM 的存在意味着可以直接把自己训的模型接进来，在真实的 Agent harness 里做评测。

### 9.2 限流与容错

- `src/llm/provider-rate-limit.ts`：provider 级别的限流；
- `src/llm/provider-rate-limit-stream.ts`：流式场景的限流；
- `packages/retry/`：统一重试库，支持指数退避，并能解析 429 响应的 `Retry-After` 头。

限流在 provider 维度做，是因为配额本身就是按 provider（更准确说是按 API key）计的。重试优先遵循服务端给出的 `Retry-After`，比自己盲猜退避时间更礼貌也更高效。

## 小结

本篇的要点：

1. **激活是惰性的**：插件按 trigger 条件按需激活，且必须通过信任策略；
2. **API 有生命周期**：注册期结束后只剩白名单方法，防止状态漂移和资源泄漏；
3. **Skill 是声明式的**：写一份 `SKILL.md` 就能扩展能力，系统提示词里只放索引；
4. **Context Engine 可替换且可隔离**：三个钩子是记忆 / 上下文实验的最佳插入点，出错自动回退；
5. **MCP 是双向桥**：OpenClaw 既消费外部工具，也把自己的工具暴露出去；
6. **工具调用两端都有扩展点**：`beforeToolCall` 管进，结果中间件管出；
7. **子 Agent 权限只收缩不扩张**：包括定时任务创建权限，lineage 全链路可审计。

### 系列回顾

三篇连起来，OpenClaw 的设计可以概括为一条主线：**把"不变的核心"和"多变的外围"彻底分开**。

- 不变的核心是 `packages/agent-core/`——一个纯逻辑、依赖注入的 Agent Loop（[第 2 篇](/blog/openclaw-02-agent-loop/)）；
- 多变的外围是 182 个扩展、52 个技能、可插拔的 Context Engine 和 MCP 工具网络（本篇）；
- 把两者粘起来的是 Gateway：路由、准入、调度、隔离、持久化（[第 1 篇](/blog/openclaw-01-architecture/)）。

对算法研究员来说，这个分层给出了清晰的实验入口：改 Agent 行为看 `agent-core`，改上下文与记忆实现一个 Context Engine，接自己的模型写一个 provider 或直接走 vLLM。

## 附录：快速上手源码

```bash
# 1. 克隆
git clone --depth 1 https://github.com/openclaw/openclaw.git

# 2. 看核心算法（不需要构建）
cat packages/agent-core/src/agent-loop.ts        # Agent Loop
cat packages/agent-core/src/types.ts             # 核心类型
ls  packages/agent-core/src/harness/compaction/  # Compaction

# 3. 看架构
cat src/gateway/agent-turn/agent-run-dispatch.ts # 调度入口
cat src/routing/resolve-route.ts                 # 路由
cat src/plugins/activation-planner.ts            # 插件激活

# 4. 看某个扩展如何实现
cat extensions/anthropic/api.ts                  # Anthropic
cat extensions/openai/api.ts                     # OpenAI

# 5. 看测试理解行为
find packages/agent-core -name "*.test.ts"       # Agent 测试
find src/plugins -name "*.test.ts"               # 插件测试
```

读源码时有一个小技巧：**先看测试再看实现**。测试往往比实现更直接地回答"给定这样的输入，期望什么行为"；而 `agent-core` 的 IO 全部靠依赖注入，它的测试天然不需要真实的 LLM 和工具环境，读起来很轻量。
