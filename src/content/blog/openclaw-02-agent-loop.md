---
title: "OpenClaw 源码解析（二）：Agent Loop 与工具执行引擎"
description: "深入 packages/agent-core：一个不依赖任何 IO 的 Agent 主循环如何调 LLM、执行工具、防止死循环、在 turn 边界压缩上下文，并用统一事件流把一切暴露给上层。最后从算法研究员视角聊聊：这套 harness 对 Agent RL 的 rollout 设计意味着什么。"
pubDate: 2026-06-15
tags: ["OpenClaw", "Agent", "Agent Loop", "工具调用", "上下文压缩", "源码解读"]
---

> **本文是《OpenClaw 源码解析》系列第 2 篇**。[上一篇](/blog/openclaw-01-architecture/)从网关视角走完了一条消息的七站旅程，本篇钻进其中最核心的一站——`packages/agent-core/`。这是整个仓库里最"算法"的部分：纯逻辑、无 IO、可独立测试。
>
> 系列目录：
> 1. [整体架构与核心数据流](/blog/openclaw-01-architecture/)
> 2. **Agent Loop 与工具执行引擎**（本文）
> 3. [插件系统与扩展机制](/blog/openclaw-03-plugin-system/)

"调 LLM → 执行工具 → 回填结果 → 再调 LLM"，Agent Loop 的骨架谁都会写，十几行就够。难的是骨架之外的东西：工具并发怎么排、模型反复调错工具怎么办、上下文爆了怎么压、用户中途插话怎么接。OpenClaw 的 `agent-core` 把这些都拆成了独立的小模块，本篇逐个过一遍。

## 一、Agent Loop 住在哪里

```text
packages/agent-core/src/
├── agent-loop.ts                    # ★ 主循环入口
├── agent-loop-steering.ts           # 流式转向控制
├── agent-loop-tool-execution.ts     # 工具执行编排
├── agent-loop-tool-recovery.ts      # 工具循环恢复
├── agent-stream-response.ts         # 流式事件发射
├── tool-batch-admission.ts          # 工具批次准入
├── tool-batch-completion.ts         # 工具批次完成
├── tool-call-outcome.ts             # 单个工具调用结果
├── tool-execution-context.ts        # 工具执行上下文
├── turn-interruption.ts             # 轮次中断处理
├── turn-taint.ts                    # 轮次污染追踪
├── reasoning.ts                     # 推理模式 (thinking)
├── operator-messages.ts             # 系统更新消息
├── errors.ts                        # 错误类型定义
├── types.ts                         # ★ 核心类型
├── harness/
│   ├── compaction/
│   │   ├── compaction.ts            # ★ 上下文压缩算法
│   │   ├── compaction-cut-point.ts  # 截断点计算
│   │   ├── compaction-details.ts    # 压缩细节
│   │   └── summarization-completion.ts # 摘要生成
│   ├── session/
│   │   └── session.ts               # Session 上下文构建
│   └── messages.ts                  # 消息格式化
└── runtime-deps.ts                  # 运行时依赖注入接口
```

文件按"一个阶段一个文件"拆分：工具执行被切成准入（`tool-batch-admission`）、编排（`agent-loop-tool-execution`）、结果（`tool-call-outcome`）、完成（`tool-batch-completion`）、恢复（`agent-loop-tool-recovery`）。读的时候建议先看 `types.ts` 把类型摸清，再看 `agent-loop.ts`，其余按需跳转。

## 二、主循环

### 2.1 伪代码

```text
runAgentLoop(prompts, context, config, emit, signal):
  messages = context.messages + prompts
  recoveryState = {criticalToolLoopSeen: false}

  emit(agent_start)
  emit(turn_start)

  for prompt in prompts:
    messages.push(prompt)
    if prompt.role == "user":
      resetToolLoopRecovery(recoveryState)

  loop:                                         ← 核心循环
    response = streamLLM(messages)              ← ① 调 LLM
    assistantMessage = collectResponse(response)
    messages.push(assistantMessage)

    toolCalls = extractToolCalls(assistantMessage)
    if no toolCalls:
      break                                     ← 无工具调用，结束

    toolResults = executeTools(toolCalls)       ← ② 执行工具
    for result in toolResults:
      messages.push(result)

    advanceToolLoopRecovery(recoveryState)      ← ③ 循环恢复检查
    if recoveryState.criticalToolLoopSeen:
      break                                     ← 检测到死循环，终止

  emit(turn_end)
  emit(agent_end)
  return newMessages
```

三个值得注意的点：

- **正常终止条件只有一个：模型不再调用工具**。没有"任务完成"的显式信号，模型输出纯文本就代表这一轮结束。
- **异常终止由循环恢复机制兜底**：`criticalToolLoopSeen` 被置位时强制退出（见 3.3 节）。另外 `AgentLoopConfig` 里还有 `maxTurns` 作为硬上限。
- **用户消息会重置恢复状态**：`resetToolLoopRecovery` 只在 `role == "user"` 时调用。用户一插话，之前累积的错误计数清零——人类给了新信息，模型理应获得新的机会。

### 2.2 实际入口签名

```typescript
// packages/agent-core/src/agent-loop.ts (简化)
export async function runAgentLoop(
  prompts: AgentMessage[],
  context: AgentContext,
  config: AgentLoopConfig,
  emit: AgentEventSink,
  signal?: AbortSignal,
  streamFn?: StreamFn,
  runtime?: AgentCoreStreamRuntimeDeps,
): Promise<AgentMessage[]> {
  const newMessages: AgentMessage[] = [];
  const recoveryState = config.toolLoopRecoveryState ?? { criticalToolLoopSeen: false };
  const state = { context: { ...context, messages: [...context.messages] } };

  await emit({ type: "agent_start" });
  await emit({ type: "turn_start" });

  // 注入用户消息
  for (const prompt of inputPrompts) {
    if (config.consumeQueuedMessageCancellation?.(prompt)) continue;
    state.context.messages.push(prompt);
    newMessages.push(prompt);
    if (prompt.role === "user") {
      resetToolLoopRecovery(recoveryState);
    }
  }

  // 核心循环在 streamAgentResponse 内部实现
  // ...
}
```

签名本身就是一份设计说明：

| 参数 | 角色 |
| :--- | :--- |
| `prompts` | 本轮新输入（通常是用户消息） |
| `context` | 已有对话历史 + 系统提示 + 工具列表 + 模型 |
| `config` | 循环行为：工具执行模式、队列模式、恢复状态、最大轮数 |
| `emit` | 事件出口，所有输出都走这里 |
| `signal` | 外部中止信号 |
| `streamFn` | 怎么调 LLM——注入的，不是写死的 |
| `runtime` | 怎么执行工具、读写文件、生成子 Agent——同样是注入的 |

函数内部先把 `context.messages` 复制一份再操作，不修改调用方传入的数组；返回值只是**本轮新增**的消息。整个函数可以看成 `prompts + context + config → newMessages` 的映射，副作用全部被挤到 `emit` 和注入的依赖里。

`consumeQueuedMessageCancellation` 处理的是一个很现实的场景：Agent 正在运行时用户又发了消息，这些消息先进入队列；如果某条排队消息在被消费前已被取消，注入时就直接跳过。配合 `QueueMode`（`all` 一次注入全部 / `one-at-a-time` 逐条注入），网关可以精细控制"运行中插话"的语义。

## 三、工具执行引擎

### 3.1 两种执行模式

```typescript
// types.ts
export type ToolExecutionMode = "sequential" | "parallel";
```

| 模式 | 行为 | 适用 |
| :--- | :--- | :--- |
| `sequential` | 逐个准备 → 执行 → 完成；Steering 可跳过后续调用 | 默认模式 |
| `parallel` | 逐个准备 → 并发执行 → 全部完成后再统一发射结果 | 多个独立工具调用 |

两者的关键差异在"可干预性"和"确定性"：

- `sequential` 每执行完一个工具都有一个检查点，外部可以在这里决定跳过剩余调用——比如第一个工具的结果已经说明后面的调用没意义了；
- `parallel` 准备阶段仍是逐个的（准入检查要串行做），执行阶段并发，结果**等全部完成后再发射**。这样 transcript 里 `tool_result` 的顺序是确定的，不会因为哪个工具先返回而乱序。

### 3.2 一次工具调用的生命周期

```text
LLM 返回 Assistant Message (含 tool_use blocks)
    │
    ▼
① Prepare (agent-loop-tool-execution.ts)
    prepareToolCallExecution()
    ├── 解析工具名 + 参数
    ├── beforeToolCall() hook → 可 block
    └── 准入检查 (tool-batch-admission.ts)
    │
    ▼
② Execute
    launchParallelToolCalls() 或逐个执行
    ├── captureAgentToolInputScope()   # 捕获工具输入上下文
    ├── 实际调用工具函数
    └── 超时 / 中止检测
    │
    ▼
③ Finalize (tool-call-outcome.ts)
    finalizeToolCallOutcome()
    ├── completeStartedToolCall()      # 正常完成
    ├── completeUnstartedToolCall()    # 未启动就失败
    ├── immediateToolCallError()       # 立即错误
    └── hidesToolCallFromChannelProgress()  # 隐藏内部工具
    │
    ▼
④ Emit Results (agent-stream-response.ts)
    emitToolResultMessage()
    ├── tool_execution_start 事件
    ├── tool_execution_end 事件
    └── ToolResultMessage 写入 transcript
    │
    ▼
⑤ Taint Check (turn-taint.ts)
    toolResultTaintsTurn()
    └── 某些工具结果会"污染"当前轮次（需要 LLM 重新评估）
    │
    ▼
⑥ Loop Recovery (agent-loop-tool-recovery.ts)
    advanceToolLoopRecovery()
    ├── 检测重复工具调用模式
    ├── 累积相同工具错误计数
    └── 超过阈值 → criticalToolLoopSeen = true → 终止循环
```

几处设计值得展开：

**Prepare 阶段的 `beforeToolCall` 钩子**。返回 `BeforeToolCallResult { block?, reason? }`，`block: true` 时工具不会真正执行。这是权限审批、策略拦截、插件干预的统一入口——上层不需要改 Agent Loop，只要挂一个钩子。

**Finalize 区分"已启动"和"未启动"**。`completeStartedToolCall` 与 `completeUnstartedToolCall` 分开处理，背后的考量是副作用：未启动就失败的调用（参数解析失败、被 block、准入不通过）一定没有副作用；已启动的调用即便失败，也可能已经写了文件、发了请求。两类失败在恢复策略和审计上的含义完全不同。

**`hidesToolCallFromChannelProgress`**。用户在 Telegram 里会看到"正在搜索……"这样的进度提示，但有些内部工具（簿记、状态同步之类）不该打扰用户，这个函数决定哪些调用不出现在渠道进度里。

**Taint Check**。`toolResultTaintsTurn` 判断某个工具结果是否"污染"了当前轮次，需要 LLM 重新评估。可以想象的场景包括：工具返回了来自外部的不可信内容（潜在的 prompt injection），或者工具改变了环境状态，使模型之前的计划失效。把"污染"作为一个显式的、可追踪的状态，比让模型自己意识到要可靠得多。

### 3.3 循环恢复：防止 Agent 陷入死循环

这是线上 Agent 最常见的失败模式之一：模型用同样的错误参数反复调用同一个工具，每次都失败，每次都"再试一次"，直到烧光 token 预算。

```typescript
// agent-loop-tool-recovery.ts (逻辑简化)
export function advanceToolLoopRecovery(state: ToolLoopRecoveryState): void {
  // 检测连续失败的相同工具调用
  if (isRepeatedToolError(state.lastToolCall, currentToolCall)) {
    state.repeatedErrorCount++;
  }

  // 超过阈值 → 标记为关键循环
  if (state.repeatedErrorCount >= MAX_REPEATED_TOOL_ERRORS) {
    state.criticalToolLoopSeen = true;
    appendToolLoopRecoveryNote(state,
      "Repeated tool error detected, terminating loop"
    );
  }
}

// 错误码: REPEATED_TOOL_ERROR_CODE = "repeated_tool_error"
// 消息: "OpenClaw stopped this run because tool-loop recovery
//        encountered another critical loop."
```

逻辑很朴素：识别"相同工具 + 重复错误"的模式，计数超过阈值就置位 `criticalToolLoopSeen`，主循环检测到后退出，并以 `repeated_tool_error` 错误码结束运行。配合 2.1 节说的"用户消息重置计数"，形成一个闭环：**模型自己卡住时系统替它刹车，人类给出新信息后再放开**。

## 四、Compaction：上下文压缩

### 4.1 问题与策略

LLM 的上下文窗口有限（比如 200K tokens），长对话迟早超限。OpenClaw 的策略是**保留最近几轮原文 + 把更早的历史压缩成摘要**。两个关键常量：

- `MAX_COMPACTION_SUMMARY_CHARS = 16,000`：摘要的最大长度；
- `CHARS_PER_TOKEN_ESTIMATE`：字符到 token 的估算系数，对 CJK 单独处理——中文一个字通常就占一个甚至更多 token，用英文的"约 4 字符 / token"去估会严重低估。

### 4.2 压缩算法

```typescript
// packages/agent-core/src/harness/compaction/compaction.ts (简化)
export async function compact(params: {
  messages: AgentMessage[];
  tokenBudget: number;
  model: Model;
  streamFn: StreamFn;
}): Promise<CompactionResult> {
  // 1. 估算当前 token 数
  const currentTokens = estimateContextTokens(params.messages);
  if (currentTokens <= params.tokenBudget) {
    return null; // 不需要压缩
  }

  // 2. 找到截断点: 保留最近的消息，压缩更早的
  const cutPoint = findCutPoint(params.messages, params.tokenBudget);

  // 3. 将被截断的历史序列化
  const historyText = serializeConversation(
    params.messages.slice(0, cutPoint)
  );

  // 4. 调用 LLM 生成摘要
  const summary = await generateSummary({
    history: historyText,
    latestUserRequest: extractLatestUserRequest(params.messages),
    model: params.model,
    streamFn: params.streamFn,
  });

  // 5. 截断摘要到安全长度
  const finalSummary = capCompactionSummary(summary);

  return {
    summary: finalSummary,
    firstKeptEntryId: params.messages[cutPoint].id,
    tokensBefore: currentTokens,
  };
}
```

最容易被忽略的是第 4 步的 `latestUserRequest` 参数：**摘要生成是以当前用户请求为条件的**。这不是一个泛泛的"总结一下历史"，而是"围绕用户现在要做的事，总结历史里有用的部分"。同样一段历史，用户在问部署问题和在问代码问题时，应该保留的信息是不同的。

返回值里的 `firstKeptEntryId` 是一个指针而不是一份拷贝——它标记了原文保留的起点。`tokensBefore` 则留给可观测性：每次压缩前有多少 token，事后一查便知。

### 4.3 截断点：必须落在 turn 边界

```typescript
// compaction-cut-point.ts (核心逻辑)
export function findCutPoint(
  messages: AgentMessage[],
  tokenBudget: number,
): number {
  // 从后往前累加 token，找到第一个超预算的位置
  let accumulated = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    accumulated += estimateTokens(messages[i]);
    if (accumulated > tokenBudget) {
      // 回退到 turn 边界（不在 tool call 中间截断）
      return findTurnStartIndex(messages, i + 1);
    }
  }
  return 0;
}
```

用公式写，先找到能放进预算的最长后缀起点：

$$
k^{\star} = \min\Big\{\,k \;:\; \sum_{j=k}^{n-1} \mathrm{tok}(m_j) \le B \Big\}
$$

其中 $m_j$ 是第 $j$ 条消息、$B$ 是 token 预算。再把 $k^{\star}$ 对齐到 turn 的起点。这里的 turn 指一个完整的"user → assistant（+ 若干 tool call / tool result）"循环。

**为什么不能在 turn 中间截断？** 最直接的原因是 API 约束：主流 provider 要求每个 `tool_result` 都必须能找到对应的 `tool_use`，配对断开请求会直接报错。更深一层的原因是语义完整性——把一次工具调用的"问"放进摘要、"答"留在原文，模型会看到一个莫名其妙出现的工具结果。

## 五、Steering：流式转向控制

```typescript
// agent-loop-steering.ts
// Steering 允许在 LLM 流式输出过程中动态修改行为

export function createStreamedSteeringConfig(): SteeringConfig {
  return {
    // 在 sequential 模式下，第一个工具执行后可跳过剩余工具
    skipRemainingToolCalls: false,

    // 检查点：允许外部逻辑（如用户中断）注入控制信号
    checkpoints: [],
  };
}

export function getSteeringAtCheckpoint(
  config: SteeringConfig,
  checkpoint: string,
): SteeringDecision {
  // 返回: continue | skip | abort
}
```

Steering 的本质是**在循环的若干检查点上开一扇窗**，让外部逻辑决定 `continue`（继续）、`skip`（跳过剩余工具调用）还是 `abort`（终止）。外部是谁？可以是用户的中断操作，也可以是插件——[第 3 篇](/blog/openclaw-03-plugin-system/)会看到插件 API 里专门有一个运行期可调用的 `steerActiveRun` 方法。

这也解释了为什么 `sequential` 是默认模式：只有串行执行，"第一个工具执行完、第二个还没开始"这样的检查点才存在。

## 六、运行时依赖注入

```typescript
// runtime-deps.ts — Agent Core 不直接依赖 IO
// 所有外部能力通过 deps 注入

export interface AgentCoreStreamRuntimeDeps {
  // 工具执行
  executeTool: (call: AgentToolCall) => Promise<ToolResultMessage>;

  // 文件操作
  readFile: (path: string) => Promise<string>;
  writeFile: (path: string, content: string) => Promise<void>;

  // 子 Agent 生成
  spawnSubAgent: (params: SpawnParams) => Promise<SpawnResult>;

  // 审批
  requestApproval: (action: string) => Promise<boolean>;

  // ...其他能力
}
```

工具怎么执行、文件怎么读写、子 Agent 怎么拉起、审批找谁要——`agent-core` 一概不知道，全部由调用方注入。结果是 Agent Loop **可以用任意 LLM provider、任意工具集、任意存储后端运行**。

对算法研究员来说这几乎是理想的实验条件：

- 注入一个读本地 JSONL 的 `streamFn`，就能离线回放历史轨迹；
- 注入一个 mock 的 `executeTool`，就能在不碰真实环境的情况下测试循环恢复、压缩等逻辑；
- 注入一个总是返回 `true` 的 `requestApproval`，就能在沙箱里跑批量 rollout。

## 七、事件流模型

```typescript
// agent-stream-response.ts 定义的事件类型:
type AgentEvent =
  | { type: "agent_start" }
  | { type: "agent_end" }
  | { type: "turn_start" }
  | { type: "turn_end" }
  | { type: "message_start"; message: AgentMessage }
  | { type: "message_delta"; delta: string }        // 流式文本
  | { type: "message_end"; message: AgentMessage }
  | { type: "tool_execution_start"; toolCall: AgentToolCall }
  | { type: "tool_execution_end"; result: ToolResultMessage }
  | { type: "thinking_start" }                       // 推理开始
  | { type: "thinking_delta"; delta: string }
  | { type: "thinking_end" }
  | { type: "error"; error: Error };
```

事件时序：

```text
agent_start → turn_start → message_start →
  [message_delta]* → [thinking_*] →
  tool_execution_start → tool_execution_end →
  message_end → [loop back to message_start] →
turn_end → agent_end
```

Agent Loop 的所有行为都通过 `AgentEventSink` 这一个出口发射，**渠道无关**。Discord 扩展可以把 `message_delta` 攒成一条消息再发，TUI 可以逐字渲染，Web 客户端可以把 `thinking_*` 折叠成一个"思考中"面板——它们消费的是同一条事件流。

`thinking_*` 与 `message_*` 是分开的事件类型，这让"推理过程要不要展示给用户"成了渠道层的决定，而不是 Agent Loop 的决定。

## 八、核心类型速查

```typescript
// types.ts (关键类型)
interface AgentMessage {
  role: "user" | "assistant" | "system" | "tool_result";
  content: string | ContentBlock[];
  timestamp?: number;
}

interface AgentContext {
  messages: AgentMessage[];           // 完整对话历史
  systemPrompt?: string;              // 系统提示词
  tools?: Tool[];                     // 可用工具列表
  model?: Model;                      // 当前模型
}

interface AgentLoopConfig {
  toolExecutionMode?: ToolExecutionMode;  // sequential | parallel
  queueMode?: QueueMode;                  // all | one-at-a-time
  toolLoopRecoveryState?: ToolLoopRecoveryState;
  maxTurns?: number;
  consumeQueuedMessageCancellation?: (msg: AgentMessage) => boolean;
}

interface AgentToolCall {
  type: "toolCall";
  toolCallId: string;
  toolName: string;
  args: Record<string, unknown>;
}

interface BeforeToolCallResult {
  block?: boolean;     // true → 阻止执行
  reason?: string;     // 阻止原因
}
```

## 九、算法研究员视角：这套 harness 对 Agent RL 意味着什么

读完 `agent-core`，回到开篇那个问题：训练环境和线上 harness 不一致怎么办？下面是我自己的几点思考。

**1. 压缩应该进入训练分布。** 线上 Agent 跑长任务时，上下文会变成"摘要 + 最近几轮原文"的形态。如果训练 rollout 从不触发压缩，模型就没见过这种输入，上线后在长任务的后半程表现会打折。一个直接的做法是在 rollout 里复用同一套 `compact()` 逻辑，用较小的 `tokenBudget` 主动触发。

**2. 循环恢复是天然的截断条件。** `criticalToolLoopSeen` 被置位的轨迹，本质上就是"模型卡死了"。在 RL 里，这类轨迹可以直接给负奖励，或者作为过程奖励的信号源——比如对"重复相同错误调用"的步骤单独惩罚。这和长轨迹的信用分配问题直接相关，可以参考我之前写的[《长轨迹 Agent RL 的信用分配：从 GRPO 到 GiGPO》](/blog/agent-rl-credit-assignment-gigpo/)。

**3. 事件流就是轨迹日志。** `message_*`、`tool_execution_*`、`thinking_*` 事件按时序排开，就是一条完整轨迹。其中 `tool_result` 是环境注入的，训练时要从 loss 中 mask 掉——这一点和[《slime 学习笔记（四）》](/blog/slime-04-search-agent-rl/)里 search agent 的 `<information>` 块是同一个道理。

**4. 依赖注入让"同一套循环跑训练和线上"成为可能。** 只要把 `streamFn` 换成训练框架的推理引擎、把 `executeTool` 换成沙箱环境，rollout 用的就是线上同一份 Agent Loop 代码。训练—部署一致性问题，在代码层面就消掉了一大半。

## 小结

1. **Agent Loop 是近似纯函数**：`prompts + context + config → newMessages`，所有 IO 通过依赖注入，副作用集中在事件出口；
2. **工具执行是六阶段流水线**：Prepare → Execute → Finalize → Emit → Taint Check → Recovery，`beforeToolCall` 钩子是统一的拦截点；
3. **循环恢复是安全网**：识别重复工具错误，超阈值终止；用户插话时重置；
4. **Compaction 以当前请求为条件、在 turn 边界截断**：不破坏 `tool_call` / `tool_result` 配对，摘要有长度上限，原文用指针标记保留起点；
5. **Steering 在检查点上开窗**：外部可注入 `continue` / `skip` / `abort`，这也是 `sequential` 成为默认模式的原因；
6. **事件流是统一输出接口**：渠道无关，推理过程与正文分离。

下一篇 [《插件系统与扩展机制》](/blog/openclaw-03-plugin-system/) 是系列收尾：182 个扩展如何按需激活、插件 API 为什么有生命周期、SKILL.md 怎么变成系统提示词、出问题的 Context Engine 如何被隔离，以及子 Agent 的权限为什么只能收缩不能扩张。
