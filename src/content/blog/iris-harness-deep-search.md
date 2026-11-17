---
title: "Iris-Harness：一个为深度搜索定制的 Harness"
description: "拆解 Iris-Harness 源码：带韧性的 ReAct 主循环、两个 Serper 工具、三套可组合的上下文管理（discard-all / retry / recency-K）、append-only 的对话日志，以及四个 benchmark 的判分与复现协议。"
pubDate: 2026-10-12
tags: ["Iris-Harness", "Harness Engineering", "深度搜索", "Agent 评测", "上下文管理", "BrowseComp", "HLE", "ReAct"]
---

> 一个公开的 benchmark 分数，既属于 agent，也属于跑它的 harness。turn 上限、上下文处理方式、判分器——任意一项都能让同一个模型的分数挪动好几个点。Iris-Harness 把这套「容易被忽略但决定成败」的工程细节完整开源，让 Iris report 里的每个数字都可被复查，也让别的系统能在同一标尺上被衡量。

本文基于 `Iris-Harness` 源码，拆解它的架构、ReAct 主循环、两个工具、三套上下文管理策略，以及四个 benchmark 的判分与复现协议。

---

## 1. 定位：harness 不是配角

Iris-Harness 是一个**单 agent 的 ReAct 评测框架**，只有两个工具：`web_search` 与 `scrape_website`。它刻意不内置模型——你自己用 SGLang / vLLM / 任意 OpenAI 兼容服务把权重跑起来，harness 只负责把 `--base-url` 指过去并驱动评测。

它想证明一件事：**同一个 agent 能力，换一套 harness，分数就不一样**。所以它把三件最容易「偷分」的事做成了可独立开关、可审计的模块：

- 上下文如何在长搜索中被裁剪 / 丢弃 / 重试；
- 答案如何被解析、如何被判对错；
- 失败如何被回滚、重试、兜底。

技术栈：Python ≥ 3.12 + Hydra 配置 + MCP 工具 + `uv` 管理。统一口径：**256K 上下文窗口、600 turn 上限、一套工具、一个判分器**，跨模型跨策略全程固定。

---

## 2. 主循环：一个带「韧性」的 ReAct

先看全景——一道题从加载到被判分，以及三套上下文管理分别在哪里介入：

```text
① 加载标准化题目（JSONL: task_id / task_question / ground_truth / metadata）
      │
② 并行 worker 跑单题（命中缓存轨迹则跳过推理）
      │
③ ReAct 主循环：搜索 → 抓取摘要 → 再判断（≤ 600 turn） ◀────────┐
      │                                                        │
④ Final Summary：裁剪到能塞进窗口，再要 \boxed{} 答案            │
      │                                                        │
⑤ 答案能解析出来?                                               │
      ├─ 不能，且还有 retry 预算                                 │
      │     → 带上「已排除了什么」的失败摘要，从头重跑这道题 ──────┘
      └─ 能 / retry 预算已耗尽
      │
⑥ self-verification（默认关）：verify → reanswer → re-verify
      │
⑦ LLM Judge 判对错
      │
⑧ pass@k 聚合 / ave@M 平均 → accuracy.txt + 终止分析
```

再放大第 ③ 步，看单轮内部怎么走、上下文管理在哪里介入：

```text
每一轮 turn:

① LLM 调用
     【recency-K】只把最近 K 个工具结果原文发给模型，更早的折叠成占位符
     │
② 这轮要调工具吗?
     ├─ 格式错 / 拒答 / 重复 query / 工具报错
     │     → 回滚本轮，重来（连续上限 5 次）→ 回到 ①
     ├─ 不要（自然结束）
     │     → 跳出循环 → Final Summary
     └─ 要
           → web_search / scrape_website
             （抓取后强制 LLM 摘要，绝不回传原文）
     │
③ 结果写回对话 + 落盘
     │
④ prompt 超过 discard 阈值?
     └─ 是 →【discard-all】对话重置为最初的问题
              （full 日志记 marker，轨迹不丢）→ 回到 ①
     │
⑤ 预判下一轮会撑爆 256K?
     ├─ 会 → 强制收尾 → Final Summary
     └─ 不会 → 回到 ①
```

两图中带【】的三处，正是第 4 节要展开的上下文管理：**recency-K** 在发送时折叠、**discard-all** 在超阈值时重置、**retry** 在缺答时携经验重跑。

入口链路很直白：`execute_task_pipeline`（`src/core/pipeline.py`）→ `Orchestrator.run_main_agent`（`src/core/orchestrator.py`）。每一轮做的事：

```text
LLM 调用 → 解析 tool_calls → 执行工具 → 回填结果 → 上下文检查 → 下一轮
```

当某轮不再请求工具（或触发结束）时跳出循环，进入 **final summary**，用 `\boxed{}` 产出最终答案。支持两种工具调用协议：`native_fc`（原生 function calling）与 `mcp_xml`（XML 标签），通过 `--tool-call-mode` 切换，且必须与模型训练方式一致。

真正体现「harness 决定分数」的，是循环里密布的**韧性逻辑**：

| 场景 | 处理 | 目的 |
| --- | --- | --- |
| 回复里混入 MCP 标签 / 出现拒答关键词 | pop 掉本轮 assistant，回滚重试 | 纠正格式与拒答，连续回滚上限 5 次 |
| 重复 query（同一工具同一检索词） | 回滚并提示换问法 | 避免空转 |
| 工具结果报错 | 回滚该轮 | 不让坏结果污染上下文 |
| 单次响应被 `max_tokens` 截断 | 以 ×1.5 放大预算重发，上限 5 次 | 区别于「瞬时错误重试」，专治 length 截断 |
| 硬上下文溢出（provider 返回 400） | **不盲目重发**（否则会 400 到死），强制结束去 summary | `is_context_length_error` 识别各家报错措辞 |

---

## 3. 两个工具：一个 Serper 后端，强制摘要

两个工具由单个 Serper MCP server（`searching_serper_mcp_server.py`）提供，全链路 `httpx.AsyncClient` 异步、连接池复用：

- **`web_search`** → Serper `/search`（Google SERP）。归一化 organic 结果为统一 schema，带指数退避重试（1/2/4/8s），并**过滤掉 HuggingFace dataset/space 链接**以防 benchmark 答案泄漏。
- **`scrape_website`** → Serper Scrape 为主、纯 `httpx` GET 为备，抓取后**强制走一次 LLM 抽取摘要**（针对调用方的 `info_to_extract`），绝不回传原始网页。工程细节堆得很满：
  - 内容上限 `MAX_SCRAPE_CHARS`（默认 ~400K 字符）先截断；
  - 摘要 LLM 支持 openai / azure / gemini / maas 四种 provider 自动识别；
  - **上下文溢出自动截断重试**（每次多砍 40960×n 字符）；
  - **复读检测**（末 50 字符重复 >5 次判退化，重试）；
  - 摘要彻底失败时**兜底回传截断原文**，不让 agent 空手而归。

「抓取必摘要」这个设计，等于把一层信息压缩固化进 harness——它直接影响 agent 看到多少噪声，因而也影响分数。

---

## 4. 核心看点：三套可组合的上下文管理

长程搜索会在难题解决前耗尽上下文，所以每个严肃系统都得有应对机制。Iris-Harness 实现了三套（灵感来自 GLM-5 §4.2.4 的分层上下文管理 HCM），且**彼此正交、可独立开关**：

| Flag | 机制 | 行为 |
| --- | --- | --- |
| `--context-discard-threshold N` | **discard-all** | 一旦 prompt 超过 *N* token，把对话**重置为最初的问题**再继续；此后每一轮（含模型自己的推理）都从它所见中丢弃，但完整轨迹仍落盘 |
| `--context-compress-limit N` | **retry** | 一个 episode 结束却没有可解析答案时，从头重跑最多 *N* 次，并把「已排除了什么」的失败摘要带进下一次（`--no-retry-summary` 可清空）。**错答不重试，只重试缺答** |
| `--keep-tool-result K` | **recency-K** | 只把最近 *K* 个工具结果原文发给模型，更早的折叠成占位符；thinking 与 tool call 永远保留。`-1` 全留，`0` 全折叠 |

report 里的四种 regime 就是它们的组合：`w/o`（全关）、`retry`、`discard-all`、`discard-all + retry`。**discard-all 是 report 的 headline 设定**，哪怕叠加 retry 分数更高。

两个关键守卫让这套机制真正「落得了地」：

- **`ensure_summary_context`**（`providers/openai_client.py`）：每轮末尾预估「下一次请求 + summary + 响应预留」是否会撑爆窗口，会就**回滚整轮**并转去收尾。它特意从尾部回溯统计本轮新增的全部 tool 结果，避免 native_fc 多工具结果被严重低估。
- **`enforce_context_budget` / `drop_last_round`**：生成 summary 前按**最小代价**弹出整个尾部轮次，直到请求能装进硬窗口；系统消息与首条任务消息永远保留。

---

## 5. 被低估的工程：append-only 的对话日志

`message_history` 这一个列表身兼两职：既是**发给模型的对话**，也是**落盘的轨迹**。而上下文管理（discard、回滚、弹轮）是**原地删**的——删完模型看着是清爽了，磁盘上的记录也一并蒸发，事后再也说不清「模型当时究竟看到了什么」。

`ConversationHistory`（`utils/conversation_markers.py`）的解法像记账：**不真删**。它本身就是模型可见的那一份，另外维护一条只增不减的 `full` 账本；每次删除不抹掉原记录，而是补一条「此处隐藏了 N 条」的备注（marker）：

```json
{"role": "context_marker", "op": "rollback",    "count": 3}
{"role": "context_marker", "op": "discard_all", "prefix_len": 2}
```

marker 只躺在 `full` 里，永不发给模型。照着账本从头重放一遍，就能精确还原模型当时的视图——即随时可断言的不变式 `replay(full) == visible`。于是干净的模型输入与**状态忠实**、可复现的完整轨迹同时到手，后者才做得了 SFT。非尾部的批量改动（insert/remove/clear…）一律禁用，从源头掐死日志漂移。

---

## 6. 评测与判分：四个 benchmark，一套严谨协议

| Benchmark | 题量 | 来源 | GT 怎么来 | 指标与判分 |
| --- | --- | --- | --- | --- |
| `browsecomp` | 1,266 | OpenAI simple-evals CSV | canary XOR 解密 | accuracy · LLM judge 回 A/B |
| `browsecomp_zh` | 289 | HF `PALIN2018/BrowseComp-ZH` | canary XOR 解密 + strip | accuracy · 中文 judge 回 A/B |
| `deepsearchqa` | 900 | HF `google/deepsearchqa` | 官方 answer 字段 | **F1** · judge 输出逐项 correctness |
| `hle-text-2158` | 2,158 | HF `cais/hle`（gated） | 官方 answer 字段 | accuracy · judge 结构化 yes/no |

**共同前提**：harness 不分发数据、也不自己标注。`prepare_data.py` 从官方源现拉，统一写成 `{task_id, task_question, ground_truth, metadata}` 的 JSONL，并**断言行数**——上游一改就显式报错，而非悄悄改变你在测什么。

**BrowseComp（英文）**：人工构造的难搜题，答案唯一但需多跳浏览。官方 CSV 的题目与答案均为 **base64 + XOR 加密**，每行自带 `canary` 作密码（`sha256(canary)` 填充至密文长度后逐字节异或），以防爬虫把答案吃进训练集。判分先 **exact match 短路**，否则走 LLM judge（prompt 取自 Tongyi DeepResearch），只许回 `A`/`B`、`max_completion_tokens=16`。判定要点：只看语义，大小写标点语序不计；模糊措辞只要**包含 GT 且无矛盾**也算对；GT 含多个方面须全中，**无「部分正确」**；数字容许精度截断（3518 / 3518.17 均对）；绕答或罗列候选判错。

**BrowseComp-ZH（中文）**：同套 canary 解密（Question/Answer/Topic 三字段），额外 `.strip()`——上游若干行带首尾空格，否则 judge 比的是空白。中文 judge 多一条 GT 约定：`【【…】】` 中同括号内逗号分隔是**同义变体**（任答其一即对），跨括号是**不同方面**（须全部答出）；名字不同译法算对（Robinson / 鲁滨逊）。

**DeepSearchQA（F1）**：来自 Google DeepMind，答案常是集合，题型按 `answer_type` 分 Single / Set Answer；4 道空集答案上游为 `null`，归一成 `"None"`。它是唯一**不做 exact match 短路**的 benchmark。judge 用官方 Kaggle prompt 输出嵌套 JSON：`Correctness Details`（每个期望项 → bool）+ `Excessive Answers`，据此派生两套指标：

- **二值判定**（喂 pass@k）：全项命中且无多余 → CORRECT；
- **F1**（主指标）：`TP=num_correct`、`FN=num_expected−num_correct`、`FP=num_excessive`，逐题算 F1 再**宏平均**，每题只认第一个带 `eval_details` 的 attempt。另报 Fully Correct / Fully Incorrect / Correct-with-Extraneous 三率。

指标从落盘结果**复算而不重跑 judge**；F1 分母是 `num_valid`，小于总题数即打 WARNING——漏题会**虚高 F1**，与全量分母的 pass@k 不可直接比。

**HLE text-only（accuracy）**：gated 数据集，需接受条款并 `hf auth login`；丢掉带 image 的行，2,500 → **2,158**（harness 无视觉工具）。**每题题干追加一句搜索引导**（先搜背景、多路径探索、别急下结论）——HLE 本是闭卷考，不加这句 agent 常凭记忆直答、根本不搜索，复现时必须照抄。judge 用官方 prompt 输出 `extracted_final_answer / reasoning / correct(yes|no) / confidence`：OpenAI、Azure 走 Pydantic structured output，Gemini、MaaS 退化为「只回 JSON」+ 正则兜底；`correct=yes` 即对，数值题容许小误差。judge 失败**绝不 exit**（worker 进程里 `SystemExit` 会穿透上层 `except Exception`，连带搞死整个 run），而是返回 NOT_ATTEMPTED 并计入错误统计。

**共性**：`NOT_ATTEMPTED` 会在判分层重试（10 次 / 间隔 5s），仍失败则计入**故障统计**而非静默算错；judge 模型独立于被测模型配置（`JUDGE_PROVIDER` / `JUDGE_MODEL_NAME`，支持 openai / azure / gemini / maas）。

**pass@k vs ave@M**：`passN` 在一次 run 里每题采 N 个答案，任一判对即过（提前停止），并额外输出 pass@1…pass@N 各级准确率；`aveM` 跑 M 次独立 pass@1 再求均值（`calculate_average_score.py`，run 数对不上会报错退出，杜绝误报）。benchmark spec 写作 `name:aveM:passN`。

**Self-verification（可选、默认关）**：答案产出后、判分前，启一个**独立的 verifier agent** 走 `verify → (判错)reanswer → re-verify`。verifier 用专用 system prompt、`answer_mode=direct` 直出 JSON verdict、且**永远看不到 ground truth**；判决不可解析时 fail-open（默认接受原答），任何异常都保留原答案，确保它只会纠错、不会帮倒忙。

---

## 7. 跑起来

```bash
uv sync
cp .env.example .env          # 填 SERPER_API_KEY + 摘要模型 + 判分模型
uv run python data/prepare_data.py

bash scripts/run_eval.sh \
  --base-url http://127.0.0.1:21234/v1 \
  --llm-config iris-mini \
  --benchmarks "browsecomp:0:1" "browsecomp_zh:0:1" \
                "deepsearchqa:0:1" "hle-text-2158:0:1" \
  --context-discard-threshold 131072 \
  --max-concurrent 32
```

结果落在 `logs/<model-tag>/<benchmark>/`：每题轨迹、`accuracy.txt`、以及解析后的完整 Hydra 配置。想先验证 key 与 endpoint，加 `--max-tasks 2 --max-concurrent 2` 做冒烟测试，再投入 2,158 题的大 run。

---

## 结语

Iris-Harness 的价值不在「又一个 ReAct agent」，而在它把**评测这件事本身**当作工程对象来认真对待：上下文怎么丢、答案怎么判、失败怎么回滚兜底、轨迹怎么状态忠实地落盘——这些平时藏在脚注里的决定，恰恰是让一个 benchmark 数字可信、可比、可复现的关键。当每个系统都报出漂亮数字时，**能被同一把标尺复查的那个，才真正有意义**。
