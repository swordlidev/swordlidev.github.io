---
title: "slime 学习笔记（四）：实战，从零训一个 Search Agent"
description: "系列收尾篇，换成动手视角：搭搜索环境、写多轮 agent 的生成函数、把官方 search-r1 示例改造到生产可用，并正面处理真实项目里绕不开的四个难题——开放类问题的奖励、长轨迹的 credit assignment、rubric 奖励设计、reward hacking。"
pubDate: 2026-08-26
tags: ["slime", "强化学习", "RLHF", "Agent", "Search Agent", "源码解读"]
---

> **本文是《slime 学习笔记》系列第 4 篇，也是收尾篇**。前三篇讲了[框架骨架](/blog/slime-01-getting-started/)、[内部机制](/blog/slime-02-core-pipeline/)和[算法设计](/blog/slime-03-algorithm-design/)，本文换成动手视角：搭好搜索环境，写出多轮 agent 的生成函数，把官方示例改造到生产可用，并正面处理真实项目里那四个绕不开的难题——**开放类问题的奖励、长轨迹的 credit assignment、rubric 奖励设计、reward hacking**（第六节）。参考实现是仓库里的 [`examples/search-r1/`](https://github.com/THUDM/slime/tree/main/examples/search-r1)。
>
> 系列目录：
> 1. [入门与整体架构](/blog/slime-01-getting-started/)
> 2. [从 prompt 到梯度的完整链路](/blog/slime-02-core-pipeline/)
> 3. [算法层：参数、Advantage 与 Loss 的设计](/blog/slime-03-algorithm-design/)
> 4. **实战：从零训一个 Search Agent**（本文）

## 一、为什么从 Search Agent 入手

在所有 agentic RL 场景里，search agent 的工程复杂度和业务价值比最好：

- **环境可控**：检索服务是个无状态 HTTP 接口，不像 sandbox 要管容器生命周期，也不像浏览器 agent 要处理页面状态。
- **奖励可验证**：问答任务有标准答案，可以做 rule-based verifier，不需要训奖励模型。
- **业务直通**：几乎所有企业都已经有检索系统（ES、向量库、内部搜索中台）。训出来的能力——「判断何时该搜、搜什么、怎么从多轮检索结果里综合答案」——可以直接接到现有 RAG 链路上。

目标形态是让模型学会这样的轨迹：

```text
<think>需要先确认这个人的出生地</think>
<search>张三 出生地</search>
<information>Doc 1(Title: ...) ...</information>
<think>还需要知道该地的所属省份</think>
<search>XX 市 所属省份</search>
<information>Doc 1(Title: ...) ...</information>
<answer>YY 省</answer>
```

注意 `<information>` 块是**环境注入**的，不是模型生成的。这一点直接决定了 `loss_mask` 怎么切——后面会详细讲。

## 二、铺垫：slime 的定制哲学

slime 的所有扩展都是同一个模式：**传一个函数路径（import path），框架用 `load_function` 动态加载**。核心代码零侵入，你不需要 fork slime。

接口按「插在哪一层」分组（完整清单见 [docs/zh/get_started/customization.md](https://github.com/THUDM/slime/blob/main/docs/zh/get_started/customization.md)）：

| 层 | 代表参数 | 用途 |
| :--- | :--- | :--- |
| 数据源 | `--data-source-path` | 换 prompt / task 来源 |
| rollout 编排 | `--rollout-function-path` | 换掉整个生成循环（最后手段） |
| **单样本生成** | **`--custom-generate-function-path`** | **agent loop 的落点** |
| **奖励** | **`--custom-rm-path`** | **verifier / 业务打分** |
| 过滤 | `--dynamic-sampling-filter-path`、`--buffer-filter-path`、`--rollout-sample-filter-path` | 丢弃无梯度信号的组、buffer 消费顺序 |
| 数据转换 | `--custom-convert-samples-to-train-data-path`、`--custom-reward-post-process-path` | 自定义张量化与奖励后处理 |
| 损失 | `--custom-loss-function-path`、`--custom-advantage-function-path`、`--custom-tis-function-path` | 自定义算法 |
| Megatron 钩子 | `--custom-megatron-init-path`、`--custom-megatron-before-train-step-hook-path` | 训练进程生命周期注入 |
| 日志 | `--custom-rollout-log-function-path` | 自定义指标 |

```text
flowchart TD
    A{"要改什么？"} --> B{"改单条轨迹<br/>怎么生成？"}
    B -- "是" --> C["--custom-generate-function-path<br/>（99% 的 agent 场景）"]
    B -- "否" --> D{"改怎么打分？"}
    D -- "是" --> E["--custom-rm-path"]
    D -- "否" --> F{"改 batch 怎么凑<br/>（并发/过滤/重试策略）？"}
    F -- "是" --> G["--rollout-function-path<br/>⚠️ 要自己实现整个循环"]
    F -- "否" --> H["看具体层：loss / 数据源 / 转换"]
```

**官方建议非常明确：agentic 场景从 `--custom-generate-function-path` + `--custom-rm-path` 开始，不要一上来就重写 rollout function。** 因为默认的 `generate_rollout` 已经帮你做好了超采样、动态过滤、并发控制、abort、partial rollout 回填这一整套（见[第 2 篇](/blog/slime-02-core-pipeline/)），重写意味着全部自己实现。

本文的 search agent，`CUSTOM_ARGS` 里**只有两行**：

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)
```

## 三、搜索环境怎么搭

这是全文最实际的部分。slime 侧只认一个检索契约，所以先把契约讲清楚，三条路径就都是它的实现。

### 3.1 统一检索契约

```python
# 请求侧（HTTP 检索服务）：POST {search_url}
# {"queries": ["..."], "topk": 3, "return_scores": False}

# 响应侧必须能被归一化成这个结构：
# [{"document": {"contents": "<title>\n<正文>"}}, ...]

async def my_search(query: str, top_k: int) -> list[dict]:
    """返回 [{"document": {"contents": ...}}]，contents 首行当 title，其余当正文。"""
```

`examples/search-r1/` 里的 `local_search` 和 `google_search` **本就是同一契约的两个实现**，彼此可以 drop-in 替换。拿到结果后由 `_passages2string` 统一拼成喂给模型的文本：

```text
Doc 1(Title: 标题) 正文...
Doc 2(Title: 标题) 正文...
```

**这个契约就是业务接入点**——你的 ES、向量库、搜索中台，只要包一个返回该结构的 async 函数，就能接进来。

### 3.2 路径 A：本地稠密检索（可复现实验）

用 FAISS + E5 检索 wiki-18 语料，完全离线。仓库提供了完整的服务端 `examples/search-r1/local_dense_retriever/retrieval_server.py`。

**关键前提：必须用独立的 conda 环境。** 检索器依赖 `faiss-gpu` 和特定版本 PyTorch，和训练环境（Megatron + SGLang）会冲突。

```bash
# 1. 建独立环境
conda create -n retriever python=3.10 && conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 \
    -c pytorch -c nvidia
pip install transformers datasets pyserini huggingface_hub uvicorn fastapi
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# 2. 下载索引与语料（注意磁盘！）
python examples/search-r1/local_dense_retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index    # 分片合并
gzip -d $save_path/wiki-18.jsonl.gz

# 3. 启动服务（监听 0.0.0.0:8000，端点 POST /retrieve）
python examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path $save_path/e5_Flat.index \
    --corpus_path $save_path/wiki-18.jsonl \
    --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu

# 4. 启动训练前务必退出该环境
conda deactivate
```

容量与运维要点（这些数字决定你能不能跑）：

| 项 | 量级 |
| :--- | :--- |
| 下载体积 | 约 60–70 GB（索引分片 `part_aa`/`part_ab` + 语料） |
| 解压合并后磁盘 | **约 132 GB** |
| 检索服务显存 | 每 GPU 约 5–7 GB |
| 首次启动耗时 | 需下载 E5 模型 + load 索引，几分钟；后续 1–2 分钟 |

两个坑：**服务进程不随 shell 关闭而终止**，重启要先 `lsof -i :8000` 找 PID 杀掉；下载脚本的 `--repo_id` 参数实际被忽略（repo id 在脚本内写死），只有 `--save_path` 生效。

### 3.3 路径 B：外部搜索 API（快速验证）

改 `generate_with_search.py` 顶部的 `SEARCH_R1_CONFIGS`：

```python
"search_backend": "google",
"google": {"api_key": os.environ["SERPER_API_KEY"], "snippet_only": True, "proxy": None},
```

底层是 serper.dev：POST `https://google.serper.dev/search`，header `X-API-KEY`，body `{"q", "num", "gl": "us", "hl": "en"}`，从响应的 `organic` 字段取结果。

`snippet_only` 这个开关的差异比看上去大：

| 取值 | 行为 | 代价 |
| :--- | :--- | :--- |
| `True` | 只用搜索结果的 title + snippet | 快、稳定、信息量小 |
| `False` | 额外**实际抓取**每个结果页的正文（并发 8，总超时 5s，随机 UA，`chardet` 探测编码），再从正文里提取与 query 相关的段落 | 信息量大，但延迟高、失败率高、有外站抓取的稳定性与合规风险 |

**生产环境建议 `snippet_only=True` 或直接走路径 C。**

### 3.4 路径 C：对接自有业务检索（推荐落地方式）

实现一个符合 §3.1 契约的函数，替掉 `search()` 里的分派即可。参考 `local_search_server.py` 的写法：

```python
async def business_search(query: str, top_k: int) -> list[dict]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
        async with sess.post(BUSINESS_SEARCH_URL, json={"q": query, "size": top_k}) as r:
            hits = (await r.json()).get("hits", [])
    # 关键：把业务字段映射成统一结构，首行 title 其余正文
    return [{"document": {"contents": f"{h['title']}\n{h['content']}"}} for h in hits]
```

注意 `local_search` 的异常处理策略是**失败返回空列表而不抛异常**。这个选择是对的——单条检索失败不该让整个 rollout 挂掉。但它只 `print` 一行，生产环境必须换成结构化日志 + 失败率监控，否则检索服务半死不活时你只会看到 reward 莫名下降。

### 3.5 三条路径选型

| | A 本地稠密检索 | B 外部搜索 API | C 自有业务检索 |
| :--- | :--- | :--- | :--- |
| 可复现性 | 高 | 低（结果随时间变） | 中 |
| 成本 | 132 GB 磁盘 + GPU | 按调用计费 | 复用现有系统 |
| 延迟稳定性 | 好 | 一般 | 取决于自身 SLA |
| 适用 | 论文复现、离线基线 | 快速验证想法 | **业务落地** |

## 四、Agent 主循环怎么写

`--custom-generate-function-path` 的签名是固定的：

```python
async def generate(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]:
    ...
```

它拿到一条待生成的 `Sample`，负责把它填完整。多轮循环的骨架：

```python
state = GenerateState(args)                      # 复用 slime 单例拿 tokenizer
url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
sampling_params["stop"] = ["</search>", "</answer>"]   # ← 关键，见下文

for turn in range(max_turns):
    output = await post(url, {"text": prompt + response, "sampling_params": sampling_params})
    # 模型生成段：参与训练
    sample.append_response_tokens(args, tokens=gen_tokens, log_probs=gen_logps, trainable=True)
    action, content = postprocess_predictions(output["text"])   # 解析 <search>/<answer>
    if action == "answer":
        break
    obs = f"\n\n<information>{await search(content)}</information>\n\n"
    # 环境注入段：不参与训练
    sample.append_response_tokens(args, tokens=tokenize(obs), trainable=False)
```

```text
sequenceDiagram
    participant G as generate()
    participant R as SGLang Router
    participant S as 检索服务

    G->>R: POST /generate（stop=["</search>","</answer>"]）
    R-->>G: "<think>...</think><search>q</search>"
    Note over G: loss_mask += [1]*n
    G->>S: search(q)  —— 受独立 SEMAPHORE 限流
    S-->>G: Doc 1(Title: ...) ...
    Note over G: loss_mask += [0]*m（环境注入不训练）
    G->>R: POST /generate（续上完整上下文）
    R-->>G: "<think>...</think><answer>A</answer>"
    Note over G: status = COMPLETED
```

### 六个必须处理对的工程细节

**① 停止词注入——最容易漏且后果最严重的一处。**

必须把 `</search>` 和 `</answer>` 放进 `sampling_params["stop"]`。源码注释里记录了不注入的后果：引擎会在闭合标签后继续吐垃圾 token，甚至编造出新的 "Question:" 段落。而由于开启 `return_logprob` 时为保 token/logprob 对齐**禁用了字符串裁剪**，这些垃圾会以 `loss_mask=1` 被当作模型输出训练进去，同时破坏格式校验导致奖励下降。slime 已设 `no_stop_trim=True`，所以闭合标签本身会保留在输出里，不用担心被吃掉。

**② loss_mask 切分。** 模型生成段 `trainable=True`，检索返回段 `trainable=False`。这是多轮 agent 的铁律——训练环境注入的文本会让模型学会"自己编检索结果"。

**③ token 与 logprob 必须等长。** 开 `return_logprob` 时直接从 `meta_info["output_token_logprobs"]` 取 `(logprob, token_id)` 对，不要先解码成字符串再重新 tokenize（会错位）。observation 段补 `0.0` 占位，并加显式断言：

```python
assert len(response_token_ids) == len(rollout_log_probs)
```

**④ 终止状态映射。** 按 `finish_reason["type"]` 映射到 `Sample.Status`：`length` → `TRUNCATED`，`abort` → `ABORTED`（要立即 return，不要再继续循环），`stop` → `COMPLETED`。工具调用失败等可恢复错误用 `FAILED`。

**⑤ 搜索限流要独立于训练并发。** 示例用模块级 `SEMAPHORE = asyncio.Semaphore(search_concurrency)`（默认 256）专门保护检索服务。这与 `GenerateState` 里控制 SGLang 并发的 semaphore 是**两套独立的闸门**——生成并发高不代表检索服务扛得住。

**⑥ 自定义函数必须能被 Ray worker 导入。** `--custom-generate-function-path generate_with_search.generate` 是个模块路径，Ray worker 进程要能 `import generate_with_search`。所以 `ray job submit` 的 `RUNTIME_ENV_JSON` 里 `PYTHONPATH` 必须包含你的代码目录：

```bash
RUNTIME_ENV_JSON="{\"env_vars\": {\"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\"}}"
```

漏了这一步会在第一次生成时报 `ModuleNotFoundError`，而且报错位置在 worker 里，不太直观。

此外示例开头有 `assert not args.partial_rollout`——这个生成函数不支持 partial rollout，因为半成品轨迹的多轮状态恢复需要额外设计。

## 五、奖励函数设计（含两个必踩的坑）

`--custom-rm-path` 的实现很薄：

```python
async def reward_func(args, sample, **kwargs):
    return compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label["ground_truth"],   # 来自 --label-key reward_model
    )
```

`qa_em_format.py` 里的打分设计思路是**分层奖励**：

| 情形 | 得分（设计意图） |
| :--- | :--- |
| 格式合法 + EM 命中 | `score`（满分） |
| EM 命中但格式非法 | `score - structure_format_score` |
| 未答对、格式合法、检索到了答案 | `structure_format_score + retrieval_score` |
| 未答对、格式合法、没检索到 | `structure_format_score` |
| 格式非法 | `final_format_score` |

格式校验 `is_valid_sequence` 是个**状态机**：先定位 `<|im_start|>assistant`，校验 think / search / information / answer 四种标签开闭配对，然后沿 `start → in_think → after_think → in_search → after_search → in_information → ... → in_answer → end` 走，标签之间只允许空白字符。

### 坑一：示例的分层奖励实际上没生效

`compute_score_em` 的签名里 `structure_format_score`、`final_format_score`、`retrieval_score` **默认全是 0**，而 `reward_func` 只传了 `format_score=0.2`——可是 `format_score` 这个参数在函数体里**从未被使用**（只出现在签名和 docstring 中）。

结果：示例跑起来的实际奖励退化成**二值 EM（0 或 1.0）**，所有格式分和检索分都是 0。代码注释里写的 `# 0.3` `# 0.2` `# 0.8` `# 0.1` 是设计意图，不是实际值。

要真正启用分层奖励，必须显式传参：

```python
score = compute_score_em(
    solution_str=sample.prompt + sample.response,
    ground_truth=sample.label["ground_truth"],
    structure_format_score=0.2, retrieval_score=0.1, final_format_score=0.1,
)
```

### 坑二：`extract_solution` 要求至少两个 `<answer>` 块

```python
if len(matches) <= 1:
    return None          # 0 个或恰好 1 个匹配都返回 None
return matches[-1].group(1).strip()
```

这不是 bug，是对 Search-R1 prompt 模板的适配——模板里含一个 `<answer>` 示例，所以模型自己的答案是第 2 个匹配。

**如果你换成自己的 prompt 且不含示例 answer，`extract_solution` 会永远返回 `None`，奖励恒为 0，模型学不到任何东西。** 这个故障非常隐蔽：训练能正常跑，只是 reward 曲线平在 0。换 prompt 时务必同步改这里。

同理，`is_valid_sequence` 里硬编码了 `<|im_start|>assistant`（Qwen chat template），换模型族要一起改。

## 六、真实搜索项目的四个硬问题

第五节讲的是「示例怎么写、哪里有坑」。但真到业务里训搜索 agent，你会撞上的是另外四个更难的问题：**问题没有标准答案、轨迹长到分不清功劳、评分标准没法写成规则、以及模型找到了捷径**。这四个问题互相纠缠——放宽奖励以适应开放问题，往往就是打开了 reward hacking 的门。

这一节讲清楚每个问题的成因、slime 里的落点、以及可落地的做法。

### 6.1 开放类问题怎么给奖励

示例用的是 EM（精确匹配），只适用于「答案是一个短实体」的事实问答。业务里的问题通常长这样：

> 「帮我对比一下这三款产品的售后政策差异」
> 「这个季度我们区域的销量下滑可能是什么原因」

这类问题**没有唯一答案字符串**，EM 直接退化成恒 0。

**第一步：先别急着上模型打分，看能不能把「开放」拆成「可验证的子约束」。**

这是最被低估的一步。很多看起来开放的问题，其实可以分解出一批硬性事实点：

| 拆解维度 | 可验证形式 | 在 slime 里的实现 |
| :--- | :--- | :--- |
| 必须提到的关键实体/数字 | 关键点召回率（覆盖了几个 must-have） | 规则函数，放进 `--custom-rm-path` |
| 不能出现的内容 | 黑名单命中即扣分（过期政策、竞品名） | 同上 |
| 必须有引用 | 答案里的断言能否对应到某篇 `<information>` | 需要在 generate 里记录检索到的文档，见下文 |
| 结构要求 | 是否按「结论 + 依据 + 来源」组织 | 状态机校验，类似 `is_valid_sequence` |

把这些组合起来，开放问题也能得到一个**连续、稠密**的分数，而且完全可复现、零成本。

**第二步：实在无法规则化的部分，才交给模型判分。** slime 这里有两条现成通路：

```bash
# 通路 A：远程奖励服务（rm_hub 内置，自带 10 次指数退避重试）
--rm-type remote_rm --rm-url http://judge-service:8000

# 通路 B：完全自己实现（可在函数内调任意 LLM / 多个 judge / 混合规则）
--custom-rm-path my_reward.open_ended_reward
```

通路 A 的协议固定在 `slime/rollout/rm_hub/__init__.py` 的 `remote_rm`，POST 过去的 body 只有三个字段：

```python
payload = {"prompt": sample.prompt, "response": sample.response, "label": sample.label}
```

**注意它不传 metadata**，所以如果你的 judge 需要看检索到的文档、轮数等信息，要么把它们塞进 `label`，要么走通路 B。

**第三步：组合与取值。** 实践中稳定的形态是加权和，并且让可验证部分占主导：

```python
async def open_ended_reward(args, sample, **kwargs):
    hard = keypoint_recall(sample.response, sample.label["must_have"])   # 规则，稠密
    cite = citation_grounding(sample)                                    # 规则，防编造
    soft = await llm_judge(sample)                                       # 模型，0~1
    return {
        "reward": 0.5 * hard + 0.2 * cite + 0.3 * soft,
        "hard": hard, "cite": cite, "soft": soft,          # 分项一并返回，便于监控
    }
```

返回 dict 是 slime 支持的——`Sample.get_reward_value` 会按 `--reward-key` 取值：

```python
def get_reward_value(self, args) -> float:
    return self.reward if not args.reward_key else self.reward[args.reward_key]
```

所以配 `--reward-key reward` 用于训练，其余分项留在 dict 里供日志分析。评测可以用 `--eval-reward-key` 指向另一个键（比如只看 `hard`，避免 judge 噪声干扰验证集结论）。

> 另一个常被忽略的选项：`slime/rollout/rm_hub/f1.py` 的 `f1_score`（`--rm-type f1`）。它按 token 级重叠算 F1，对「答案是短语而非单个实体」的场景比 EM 宽松得多，而且是纯规则。开放度不高的任务先试它，往往够用。

### 6.2 长搜索轨迹的 credit assignment

这是搜索 agent 最本质的算法困难。一条轨迹可能是：

```text
think → search(q1) → info → think → search(q2) → info → ... → search(q7) → info → answer
```

最后答对了，奖励 +1。**但这 1 分该记在谁头上？** 可能第 3 轮那个检索词是关键突破，前两轮纯属浪费；也可能前六轮都在绕路，最后一轮瞎蒙对了。

**先看清 slime 实际在做什么。** GRPO 路线下（[第 3 篇](/blog/slime-03-algorithm-design/) 第五节），`get_grpo_returns` 把序列级标量 advantage **原封不动广播到每个 token**：

```python
advantages = [torch.ones_like(kl[i]) * reward[i] for i in range(...)]
```

也就是说，**框架默认不做任何 token 级或步骤级的功劳分配**——整条轨迹里每个可训练 token 拿到完全相同的 advantage。这是 outcome-only supervision 的标准形态，优点是无偏、不需要过程标注，缺点就是方差随轨迹变长而迅速上升。

那实践里怎么办？按「改动成本」从低到高有四条路：

**路径一：靠组内对比，把轮数当成可比较的变量（零成本，首选）。**

GRPO 的 baseline 是同一 prompt 的其它采样。如果 8 条采样里有的 3 轮答对、有的 7 轮答对，**只要你的奖励对轮数敏感**，组内对比就自动完成了粗粒度的功劳分配。做法是加一个轻微的步数惩罚：

```python
reward = correctness - lambda_steps * num_search_calls      # lambda 取很小，如 0.02
```

这比任何 token 级分配都简单，而且直接对齐业务目标（少调用检索 = 省钱 = 低延迟）。**但 λ 必须小**，否则模型学会「不搜直接猜」——这是 6.4 节的典型 hacking 之一。

**路径二：把一条长轨迹拆成多个 Sample，共享 outcome（中等成本）。**

`--custom-generate-function-path` 的返回类型是 `Sample | list[Sample]`，允许一条 rollout 产出多条训练样本。`_post_process_rewards` 对这种「多段」轨迹有专门支持：

```python
# 同一 rollout 的多个 segment 通过 rollout_id 关联
rid = sample.rollout_id if sample.rollout_id is not None else sample.index
...
if group != sample.group_index or previous_reward != reward:
    raise ValueError("Segments of rollout {rid} must share a prompt group and outcome reward")
```

注意这条**硬约束**：同一 `rollout_id` 的所有 segment **必须共享同一个 group 和同一个 outcome reward**。换句话说，slime 的多段机制是为「把长轨迹切成多个可训练片段、但用同一个结果分」设计的，**不是**给你按段打不同分的。组内统计按 unique rollout 算，再把结果复制回每个 segment——这样切 10 段不会让这条轨迹在组内的权重变成 10 倍。

这条路径的真实收益在于**显存和打包**：把 10 万 token 的长轨迹切成若干段，每段独立进 microbatch，比整条塞进去友好得多。同时 `rollout_mask_sums` 会把同一 rollout 所有 segment 的 mask 总数加起来当分母（见 `batch_builder.py`），保证 per-token 口径下这条轨迹整体仍然只算一次均值。

**路径三：过程奖励（process reward），给中间步骤直接打分（高成本）。**

在奖励函数里对每一轮检索单独评估，例如「这一轮的检索结果里是否包含了最终答案所需的信息」，然后把分数汇总。示例 `qa_em_format.py` 里的 `retrieval_score` 就是这个思路的最简版——检索结果里出现了 golden answer 就给分。

它的价值在**冷启动**：答案还答不对的阶段，「搜到了相关文档」这个信号能提供梯度（§8 改造清单第 11 条也提到这点）。但要警惕它制造新的 hacking 面——见 6.4。

**路径四：真正的 token 级功劳分配（最高成本）。**

用 `--custom-advantage-function-path` 完全接管 advantage 计算。签名是：

```python
def custom_fn(args, rollout_data) -> None:
    # 必须就地写入 rollout_data["advantages"] 和 rollout_data["returns"]
```

这里可以实现按轮次折扣、按注意力归因、或用一个 critic 给 token 级估值。但要清楚代价：

- 需要自己处理 CP 切分（advantage 的形状必须和 CP-local 的 log_probs 对齐）；
- 需要自己处理归一化；
- 一旦偏离标准估计器，调试难度陡增。

**除非你已经确认方差是主要瓶颈，否则不建议从这里开始。** 更务实的顺序是：路径一 → 加过程奖励 → 实在不行才动 advantage。

还有一个纯工程但影响巨大的点：**`loss_mask` 本身就是一种功劳分配**。环境注入段设 `trainable=False` 已经是在说「这些 token 不该为结果负责」。同理，如果你判定某条轨迹的某一段是明显的无效探索，可以用 `--rollout-sample-filter-path` 把整条样本的 `remove_sample` 置位（它会把 loss_mask 全部清零），但注意文档里的提醒：**这不影响该样本参与 advantage 归一化**。

### 6.3 用 rubric 设计奖励

当「可验证子约束」也写不出来、又不想让 judge 给一个说不清的 0.7 分时，rubric 是目前最可控的方案：**把评分标准拆成一组彼此独立、可逐条判定的条目，让 judge 只做二元判断，分数由代码算。**

对比一下两种 judge 用法：

| | 整体打分 | rubric 逐条判定 |
| :--- | :--- | :--- |
| judge 输出 | 一个 0–10 的分 | 每条 `yes/no` |
| 可解释性 | 差，分数漂移无从归因 | 好，能看出扣在哪条 |
| 稳定性 | 差，同一答案多次打分能差 2–3 分 | 好，二元判断一致性高得多 |
| 可审计 | 难 | 能和人工标注逐条对齐 |
| 调整成本 | 调 prompt，全盘重标 | 改单条权重即可 |

一个搜索 agent 的 rubric 骨架（搜索任务要同时覆盖**结果**与**过程**）：

```python
RUBRIC = [
    # (条目 id, 权重, 是否为否决项)
    ("answer_addresses_question",  0.30, False),   # 答到点上
    ("all_claims_grounded",        0.25, True),    # 每个断言都有检索来源 ← 否决项
    ("no_fabricated_citation",     0.00, True),    # 没有编造来源 ← 纯否决
    ("covers_required_aspects",    0.20, False),   # 覆盖必要维度
    ("search_queries_reasonable",  0.15, False),   # 检索词与问题相关（过程分）
    ("no_redundant_search",        0.10, False),   # 没有大量重复检索（过程分）
]

async def rubric_reward(args, sample, **kwargs):
    judgements = await judge_rubric(sample, RUBRIC)      # 返回 {item_id: bool}
    # 否决项优先：任一不通过直接判 0，不进入加权
    for item_id, _, is_veto in RUBRIC:
        if is_veto and not judgements[item_id]:
            return {"reward": 0.0, "veto": item_id, **judgements}
    score = sum(w for item_id, w, _ in RUBRIC if judgements[item_id])
    return {"reward": score, **judgements}
```

几个实践要点：

1. **否决项（veto）必须单独处理，不能只是给个低权重。** 「编造引用」这类行为一旦可以用其它条目的高分抵消，模型就会学会「多数条目做好 + 偷偷编一条来源」。
2. **条目之间要尽量正交。** 如果两条判定高度相关（比如「有引用」和「引用正确」），实际效果是给同一件事双倍权重，无意中扭曲了优化方向。
3. **judge 的输入要给足。** 这正是 6.1 提到的 `remote_rm` 只传三字段的限制——rubric 里的 `all_claims_grounded`、`no_redundant_search` 必须看到检索到的文档和检索词序列。所以 rubric 场景基本只能走 `--custom-rm-path`，自己在函数里组装 judge 输入（从 `sample.response` 里解析 `<information>` 块，或在 generate 阶段把文档存进 `sample.metadata`）。
4. **rubric 要和 `--group-rm` 配合考虑。** 开 `--group-rm` 后走 `batched_async_rm`，一次拿到整组。对 rubric 的好处是可以做**组内相对判定**（「这 8 个答案里哪些覆盖得更全」），judge 做比较题比打绝对分稳定得多。
5. **分项全部回传，用于监控。** 返回 dict + `--reward-key` 取总分，各条目通过率进日志。**哪一条的通过率在异常快速上升，往往就是 hacking 正在发生的地方。**

最后提醒：rubric judge 是外部服务，会成为 rollout 的长尾来源。用 `slime/observability/trace_utils.py` 的 `trace_span` 包住 judge 调用，并像保护检索服务一样给它独立的并发闸门（§4 细节⑤）。

### 6.4 Reward hacking：搜索任务的典型姿势与防法

Reward hacking 的定义是：**模型找到了让奖励升高、但并未真正完成任务的策略。** 它在搜索 agent 里特别容易发生，因为奖励函数要同时评价「过程」和「结果」，而过程信号往往比结果信号好骗。

#### 常见姿势

| 姿势 | 怎么发生 | 触发它的奖励设计 |
| :--- | :--- | :--- |
| **格式刷分** | 输出完美的 think/search/answer 结构，答案内容随便填 | 格式分占比过高 |
| **跳过检索直接猜** | 不调用检索就答，赌模型自身的参数知识 | 步数惩罚 λ 太大 |
| **检索词复制问题** | 每轮都把原问题整段当 query，不做分解 | 只奖励「调用了检索」，不评价检索词质量 |
| **刷过程分** | 疯狂检索，让 golden answer 碰巧出现在某篇文档里拿 `retrieval_score` | 过程奖励按「文档里是否出现答案」给分 |
| **答案堆砌** | 把所有检索到的内容全列出来，赌 EM/F1 命中 | token 级 F1、关键点召回（无精确率约束） |
| **编造引用** | 写出格式正确但不存在的来源 | rubric 没把「引用真实性」设为否决项 |
| **judge 话术** | 学会 judge 偏好的措辞（「综上所述」「根据权威资料」），内容不变 | 单一 judge + 整体打分 |
| **长度膨胀** | 答案越长越容易蹭上分 | 召回类指标 + per-token loss 口径 |

注意最后一条和[第 3 篇](/blog/slime-03-algorithm-design/) 第九节相关：开了 `--calculate-per-token-loss` 后长序列权重更大，如果奖励又偏好长答案，两者会**叠加放大**长度膨胀。

#### 怎么防

**① 奖励设计层面——这是主战场。**

- **精确率与召回率要配对出现。** 只给召回（覆盖了几个关键点）必然导致堆砌，必须同时惩罚无关内容。F1 本身就是这个思路（`f1_score` 同时返回 precision/recall）。
- **过程分设上限，且必须由结果分「解锁」。** 比较稳的形态是：过程分只在结果分非零时才给，或者过程分总和严格小于结果分。这样「只刷过程」的收益天花板很低。
- **否决项用硬规则，不要交给 judge。** 「引用的文档 ID 是否真实存在于本轮检索结果中」是可以用代码精确判定的——不要让 judge 去猜。
- **步数惩罚取小值，并同时监控「零检索答题率」。** 这个指标一旦上升就说明 λ 设大了。

**② 监控层面——hacking 的特征是「指标分化」。**

这是最实用的识别手段：**把奖励分项全部回传并分别记录**（6.1/6.3 都强调了这点），然后盯住它们的**背离**：

```text
格式分 ↑↑  但 EM/judge 分持平  →  格式刷分
过程分 ↑↑  但结果分持平        →  刷检索
训练 reward ↑↑ 但 eval 指标持平/下降 → 过拟合到奖励函数本身
答案平均长度 ↑↑ 且 F1 precision ↓ → 答案堆砌
检索调用数 ↓↓ 且 reward 持平     →  跳过检索在蒙对
```

**最硬的一道防线是：验证集用一套和训练不同的奖励口径。** 配 `--eval-reward-key` 指向更严格的指标（比如只看 `hard` 规则分，不含 judge 和格式分），这样「训练 reward 涨、eval 不涨」会立刻暴露。只用同一个函数既训又评，等于没有体检。

**③ 算法层面的辅助手段。**

- **KL 约束**（`--use-kl-loss --kl-loss-coef`）限制策略偏离 ref 的程度，能延缓极端退化行为的出现。示例配 `0.001` 这种小值就是在这个考虑上。
- **`--ref-update-interval`** 若设得过于频繁，KL 锚点会跟着模型一起漂，约束作用减弱——怀疑 hacking 时可以先把它关掉，让 ref 固定为初始权重。
- **熵监控**。`entropy_loss` 塌缩（快速下降到很低）常常伴随模型收敛到某个单一的刷分模板。
- **`--dynamic-sampling-filter-path` 的副作用要注意**：它丢掉组内方差为 0 的组。当 hacking 成熟后，整组会**全部拿高分**，方差同样为 0，于是这些样本被静默丢弃——表现是「reward 很高但有效样本数在下降」。把过滤率也纳入监控。

**④ 流程层面。**

定期人工抽样看完整轨迹。`compute_score_em` 自带 1/64 概率打印样例就是为此准备的（§9 调试第 3 步）。所有自动化指标都可能被绕过，**读几条真实轨迹是发现新 hacking 姿势最快的方式**——而且新姿势总会出现。

#### 一张取舍图

```text
flowchart TD
    A["奖励信号稀疏<br/>（只有二值结果分）"] --> B["冷启动无梯度<br/>GRPO 组内全零"]
    A2["奖励信号稠密<br/>（格式分+过程分+judge）"] --> B2["容易被 hack"]
    B --> C["加过程分/格式分"]
    B2 --> D["加约束：精确率惩罚<br/>否决项、过程分上限"]
    C --> E["稠密但受控<br/>+ 分项监控 + 独立 eval 口径"]
    D --> E
```

**结论是：稀疏和稠密都不是答案，「稠密 + 受约束 + 可观测」才是。** 奖励函数不是写一次就完的——它会和模型形成对抗关系，需要随训练迭代。

## 七、启动配置拆解

参数分组与[第 1 篇](/blog/slime-01-getting-started/)讲的一致，算法参数的原理见[第 3 篇](/blog/slime-03-algorithm-design/)，这里只说 search agent 特有的几处（基于 `examples/search-r1/run_qwen3_4B.sh`）：

| 配置 | 值 | 为什么 |
| :--- | :--- | :--- |
| `--rollout-max-response-len` | `131072` | 多轮累积上下文很长：每轮 think + search + 3 篇文档。必须给足，否则大量轨迹 `TRUNCATED` |
| `--n-samples-per-prompt` | `8` | GRPO 需要组内对比，组太小优势估计噪声大 |
| `--rollout-batch-size` / `--global-batch-size` | `32` / `256` | 32 prompt × 8 采样 = 256，正好一个训练 batch |
| `--input-key` / `--label-key` | `prompt` / `reward_model` | 对齐 Search-R1 的 parquet 字段。`label` 里取 `["ground_truth"]["target"]` |
| `--context-parallel-size` | `2` | 长序列靠 CP 切分 |
| `--advantage-estimator` + KL | `grpo` + `--use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl` | agent 任务容易崩，小 KL 约束保稳 |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.28` | 非对称 clip，给正向更新更大空间 |
| 资源 | `--actor-num-gpus-per-node 4 --rollout-num-gpus 4 --colocate` | 8 卡单机，训练与生成共享 |

如果要开 TIS（缓解训推不一致，原理见[第 3 篇](/blog/slime-03-algorithm-design/) 第八节），需要**三处同时改**，缺一不可：

1. `generate_with_search.py` 里 `"return_logprob": True`；
2. `GRPO_ARGS` 加 `--use-tis`；
3. `CUSTOM_ARGS` 加 `--custom-config-path` 与 `--custom-tis-function-path`。

代价是收集 logprob 会禁用响应后处理（为保对齐），且增加计算开销。

## 八、从示例到业务：改造清单

示例是为复现 Search-R1 论文写的，直接上业务还差这些。按优先级排序：

| # | 改造项 | 示例现状 | 生产要求 |
| :-- | :--- | :--- | :--- |
| 1 | **密钥管理** | `api_key` 硬编码在 `SEARCH_R1_CONFIGS` 里 | **必须改为从环境变量读取**，绝不能进代码仓库 |
| 2 | **检索目标白名单** | `search_url` 完全可配置 | 若 URL 来自配置下发或数据驱动，**必须做目标白名单校验，默认拒绝内网网段**（防 SSRF）。固定常量 URL 则风险可控 |
| 3 | **奖励分层生效** | `format_score` 参数未被使用，退化成二值 EM | 显式传 `structure_format_score` / `retrieval_score` / `final_format_score` |
| 4 | **answer 提取逻辑** | 要求 ≥2 个 `<answer>` 块 | 换 prompt 后必须同步改，否则 reward 恒 0 |
| 5 | **格式校验适配** | 硬编码 Qwen 的 assistant 起始标记 | 换模型族时同步改 chat template 标记 |
| 6 | **外网抓取** | `snippet_only=False` 会实抓外部页面 | 业务环境用 `snippet_only=True` 或走自有检索，规避稳定性与合规风险 |
| 7 | **并发容量规划** | `search_concurrency=256` 写死 | 按检索服务实测 QPS 定值，与 rollout 并发联合压测 |
| 8 | **失败可观测** | 异常时 `print` 并返回 `[]` | 结构化日志 + 失败率/延迟监控。空结果会让 agent 看到空 `<information>`，静默劣化 |
| 9 | **token 预算** | `max_turns=2` | 与 `--rollout-max-response-len` 联合规划：轮数 × (think + 文档长度) 要留余量 |
| 10 | **语料与 retriever** | wiki-18 + `intfloat/e5-base-v2` | 中文/业务场景换索引与 embedding 模型；retriever 与策略模型 tokenizer 无关，但召回质量直接决定奖励上限 |
| 11 | **verifier 替换** | EM 精确匹配 | 换业务 verifier（结构化字段比对、单元测试、LLM judge），但**保留"格式分 + 过程分 + 结果分"的分层结构**。开放类问题见 §6.1、rubric 见 §6.3 |
| 12 | **慢检索定位** | 无专门埋点 | 用 `slime/observability/trace_utils.py` 的 `trace_span` 包住检索调用，定位长尾 |
| 13 | **奖励分项可观测** | 只返回一个标量 | 奖励函数返回 dict + `--reward-key` 取总分，各分项进日志。**这是识别 reward hacking 的前提**（§6.4） |
| 14 | **eval 用独立口径** | 训练与评测同一函数 | 配 `--eval-reward-key` 指向更严格的指标，避免"训练 reward 涨、真实能力不涨"无从察觉（§6.4） |

第 11 条值得多说一句：分层奖励不只是给部分分，它在**冷启动阶段提供梯度信号**。如果只有二值 EM，模型在学会正确格式之前所有样本都是 0 分，GRPO 组内全零 → 优势全零 → 没有梯度（这正是[第 3 篇](/blog/slime-03-algorithm-design/) 提到的 GRPO 冷启动陷阱）。这时动态过滤（`--dynamic-sampling-filter-path`）会把这些组全丢掉，训练直接停滞。给格式分能让模型先学会"按协议输出"，再学"答对"。

## 九、调试与验证路径

别一上来就端到端跑。按这个顺序，每步都能把问题隔离在小范围内：

1. **单独验检索服务**。直接 `curl -X POST http://127.0.0.1:8000/retrieve -d '{"queries":["test"],"topk":3}'`，确认返回结构能被 `_passages2string` 消费。
2. **只跑生成，dump 轨迹**：
   ```bash
   python train.py --debug-rollout-only --save-debug-rollout-data /tmp/r_{rollout_id}.pt ...
   ```
   然后重点检查三件事：轨迹里 `<information>` 块是否正常注入；`loss_mask` 在 observation 段是否为 0；`len(tokens) == len(loss_mask)` 且与 `rollout_log_probs` 等长。
3. **确认 reward 分布不是常数**。如果全是 0，九成是 §5 的两个坑之一。`compute_score_em` 自带 1/64 概率打印样例（golden answer / 提取到的 answer / 完整轨迹），直接看这个打印最快。
4. **只跑训练**：`--debug-train-only --load-debug-rollout-data` 用固定数据反复跑，调并行配置和显存。
5. **最后端到端**，关注 `TRUNCATED` 比例（高则加长度预算）和检索调用的 P99 延迟。

遇到 Ray 启动卡住可以 `rm -rf /root/.cache`；检索服务无响应用 `lsof -i :8000` + `nvidia-smi` 确认进程和显存状态。

## 小结

把 search agent 跑起来，真正需要你写的只有两个函数：

- `generate()` —— 多轮循环 + 停止词注入 + `loss_mask` 切分 + token/logprob 对齐；
- `reward_func()` —— 分层奖励 + 格式校验。

其余的超采样、动态过滤、并发控制、abort、权重同步，slime 的默认 `generate_rollout` 已经包办。

而把它从示例变成业务可用，重点在三件事：**把检索换成自有系统**（统一契约让这件事只需改一个 async 函数）、**让分层奖励真正生效**（示例里它是失效的）、**把失败和延迟变得可观测**（检索是外部依赖，静默失败会以"reward 下降"的形式出现，极难归因）。

而真正决定项目成败的，是第六节那四个问题。它们可以收敛成一句话：**奖励函数是你和模型之间的对抗界面。**

- **开放问题**先尽力拆成可验证子约束，模型打分只补最后一块，且让规则部分占主导（§6.1）；
- **长轨迹的功劳分配**，slime 默认是整条广播同一个 advantage——优先靠「组内对比 + 轻微步数惩罚」解决，不要一上来就动 `--custom-advantage-function-path`（§6.2）；
- **rubric** 的价值在于把「judge 给个说不清的 0.7」变成「十条二元判定 + 代码算分」，可解释、可审计、可单条调整，否决项必须用硬规则兜底（§6.3）；
- **reward hacking 不是能被一次性消除的 bug，而是持续存在的压力**。防线是三层：奖励设计上让精确率与召回率配对、过程分由结果分解锁；监控上盯分项之间的「背离」；流程上定期人工读轨迹（§6.4）。

延伸方向，按需查文档：

| 想做什么 | 去哪 |
| :--- | :--- |
| 生成与训练完全异步、不互相等待 | [`examples/fully_async/`](https://github.com/THUDM/slime/tree/main/examples/fully_async)、[docs/zh/advanced/rollout-scheduling.md](https://github.com/THUDM/slime/blob/main/docs/zh/advanced/rollout-scheduling.md) |
| 代码 agent / 工具调用 / 多 agent | [`examples/coding_agent_rl/`](https://github.com/THUDM/slime/tree/main/examples/coding_agent_rl)、[`examples/retool/`](https://github.com/THUDM/slime/tree/main/examples/retool)、[`examples/multi_agent/`](https://github.com/THUDM/slime/tree/main/examples/multi_agent) |
| 多模态 agent | [`examples/geo3k_vlm_multi_turn/`](https://github.com/THUDM/slime/tree/main/examples/geo3k_vlm_multi_turn) |
| 权重同步只传增量 | [docs/zh/advanced/delta-weight-sync.md](https://github.com/THUDM/slime/blob/main/docs/zh/advanced/delta-weight-sync.md) |
| 超大规模下的数据持久化与容错 | [docs/zh/advanced/straw.md](https://github.com/THUDM/slime/blob/main/docs/zh/advanced/straw.md)、[fault-tolerance.md](https://github.com/THUDM/slime/blob/main/docs/zh/advanced/fault-tolerance.md) |
| 接外部推理引擎（含 vLLM） | [docs/zh/advanced/external-rollout-engines.md](https://github.com/THUDM/slime/blob/main/docs/zh/advanced/external-rollout-engines.md) |

---

系列到此结束：[第 1 篇](/blog/slime-01-getting-started/) 建立骨架认知，[第 2 篇](/blog/slime-02-core-pipeline/) 讲透内部机制，[第 3 篇](/blog/slime-03-algorithm-design/) 拆解算法设计，本篇动手落地。
