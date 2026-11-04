---
title: "长轨迹 Agent RL 的信用分配：从 GRPO 到 GiGPO"
description: "一条 20 步的 agent 轨迹只有一个终局 0/1 奖励，梯度该发给谁？本文梳理信用分配的六大类解法，并以 GiGPO（NeurIPS 2025）为主线，从论文公式一路走到 verl-agent 的检索服务搭建、训练参数，以及不依赖任何框架的自研实现——含一个官方源码校对出来的易错点。"
pubDate: 2026-02-18
tags: ["强化学习", "Agent", "GiGPO", "信用分配", "Search Agent", "verl-agent"]
---

> **本文解决一个问题**：多轮 Agent（搜索、工具调用、网页操作）训练时，轨迹动辄十几步甚至上百步，但奖励往往只有最后那一个 0/1。这个信号怎么分给中间的每一步？
>
> 全文两部分：**第一部分**给信用分配（credit assignment）的方法全景和选型决策树；**第二部分**以 GiGPO 为主线落地一个 Search Agent——从公式到检索服务、训练脚本、调参，再到不用 verl-agent 时的自研实现。第三、七节是自研实现时最容易踩的坑（已按官方 `verl-agent` 源码校对）。
>
> 前置知识：[GRPO 与 PPO 的优势估计](/blog/verl-04-ppo-grpo-core/)；Search Agent 的工程侧（检索服务、loss_mask、分层奖励）可对照[《slime 学习笔记（四）》](/blog/slime-04-search-agent-rl/)。

## 一、问题定义：信用分配到底在解决什么

**信用分配 = 当只有最终结果的好/坏信号时，搞清楚多步轨迹中「哪一步真正重要」。**

```text
Agent 执行 20 步 → 最终成功 ✅
  第 3 步：打开冰箱（关键！）
  第 7 步：看了一下窗户（无用）
  第 15 步：拿起鸡蛋（关键！）
  ...其余步骤：可有可无

问题：奖励信号只有一个 ✅ / ❌，如何把功劳/责任分配到每一步？
```

信用分配有三个层级，Agent 场景卡在中间那层：

| 层级 | 粒度 | 典型场景 |
| :--- | :--- | :--- |
| **Token 级** | 每个 token 贡献多少 | 单轮推理（数学 CoT） |
| **Turn / Step 级** | 每个动作轮次贡献多少 | 多轮 Agent（搜索、工具调用） |
| **Episode 级** | 整条轨迹好不好 | GRPO 基线（所有 step 共享同一个 advantage） |

推理 RL（数学/代码）之所以用 episode 级就够了，是因为它本质上只有「一轮」。一旦进入 Agent 场景，难度是**质变**的：

| 对比维度 | 推理 RL | Agent RL |
| :--- | :--- | :--- |
| 轨迹长度 | 500 ~ 30K tokens | 10K ~ 500K+ tokens |
| 轮次数 | 1 轮 | 10 ~ 100+ 轮 |
| 环境交互 | 无（纯文本生成） | 有（工具返回、网页状态） |
| 状态转移 | 确定性（token 接 token） | 随机性（环境响应不可控） |
| 奖励类型 | 二值（对/错） | 稀疏（最终成功/失败） |

轨迹变长带来的直接后果是**信号稀释**：一个 0/1 奖励摊到 20 步上，每步分到的有效梯度微乎其微，实用中最典型的表现就是 agent 学着学着「忘了调工具」。

## 二、六大解法全景

按**核心机制**划分（参考综述 [arXiv:2604.09459](https://arxiv.org/abs/2604.09459)，69 篇论文库）：

```text
                        信用分配方法
                            │
        ┌───────────┬───────┼───────┬──────────┬──────────┐
        │           │       │       │          │          │
   ① 轨迹级      ② 同状态   ③ 学习式  ④ 特权    ⑤ 图结构   ⑥ 手工
   Advantage     分组对比    PRM     Critic     分析       Reward
   (baseline)   (Critic-  (Process  (训练时    (状态转    Shaping
                 Free)    Reward)   特权信息)   移图)
```

### ❶ 轨迹级 Advantage（baseline）

**代表**：GRPO、REINFORCE、RLOO。同一个 prompt 采样 N 条轨迹，按终端奖励的相对排名给**所有 token 同一个 advantage**。

```text
轨迹 A: 成功 → advantage = +1 → 20 步全部 +1
轨迹 B: 失败 → advantage = -1 → 20 步全部 -1
```

问题是第 3 步的关键决策和第 17 步的无关操作拿到完全相同的信号。适合短轨迹（< 5 步）或作为 baseline。

### ❷ 同状态分组对比（Critic-Free Step-Level）

**核心思想**：如果多条轨迹经过了同一个状态，那么从该状态出发的**不同动作**就构成一个天然对照组，可以直接比较好坏。

- **GiGPO**（[arXiv:2505.10978](https://arxiv.org/abs/2505.10978)，NeurIPS 2025）：识别「锚状态」，同状态的不同动作组成 step-level group，组内对比出 micro advantage。不需要 Critic、不需要额外 rollout，**零额外计算开销**。论文报告 ALFWorld +12%、WebShop +9%（vs GRPO）。
- **HGPO**（[arXiv:2602.22817](https://arxiv.org/abs/2602.22817)，ICLR 2026）：GiGPO 的升级版，把分组从「扁平」扩展为层次分组，覆盖状态重合更稀疏的场景。

局限很明确：**依赖状态重叠**。ALFWorld 里多条轨迹都到过「厨房」很正常，但在开放式语言环境（搜索结果各不相同）里，状态几乎不会完全相同——这正是第四、五节要解决的问题。

### ❸ 学习式 Process Reward Model

- **iStar / OPRL**（[arXiv:2509.19199](https://arxiv.org/abs/2509.19199)）：用终端奖励对轨迹排序构造正负样本，用 DPO 目标训 PRM，**不需要 step-level 人工标注**；隐式 step reward 是 potential-based shaping，理论上不改变最优策略。粒度是 turn 级而非 token 级，避免过细粒度方差。适用于**不可验证奖励**的开放式环境（SOTOPIA 上目标完成率 +14%）。
- **SWEET-RL**（[arXiv:2503.15478](https://arxiv.org/abs/2503.15478)）：训一个**特权 Critic**——训练时能看到完整未来轨迹和金标准，推理时不参与。非对称训练是关键设计，打分用 token-level mean log-prob 而非回归头。

### ❹ Turn-Level MDP 建模

**MT-GRPO**（[arXiv:2505.11821](https://arxiv.org/abs/2505.11821)）把多轮交互显式建模成 turn-level MDP，结论非常实用：

| 奖励结构 | 说明 | 效果 |
| :--- | :--- | :--- |
| Terminal | 只在最后一步给奖励 | 最差 |
| Delayed | 中间有部分奖励但延迟 | 中等 |
| **Per-turn** | **每轮都有即时奖励** | **最好** |

**如果你能为每一步设计哪怕粗糙的中间奖励，就应该用**——收益远大于成本。另外，轨迹级 advantage 的方法在长轨迹上会「忘记调工具」，turn-level 方法不会。

### ❺ 图结构分析

**GEPO**（[arXiv:2510.26270](https://arxiv.org/abs/2510.26270)）把多条轨迹构建成**状态转移图**，用拓扑结构判断状态重要性：

```text
多条轨迹 → 状态转移图
  → TCC(state) = 中介中心性(betweenness) × 与任务的语义相似度
  → 轨迹级：用 TCC 做自适应折扣
  → Step 级：按后继状态的 TCC 缩放 credit
```

直觉很好：处在「交通枢纽」位置、且语义上跟任务相关的状态，更可能是关键决策点。ALFWorld +1.1%、WebShop +3.2%、QA +3.8%（vs 最强 baseline），开销很低。

### ❻ 手工 Reward Shaping

| 方法 | Reward 设计 | 场景 |
| :--- | :--- | :--- |
| Search-P1 | path-centric：检索相关性 + 去重 | Agentic RAG |
| StepSearch | 信息增益 − 冗余惩罚 + query-intent 对齐 | 多跳检索 |
| ARTIST | Function Reward（工具调用正确性）+ State Reward（状态追踪） | 多轮工具使用 |
| G-RA | 门控奖励：中间奖励须过长期目标阈值才累积 | 防 reward hack |

最容易落地，但需要领域知识。建议从简单规则起步：工具调用成功 +1、检索命中 +小奖励。注意越精细的中间奖励越容易被 hack，需要和 [rubric 类结构化评估](/blog/rubrics-for-llm-eval-and-rl/) 结合。

### 全景对比与选型决策树

| 方法 | 信用粒度 | 需要 Critic | 额外标注 | 额外计算 | 适用范围 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GRPO（baseline） | Episode | ❌ | 无 | 无 | 短轨迹 |
| **GiGPO** | Step（同状态分组） | ❌ | 无 | **≈0** | 有限状态空间 |
| **HGPO** | Step（层次分组） | ❌ | 无 | ≈0 | 有限状态空间 |
| iStar / OPRL | Turn（在线 PRM） | ✅（共享骨干） | 无 | 中等 | **开放式 + 不可验证** |
| SWEET-RL | Turn（特权 Critic） | ✅（特权信息） | 无 | 中等 | 协作推理 |
| MT-GRPO | Turn（MDP 建模） | PPO 时需要 | **需要中间奖励** | 低 | 有中间奖励的场景 |
| GEPO | Step（图结构） | ❌ | 无 | 低 | 中等状态空间 |
| 手工 Shaping | Turn（规则） | ❌ | 无 | 无 | 特定领域 |

```text
你的 Agent RL 场景是？
│
├─ 轨迹短（≤ 5 步）、奖励密集
│   └→ 直接用 GRPO / REINFORCE
│
├─ 轨迹长（10+ 步）、奖励稀疏
│   ├─ 能设计每步中间奖励吗？
│   │   ├─ 能 → MT-GRPO（PPO + per-turn reward）
│   │   └─ 不能 ↓
│   ├─ 环境状态空间有限（ALFWorld / WebShop 类）？
│   │   ├─ 是 → GiGPO / HGPO（零额外开销）
│   │   └─ 否（开放式语言环境）↓
│   ├─ 能承受训一个 PRM 的开销吗？
│   │   ├─ 能 → iStar/OPRL（最通用，含不可验证奖励）
│   │   │      或 SWEET-RL（训练时有特权信息）
│   │   └─ 不能 → GEPO（利用采样图结构）
│   └─ 以上都不方便？
│       └→ 手工 Reward Shaping 起步，后续升级为学习式方法
```

## 三、GiGPO 逐公式拆解

GiGPO 的结构是「组中有组（group-in-group）」：外层是 GRPO 的 episode 组，内层是锚状态构成的 step 组。

**① Episode-level advantage（同 GRPO，论文 Eq.3）**

$$
A^{E}(\tau^{(i)}) = \frac{R(\tau^{(i)}) - \mathrm{mean}\big(\{R(\tau^{(j)})\}\big)}{\mathrm{std}\big(\{R(\tau^{(j)})\}\big)}
$$

**② 每步的折扣回报（Eq.5）**

$$
R_t^{(i)} = \sum_{k=t}^{T} \gamma^{\,k-t}\, r_k^{(i)}, \quad \gamma \in (0, 1]
$$

Search 场景通常只有终端奖励，此时 $R_t^{(i)} = \gamma^{\,T-t} \cdot R(\tau^{(i)})$。

**③ 锚状态分组（Eq.6，整个方法的核心）**

$$
G^{S}(\tilde{s}) = \Big\{\, \big(a_t^{(i)},\; R_t^{(i)}\big) \;\Big|\; s_t^{(i)} = \tilde{s} \,\Big\}
$$

请注意这个集合的条件：**只有 $s_t^{(i)} = \tilde{s}$，没有 $t$**。也就是说，一个 step group 是「所有轨迹中到达该状态的所有 transition」，与它发生在第几步无关。官方实现里 batch 的每一行就是一个 transition，聚类全程不涉及 step 号（见第七节的易错点）。

**④ Step-level relative advantage（Eq.7）**：在 $G^S(\tilde{s})$ 内做同样的标准化，把 $(a_t^{(i)}, R_t^{(i)})$ 的回报和组内均值比。组内只有 1 个成员时 advantage 自然为 0。

**⑤ 合并（Eq.8）**

$$
A_t^{(i)} = A^{E}(\tau^{(i)}) + w \cdot A^{S}\big(a_t^{(i)}\big)
$$

$w$ 就是实现里的 `step_advantage_w`，$w=0$ 时退化为纯 GRPO。

## 四、Search Agent：锚状态几乎不匹配怎么办

### 交互流程

```text
Step 1: 用户提问 "谁发明了电话？"
  → Agent 思考 → 生成 search query: "invention of telephone"
  → 环境返回检索结果（observation）

Step 2: Agent 看到检索结果
  → 思考"信息不够" → 生成新 query: "Alexander Graham Bell telephone patent"
  → 环境返回新检索结果

Step 3: Agent 认为信息充分
  → 生成最终答案 → 环境给出奖励 ✅ / ❌
```

每一步 Agent 生成的「思考 + query / answer」是 **action**，环境返回的检索结果是 **observation**。

### 双层 Advantage 在搜索里的样子

```text
Episode-level（宏观）：同一问题采样 N 条轨迹，按最终对错排序
  → 成功轨迹 advantage > 0，失败轨迹 < 0（跟 GRPO 一样）

Step-level（微观，GiGPO 独有）：
  轨迹 A 的第 t 步 和 轨迹 B 的第 t' 步 到达了同一（或相似的）状态 → 锚状态
  注意 t 和 t' 不必相等，步号不参与分组
  → 它们从该状态出发的不同 action 构成「step group」→ 组内对比

最终：A_total = A_episode + λ × A_step
```

### 相似度匹配

GiGPO 原始设计靠**环境状态完全匹配**识别锚状态（ALFWorld 里多条轨迹都到了「厨房」）。但 Search 场景中不同轨迹的检索结果文本几乎不会完全相同，完全匹配会退化成纯 GRPO。

`verl-agent` 在 2025.08 引入了 **similarity-based anchor state matching**：不要求完全匹配，而是算状态文本相似度（默认 `SequenceMatcher` 比率），超过阈值（如 0.9）即视为同一状态。对应开关是 `enable_similarity` + `similarity_thresh`。

### 一个具体例子

```text
问题: "第一个登月的人是谁？"  采样 5 条轨迹

轨迹 1: search("first moon landing")   → 读结果 → search("Apollo 11 crew") → 答 "Neil Armstrong" ✅
轨迹 2: search("first person on moon") → 读结果 → 答 "Buzz Aldrin" ❌
轨迹 3: search("moon landing history") → 读结果 → search("Neil Armstrong") → 答 "Neil Armstrong" ✅
轨迹 4: search("first moon landing")   → 读结果 → 答 "Armstrong" ✅
轨迹 5: search("Apollo mission")       → 读结果 → search("moon landing") → 答 "Unknown" ❌

Episode-level：轨迹 1,3,4 的 A_episode > 0；轨迹 2,5 < 0

Step-level（GiGPO 独有）：
轨迹 1 Step1 与 轨迹 4 Step1 的检索结果相似度 0.98 > 0.9 → 匹配，归为一组
  轨迹 1 后续 search("Apollo 11 crew") → 成功 → A_step > 0
  轨迹 4 直接回答（信息不足）         → 部分成功 → A_step ≈ 0
  结论：从这个状态出发，继续搜索比直接回答更好
```

最后这条信号在纯 GRPO 里**完全丢失**——两条轨迹都是 R=1，所有 step 拿到同一个 advantage，模型学不到「这一步该继续搜」。

## 五、verl-agent 实战：环境、检索服务与训练脚本

> 参考实现：[github.com/langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)

### 5.1 环境搭建

```bash
# 1. 基础环境
conda create -n verl-agent python==3.12 -y
conda activate verl-agent
pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# 2. 安装 verl-agent
git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent && pip install -e .

# 3. 安装 Search 环境
cd ./agent_system/environments/env_package/search/third_party
pip install -e . && pip install gym==0.26.2

# 4. 准备数据集
cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
# 数据保存到 ~/data/searchR1_processed_direct/
```

### 5.2 检索服务

faiss-gpu 与训练依赖容易冲突，官方建议**单独开一个 conda 环境**：

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
conda install numpy==1.26.4
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi

# 下载索引
local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz

# 启动（约占用 6GB 显存，训练规划时要预留）
bash examples/search/run_search_server.sh
```

### 5.3 训练脚本关键参数

```bash
# ===== GiGPO 核心参数 =====
algorithm.adv_estimator=gigpo                # 使用 GiGPO
algorithm.gigpo.step_advantage_w=1.0         # A = A_episode + 1.0 * A_step
algorithm.gigpo.mode="mean_std_norm"         # 或 "mean_norm"
algorithm.gigpo.enable_similarity=True       # Search 场景核心开关！
algorithm.gigpo.similarity_thresh=0.9

# ===== 环境参数 =====
env.env_name=search
env.max_steps=4                              # 每条轨迹最多 4 轮
env.rollout.n=5                              # group_size
env.history_length=4                         # 保留最近 4 轮历史
env.search.search_url='http://<IP>:<PORT>/retrieve'

# ===== 其他 =====
algorithm.gamma=0.95                         # 折扣因子
actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.use_kl_loss=True     # KL 正则
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.use_invalid_action_penalty=True   # 无效动作惩罚
actor_rollout_ref.actor.invalid_action_penalty_coef=0.01
data.train_batch_size=256
data.max_prompt_length=4096
data.max_response_length=512
trainer.n_gpus_per_node=8
```

启动：`bash examples/gigpo_trainer/run_search.sh vllm`。

## 六、调参指南

**`similarity_thresh`（最关键）**

| 阈值 | 效果 | 适用 |
| :--- | :--- | :--- |
| 0.95 | 只匹配几乎相同的状态 → step group 很少 → 接近纯 GRPO | 状态差异大的场景 |
| **0.90** | **平衡点（官方默认）** | **Search 场景推荐** |
| 0.85 | 更宽松 → group 更多 → 粒度更细，可能有噪声 | 匹配率过低时 |
| < 0.80 | 不同状态被错误归组 | 不推荐 |

先用 0.9 跑一轮，观察 `step_group_count` / `step_group_avg_size`；如果绝大多数 step 没匹配上，再降到 0.85。

**`group_size`（`env.rollout.n`）**：越大，episode advantage 越准、命中同状态概率越高，但推理开销线性增长。4 卡 → 4，8 卡 → 5~8，Search 场景 4~6 通常够用。

**`max_steps`**：4 适合单跳/简单多跳，6 适合 MuSiQue / HotpotQA 级别的复杂多跳，8+ 训练慢且信用分配更难。从 4 起步，若模型总把步数用完还答不对再加到 6。

**`step_advantage_w`**：0 退化为纯 GRPO，0.5 温和，1.0 官方默认（step 与 episode 等权），2.0 激进。训练不稳就降到 0.5。

**归一化模式**

| Mode | 公式 | 特点 |
| :--- | :--- | :--- |
| `mean_std_norm` | $(R - \mathrm{mean}) / \mathrm{std}$ | GRPO 式，可能有 difficulty bias |
| `mean_norm` | $R - \mathrm{mean}$ | RLOO 式，无 difficulty bias，长轨迹更稳定 |

Search 场景官方默认 `mean_std_norm`；loss 震荡时换 `mean_norm`。

## 七、不用 verl-agent：自己实现时要注意什么

如果你在自己的训练管道里实现（例如基于 [slime](/blog/slime-03-algorithm-design/)），核心逻辑不长，但有三个坑。

### 7.1 数据结构

```python
@dataclass
class Trajectory:
    prompt: str              # 用户问题
    steps: List[Step]        # 多轮交互
    reward: float            # 最终奖励（0 或 1）

@dataclass
class Step:
    state: str               # 锚观测：当前 env 观测（检索结果），不要拼历史
    action: str              # Agent 生成的文本（思考 + query/answer）
    action_log_probs: Tensor # 每个 token 的 log prob
```

### 7.2 核心实现

```python
def compute_gigpo_advantages(trajectories, step_w=1.0, sim_thresh=0.9):
    """trajectories: 同一个 prompt 的 N 条轨迹"""
    rewards = [t.reward for t in trajectories]

    # ① Episode-level advantage（和 GRPO 一样）
    mean_r, std_r = np.mean(rewards), np.std(rewards) + 1e-8
    episode_advs = [(r - mean_r) / std_r for r in rewards]

    # ② Step-level advantage（GiGPO 独有）
    # 官方 verl-agent 的 build_step_group：先按 prompt 切开，再对该 prompt 的
    # 全部 transition 按 anchor_obs 聚类，全程没有 step_idx 维度。
    # 论文 Eq.6：G^S(s̃) = {(a_t^(i), R_t^(i)) | s_t^(i) = s̃} —— 只按状态分组，与 t 无关，
    # 所以同一个状态出现在轨迹 A 第 2 步、轨迹 B 第 5 步也要归为一组。
    entries = []  # 每个元素 = 一个 transition
    for traj_idx, traj in enumerate(trajectories):
        for step_idx, step in enumerate(traj.steps):
            entries.append({
                "traj_idx": traj_idx,
                "step_idx": step_idx,
                "state": step.state,          # 锚观测（检索结果，不含历史）
                "future_return": compute_future_return(traj, step_idx),
            })

    # 锚状态分组：跨轨迹 + 跨步（对齐官方 build_step_group）
    groups = build_similarity_groups(entries, sim_thresh)

    step_advs = {}
    for group in groups:
        returns = [e["future_return"] for e in group]
        mean_g, std_g = np.mean(returns), np.std(returns) + 1e-8
        for e in group:
            # 组内只有 1 个成员时 mean_g == 自身 return → A_step 自然为 0
            step_advs[(e["traj_idx"], e["step_idx"])] = \
                (e["future_return"] - mean_g) / std_g

    # ③ 合并
    final_advs = {}
    for traj_idx, traj in enumerate(trajectories):
        for step_idx in range(len(traj.steps)):
            final_advs[(traj_idx, step_idx)] = \
                episode_advs[traj_idx] + step_w * step_advs.get((traj_idx, step_idx), 0.0)
    return final_advs


def build_similarity_groups(entries, threshold):
    """
    锚状态分组：范围是「同一 prompt 下的所有 transition」，
    轨迹维和步维都不参与分组 key，只按状态聚类；跨 prompt 不分组。
    复杂度 M = N 条轨迹 × L 步，两两比较 O(M²)，M 大时可用 FAISS / MinHash 预筛。
    """
    # 方法1: sentence-transformers 编码 + cosine（推荐，可批量编码）
    # 方法2: n-gram overlap / jaccard / SequenceMatcher（官方默认）
    # 方法3: 结构化状态 ID 完全匹配（最快）
    groups, used = [], set()
    for i, e1 in enumerate(entries):
        if i in used:
            continue
        group = [e1]
        used.add(i)
        for j, e2 in enumerate(entries):
            if j in used:
                continue
            if cosine_similarity(encode(e1["state"]), encode(e2["state"])) > threshold:
                group.append(e2)
                used.add(j)
        groups.append(group)
    return groups


def compute_future_return(traj, step_idx, gamma=0.95):
    """从 step_idx 开始的折扣累积奖励"""
    remaining = len(traj.steps) - step_idx - 1
    return (gamma ** remaining) * traj.reward
```

### 7.3 三个易错点

**坑 1：把分组写成「按步号分组」。** 直觉写法是先 `for step_idx` 再 `for traj_idx`，只在同一步号内聚类。这是错的：

- 同一状态在不同轨迹里出现在不同步号是常态（轨迹长度不齐时更是必然），按步号分组会大量漏配；
- 按步号分组退化成「同一层内比较」，丢掉了 GiGPO 靠状态重合做信用分配的核心机制；
- 极端情况下每个 step_idx 只有 1 个样本 → step group 全是 singleton → step advantage 恒为 0 → **静默退化成纯 GRPO**（loss 看着在降，其实没训到东西）。

官方 `verl-agent/gigpo/core_gigpo.py` 里，`build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)` 先按 `index`（prompt uid）切开，然后在这个 prompt 的**全部 transition** 上按 `anchor_obs` 聚类，全程没有 `step_idx`。正确做法是：**摊平所有轨迹的所有步 → 按状态做一次全局聚类 → 组内归一化**。另外官方对 singleton 组并不跳过（`step_norm_reward` 里 size=1 时 mean 等于自身 → A_step=0），与上面写法等价。

**坑 2：把完整对话历史拼进 `state`。** 参与聚类的应该是**纯环境观测**（本次检索结果）。拼了历史之后，两条轨迹即使到达同一状态，也会因前缀不同而永远匹配不上，相似度分组直接失效。

**坑 3：忘记 token masking。** 环境返回的 observation token（检索到的文档）必须 mask 掉，只对 agent 生成的 action token 算 loss，否则梯度会泄漏到不可控内容上。

```python
for traj in batch:
    for step in traj.steps:
        loss_mask[observation_token_positions] = 0
        loss_mask[action_token_positions] = 1
```

## 八、局限、进阶组合与实验数字

| 局限 | 说明 | 应对 |
| :--- | :--- | :--- |
| 状态匹配率低 | Search 场景中轨迹很少到达相似状态 | ① 调低 `sim_thresh` ② 增大 `group_size` ③ 结合手工 step reward |
| 相似度计算开销 | 需要编码 + 两两比较 | ① 只对 query 部分算相似度 ② 用快速哈希近似 / FAISS 预筛 |
| 短轨迹收益有限 | `max_steps=2~3` 时对比空间很小 | 增加 `max_steps ≥ 4` |
| 二值奖励信息有限 | 终端 0/1 让 future return 区分度低 | 用 F1 或 partial match 做连续奖励 |

**进阶组合**：

1. **GiGPO + 手工 step reward**：检索命中 gold evidence 时给 +小奖励；
2. **GiGPO + StepSearch 式 shaping**：信息增益 reward + 冗余惩罚；
3. **GiGPO → iStar/OPRL**：先用 GiGPO 零成本冷启动，再切到学习式 PRM 做精细化训练。

**论文报告的 Search-R1 结果**（7 个 QA 数据集平均）：

| 方法 | Qwen2.5-3B | Qwen2.5-7B |
| :--- | :--- | :--- |
| R1-Instruct（无搜索） | 27.1% | 22.4% |
| Search-R1（GRPO） | 32.5% | 38.5% |
| ZeroSearch | 31.7% | 39.1% |
| **GiGPO（sim=0.9）** | **42.1%** | **47.2%** |

3B 上比 Search-R1 高 9.6 个点，7B 上高 8.7 个点。

## 小结

- **别拿纯 episode-level advantage 训长轨迹 agent**。GRPO 在 10+ 步任务上信号稀释严重，典型症状就是「忘记用工具」。
- **先问自己能不能设计中间奖励**。MT-GRPO 的结论很硬：per-turn reward 远好于 terminal reward，哪怕规则很粗糙。
- **有状态重叠就用 GiGPO**：不增加任何 rollout 和模型开销，只在已有采样数据上加一层分组，是性价比最高的一次升级。状态重合稀疏时看 HGPO。
- **开放、不可验证的场景走 iStar/OPRL**，通用性最好，代价是要训一个 PRM。
- **自己实现时盯住分组口径**：摊平所有 transition 后按状态全局聚类，步号不参与分组（Eq.6），锚观测不要拼历史，observation token 必须 mask。

顺便说一句，信用分配不只是训练问题。知道「哪一步出了问题」对 agent 的可解释性和持续迭代同样关键，这也是它在评估侧的价值。

## 参考资源

| 资源 | 链接 |
| :--- | :--- |
| GiGPO（NeurIPS 2025） | [arXiv:2505.10978](https://arxiv.org/abs/2505.10978) |
| HGPO（ICLR 2026） | [arXiv:2602.22817](https://arxiv.org/abs/2602.22817) |
| iStar / OPRL | [arXiv:2509.19199](https://arxiv.org/abs/2509.19199) |
| SWEET-RL | [arXiv:2503.15478](https://arxiv.org/abs/2503.15478) |
| MT-GRPO | [arXiv:2505.11821](https://arxiv.org/abs/2505.11821) |
| GEPO | [arXiv:2510.26270](https://arxiv.org/abs/2510.26270) |
| 信用分配综述（69 篇） | [arXiv:2604.09459](https://arxiv.org/abs/2604.09459) |
| verl-agent（GiGPO 官方实现） | [github.com/langfengQ/verl-agent](https://github.com/langfengQ/verl-agent) |
| Search 训练脚本 | [run_search.sh](https://github.com/langfengQ/verl-agent/blob/master/examples/gigpo_trainer/run_search.sh) |
| Search-R1 原始框架 | [github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) |
| 方法目录（living repo） | [Awesome-Credit-Assignment-in-LLM-RL](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL) |
