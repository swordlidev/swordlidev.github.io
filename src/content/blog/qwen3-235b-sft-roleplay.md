---
title: "Qwen3-235B-A22B 64 卡全参 SFT 实战复盘：角色扮演 Agent"
description: "在 64 卡 RoCE 集群上对 Qwen3-235B-A22B（MoE）做角色扮演全参 SFT 的完整复盘：环境、网络、数据、模型、权重转换、训练到踩坑七步链路，附实跑数字与参数。"
pubDate: 2025-10-05
tags: ["Qwen3", "SFT", "分布式训练", "MoE", "角色扮演", "实战"]
---

> 对象：在 64 卡（8 节点 × 8 卡）RoCE 集群上，对 **Qwen3-235B-A22B（MoE）** 做角色扮演 **全参 SFT**。
> 本文从 **环境 → 网络 → 数据 → 模型 → 权重转换 → SFT 训练 → 踩坑** 七个方面复盘整条链路，所有数字 / 参数均来自实跑日志与脚本。
> 配套脚本（仓库根目录）：`run_convert_qwen235b.sh`（HF→mcore 转换）、`run_sft_qwen235b.sh`（64 卡 SFT）、`net_env.sh`（RoCE 固化）、`te-venv/te_env.sh`（TE/cuDNN 环境）。

这是一篇把 235B MoE 大模型"真正跑起来"的工程复盘。如果你对其中的并行概念（TP/PP/DP/EP、RoCE、分布式优化器等）还不熟悉，建议先读我的[《Megatron-LM 分布式训练》系列](/blog/megatron-lm-01-overview/)打底，本文则是这些概念在一次真实任务里的落地。

## 项目背景：角色扮演（Role-play）

### 为什么做这件事

我们要把通用大模型 **Qwen3-235B-A22B** 微调成一个**高质量的角色扮演 Agent**——即给定一张「角色卡（人物设定 + 世界观 + 开场白）」后，模型能**始终以第一人称、沉浸式地出演该 NPC 角色**：既不跳戏（OOC）、不替用户发言，又能推动剧情、保持长程一致性。SFT 的训练信号正来自一批**经过质量筛选的优质多轮角色扮演对话轨迹**。

### 数据来源：Role-play Benchmark（MiniMax）

数据取自 `data/role-play-bench`（HuggingFace `MiniMaxAI/role-play-bench`，CC BY 4.0 / BY-SA 4.0）。这是一个评测**角色扮演 Agent** 的中英双语基准，核心理念是 **Situated Reenactment（情境重演）**：在特定的 `{世界观} × {剧情}` 坐标、面向给定 `{用户偏好}` 下考察 NPC 表现。它有几个与训练强相关的设计：

- **非可验证任务 + 负向评测（Flooring）**：角色扮演没有唯一"正确答案"（例如傲娇角色被问"你喜欢我吗"），因此评测不定义"完美回答"，而是**检测 misalignment**——哪些明显不符合人设。
- **Model-on-Model 自博弈**：用精心设计的 User Simulator 与被测模型多轮对话，生成对话轨迹（每会话 **100 轮**、每场景 **3 次**采样、按 **20 轮切块**做 LLM-as-Judge）。
- **三大目标 / 六维打分（0–100）**：

  | 核心目标（权重） | 关注点 | 维度 |
  |---|---|---|
  | **Worlds（50%）** | 世界观一致性与逻辑，防灾难性遗忘/人物混淆/时空错误 | basics、logic、knowledge |
  | **Stories（25%）** | 叙事推进、表达多样性，避免机械重复/停滞/OOC | diversity、content_logic |
  | **User Preferences（25%）** | 交互质量，不替用户发言、不自说自话、不过度拒绝 | interaction |

- **数据规模**：中文 50 seeds / 1650 对话，英文 45 seeds / 1485 对话；三张表——`seeds`（角色卡+场景）、`dialogues`（多轮自博弈轨迹）、`evaluations`（六维分数）。

> 说明：该 benchmark 本身是**评测集**；我们是把其中**高分对话轨迹**反过来当作 **SFT 训练语料**，让 235B 学习这些优质角色扮演行为。

### 从 benchmark 到 SFT 语料（`build_sft.py` 做了什么）

脚本把三张表 join 后，按 benchmark 的加权总分做**质量筛选**，再转成 Megatron 原生 SFT 的 `{"messages":[...]}`：

1. **质量门槛**：`overall ≥ 80`（`0.5·Worlds + 0.25·Stories + 0.25·Preferences`）、`basics ≥ 85`、`logic ≥ 60`；可按 `model_name` 限定来源、按 seed 限量、按轮数截断。
2. **角色映射**：训练模型**成为 AI 角色** → AI 轮映射为 `assistant`（**计损失**），user 轮为 `user`（**掩码**）。
3. **结构规整**（满足 Megatron 严格模板）：首轮 `system`（角色卡，**开场白 prologue 折叠进 system**），其后 **严格 user/assistant 交替**，丢弃开头多余的 assistant 与结尾悬空的 user，**保证以 assistant 轮结尾且 assistant 内容非空**。
4. **产物**：中英合并后 shuffle、2% 作验证集 → `sft_all_train.jsonl`（**806 条**）+ `sft_all_val.jsonl`（**16 条**），即本次 SFT 的输入。

这也解释了后面第 3 节两个工程取舍的来源：数据天然是**多轮长对话**（→ THD packing）、且设计上**只在 assistant 轮计损失**（→ 掩码方案的取舍）。

---

## 0. 结论速览（TL;DR）

| 环节 | 状态 | 关键数字 |
|---|---|---|
| 环境（TE + FP8 + cuDNN 9.27 隔离） | ✅ | cuDNN 9.27 / torch 2.8.0+cu128 / NCCL 2.27.3 |
| 64 卡 RoCE all-reduce 基准 | ✅ | busbw **~184 GB/s**（1GB 消息，8×200Gb rail-optimized） |
| 64 卡多机 TE 冒烟（bf16/FP8） | ✅ | loss 11.2→7.9，~656 ms/iter |
| 235B shape 冒烟（真实架构 + mock） | ✅ | 235.10B 参数，峰值 **79 GB/卡**，不 OOM |
| HF → mcore 权重转换 | ✅ | 8 分片 **439 GiB≈470GB**，耗时 **~67 分钟** |
| 64 卡全参 SFT（50/100 步） | ✅ | lm loss **2.20 → 0.83** |

**一句话**：环境 / 网络 / 数据 / 模型 / 转换 / 训练这条重链路**全部打通且健康**。

---

## 1. 环境配置

### 1.1 硬件与集群拓扑

- **8 节点 × 8 卡 = 64 GPU**（每卡约 96GB）。
- master-0 + worker-0..6；节点名形如 `train-...-worker-N`。
- 免密 ssh、`pdsh` 扇出、共享存储（Lustre）全节点路径一致（`te-venv/`、`cudnn-ext/`、数据与脚本均无需各节点单独部署）。

### 1.2 软件栈

| 组件 | 版本/说明 |
|---|---|
| PyTorch | 2.8.0+cu128 |
| NCCL | 2.27.3 + cuda12.9 |
| Transformer Engine | pytorch 后端，FP8 路径可用 |
| cuDNN | **9.27**（关键，见下） |
| Megatron-LM | 本仓库自带（含 Qwen3 官方配置 + 原生 SFT 通路） |
| 权重转换 | nvidia-modelopt 0.47.0（`examples/post_training/modelopt`） |

### 1.3 关键环境隔离：cuDNN 9.27 预载（`te-venv/te_env.sh`）

TE 运行时要求 **cuDNN ≥ 9.12**，而 venv 自带 9.10.2 → 用 `LD_PRELOAD` 抢先载入 `cudnn-ext` 里的 9.27，并设 `CUDNN_HOME/CUDNN_PATH` 让 TE 的 `_load_cuda_library` 命中系统路径：

```sh
export LD_LIBRARY_PATH=".../cudnn-ext/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDNN_HOME=".../cudnn-ext/nvidia/cudnn"
export CUDNN_PATH=".../cudnn-ext/nvidia/cudnn"
export LD_PRELOAD=".../cudnn-ext/nvidia/cudnn/lib/libcudnn.so.9${LD_PRELOAD:+:$LD_PRELOAD}"
```

### 1.4 启动方式：master 单点 pdsh 扇出 + c10d rendezvous

- 只在 master-0 执行脚本，由 `pdsh -w master-0,worker-[0-6]` 把同一条 `torchrun` 推到 8 节点。
- `--rdzv_backend=c10d --rdzv_endpoint=<master_ip>:port`：各节点**自动协商 node_rank**，无需手工编号；endpoint 用 master **IP**（规避 worker 不认 master 主机名）。
- 每节点先 `source te_env.sh + net_env.sh`，再用 venv python 跑。

---

## 2. 网络（RoCE / NCCL）—— `net_env.sh`

实测固化（由 64 卡 all-reduce 基准得出）：

```sh
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5,vrdma,shca      # 实测命中 mlx5_bond_0..7
export NCCL_IB_GID_INDEX=3              # RoCE v2
export NCCL_IB_TIMEOUT=24
export NCCL_SOCKET_IFNAME=^lo,docker0,veth,cni0,flannel   # 控制面走管理网，排除虚拟网卡
```

- 每节点 **8× `mlx5_bond_0..7`**，各 200Gb，**rail-optimized**（每卡一张网卡，PCI 亲和）。
- **实测 busbw ~184 GB/s**（1GB 消息，接近 8×200Gb 线速的健康水平；TCP 对照仅 ~5 GB/s，RoCE 快 ~35×）。
- 工程铁律：**TP 绝不跨机**（走节点内 NVLink）；DP/PP/EP 的跨机通信才走 RoCE。`CUDA_DEVICE_MAX_CONNECTIONS=1`。

---

## 3. 数据

- 数据集：`data/role-play-bench/sft/`，**角色扮演多轮对话 SFT**。
- 格式：Megatron 原生 SFT 的 `{"messages":[{"role":...,"content":...}, ...]}`（jsonl，开箱即用，`SFTLowLevelDataset` 用 `load_dataset("json")` 读）。
- 规模：**训练 806 条 + 验证 16 条**多轮对话。
- 训练目标（设计）：只在 **assistant 轮**计损失（`build_sft.py` 按 assistant-only 掩码设计）。
- 打包：Megatron `--sft` 走 **THD packed**（`cu_seqlens` 变长注意力，样本拼接），因此**不能**在 dataloader 里生成 attention mask（见踩坑 §7）。

### 掩码方案的取舍（重要）

Megatron 内置 SFT prompt format 只有 `nemotron-* / identity / default` 几种：

- 带掩码的 `nemotron-*` 用 **Nemotron 词表**，与 Qwen3 不兼容；
- 能用 Qwen3 原生 `<|im_start|>` 模板的 `default` **不做掩码**（全 token 计损失）；
- Qwen3 chat 模板**无 `{% generation %}` 块** → 无法用 HF 原生 assistant 掩码。

---

## 4. 模型架构（Qwen3-235B-A22B，参数取自仓库官方配置）

```
--num-layers 94  --hidden-size 4096  --ffn-hidden-size 12288
--num-attention-heads 64  --group-query-attention --num-query-groups 4  --kv-channels 128
--qk-layernorm  --normalization RMSNorm --norm-epsilon 1e-6  --swiglu
--disable-bias-linear  --untie-embeddings-and-output-weights
--position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0
--num-experts 128 --moe-ffn-hidden-size 1536 --moe-router-topk 8 --moe-router-dtype fp32
--moe-aux-loss-coeff 1e-3 --moe-router-load-balancing-type aux_loss
--moe-token-dispatcher-type alltoall --moe-layer-freq 1 --moe-layer-recompute
--seq-length 4096 --max-position-embeddings 40960  （vocab 151936）
```

### 4.0 名字怎么读：`235B-A22B`

- **235B** = 总参数量（含全部 128 个专家），决定**模型容量**与**存储/显存**占用。
- **A22B** = Active 22B，每个 token 实际只激活 **≈22B** 参数（top-8 专家 + 注意力等共享部分），决定**单 token 计算量（FLOPs）**。
- 这正是 MoE 的精髓：**容量（235B）与算力（22B/token）解耦**——用稀疏激活把参数堆大，但每步只算一小部分。

### 4.1 规模与结构

| 参数 | 值 | 说明 |
|---|---|---|
| `--num-layers` | 94 | Transformer 解码层数 |
| `--hidden-size` | 4096 | 隐藏维 d_model（残差流宽度） |
| `--ffn-hidden-size` | 12288 | 稠密 FFN 中间维（HF config 的 `intermediate_size`）。本模型为**全 MoE**，专家 FFN 宽度另由 `--moe-ffn-hidden-size` 决定；此值作架构基准保留 |
| `--seq-length` | 4096 | 训练截断序列长度（本次 SFT；可调大到 8192） |
| `--max-position-embeddings` | 40960 | 模型支持的最大位置（Qwen3 原生上限） |
| vocab | 151936 | 词表大小（`--make-vocab-size-divisible-by 1187` 对齐后） |

### 4.2 注意力：GQA + QK-Norm

| 参数 | 值 | 说明 |
|---|---|---|
| `--num-attention-heads` | 64 | Query 头数 |
| `--group-query-attention` | — | 启用 **GQA**（分组查询注意力） |
| `--num-query-groups` | 4 | **KV 头/组数 = 4**：64 个 Q 头共享 4 组 K/V（每 16 个 Q 共享 1 组）→ 大幅压缩 KV cache。**这 4 正是转换阶段 TP≤4 的根源**（§5.2 / §7-#4） |
| `--kv-channels` | 128 | 每个注意力头维度 head_dim=128。注意 64×128=8192 ≠ hidden 4096 → Qwen3 的 **head_dim 与 hidden 解耦**，独立设定 |
| `--qk-layernorm` | — | 对 Q、K 各做一次归一化（Qwen3 的 **QK-Norm**），稳定注意力数值、利于大模型训练稳定性 |

### 4.3 归一化 / 激活 / 偏置 / 词嵌入

| 参数 | 值 | 说明 |
|---|---|---|
| `--normalization RMSNorm` | — | 用 RMSNorm（比 LayerNorm 省算、无需减均值），Qwen3/LLaMA 系标准 |
| `--norm-epsilon` | 1e-6 | 归一化数值稳定项 |
| `--swiglu` | — | FFN 用 **SwiGLU** 门控激活（gated linear unit，比 GeLU 效果好） |
| `--disable-bias-linear` | — | 所有 Linear **不带 bias**（现代 LLM 惯例，省参省算） |
| `--untie-embeddings-and-output-weights` | — | 输入 embedding 与输出 lm-head **不共享权重**（解绑，各自独立训练） |

### 4.4 位置编码：RoPE

| 参数 | 值 | 说明 |
|---|---|---|
| `--position-embedding-type rope` | — | 旋转位置编码 RoPE（相对位置，外推友好） |
| `--rotary-base` | 1000000 | RoPE 的 θ/base=1e6（**高 base 利于长上下文**，配合 40960 位置） |
| `--rotary-percent` | 1.0 | 100% 的头维度都施加 RoPE |

### 4.5 MoE（本模型核心）

| 参数 | 值 | 说明 |
|---|---|---|
| `--num-experts` | 128 | 每个 MoE 层 128 个专家 FFN |
| `--moe-ffn-hidden-size` | 1536 | **单个专家**的 FFN 中间维（专家被切得很"瘦"，靠数量堆容量） |
| `--moe-router-topk` | 8 | 每个 token 路由到 **top-8** 专家（稀疏激活，对应 A22B） |
| `--moe-router-dtype` | fp32 | router 的 gating 用 fp32 计算，**防止路由打分抖动/数值不稳** |
| `--moe-router-load-balancing-type` | aux_loss | 用**辅助损失**做负载均衡（防专家塌缩，日志中的 `load_balancing_loss`） |
| `--moe-aux-loss-coeff` | 1e-3 | 上述辅助损失的系数（软约束，太大伤主任务、太小不均衡） |
| `--moe-token-dispatcher-type` | alltoall | 专家并行 EP 的 token 分发用 **all-to-all**（标准方案，跨机走 RoCE） |
| `--moe-layer-freq` | 1 | **每一层都是 MoE 层**（频率=1；非隔层） |
| `--moe-layer-recompute` | — | 对 MoE 层做激活**重计算**（用算力换显存，MoE 激活是大头） |

### 4.6 实测

- 构建后**总参数 235.10B**，每卡最重分片 **4.07B**（与 `TP8/PP2/EP4` 的显存测算一致）。
- MoE：128 专家 / 每 token top-8；专家 dispatch/combine 的两次 all-to-all 走已实测的 ~184 GB/s RoCE。
- 架构参数在**转换**与 **SFT** 两个脚本里必须**逐字一致**，否则 mcore 权重加载会 shape/结构 mismatch。

---

## 5. 权重转换（step1：HF → mcore）—— `run_convert_qwen235b.sh`

### 5.1 做了什么

- 输入：HF 权重 **118 个 safetensors（~459GB）** + tokenizer。
- 工具：modelopt `convert_model.py`（仓库自带 Qwen3-235B-A22B 专用 conf）。
- **环境保护**：用 `pip freeze` 生成版本约束 `-c`，锁死 torch/TE/cuDNN/NCCL 不被 modelopt 依赖降级；装完立即自检（cuDNN 9.27 / TE / modelopt import 全绿才继续）；`MLM_SKIP_INSTALL=1` 禁止 conf 脚本二次无约束安装。
- 并行：**TP=4 / PP=2 / EP=1 / ETP=4**（见 §5.2 原因），单机 8 卡（`TP*PP=8`，每卡 ~59GB 权重）。
- 输出：`torch_dist` 分布式 ckpt，**8 个 `.distcp` 分片，共 439 GiB≈470GB**，耗时 **~67 分钟**（共享存储 + MoE 专家多）。dist-ckpt 与并行无关，训练时自动重切。

### 5.2 两处必改（否则必崩）

1. **`--no-gradient-accumulation-fusion`**：本环境未装 APEX 的 `fused_weight_gradient_mlp_cuda`，否则建模即 `RuntimeError`。通过 `MLM_EXTRA_ARGS` 注入。
2. **TP 必须 ≤ 4**：Qwen3-235B 的 GQA 只有 **4 个 KV 组**，modelopt 导入器断言 `num_query_groups(4) % TP == 0`（`megatron_importer.py:_qkv_merging`）。原设 TP=8 → `AssertionError`；改 **TP=4/PP=2** 后 `TP*PP=8` 既满足约束又装得下。

---

## 6. SFT 训练（step2：64 卡全参）—— `run_sft_qwen235b.sh`

### 6.1 配置

| 项 | 值 |
|---|---|
| 并行 | **TP=8（节点内 NVLink） / PP=2（47/47 层） / EP=4 / DP=4** + sequence-parallel + 分布式优化器 |
| 加载 | `--finetune --load <mcore> --no-load-optim --no-load-rng`；torch_dist 自动 **TP4/PP2/EP1 → TP8/PP2/EP4** 重切 |
| 精度/优化器 | bf16；lr 1e-5，cosine，warmup 0.03，min-lr 1e-6；clip 1.0；adam (0.9,0.95) |
| batch/seq | micro-batch 1，global-batch 32，seq 4096，train-iters 100 |
| SFT | `--sft --sft-tokenizer-prompt-format default --tokenizer-type SFTTokenizer --tokenizer-model <HF>` |
| 存档 | `--save <sft> --save-interval 50`，`--ckpt-format torch_dist` |
| 换挡 | `RECOMPUTE_FULL=1`（整层重算）/ `USE_FP8=1`（hybrid）/ `GBS/SEQ/TRAIN_ITERS` 可调 |

### 6.2 训练结果（iter 1→50）

| iter | lm loss | load_balancing_loss | grad norm |
|---|---|---|---|
| 1 | 2.195 | 4.16 | 12.1 |
| 10 | 1.378 | 3.43 | 1.37 |
| 20 | 1.329 | 2.37 | 0.81 |
| 30 | 1.152 | 2.08 | 0.83 |
| 40 | 0.894 | 2.03 | 0.96 |
| 50 | 0.834 | 1.78 | 0.96 |

- lm loss **单调下降、0 NaN、0 skip**，loss scale 恒 1.0（bf16 稳定）。
- **起步 2.2 而非冒烟的 ~12** → 证明**预训练权重确实加载成功**（关键验证点）。
- `load_balancing_loss` 4.16→1.78 → MoE 路由在学会均衡；grad norm 12→~1 → 优化健康。

---

## 7. 踩坑总表（按出现顺序）

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 1 | `import transformer_engine` 报 cuDNN 过旧 | venv 自带 9.10.2 < TE 要求的 9.12 | `LD_PRELOAD` 预载 `cudnn-ext` 的 9.27（`te_env.sh`） |
| 2 | 基准/实验 `set_device` 即 CUDA OOM，拖垮多 rank 会合 | 卡被训练占满，新进程连 CUDA context 都挤不进 | 跑前 `nvidia-smi` 预检空闲显存，不足即中止，不抢卡 |
| 3 | 转换建模即 `RuntimeError: fused_weight_gradient_mlp_cuda not found` | 未装 APEX，`gradient_accumulation_fusion=True` | 全程加 `--no-gradient-accumulation-fusion` |
| 4 | 转换导权重 `AssertionError: num_query_groups % tp_size == 0` | GQA 仅 4 组，TP=8 不整除（modelopt 导入器约束） | 转换用 **TP=4/PP=2**（`TP*PP=8` 仍装得下） |
| 5 | SFT 加载 `RuntimeError: Missing key ..._extra_state/shard_0_1` | TE 的 `_extra_state`（FP8 占位，bf16 无实义）是 `ShardedObject`，不能跨 TP 自动重切 | 加 `--dist-ckpt-strictness log_all`（剔除多余键，仅打日志） |
| 6 | 一堆 NCCL `store->get` 失败 + rendezvous 崩 | **假象**：rank0 在数据加载崩 → 它托管的 TCPStore 关闭，其余 rank 建组失败 | 追根因，别被次生报错带偏 |
| 7 | rank0 `sft_dataset.__getitem__` `AssertionError` | THD packed 要求 `create_attention_mask=False` 且 `reset_attention_mask=False` | 加 `--no-create-attention-mask-in-dataloader`，且**不要**带 `--reset-attention-mask` |
| 8 | iter50 存档目录为空、全节点 ssh 255 | 3.3TB 同步存档窗口内被外部终止 | 守护式启动 + 足够墙上时限 +（可选）异步存档 |

**方法论**：多机大模型里，**NCCL/rendezvous/ssh 退出码这类报错经常是"次生灾害"**——先定位**最早崩的那个 rank 的真实异常**（往往在数据、建模或加载里），再修；修完一个错常会露出下一个，逐层推进。

---

## 8. 复现路径（从零到 SFT）

```bash
# 前置：环境/网络/多机 TE/235B shape 均已验证（见 §0 阶梯）

# step1：HF → mcore 权重转换（单机 8 卡，~1 小时，一次性）
bash run_convert_qwen235b.sh
#   产物：.../checkpoints/Qwen/Qwen3-235B-A22B-mcore/（8×distcp，~470GB）

# step2：64 卡全参 SFT（守护式启动，建议 tmux / nohup）
tmux new -s sft
bash run_sft_qwen235b.sh
#   换挡示例：
#   RECOMPUTE_FULL=1 bash run_sft_qwen235b.sh          # 显存紧张
#   USE_FP8=1 bash run_sft_qwen235b.sh                 # FP8 提速
#   TRAIN_ITERS=200 GBS=64 SEQ=8192 bash run_sft_qwen235b.sh
```

---

## 小结

从环境隔离（cuDNN 9.27）、RoCE 网络固化（busbw ~184 GB/s），到数据构造（benchmark 高分轨迹反哺 SFT）、235B MoE 的权重转换（TP≤4 的 GQA 约束）与 64 卡全参训练（lm loss 2.2→0.83），整条链路的每个环节都有各自的"必踩之坑"。核心方法论只有一句：**在多机大模型里，先找最早崩的那个 rank 的真实异常，再逐层推进**。关于这些并行维度的原理，可回看[《Megatron-LM 分布式训练》系列](/blog/megatron-lm-01-overview/)。
