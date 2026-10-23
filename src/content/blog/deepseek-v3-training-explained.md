---
title: "DeepSeek-V3 实战解读：拆解真实训练脚本"
description: "以 DeepSeek-V3 的真实 Megatron 训练脚本为样本，把每个训练参数对回背后的原理：MLA、细粒度 MoE、MTP、EP+ZeRO-3 的并行哲学；并横向对比 Qwen3-235B（两种 MoE 路线）、给出主流百亿/万亿模型并行地图，以及预训练→SFT→RL 的完整训练链路。"
pubDate: 2025-05-15
tags: ["DeepSeek-V3", "分布式训练", "MoE", "MLA", "Megatron-LM", "大模型"]
---

> 素材：Megatron 仓库 `examples/megatron_fsdp/sbatch_mfsdp_deepseek_v3.sh`（DeepSeek-V3 的真实训练脚本）。本文把每个训练参数对回它背后的原理，代码引用用 `文件:行号` 标注，可直接在仓库定位。
>
> 第十、十一节扩展为**横向对比**：Qwen3-235B vs DeepSeek-V3（两种 MoE 并行哲学），并附 Kimi K2 等主流大模型（100B+）并行地图；第十二节梳理 DeepSeek-V3 的**完整训练链路**（预训练 → SFT → RL）。

如果你还不熟悉其中的并行概念（TP/PP/EP、ZeRO、Checkpoint 等），建议先读我的[《Megatron-LM 分布式训练》系列](/blog/megatron-lm-01-overview/)；另一篇[《Qwen3-235B-A22B 64 卡全参 SFT 实战》](/blog/qwen3-235b-sft-roleplay/)则是本文第十节对比对象在真实任务里的落地。

## 一、DeepSeek-V3 全局概览

| 指标 | 值 | 脚本出处 |
|---|---|---|
| 总参数 | **671B** | 61 层 × 256 专家堆出来 |
| 激活参数 | **37B** | 每 token 只过 top-8 专家 |
| 层数 | 61 | `--num-layers 61` |
| hidden | 7168 | `--hidden-size 7168` |
| dense FFN | 18432 | `--ffn-hidden-size 18432` |
| 注意力头 | 128 | `--num-attention-heads 128` |
| 序列长（预训练） | 4096 | `--seq-length 4096` |

三个标志性创新：**MLA**（省显存的注意力）、**细粒度 MoE**、**MTP**（多 token 预测）。

> **为什么 `seq-length` 只有 4096，却支持 128K 上下文？**
> 因为 attention 开销 ∝ $s^2$，大模型采用**两阶段训练**：① 主预训练用短序列（4K）高效学知识（占绝大部分 token）；② 再用少量长序列 + RoPE 缩放（脚本 `--rotary-scaling-factor 40`，YaRN）把窗口扩到 128K。
> 序列短是 `--context-parallel-size 1`（CP=1）的直接原因——激活压力不大，不必切序列；只有长序列阶段才需开 CP。

## 二、最反直觉的一点：几乎不用 TP/PP

传统认知是「TP×PP×DP 三维并行」。但 DeepSeek-V3 的并行配置：

**`sbatch_mfsdp_deepseek_v3.sh:22-33`**

```bash
TP=${TP:-1}
EP=${EP:-8}
...
    --tensor-model-parallel-size ${TP}          # TP=1 !
    --expert-model-parallel-size ${EP}          # EP=8
    --expert-tensor-parallel-size 1             # ETP=1
    --context-parallel-size 1                   # CP=1
```

**TP=1、无 PP**，取而代之：

**`sbatch_mfsdp_deepseek_v3.sh:142-145`**

```bash
        --use-megatron-fsdp
        --data-parallel-sharding-strategy ${SHARDING_STRATEGY}   # optim_grads_params
        ...
        --use-distributed-optimizer
```

`optim_grads_params` = 切 **优化器状态 + 梯度 + 参数** = **ZeRO-3（Megatron-FSDP）**。

**为什么敢这么配**：

- MLA 让单层激活/KV 很小 → 不需要 TP 切层内；
- MoE 参数大头被 **EP=8** 切到 8 份 → 专家显存解决；
- 剩下用 **ZeRO-3** 分片 → 省到底。

> 对比记忆：**稠密大模型（Llama-405B）走 TP×PP×DP；稀疏 MoE 大模型（DeepSeek-V3）走 EP + ZeRO-3。** 并行策略看模型结构，不套公式。

## 三、MLA（Multi-head Latent Attention）：省显存的注意力

**`sbatch_mfsdp_deepseek_v3.sh:107-114`**

```bash
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
```

**核心**：标准 MHA 要为每 token 存完整 K、V（KV cache 巨大）。MLA 把 K/V **先压成低秩潜向量**，用时再投影回来：

- `kv-lora-rank 512`：K/V 压到 **512 维潜向量**（而非 128 heads × 128 dim）→ **KV cache 砍到约 1/10**；
- `q-lora-rank 1536`：Q 也低秩分解，省参数/激活；
- `qk-pos-emb-head-dim 64`：RoPE 单独走一小块维度（**解耦 RoPE**，MLA 的关键技巧）。

实现：`megatron/core/transformer/multi_latent_attention.py`。

> MLA 从**架构层**省激活/KV，与重计算从**计算层**省互补。单层够省，所以敢 TP=1。

## 四、细粒度 MoE：工业级配置

**`sbatch_mfsdp_deepseek_v3.sh:87-106`**

```bash
    --num-experts 256
    --moe-layer-freq [0]*3+[1]*58
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-token-dispatcher-type flex
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
```

| 参数 | 含义 | 说明 |
|---|---|---|
| `num-experts 256` + `topk 8` | 256 专家，每 token 选 8 个 | 稀疏激活：671B 参数只激活 37B |
| `moe-layer-freq [0]*3+[1]*58` | **前 3 层 dense，后 58 层 MoE** | 浅层稠密更稳，深层 MoE 扩容量 |
| `moe-shared-expert-...` | 有**共享专家**（所有 token 都过） | shared expert |
| `moe-ffn-hidden-size 2048` | 每专家很小（vs dense 18432） | **细粒度专家**：切细、组合多 |
| `token-dispatcher-type flex` | DeepEP 融合 all-to-all | 三种 dispatcher 里最快 |
| `score-function sigmoid` + `enable-expert-bias` | **用 bias 做负载均衡** | DeepSeek「无辅助损失负载均衡」 |
| `num-groups 8` + `group-topk 4` | **分组路由**（node-limited routing） | 限制专家跨节点数，省跨机 all-to-all |

**最值得学的两点**：

1. **分组路由（`num-groups 8`, `group-topk 4`）**：256 专家分 8 组，每 token 先选 4 组、再组内选专家。**目的是限制 token 要去的专家分散在多少节点上** → 直接减少 EP all-to-all 跨机通信量。算法为通信让路的典范。
2. **bias 负载均衡（`enable-expert-bias`）**：不靠 aux_loss（会干扰主 loss），而是给每专家一个可动态调整的 bias——过载调低、冷门调高，`bias-update-rate 1e-3` 控制调速。既均衡又不伤质量。

## 五、MTP（Multi-Token Prediction）：一次预测多个 token

**`sbatch_mfsdp_deepseek_v3.sh:115-116`**

```bash
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
```

标准 GPT 每位置只预测下一个 token。MTP 额外加小模块（`mtp-num-layers 1`）**同时预测再下一个 token**，作辅助目标（`loss-scaling-factor 0.1` 是权重）。

好处：

- **训练**：监督信号更密集 → 收敛更快、效果更好；
- **推理**：该模块可直接做**投机解码（speculative decoding）** → 推理提速。

## 六、显存 / 精度 / 性能

**`sbatch_mfsdp_deepseek_v3.sh:48-49`**

```bash
    --recompute-granularity selective
    --recompute-modules mlp moe mla_up_proj layernorm
```

- **selective 重计算**：精挑 `mlp / moe / mla_up_proj / layernorm`（存起来亏、重算便宜）；
- `--bf16`（L134）：首选精度；
- `--overlap-grad-reduce --overlap-param-gather`（L35-36）：通信-计算 overlap；
- `--moe-grouped-gemm --moe-permute-fusion`（L96/105）：256 个专家小 GEMM 合成一个大 GroupedGEMM，提 MFU；
- `--manual-gc --manual-gc-interval 10`（L46-47）：手动控制 GC，避免某卡随机 GC 拖慢集群同步点（大规模 MFU 杀手之一）。

## 七、通信与容错

**`sbatch_mfsdp_deepseek_v3.sh:3-7`**

```bash
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
...
export NCCL_P2P_NET_CHUNKSIZE=2097152
```

- `NCCL_IB_SL=1`：RoCE Service Level（配合交换机 QoS）；
- `NCCL_IB_TIMEOUT=19`：IB/RoCE 超时，大集群调大防误判；
- `NCCL_P2P_NET_CHUNKSIZE`：P2P 分块大小调优。

容错 / ckpt：

**`sbatch_mfsdp_deepseek_v3.sh:119-123`**

```bash
    --auto-detect-ckpt-format
    --load ${OUTPUT_PATH}/checkpoints
    --save ${OUTPUT_PATH}/checkpoints
    --save-interval 100
    --dist-ckpt-strictness log_all
```

- `--ckpt-format fsdp_dtensor`（L148）：FSDP 分布式 ckpt 格式（ShardedTensor 思想，详见[系列第 8 篇](/blog/megatron-lm-08-checkpoint-ft/)）；
- `--save-interval 100`：每 100 步存一次（频率权衡）。

## 八、一张表：DeepSeek-V3 技术全景

| 维度 | DeepSeek-V3 怎么体现 |
|---|---|
| 并行网格 | TP=1, EP=8, CP=1 + FSDP（而非 TP×PP×DP） |
| 迭代生命周期 | GBS=2048, MBS=4 → 梯度累加 |
| TP | **故意不用**（MLA 让单层够小） |
| PP | **故意不用**（靠 ZeRO-3 分片参数） |
| DP/ZeRO | **Megatron-FSDP = ZeRO-3**（`optim_grads_params`） |
| 通信 | EP 的 all-to-all + 分组路由省跨机；RoCE SL/timeout 调优 |
| EP/CP | 256 专家 top-8 + 共享专家 + 分组路由 + flex dispatcher |
| 显存/MFU | MLA 省 KV + selective 重计算 + GroupedGEMM + bf16 |
| 容错 | fsdp_dtensor ckpt + save-interval 100 |

## 九、核心收获

- 并行策略**没有标准答案**：DeepSeek-V3 用 **MLA（省 attention）+ 细粒度 MoE（堆容量）+ EP+ZeRO-3（铺专家、分片参数）** 刻意绕开传统 TP/PP；
- 关键在于**看模型结构选并行**：MLA 让单层小 → 不用 TP；MoE 参数靠 EP 切 → 不用 PP；
- **算法为通信让路**：分组路由限制专家跨节点数，直接压 EP 的跨机 all-to-all；
- **负载均衡可以不用 aux_loss**：bias 动态调整，不伤主 loss；
- 这份脚本是 64 卡集群以后跑 MoE 模型的最佳模板。

## 十、横向对比：Qwen3-235B vs DeepSeek-V3（两种 MoE 并行哲学）

> 素材：Megatron 官方配置 `examples/post_training/modelopt/conf/Qwen/Qwen3-235B-A22B.sh`（Qwen3-235B-A22B 的官方模型定义）。
> **两个都是 MoE 大模型，架构选择不同 → 推出截然不同的并行范式**，这是判断「怎么选并行」的最佳对照。

### 10.1 架构对比

| | **Qwen3-235B-A22B**（Megatron 官方 conf） | **DeepSeek-V3** |
|---|---|---|
| 总参数 / 激活 | 235B / 22B | 671B / 37B |
| 层数 | 94 | 61 |
| hidden | 4096 | 7168 |
| **注意力** | **GQA**（64 头 / 4 KV 组） | **MLA**（低秩压缩 KV） |
| 专家数 / topk | 128 / 8 | 256 / 8 |
| MoE 层分布 | **每层都是 MoE** | 前 3 层 dense，后 58 层 MoE |
| 共享专家 | 无 | 有 |
| 负载均衡 | **aux_loss** | **bias 动态调整**（无辅助损失） |
| 分组路由 | 无 | 有（node-limited） |
| MTP | 无 | 有 |
| dispatcher | `alltoall` | `flex`（DeepEP） |

（Qwen 架构出处：`Qwen3-235B-A22B.sh:21-36`——`num-layers 94`、`hidden-size 4096`、`ffn-hidden-size 12288`、`num-query-groups 4`、`num-experts 128`、`moe-ffn-hidden-size 1536`、`moe-router-topk 8`、`moe-token-dispatcher-type alltoall`）

### 10.2 并行策略对比（核心差异）

**关键事实**：Megatron 官方这份 Qwen3-235B conf 是 **modelopt（后训练/量化/导出）的模型定义**，它**只固定架构 + 少量并行开关，不写死 TP/PP/EP**——并行维度在启动时按集群规模由环境变量传入（Megatron 刻意把「模型架构」和「并行度」解耦）。官方文件里**明确写死**的并行相关项只有：

**`Qwen3-235B-A22B.sh:12-48`**

```bash
    --micro-batch-size 1 \
    ...
    --moe-token-dispatcher-type alltoall \   # EP 用 all-to-all
    --moe-layer-recompute \                   # MoE 层重计算省激活
    ...
    --sequence-parallel \                     # 开 SP → 意味着搭配 TP>1
```

从这几个开关能反推它的**并行范式**（由架构决定，不是文件写死的具体数字）：

| | **Qwen3-235B（官方 conf 推出的范式）** | **DeepSeek-V3** |
|---|---|---|
| TP | **需 >1**（开了 `--sequence-parallel`，SP 寄生于 TP） | **1** |
| SP | 默认开（`:48`）→ 必然配 TP | 空转（TP=1） |
| EP | 需要（128 专家，`alltoall` 分发） | 8 |
| MoE 重计算 | 默认开（`--moe-layer-recompute`，`:36`） | selective 重计算 |
| 分片 | 典型用分布式优化器（**ZeRO-1**）即可 | **Megatron-FSDP（ZeRO-3）** |
| 路线 | **TP 切注意力 + EP 铺专家 + ZeRO-1** | **EP + ZeRO-3，零 TP/PP** |

> 换句话说：官方没给你一个「TP=x PP=y EP=z」的死数字，但它用 `--sequence-parallel` + `--moe-layer-recompute` + `alltoall` 告诉你——**这个模型就是要靠 TP（切注意力）+ EP（铺专家）来跑**，和 DeepSeek 的「零 TP/PP」路线泾渭分明。具体 TP/EP/PP 取多少，按你的卡数和显存现算（见 10.4）。

### 10.3 为什么 Qwen 要用 TP、DeepSeek 却敢 TP=1？—— 根源在注意力

- Qwen3 用 **GQA（标准注意力）** → 单层 attention 的激活/参数仍较大，且官方默认开 `--sequence-parallel` → **必须搭配 TP 切层内**；
- DeepSeek 用 **MLA（KV 压成 512 维潜向量）** → 单层已够小 → 不需要 TP，直接 `TP=1` 靠 ZeRO-3 分片。

这正是本文第三节 MLA 的深层意义：**MLA 不只省推理 KV cache，它直接决定了你能用的并行策略**。

### 10.4 显存账：为什么 64 卡能跑 Qwen235B，却跑不了 DeepSeek 671B

按 16 字节/参数（全参训练，bf16 权重+梯度 + fp32 master + Adam m/v）：

| 模型 | 参数状态总量 | 64 卡（约 6.1TB）装得下？ |
|---|---|---|
| **Qwen3-235B** | 235B×16 = **3.76 TB** | ✅ 装得下（≈58.7GB/卡状态 + 重计算压激活） |
| **DeepSeek-V3 671B** | 671B×16 = **10.7 TB** | ❌ 差近 2 倍，需 128+ 卡才装得下状态 |

> 这就是 Qwen3-235B 能在 64 卡量级全参训练、而 DeepSeek-V3 671B 不能的根本原因——235B 的参数状态刚好能压进 64×96GB（再靠官方默认的 `--moe-layer-recompute` 把激活压下去）。这也解释了 10.2 为什么 Qwen 用 ZeRO-1 就够、DeepSeek 必须上 ZeRO-3。（Qwen3-235B 的 64 卡实战见[这篇](/blog/qwen3-235b-sft-roleplay/)。）

## 十一、主流大模型并行地图（100B 以上，含 Kimi K2 等）

把「还有哪些大模型」系统化成一张地图（**只收录 100B+ 的主流大模型**），归类维度：**稠密/MoE、注意力类型（MHA/GQA/MLA）、并行倾向**。

### 11.1 稠密模型（Dense，100B+）

> 注：100B+ 的**纯稠密**模型已越来越少——参数一旦上百 B，几乎都转向 MoE（把容量和算力解耦）。仍在坚持大稠密的代表：

| 模型 | 规模 | 注意力 | 并行典型 |
|---|---|---|---|
| **Llama 3.1 405B** | 405B | GQA | TP×PP×DP 经典 3D |
| **NVIDIA Nemotron-4 340B** | 340B | GQA | TP×PP×DP |
| **Falcon-180B** | 180B | MQA | TP×PP×DP |

### 11.2 MoE 模型（稀疏，100B+）

| 模型 | 总参/激活 | 注意力 | 专家 | 特点与路线 |
|---|---|---|---|---|
| **Qwen3-235B-A22B** | 235B/22B | **GQA** | 128 / top8 | aux_loss；**TP 切注意力 + EP + ZeRO-1** |
| **DeepSeek-V3 / R1** | 671B/37B | **MLA** | 256 / top8 | 细粒度+分组路由+MTP；走 **EP + ZeRO-3** |
| **Kimi K2**（Moonshot，2025.7） | **1.04T/32B** | **MLA**（hidden 7168, 64 头） | **384 / top8** | 超稀疏 MoE，走 DeepSeek 类路线；**MuonClip 优化器**（QK-Clip 解决 attention logit 爆炸），15.5T token 预训练 |
| **Grok-1** | 314B | — | 8 / top2 | xAI |
| **Mixtral 8×22B** | 141B/39B | GQA | 8 / top2 | 最早出圈的 MoE，**粗粒度专家** |
| **DBRX** | 132B | GQA | 16 / top4 | Databricks |

> **Kimi K2 的看点**：参数量（1.04T）比 DeepSeek-V3（671B）还大，架构选择却高度一致——**MLA + 超稀疏 MoE（384 专家）**。这进一步印证「**MLA + 大规模 MoE**」已成为万亿级模型的主流范式。它唯一的大创新在优化器（MuonClip，替代 Adam），不在并行结构。显存上 1.04T×16B≈16.6TB，训练需要比 DeepSeek-V3 更多的卡（数百张起步）。

### 11.3 三条判断规律（可预测任意新模型的并行策略）

1. **看注意力**：用 **MLA**（DeepSeek / Kimi K2）→ 敢省 TP（TP=1）；用 **GQA**（Qwen / Llama / Mixtral）→ 通常仍需 TP 切 attention。
2. **看是否 MoE**：MoE → 一定要 **EP** 铺专家；稠密 → TP×PP×DP 老三样。
3. **看参数总量 vs 集群显存**：装得下就用 **ZeRO-1**（省通信，如 Qwen3-235B）；装不下就上 **ZeRO-3 / FSDP**（如 DeepSeek 671B、Kimi 1T）。

> 一句话：**并行策略 = f(注意力类型, 是否 MoE, 参数量/显存)**。Qwen3-235B（GQA → TP 切注意力 + EP + ZeRO-1）和 DeepSeek-V3（MLA → EP8 + ZeRO-3）正好是这个函数的两个端点。

## 十二、DeepSeek-V3 完整训练链路：预训练 → SFT → RL

> 前九节拆的是**预训练脚本**（`sbatch_mfsdp_deepseek_v3.sh`），但那只是完整训练的**第一段**。在 Megatron 里把一个 DeepSeek-V3 从头训到能用，要走三段，**三个入口脚本、三套工具链**。

### 12.1 全景：三个阶段 = 三个入口

| 阶段 | 入口脚本 | 真实配置 | 分片路线 |
|---|---|---|---|
| **① 预训练** | `pretrain_gpt.py` | `examples/megatron_fsdp/sbatch_mfsdp_deepseek_v3.sh` | **Megatron-FSDP（ZeRO-3）** |
| **② SFT** | `examples/post_training/modelopt/finetune.py` | `finetune.sh` + `conf/deepseek-ai/DeepSeek-R1.sh` | 分布式优化器（ZeRO-1）+ N-D 并行 |
| **③ RL** | `train_rl.py`（GRPO） | `examples/rl/`（示例为 Qwen，机制通用） | 分布式优化器 + 内置推理引擎 |

> **一个重要事实**：Megatron 官方仓库里，预训练和 SFT 都有 DeepSeek 专属配置；RL 有通用 **GRPO** 实现，但随仓库给的示例是 **Qwen 2.5 32B**——GRPO 正是 DeepSeek-R1 带火的算法，Megatron 实现了它，只是没附 DeepSeek-V3 的 RL 启动脚本。

### 12.2 ① 预训练：next-token + MTP（最重的一段）

入口 `pretrain_gpt.py`（`sbatch_mfsdp_deepseek_v3.sh:191`），在海量原始语料上做 next-token 预测，从零学语言与知识。特征参数：

| 维度 | 值 | 出处 |
|---|---|---|
| 训练量 | `train-samples 585937500`（≈十几 T token） | `:43` |
| 学习率 | `lr 3.9e-6`，warmup 1.536M 样本，cosine→`3.9e-7` | `:79-84` |
| weight-decay | **0.1**（预训练强正则） | `:77` |
| MTP | `mtp-num-layers 1`（第五节） | `:115` |
| 并行/分片 | TP=1 / EP=8 / CP=1 + **FSDP ZeRO-3** | `:30-36` |

> 就是前九节拆的那套（671B 全参、ZeRO-3、EP=8、MLA、MTP）。官方用 2048×H800 训约两个月——三段里唯一的「重量级」。

### 12.3 ② SFT：加载预训练权重，在指令数据上监督微调

入口 `finetune.py`（`finetune.sh:86`），用 `--finetune` 加载预训练 checkpoint（`--load`），在**指令数据集**上学「听指令、按对话格式答」。和预训练最关键的 5 个差异：

| 维度 | 预训练 | **SFT** | SFT 出处 |
|---|---|---|---|
| 权重 | 从零 | **`--finetune` 加载预训练权重** | `finetune.sh:17` |
| 数据 | 原始语料 | **HF 指令集**（默认 Magpie 对话） | `finetune.sh:27,37` |
| 训练量 | 5.86 亿样本 | **仅 12.8 万样本**（小一个数量级） | `finetune.sh:33` |
| 学习率 | 3.9e-6 | **5e-5**（短跑更激进），warmup=0 | `finetune.sh:63` |
| 样本处理 | 普通拼接 | **reset-position-ids / reset-attention-mask / eod-mask-loss** | `finetune.sh:50-52` |

那三个样本处理开关是 SFT 的灵魂：

- **`--reset-position-ids` / `--reset-attention-mask`**：多条短对话**打包（packing）**进一条 4096 序列省算力，这两个开关保证每条对话 position 从 0 重数、attention 不跨样本串味；
- **`--eod-mask-loss`**：EOD/padding 位不计 loss，只对真正的回答内容算损失。

> 模型架构由 `conf/deepseek-ai/DeepSeek-R1.sh` 定义，和预训练**完全一致**（61 层 MLA、256 专家、`moe-layer-freq [0]*3+[1]*58`，`DeepSeek-R1.sh:16-45`），区别在用 `alltoall` 分发 + `--moe-layer-recompute`，分片走 **ZeRO-1** 而非 FSDP（SFT 只跑 12.8 万样本，不必那么激进省显存）。

### 12.4 ③ RL：GRPO 强化（DeepSeek-R1 的招牌）

入口 `train_rl.py`，Megatron 实现 **GRPO（Group Relative Policy Optimization）**——DeepSeek-R1 带火的算法。核心是「**边生成边训练**」：

```
每个 step：
  1. 对一批 prompt，模型自己生成多个回答（rollout）   ← 内置推理引擎
  2. 环境给每个回答打分（reward，如数学答案对不对）     ← environment
  3. 组内比较同一 prompt 的多个回答谁好谁坏（group relative）
  4. 用相对优势更新策略                                ← GRPO loss
```

特征参数（`examples/rl/README.md`）：

| 维度 | 值 |
|---|---|
| 组大小（每 prompt 几个答案） | `grpo-group-size 16` |
| 每步 prompt 数 | `grpo-prompts-per-step 64` |
| 裁剪区间（PPO 式） | `clamp-eps 0.2~0.28` |
| 学习率 | **1e-6**（RL 阶段极小） |
| 关键开关 | `--perform-rl-step` + 内置 `inference-dynamic-batching`/`cuda-graph`（生成 rollout） |

> **诚实说明**：这份 RL 示例训的是 **Qwen 2.5 32B**（数据 DAPO-Math-17k，评测 AIME2024），不是 DeepSeek-V3——Megatron 未附 DeepSeek-V3 的 GRPO 配置。但机制通用：把模型架构换成 DeepSeek-V3 那套即可，DeepSeek-R1 自己正是用这套 GRPO 训出来的。（GRPO 在图像生成上的变体可参见[《图像生成与强化学习》](/blog/image-generation-and-rl/)。）

### 12.5 三段串起来 + checkpoint 流转

```
原始语料              指令数据集             数学/代码环境
   │                    │                      │
   ▼                    ▼                      ▼
[① 预训练] ──权重──► [② SFT] ──权重──► [③ RL / GRPO]
pretrain_gpt.py      finetune.py          train_rl.py
FSDP ZeRO-3          ZeRO-1 + N-D并行      ZeRO-1 + 内置推理
lr 3.9e-6 / wd 0.1   lr 5e-5 / wd 0        lr 1e-6
next-token + MTP     指令监督(mask loss)   reward 强化
十几 T token         12.8 万样本           按 step 生成 rollout
```

两类 **checkpoint 转换**衔接各段（Megatron 特有的工程环节）：

1. **HF ↔ mcore**：用 Megatron-Bridge 把 HuggingFace 权重导入成 Megatron 分布式 checkpoint（SFT/RL 的前提）；
2. **`torch_dist` ↔ `fsdp_dtensor`**：预训练用 FSDP（`fsdp_dtensor` 格式），SFT 用 N-D 并行（`torch_dist` 格式），两种格式靠 `examples/megatron_fsdp/sbatch_checkpoint_convert.sh` 互转。

> 一句话：**预训练（FSDP、最重、next-token+MTP）→ SFT（加载权重、指令监督、ZeRO-1，靠 reset/mask 处理打包对话）→ RL（GRPO 边生成边打分，DeepSeek-R1 招牌）**，三段换三个入口，靠 checkpoint 转换衔接。

## 附：关键相关实现

| 主题 | 文件 |
|---|---|
| MLA | `megatron/core/transformer/multi_latent_attention.py` |
| MoE 层 / 路由 / 分发 | `megatron/core/transformer/moe/{moe_layer,router,token_dispatcher}.py` |
| MTP | `docs/user-guide/features/multi_token_prediction.md` |
| Megatron-FSDP（ZeRO-3） | `megatron/core/distributed/fsdp/` |
| 预训练脚本（阶段①） | `examples/megatron_fsdp/sbatch_mfsdp_deepseek_v3.sh` → `pretrain_gpt.py` |
| SFT 脚本 / DeepSeek 配置（阶段②） | `examples/post_training/modelopt/finetune.sh`、`finetune.py`、`conf/deepseek-ai/DeepSeek-R1.sh` |
| RL / GRPO（阶段③） | `train_rl.py`、`examples/rl/README.md`、`megatron/rl/rl_utils.py` |
| checkpoint 转换（torch_dist↔fsdp_dtensor） | `examples/megatron_fsdp/sbatch_checkpoint_convert.sh` |
| proxy 测试配置（含并行变体） | `tests/functional_tests/test_cases/mixtral/deepseekv3_proxy_flex_*` |
| Qwen3-235B-A22B 官方模型配置 | `examples/post_training/modelopt/conf/Qwen/Qwen3-235B-A22B.sh` |
