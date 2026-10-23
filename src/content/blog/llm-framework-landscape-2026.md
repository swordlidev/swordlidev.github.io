---
title: "大模型框架全景调研（2026）：训练 · 微调 · RLHF · 推理四大赛道"
description: "一张面向 2024–2026 的大模型框架地图：训练（Megatron-Core / DeepSpeed / FSDP2）、微调（LLaMA-Factory / Unsloth）、RLHF 后训练（veRL / OpenRLHF / Slime）、推理（vLLM / SGLang / TensorRT-LLM / llama.cpp）四大赛道的核心技术、横向对比与选型建议。"
pubDate: 2026-03-23
tags: ["大模型", "分布式训练", "微调", "RLHF", "推理框架", "框架调研"]
---

> **调研时间**：2026 年 3 月　｜　**覆盖范围**：2024–2026 年主流大模型训练 / 微调 / RLHF / 推理框架。

大模型技术栈在 2024–2026 这两年里经历了一轮剧烈洗牌：训练侧 PyTorch 原生方案（FSDP2）正在追平老牌框架、后训练从传统 RLHF 转向 GRPO / Agentic RL、推理侧 KV Cache 管理成为核心战场。本文把四大赛道的主流框架放进同一张地图，逐一拆解核心技术、做横向对比，并给出可落地的选型建议。

> 本文是"全景地图"视角。本站另有几篇深入单个框架的源码解读可配合阅读：[Megatron-LM 分布式训练系列](/blog/megatron-lm-01-overview/)、[verl 源码解读系列](/blog/verl-01-architecture/)（含 [RL 算法详解](/blog/verl-06-rl-algorithms/)）、[DeepSeek-V3 训练脚本拆解](/blog/deepseek-v3-training-explained/)、[Qwen3-235B 全参 SFT 实战](/blog/qwen3-235b-sft-roleplay/)。

## 一、总览：框架全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        大模型技术栈全景                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   预训练      │  │   后训练      │  │   推理部署    │              │
│  │  (Pre-train)  │  │ (Post-train) │  │  (Inference)  │              │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤              │
│  │★Megatron-LM  │  │★veRL         │  │★vLLM         │              │
│  │★DeepSpeed    │  │★OpenRLHF     │  │★SGLang       │              │
│  │★FSDP2        │  │★Slime        │  │★TensorRT-LLM │              │
│  │ TorchTitan   │  │ TRL          │  │★llama.cpp    │              │
│  │ veScale      │  │ NeMo RL      │  │ LMDeploy     │              │
│  │ NeMo         │  │              │  │ TGI / Ollama │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────────────────┐               │
│  │   微调        │  │        硬件 & 编排层          │               │
│  │ (Fine-tune)  │  ├──────────────────────────────┤               │
│  ├──────────────┤  │ NVIDIA H100/H200/B200/GB300  │               │
│  │★LLaMA-Factory│  │ AMD MI300X / 昇腾 910B       │               │
│  │★Unsloth      │  │ Ray / K8s / Slurm            │               │
│  │ ms-swift     │  │ CUDA / ROCm / CANN           │               │
│  │ Axolotl      │  └──────────────────────────────┘               │
│  └──────────────┘                   ★ = 本文重点介绍               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、训练框架

### 2.1 Megatron-LM / Megatron-Core ⭐

> **业界地位**：大规模预训练的事实标准。GPT-4、Llama 3、混元等万亿参数模型的训练基底。

| 维度 | 说明 |
|------|------|
| **开发方** | NVIDIA |
| **GitHub** | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| **定位** | 超大规模预训练的分布式训练框架 |
| **适用规模** | 百亿 → 万亿参数 |

#### 核心技术深度解析

**1) 3D 并行 — 大模型分布式训练的基石**

```
                ┌─────────────────────────────┐
                │    3D Parallelism (3D 并行)   │
                ├─────────┬─────────┬─────────┤
                │  TP     │  PP     │  DP     │
                │张量并行  │流水线并行│数据并行  │
                ├─────────┼─────────┼─────────┤
                │层内切分  │层间切分  │数据切分  │
                │单层权重  │不同层放  │同一模型  │
                │分到多卡  │不同GPU  │不同数据  │
                └─────────┴─────────┴─────────┘
```

- **张量并行 (TP)**：将单层的权重矩阵按列/行切分到多张 GPU。Megatron 的 TP 实现是最成熟的，通过精心设计的 `ColumnParallelLinear` / `RowParallelLinear` 最小化通信量（仅需 AllReduce）。
- **流水线并行 (PP)**：将模型的不同层放在不同 GPU 上，采用 **1F1B（One Forward One Backward）调度**减少流水线气泡。Megatron 的 virtual pipeline 进一步将气泡率从 `(p-1)/p` 降低到 `(p-1)/(p*m)`。
- **数据并行 (DP)**：最基础的并行，复制模型到多卡，各自处理不同数据后同步梯度。

> 这些并行维度的内部原理，本站 [Megatron-LM 分布式训练系列](/blog/megatron-lm-01-overview/) 用 8 篇做了逐一拆解（[TP](/blog/megatron-lm-02-tensor-parallel/) / [PP](/blog/megatron-lm-03-pipeline-parallel/) / [DP+ZeRO](/blog/megatron-lm-04-data-parallel-zero/) / [MoE EP+CP](/blog/megatron-lm-06-moe-ep-cp/)）。

**2) 高级并行策略**

| 策略 | 解决的问题 | 核心思路 |
|------|-----------|---------|
| **序列并行 (SP)** | LayerNorm/Dropout 的激活显存 | 沿序列维度切分，与 TP 互补 |
| **上下文并行 (CP)** | 超长序列（100K+ tokens） | Ring Attention，序列维度跨卡分布 |
| **专家并行 (EP)** | MoE 模型的专家路由 | 不同专家放不同 GPU，Token 动态路由 |
| **分布式优化器** | 优化器状态显存占用 | 跨 DP ranks 分片 Adam 状态 |

**3) Megatron-Core（2024年起独立模块化）**

Megatron-LM 的核心问题是"模型代码与并行逻辑高度耦合"。Megatron-Core 将并行策略抽象为独立模块：

```python
# Megatron-Core 的模块化使用示例
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig, TransformerLayer

config = TransformerConfig(
    tensor_model_parallel_size=8,   # TP=8
    pipeline_model_parallel_size=4, # PP=4
    expert_model_parallel_size=8,   # EP=8
    sequence_parallel=True,
    use_fp8=True,  # H100 FP8 训练
)
```

**4) FP8 训练**

在 Hopper 架构（H100/H200）上，Megatron-Core 原生支持 FP8 混合精度训练，通过 Transformer Engine 实现自动精度管理，典型场景下可带来 **30-40% 的训练加速**，且精度损失可控。

#### 典型使用场景

| 场景 | 配置参考 |
|------|---------|
| 7B Dense 模型 | 8×H100, TP=8, PP=1 |
| 70B Dense 模型 | 64×H100, TP=8, PP=4, DP=2 |
| MoE-671B（类 DeepSeek-V3） | 512×H100, TP=8, PP=4, EP=16, DP=1 |
| 超长上下文（128K） | + CP=8 |

> 类 DeepSeek-V3 的真实 Megatron 训练脚本逐参数拆解，见 [《DeepSeek-V3 实战解读》](/blog/deepseek-v3-training-explained/)。

#### 优势与局限

| ✅ 优势 | ❌ 局限 |
|---------|---------|
| 性能天花板最高，NVIDIA GPU 极致优化 | 学习曲线陡峭，代码复杂 |
| 最完整的并行策略组合（TP+PP+SP+CP+EP） | 与 NVIDIA 硬件强绑定 |
| 万亿级工业验证 | 模型与并行策略耦合（Core 有改善） |
| FP8 训练原生支持 | 社区贡献门槛高 |

---

### 2.2 DeepSpeed ⭐

> **业界地位**：使用最广泛的分布式训练库，HuggingFace 生态标配。几乎所有中小团队的大模型训练首选。

| 维度 | 说明 |
|------|------|
| **开发方** | Microsoft |
| **GitHub** | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) ，65K+ Stars |
| **定位** | 通用分布式训练优化库 |
| **适用规模** | 数亿 → 千亿参数 |

#### 核心技术深度解析

**1) ZeRO 优化器 — DeepSpeed 的灵魂**

ZeRO（Zero Redundancy Optimizer）是 DeepSpeed 最核心的创新，解决的是"传统数据并行中每张卡都完整存储模型状态"造成的显存浪费。

```
训练一个 7B 模型（FP16），每张卡需要：
  模型参数:        14 GB (7B × 2 bytes)
  梯度:            14 GB
  优化器状态(Adam): 28 GB (FP32 params + momentum + variance)
  ─────────────────────────
  总计:            56 GB  ← 单卡放不下！

ZeRO 的解法：把这些状态分片到多张卡上
```

| ZeRO Stage | 分片内容 | 显存节省 | 通信开销 |
|------------|---------|---------|---------|
| **ZeRO-1** | 优化器状态 | ~4× | 与 DDP 相同 |
| **ZeRO-2** | 优化器状态 + 梯度 | ~8× | 略增 |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | ~N× (N=GPU数) | 显著增加 |
| **ZeRO-Infinity** | + NVMe offload | 几乎无限 | 受 NVMe 带宽限制 |

**实际选择指南**：
- 8 卡 A100 训练 7B → ZeRO-2 足够
- 8 卡 A100 训练 13B → ZeRO-3
- 8 卡 A100 训练 70B → ZeRO-3 + CPU offload（会较慢）
- 要训更大的 → 加机器，或上 Megatron

**2) ZeRO++（2023年后发布）**

| 优化 | 机制 | 效果 |
|------|------|------|
| **qwZ** | 量化权重通信（INT8） | 全通信量减半 |
| **hpZ** | 层次化 All-Gather（机内全收集、机间部分收集） | 跨机通信降 4× |
| **qgZ** | 量化梯度通信 | 梯度同步通信减半 |

**3) 与 PyTorch 的集成方式**

```python
# DeepSpeed 的使用极其简单——在已有 PyTorch 代码上加 3 行
import deepspeed

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config="ds_config.json"  # JSON 配置驱动
)

# 原来的 loss.backward() + optimizer.step() 不变
model.backward(loss)
model.step()
```

```json
// ds_config.json — 典型 ZeRO-2 配置
{
  "zero_optimization": { "stage": 2 },
  "bf16": { "enabled": true },
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 2
}
```

**4) DeepSpeed-MoE**

DeepSpeed 对 MoE 架构有专门优化：Expert Parallelism、Expert Slicing、PR-MoE（Pyramid-Residual MoE）等，用于训练 MoE 架构模型。

#### 优势与局限

| ✅ 优势 | ❌ 局限 |
|---------|---------|
| ZeRO 在有限资源下效果极好 | 超大规模（万亿级）性能不及 Megatron |
| 与 PyTorch 兼容性最好，配置驱动 | TP/PP 成熟度不如 Megatron |
| HuggingFace / LLaMA-Factory 深度集成 | 版本碎片化，不同版本 API 差异大 |
| 支持 AMD ROCm / Intel GPU | ZeRO-3 的通信开销在大规模下较高 |
| 社区最活跃，文档丰富 | |

#### Megatron vs DeepSpeed — 怎么选？

| 维度 | Megatron-LM | DeepSpeed |
|------|------------|-----------|
| **核心思路** | 精细切分模型（TP+PP） | 分片冗余状态（ZeRO） |
| **适合规模** | 千亿 → 万亿 | 数亿 → 千亿 |
| **代码侵入性** | 高，需改模型代码 | 低，配置驱动 |
| **性能天花板** | 更高 | 略低 |
| **易用性** | 较难 | 简单 |
| **最佳搭配** | 专职训练团队 + NVIDIA GPU | 中小团队 + 快速迭代 |
| **可以组合使用** | ✅ Megatron-DeepSpeed = Megatron TP/PP + DeepSpeed ZeRO DP |

---

### 2.3 PyTorch FSDP / FSDP2 ⭐

> **业界地位**：PyTorch 原生方案，2025年后 FSDP2 快速成熟，正在成为"不想引入额外依赖"的团队的首选。

| 维度 | 说明 |
|------|------|
| **开发方** | Meta (PyTorch Team) |
| **集成于** | PyTorch 原生（`torch.distributed`） |
| **定位** | PyTorch 原生全分片数据并行 |

#### FSDP1 vs FSDP2

| 维度 | FSDP1 | FSDP2（2025年推出） |
|------|-------|-----|
| 分片粒度 | per-module（整层） | per-parameter（更细） |
| torch.compile | 不完全支持 | 深度集成 |
| DTensor 后端 | 不支持 | 原生支持 |
| 混合并行 | FSDP only | FSDP + TP（2D 并行） |
| FP8 | 不支持 | 支持 |
| 代码复杂度 | 较高 | 更简洁 |

#### 核心优势

- **零额外依赖**：PyTorch 内置，`pip install torch` 即可用
- **与 `torch.compile` 深度集成**：可享受编译器优化加速
- **DTensor 后端**：支持 TP + FSDP 的 2D 并行
- **FSDP2 性能已追平 DeepSpeed ZeRO-3**（在同等配置下）

#### 局限

- 缺少流水线并行 (PP)，超大规模（千亿+）不够用
- 专家并行 (EP) 支持仍在完善
- 生态工具链不如 DeepSpeed 完善

#### 适用场景

中小规模训练（单机到数十节点）、纯 PyTorch 生态、快速原型验证。

---

### 2.4 其他训练框架简介

**TorchTitan**（Meta）— PyTorch 原生的大规模预训练参考实现，基于 FSDP2 + DTensor，支持 4D 并行（TP+PP+FSDP+CP）。已在 Llama 3.1 400B 上验证，是 PyTorch 最新分布式特性的"实验田"。适合追前沿 feature 的研究者，但工业验证深度不及 Megatron。[GitHub](https://github.com/pytorch/torchtitan)

**veScale**（ByteDance Seed）— 解耦模型定义与并行策略的训练系统。用户写单卡代码，系统自动并行化。基于 DTensor，理论上很优雅，但社区规模较小，超大规模工业案例较少。[GitHub](https://github.com/volcengine/veScale)

**NeMo Framework**（NVIDIA）— 端到端大模型平台，底层基于 Megatron-Core，覆盖 Pre-training → SFT → RLHF → Deployment 全流程。YAML 配置驱动，适合 NVIDIA 全栈企业用户。[GitHub](https://github.com/NVIDIA/NeMo)

**Megatron-DeepSpeed**（Microsoft+NVIDIA）— 融合 Megatron TP/PP + DeepSpeed ZeRO，历史上训练了 Megatron-Turing NLG 530B。2025年后两个框架各自独立发展加速，该融合方案更新节奏放缓。新项目建议直接选 Megatron-Core 或 DeepSpeed。

---

### 2.5 训练框架对比总结

| 框架 | 开发方 | 最大验证规模 | 并行策略 | 易用性 | 硬件 | 推荐度 |
|------|--------|-----------|---------|--------|------|-------|
| **Megatron-Core** | NVIDIA | 万亿级 | TP+PP+DP+SP+CP+EP | ⭐⭐ | NVIDIA | 🔥🔥🔥🔥🔥 |
| **DeepSpeed** | Microsoft | 千亿级 | ZeRO+TP+PP | ⭐⭐⭐⭐ | 多平台 | 🔥🔥🔥🔥🔥 |
| **FSDP2** | Meta/PyTorch | 百亿级 | FSDP+TP | ⭐⭐⭐⭐⭐ | 通用 | 🔥🔥🔥🔥 |
| TorchTitan | Meta/PyTorch | 400B | TP+PP+FSDP+CP | ⭐⭐⭐ | 通用 | 🔥🔥🔥 |
| veScale | ByteDance | 百亿级 | 自动并行 | ⭐⭐⭐⭐ | NVIDIA | 🔥🔥 |
| NeMo | NVIDIA | 万亿级 | Megatron 全量 | ⭐⭐⭐ | NVIDIA | 🔥🔥🔥 |

---

## 三、微调框架

### 3.1 LLaMA-Factory ⭐

> **业界地位**：最流行的开源微调框架，GitHub 50K+ Stars，研究者和工程师的首选。

| 维度 | 说明 |
|------|------|
| **GitHub** | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，50K+ Stars |
| **定位** | 低代码、全功能大模型微调框架 |

#### 为什么它最流行？

1. **纯 YAML 驱动** — 改配置文件就能训练，零代码修改
2. **模型覆盖最全** — 100+ 主流开源模型（Llama/Qwen/Mistral/GLM/DeepSeek/...）
3. **全链路集成** — SFT + DPO + PPO + GRPO + KTO 一站式
4. **Web UI** — 内置 LLaMA Board，可视化训练/评估/对话
5. **灵活的分布式** — DeepSpeed ZeRO 全阶段 + FSDP2 + DDP

#### 训练方式支持

| 方式 | 显存需求 (7B) | 适用场景 |
|------|-------------|---------|
| **全参数微调** | ~56 GB | 充足资源 + 追求最优效果 |
| **LoRA** | ~16 GB | 常规微调首选 |
| **QLoRA (4-bit)** | ~6 GB | 单卡消费级 GPU |
| **DoRA** | ~18 GB | LoRA 的改进，效果更好 |

#### 典型用法

```bash
# 单卡 QLoRA 微调 Qwen2.5-7B — 一行命令
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --dataset alpaca_zh \
  --finetuning_type lora \
  --quantization_bit 4 \
  --output_dir ./output
```

---

### 3.2 Unsloth ⭐

> **业界地位**：极致显存优化的"省钱神器"，2025年后成为资源受限场景的标配搭档。

| 维度 | 说明 |
|------|------|
| **GitHub** | [unslothai/unsloth](https://github.com/unslothai/unsloth)，30K+ Stars |
| **定位** | 训练加速 + 极致显存优化 |

#### 核心卖点

- **2-5× 训练加速**：手写 Triton 内核优化反向传播
- **显存减少 60-80%**：7B QLoRA 仅需 4-7 GB 显存
- **单卡 16GB 即可微调 7B 模型**
- 支持 Llama / Qwen / Mistral / Phi / Gemma 等主流架构
- 免费开源版 + Pro 商业版（支持多卡）

#### 局限

多 GPU 支持有限（Pro 版改善），模型覆盖面不如 LLaMA-Factory。通常作为 LLaMA-Factory 的**加速后端**使用，而非单独使用。

---

### 3.3 其他微调框架简介

**ms-swift**（阿里 ModelScope）— ModelScope 生态的微调框架，支持 300+ 模型，特色是多模态微调（VL/Audio）和 Megatron 后端集成（Megatron-Bridge）。ModelScope 生态用户首选。[GitHub](https://github.com/modelscope/ms-swift)

**Axolotl**（社区）— 面向研究者的高度可定制微调框架，YAML 驱动，灵活性极高。适合论文复现和高度定制化实验，但上手门槛高于 LLaMA-Factory。[GitHub](https://github.com/axolotl-ai-cloud/axolotl)

---

### 3.4 微调框架对比总结

| 框架 | 易用性 | 模型覆盖 | 显存优化 | 多GPU | 推荐度 |
|------|--------|---------|---------|-------|-------|
| **LLaMA-Factory** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🔥🔥🔥🔥🔥 |
| **Unsloth** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 🔥🔥🔥🔥 |
| ms-swift | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🔥🔥🔥 |
| Axolotl | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🔥🔥🔥 |

> 一次 64 卡 Qwen3-235B-A22B（MoE）全参 SFT 的完整实战复盘（环境/数据/权重转换/踩坑），见 [《Qwen3-235B 角色扮演 SFT 实战》](/blog/qwen3-235b-sft-roleplay/)。

---

## 四、RLHF / 后训练框架

> 📌 **范式演进**：2025-2026年，后训练从传统 RLHF (PPO+Reward Model) 向 **GRPO/DAPO**（无需 Reward Model）和 **Agentic RL**（多轮交互强化学习）快速演进。三大框架 veRL、OpenRLHF、Slime 形成"三足鼎立"格局。这些算法的统一框架与源码细节，见 [《verl 源码解读（六）：RL 算法详解》](/blog/verl-06-rl-algorithms/)。

### 4.1 veRL ⭐

> **业界地位**：2025-2026年最活跃的开源 RL 框架，学术论文引用最多，被大量工业项目采用。

| 维度 | 说明 |
|------|------|
| **开发方** | ByteDance Seed Team |
| **GitHub** | [volcengine/verl](https://github.com/volcengine/verl) |
| **论文** | HybridFlow (EuroSys 2025) |
| **定位** | 灵活高效的 LLM RL 训练框架 |

#### 核心架构 — HybridFlow

veRL 的核心创新是 **HybridFlow**：用单控制器（Single Controller）编排全局流程，用多工作者（Multi Worker）异构执行计算，实现控制流和计算流的解耦。

```
┌─────────────────────────────────────────────────┐
│              veRL HybridFlow 架构                 │
│                                                   │
│  ┌─────────┐  编排   ┌──────────────────────┐   │
│  │ Single  │───────→│  Worker Group 1       │   │
│  │Controller│        │  (Actor: FSDP/Megatron)│   │
│  │         │───────→│  Worker Group 2       │   │
│  │ (Ray)   │        │  (Rollout: vLLM/SGLang)│   │
│  │         │───────→│  Worker Group 3       │   │
│  │         │        │  (Critic: FSDP)        │   │
│  └─────────┘        └──────────────────────┘   │
│                                                   │
│  关键：训练后端和推理引擎可自由组合               │
│  - 训练: FSDP / Megatron-LM                      │
│  - 推理: vLLM / SGLang                           │
└─────────────────────────────────────────────────┘
```

> HybridFlow 的单控制器编排、DataProto 数据总线、Rollout 混合引擎等机制，本站 [verl 源码解读系列](/blog/verl-01-architecture/) 做了逐层拆解。

#### 核心技术

| 技术 | 说明 |
|------|------|
| **训练后端灵活** | FSDP / Megatron-LM 可选 |
| **推理引擎灵活** | vLLM / SGLang 可选 |
| **算法覆盖最广** | PPO / GRPO / DAPO / DPO / ReMax / PRIME / ... |
| **Agentic RL** | 支持多轮工具调用强化学习 |
| **权重同步优化** | 训练→推理切换的通信开销最小化 |
| **吞吐提升** | 对比 baseline 提升 **1.5-20×** |

#### 为什么选 veRL？

1. **灵活性最高**：训练后端和推理引擎都可自由组合，适应不同硬件和规模
2. **算法最全**：几乎所有主流 RL 算法都支持
3. **Agentic RL 先行者**：多轮 Agent 交互强化学习支持最好
4. **社区最活跃**：被大量学术论文采用，Issue 响应快

---

### 4.2 OpenRLHF ⭐

> **业界地位**：最容易上手的大规模 RLHF 框架，70B+ 全量 PPO 训练的首选方案。

| 维度 | 说明 |
|------|------|
| **GitHub** | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) |
| **定位** | 基于 Ray + DeepSpeed + HF Transformers 的高性能 RLHF 框架 |

#### 核心技术

- **Ray 分布式编排** — Actor / Critic / Reward / Reference 四模型协调
- **DeepSpeed ZeRO-3** — 支持 **70B+ 全量 PPO 训练**（多节点）
- **vLLM 集成** — 高效 rollout 采样
- **算法支持** — PPO / DPO / GRPO / KTO / 迭代 DPO / RLVR
- **LoRA + RingAttention** — 长序列 + 参数高效微调

#### 与 veRL 的关键差异

| 维度 | veRL | OpenRLHF |
|------|------|----------|
| 训练后端 | FSDP / Megatron | DeepSpeed |
| 架构理念 | HybridFlow（控制/计算解耦） | Ray 多控制器 |
| 灵活性 | 更高（可换后端） | 中等（绑定 DeepSpeed） |
| 上手难度 | 中等 | **更容易** |
| 文档质量 | 好 | **非常好** |
| Agentic RL | ✅ 成熟 | 🔄 开发中 |

**选择建议**：追求灵活性和前沿算法 → veRL；追求快速上手和稳定可靠 → OpenRLHF。

---

### 4.3 Slime ⭐

> **业界地位**：智谱 AI 开源的 RL 框架，专为 RL Scaling 和大规模 MoE 设计，GLM-4.5 全系列的训练基底。

| 维度 | 说明 |
|------|------|
| **开发方** | 智谱 AI / 清华 THUDM |
| **GitHub** | [THUDM/slime](https://github.com/THUDM/slime) |
| **定位** | 面向 RL Scaling 的 LLM 后训练框架 |

#### 核心架构 — Megatron + SGLang 原生融合

Slime 的独特定位是将 **Megatron-LM**（训练）和 **SGLang**（推理/采样）深度打通，而非像 veRL 那样通过抽象层适配多种后端：

```
┌──────────────────────────────────────────┐
│              Slime 架构                    │
│                                            │
│  ┌──────────────┐    ┌──────────────┐    │
│  │  Megatron-LM  │←→│   SGLang      │    │
│  │  (训练引擎)    │   │  (采样引擎)   │    │
│  │  BF16 训练状态 │   │  FP8 推理     │    │
│  └──────────────┘    └──────────────┘    │
│         ↕ 异步并行，互不阻塞                │
│                                            │
│  特色：大规模 MoE RL 训练验证               │
│  特色：100% 可复现的 RL 训练               │
└──────────────────────────────────────────┘
```

#### 核心技术

| 技术 | 说明 |
|------|------|
| **异步 RL** | 推理和训练引擎解耦并行，消除 GPU 空闲 |
| **MoE RL** | 较早验证大规模 MoE 模型 RL 训练的框架 |
| **FP8 采样** | SGLang FP8 KV Cache，长上下文 rollout 效率高 |
| **100% 可复现** | 与 SGLang 团队合作实现确定性训练 |
| **RL 算法** | GRPO / DAPO / PPO（通过 `--advantage-estimator` 切换） |

#### 生产验证

- 训练 **GLM-4.5 Ultra/Plus** 全系列模型
- RL 训练周期：**2 周 → 3 天**
- MT-Bench：**8.2 → 8.9**（+8.5%）

#### 硬件适配

NVIDIA GPU（原生）/ AMD ROCm（Day-0）/ 摩尔线程 MUSA

#### 与 veRL/OpenRLHF 的差异

- **训练后端固定为 Megatron-LM**（不支持 FSDP/DeepSpeed）—— 灵活性低但性能天花板高
- **推理后端固定为 SGLang** —— 前缀缓存和结构化生成最优
- **MoE 支持最成熟** —— 如果训练 MoE 模型做 RL，Slime 是首选

---

### 4.4 其他 RLHF 框架简介

**TRL**（HuggingFace）— HF 生态标准 RL 库。与 Transformers 无缝集成，支持 PPO/DPO/KTO/ORPO/GRPO，代码简洁、教程丰富。**最适合中小规模实验和快速验证**，但超大规模性能不及以上三者。[GitHub](https://github.com/huggingface/trl)

**NeMo RL**（NVIDIA，原 NeMo-Aligner）— NVIDIA NeMo 生态的对齐模块，底层 Megatron-Core，性能极致。支持 PPO/DPO/SteerLM/GRPO。适合已在 NeMo 全栈的团队，独立使用门槛较高。

---

### 4.5 RLHF框架对比总结

| 框架 | 训练后端 | 推理引擎 | 最大规模 | MoE | Agentic RL | 易用性 | 推荐度 |
|------|---------|---------|---------|-----|------------|--------|-------|
| **veRL** | FSDP/Megatron | vLLM/SGLang | 数百B | ✅ | ✅ | ⭐⭐⭐ | 🔥🔥🔥🔥🔥 |
| **OpenRLHF** | DeepSpeed | vLLM | 70B+ | 🔄 | 🔄 | ⭐⭐⭐⭐ | 🔥🔥🔥🔥🔥 |
| **Slime** | Megatron-LM | SGLang | 数百B(MoE) | ⭐⭐⭐⭐⭐ | 🔄 | ⭐⭐⭐ | 🔥🔥🔥🔥 |
| TRL | HF Trainer | HF/vLLM | 数十B | ❌ | ❌ | ⭐⭐⭐⭐⭐ | 🔥🔥🔥 |
| NeMo RL | Megatron-Core | TRT-LLM | 数百B | ✅ | 🔄 | ⭐⭐ | 🔥🔥🔥 |

> 🔄 = 开发中/部分支持

---

## 五、推理框架

### 5.1 vLLM ⭐

> **业界地位**：LLM 推理服务的事实标准。GitHub 50K+ Stars，最广泛的模型兼容性，"推理服务选 vLLM"几乎成为行业共识。

| 维度 | 说明 |
|------|------|
| **开发方** | vLLM 社区 / PyTorch Foundation |
| **GitHub** | [vllm-project/vllm](https://github.com/vllm-project/vllm)，50K+ Stars |
| **定位** | 通用高性能 LLM 推理引擎 |

#### 核心技术深度解析

**1) PagedAttention — vLLM 的灵魂创新**

传统推理引擎为每个请求预分配连续 KV Cache，导致大量内存碎片和浪费。PagedAttention 借鉴操作系统虚拟内存的分页思想：

```
传统方式：为每个请求预分配固定大小的连续内存
  ┌────────────────────────────────────┐
  │ Request A: [KV][KV][KV][   浪费   ] │  ← 预分配 max_len，实际用不满
  │ Request B: [KV][KV][  大量浪费    ] │
  └────────────────────────────────────┘
  显存利用率: ~50-60%

PagedAttention：按需分配固定大小的"页"
  ┌─────────────────────────────┐
  │ Page Pool: [A][B][A][B][A]... │  ← 按需分配，零碎片
  └─────────────────────────────┘
  显存利用率: ~94%
```

**效果**：相同显存下可服务 **2-4× 更多并发请求**。

**2) Continuous Batching（连续批处理）**

传统 static batching 必须等一批请求全部完成才能处理下一批。Continuous Batching 允许在每个 decode step 动态插入新请求、移除已完成请求。

**3) V1 引擎（2025-2026年）**

vLLM V1 是架构重写版本：
- 统一调度器（Unified Scheduler）
- Prefix Caching（前缀缓存）
- 更高效的内存管理
- 性能进一步提升 20-30%

**4) 其他关键特性**

| 特性 | 说明 |
|------|------|
| Speculative Decoding | 投机解码，小模型草稿 + 大模型验证 |
| TP / PP | 多卡推理 |
| OpenAI 兼容 API | `vllm serve model_name` 一行启动 |
| 量化 | AWQ / GPTQ / FP8 / INT8 |
| 多模态 | 支持 VLM（视觉语言模型） |
| 硬件 | NVIDIA / AMD / 国产 GPU |

#### 性能参考

| 场景 | 指标 |
|------|------|
| Llama 70B, A100 80G, 高并发 | 吞吐 ~1800 tokens/s |
| 显存利用率 | 94%（PagedAttention） |
| 对比 TGI | 吞吐高 ~18% |

#### 一行部署

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4
# 即可通过 OpenAI 兼容 API 访问
```

---

### 5.2 SGLang ⭐

> **业界地位**：2025年增长曲线超过 vLLM 的后起之秀。在 RAG、结构化生成、RL rollout 场景中性能领先。

| 维度 | 说明 |
|------|------|
| **开发方** | SGLang 社区（UC Berkeley 发起） |
| **GitHub** | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| **定位** | 高性能 LLM/VLM 推理 + 结构化生成 |

#### 核心技术深度解析

**1) RadixAttention — 比 PagedAttention 更高效的前缀复用**

```
场景：多轮对话，每轮都带相同系统 prompt

vLLM (PagedAttention):
  Turn 1: [System Prompt KV] + [User1 KV]     ← 计算 system prompt
  Turn 2: [System Prompt KV] + [User2 KV]     ← 可以 cache，但需显式管理

SGLang (RadixAttention):
  用基数树(Radix Tree)自动管理所有前缀的 KV Cache
  ┌─ "You are a helpful" ─┬─ "User: Hello"  → [cached]
  │                        └─ "User: Help me" → [cached]
  └─ "You are a coder"   ─── "User: Code"    → [cached]
  自动匹配最长共享前缀，零配置
```

**效果**：在 RAG / 多轮对话 / few-shot 等共享前缀场景下，**前缀计算节省 80-95%**。

**2) Prefill-Decode 分离（PD Disaggregation）**

```
传统推理：Prefill 和 Decode 在同一 GPU 上交替执行
  ← Prefill 是 compute-bound，需要大算力
  ← Decode 是 memory-bound，需要大带宽
  两者互相干扰，资源利用率低

SGLang PD 分离：
  GPU Pool A (高算力): 专门做 Prefill
  GPU Pool B (高带宽): 专门做 Decode
  各自独立扩缩容
```

**3) 结构化输出 — SGLang 的杀手级特性**

```python
# SGLang 原生支持约束解码
response = model.generate(
    prompt="Extract user info as JSON:",
    json_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
)
# 100% 保证输出合法 JSON，且几乎零性能损失
```

这对 Agent / Tool-calling / Function-calling 场景是刚需。

**4) HiCache（分层 KV Cache）**

GPU → CPU → SSD 三级 KV Cache 缓存，自动在热/温/冷之间迁移，有效扩展 KV Cache 容量。

#### vLLM vs SGLang — 怎么选？

| 维度 | vLLM | SGLang |
|------|------|--------|
| **KV Cache 管理** | PagedAttention（分页） | RadixAttention（基数树） |
| **前缀复用** | 支持（V1 引擎） | **原生更高效** |
| **结构化生成** | 支持 | **原生支持，性能更优** |
| **PD 分离** | 部分支持 | **原生支持** |
| **模型兼容性** | **最广** | 快速追赶 |
| **生态成熟度** | **最成熟** | 快速追赶 |
| **RL rollout** | 支持 | **与 Slime/veRL 集成更深** |
| **选择建议** | 通用场景首选 | RAG/Agent/RL 场景首选 |

---

### 5.3 TensorRT-LLM ⭐

> **业界地位**：NVIDIA GPU 上的性能天花板。高并发吞吐领先 vLLM 5-6%，P99 延迟控制最优。

| 维度 | 说明 |
|------|------|
| **开发方** | NVIDIA |
| **GitHub** | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| **定位** | NVIDIA GPU 极致性能推理引擎 |

#### 核心技术

| 技术 | 说明 |
|------|------|
| **TensorRT 编译器** | 模型图优化 + 内核融合，编译时优化 |
| **In-flight Batching** | 请求级动态批处理 |
| **FP8 / INT4-AWQ** | Hopper/Blackwell 架构极致量化加速 |
| **定制 CUDA 内核** | FlashAttention / MoE 专用内核 |
| **Speculative Decoding** | 多种投机解码策略 |
| **多节点推理** | TP + PP 跨机推理 |

#### 性能参考

| 对比 | TensorRT-LLM | vLLM | TGI |
|------|-------------|------|-----|
| 高并发吞吐 | **基准** | -5~6% | -18% |
| P99 延迟 | **390ms** | ~410ms | ~620ms |
| 显存利用率 | 92% | 94% | 85% |

#### 优势与局限

- ✅ NVIDIA GPU 上绝对性能最优
- ✅ FP8/INT4 量化支持最完善
- ✅ 延迟控制最好
- ❌ **仅支持 NVIDIA GPU**
- ❌ 模型需要编译转换，流程复杂
- ❌ 开源程度不如 vLLM/SGLang，灵活性低

**选择建议**：已确定 NVIDIA GPU + 对延迟吞吐有极致要求 + 有专人维护 → TensorRT-LLM。否则 vLLM/SGLang 更省心。

---

### 5.4 llama.cpp ⭐

> **业界地位**：端侧/本地推理的事实标准。80K+ Stars，让大模型在笔记本、手机、树莓派上跑起来。

| 维度 | 说明 |
|------|------|
| **开发方** | Georgi Gerganov / ggml.ai |
| **GitHub** | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，80K+ Stars |
| **定位** | 纯 C/C++ 跨平台推理框架 |

#### 为什么它重要？

llama.cpp 和上面的云端推理框架解决的是**完全不同的问题**：

| 维度 | vLLM/SGLang/TRT-LLM | llama.cpp |
|------|---------------------|-----------|
| 目标 | 云端高并发服务 | 本地/端侧单用户推理 |
| 依赖 | Python + CUDA | **纯 C/C++，零依赖** |
| 硬件 | 需要 GPU 服务器 | **CPU / 消费级 GPU / 手机 / 浏览器** |
| 量化 | FP8/INT4 | **2/3/4/5/6/8-bit 极致量化** |

#### 核心特点

- **GGUF 格式**：统一的量化模型格式，已成为本地推理的事实标准
- **极致量化**：Q2_K 到 Q8_0，在精度和速度间灵活权衡
- **全平台支持**：x86 / ARM / CUDA / Metal / Vulkan / SYCL / WebAssembly
- **CPU + GPU 混合推理**：部分层放 GPU、部分放 CPU
- **Apple Silicon 优化**：在 Mac 上通过 Metal 获得出色性能

#### 性能参考（Apple M2 Max, 32GB）

| 模型 | 量化 | 推理速度 |
|------|------|---------|
| Llama 3.1 8B | Q4_K_M | ~40 tok/s |
| Qwen2.5 32B | Q4_K_M | ~12 tok/s |
| Llama 3.1 70B | Q4_K_M | ~5 tok/s |

---

### 5.5 其他推理框架简介

**LMDeploy**（上海 AI Lab）— InternLM 团队的推理框架，特色是 TurboMind C++ 引擎和**昇腾 GPU 适配**。国产 GPU 推理首选。支持 AWQ/W4A16/KV INT8 量化。[GitHub](https://github.com/InternLM/lmdeploy)

**TGI / Text Generation Inference**（HuggingFace）— HF 生态的生产级推理服务，Rust + Python 实现。HuggingFace Hub 一键部署，稳定性好。吞吐比 vLLM 低约 18%，但作为 HF 生态的默认推理后端仍有大量用户。[GitHub](https://github.com/huggingface/text-generation-inference)

**Ollama**（社区）— 本地运行大模型的最简方案。`ollama run llama3` 一行命令。底层基于 llama.cpp（Mac 上优先 MLX）。适合个人开发者体验和非生产环境。**不适合生产部署**。[GitHub](https://github.com/ollama/ollama)

**MLC-LLM**（陈天奇团队）— 基于 Apache TVM 编译器的跨平台推理，特色是浏览器端推理（WebGPU）和移动端部署（iOS/Android）。适合边缘部署场景。[GitHub](https://github.com/mlc-ai/mlc-llm)

---

### 5.6 推理框架对比总结

| 框架 | 定位 | 吞吐 | 延迟 | 易用性 | 硬件 | 推荐度 |
|------|------|------|------|--------|------|-------|
| **vLLM** | 云端通用推理 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 多平台 | 🔥🔥🔥🔥🔥 |
| **SGLang** | RAG/Agent/结构化 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 多平台 | 🔥🔥🔥🔥🔥 |
| **TensorRT-LLM** | NVIDIA 极致性能 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | NVIDIA | 🔥🔥🔥🔥 |
| **llama.cpp** | 端侧/本地推理 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | **全平台** | 🔥🔥🔥🔥🔥 |
| LMDeploy | 国产 GPU 推理 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | NVIDIA/昇腾 | 🔥🔥🔥 |
| TGI | HF 生态部署 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 多平台 | 🔥🔥🔥 |
| Ollama | 本地体验 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 全平台 | 🔥🔥🔥 |

---

## 六、关键技术趋势

### 6.1 训练侧

1. **从 RLHF 到 Agentic RL**：后训练从单步偏好对齐（PPO/DPO）演进到多轮交互强化学习，模型在环境中执行工具调用、代码运行等真实任务并获得奖励。veRL 在这一方向走得最远。

2. **异步 RL 训练**：AReaL、Slime 等框架探索 rollout-train 异步流水线，消除经典 RL "三阶段串行等待"的 GPU 空闲问题，GPU 利用率提升 30-50%。

3. **FP8/FP4 低精度训练**：Hopper (H100) FP8 已成熟落地，Blackwell (B200) FP4 开始实验。Megatron-Core + Transformer Engine 是目前最成熟的 FP8 训练方案。

4. **超长上下文训练**：Context Parallel、RingAttention 等技术使百万级 token 训练成为可能。Megatron 的 CP 和 DeepSpeed-Ulysses 是两条主要技术路线。

5. **PyTorch 原生化**：FSDP2、TorchTitan、torch.compile 让纯 PyTorch 方案逐步追赶 Megatron/DeepSpeed 的性能。

### 6.2 推理侧

1. **KV Cache 管理成为核心战场**：PagedAttention → RadixAttention → HiCache（分层缓存），KV Cache 管理决定吞吐和成本。

2. **PD 分离架构成熟**：Prefill（计算密集）和 Decode（访存密集）拆分到不同 GPU 池，独立扩缩容。SGLang 在这方面领先。

3. **结构化输出成为刚需**：Agent / Tool-calling 场景要求 100% 合法的 JSON/函数调用输出。SGLang 的约束解码是目前最优方案。

4. **投机解码 (Speculative Decoding)**：小模型快速生成 + 大模型验证，解码速度提升 2-3×。vLLM / TensorRT-LLM / SGLang 均支持。

5. **国产 GPU 适配加速**：昇腾 910B/910C 的推理框架适配（LMDeploy、vLLM）推进迅速。

### 6.3 生态趋势

- **Ray 成为编排层事实标准**：OpenRLHF、veRL、vLLM 均深度集成 Ray。
- **训练-推理一体化**：veRL / Slime 将训练后端和推理引擎在 RL pipeline 中打通。
- **框架趋同**：各框架互相吸收对方优势（如 vLLM 加入 prefix caching，SGLang 加入 PagedAttention 兼容）。

---

## 七、选型建议

### 7.1 预训练

```
├── 万亿级 + NVIDIA GPU + 专职团队
│   └── → Megatron-Core / NeMo
├── 千亿级 + 有限资源 + 快速迭代
│   └── → DeepSpeed ZeRO-3 (+ 可选 TP)
├── 百亿级 + 纯 PyTorch + 快速验证
│   └── → FSDP2 / TorchTitan
└── 多硬件平台 (AMD/Intel)
    └── → DeepSpeed / FSDP2
```

### 7.2 微调

```
├── 通用微调 + Web UI + 全流程
│   └── → LLaMA-Factory
├── 单卡 16GB + 极致省显存
│   └── → Unsloth (+ LLaMA-Factory)
├── ModelScope 生态 / 多模态
│   └── → ms-swift
└── 论文复现 / 高度定制
    └── → Axolotl
```

### 7.3 RLHF / 后训练

```
├── 灵活后端 + Agentic RL + 前沿算法
│   └── → veRL
├── 大规模 MoE RL + Megatron 性能
│   └── → Slime
├── 快速上手 + 70B PPO + 稳定可靠
│   └── → OpenRLHF
├── HuggingFace 生态 + 中小规模实验
│   └── → TRL
└── NVIDIA 全栈
    └── → NeMo RL
```

### 7.4 推理

```
├── 通用在线服务（首选）
│   └── → vLLM
├── RAG / Agent / 结构化输出 / RL rollout
│   └── → SGLang
├── NVIDIA 极致延迟吞吐（专人维护）
│   └── → TensorRT-LLM
├── 国产 GPU (昇腾)
│   └── → LMDeploy
├── 本地/端侧/嵌入式/跨平台
│   └── → llama.cpp
└── 个人体验（非生产）
    └── → Ollama
```

---

## 八、参考资源

### 核心仓库

| 类别 | 框架 | GitHub |
|------|------|--------|
| 训练 | Megatron-LM | https://github.com/NVIDIA/Megatron-LM |
| 训练 | DeepSpeed | https://github.com/microsoft/DeepSpeed |
| 训练 | TorchTitan | https://github.com/pytorch/torchtitan |
| 训练 | veScale | https://github.com/volcengine/veScale |
| 训练 | NeMo | https://github.com/NVIDIA/NeMo |
| 微调 | LLaMA-Factory | https://github.com/hiyouga/LLaMA-Factory |
| 微调 | Unsloth | https://github.com/unslothai/unsloth |
| 微调 | ms-swift | https://github.com/modelscope/ms-swift |
| 微调 | Axolotl | https://github.com/axolotl-ai-cloud/axolotl |
| RL | veRL | https://github.com/volcengine/verl |
| RL | OpenRLHF | https://github.com/OpenRLHF/OpenRLHF |
| RL | Slime | https://github.com/THUDM/slime |
| RL | TRL | https://github.com/huggingface/trl |
| 推理 | vLLM | https://github.com/vllm-project/vllm |
| 推理 | SGLang | https://github.com/sgl-project/sglang |
| 推理 | TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM |
| 推理 | LMDeploy | https://github.com/InternLM/lmdeploy |
| 推理 | TGI | https://github.com/huggingface/text-generation-inference |
| 推理 | llama.cpp | https://github.com/ggml-org/llama.cpp |
| 推理 | Ollama | https://github.com/ollama/ollama |
| 推理 | MLC-LLM | https://github.com/mlc-ai/mlc-llm |

### 核心论文

| 论文 | 会议 | 关联框架 |
|------|------|---------|
| Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | SC 2021 (Best Student Paper) | Megatron |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | SC 2020 | DeepSpeed |
| TorchTitan: One-stop PyTorch native solution for production ready LLM pretraining | arXiv 2410.06511 | TorchTitan |
| HybridFlow: A Flexible and Efficient RLHF Framework | EuroSys 2025 | veRL |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | SOSP 2023 | vLLM |
| SGLang: Efficient Execution of Structured Language Model Programs | arXiv 2312.07104 | SGLang |

---

> **免责声明**：框架生态迭代迅速，以上信息基于 2026 年 3 月时点的公开资料整理，建议以各项目最新文档和 release 为准。
