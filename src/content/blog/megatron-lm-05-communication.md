---
title: "Megatron-LM 分布式训练（五）：通信与网络实测（NCCL / RoCE）"
description: "并行策略的上限由物理网络决定。本文介绍 NVLink 与 RoCE、GPUDirect RDMA，讲解 NCCL 在机内 NVLink + 机间多网卡 RoCE 上的跑法、关键调优旋钮、busbw 健康标准，以及带宽如何决定通信能否被计算隐藏。"
pubDate: 2026-09-19
tags: ["Megatron-LM", "分布式训练", "NCCL", "RoCE", "网络"]
---

> **本文是《Megatron-LM 分布式训练》系列第 5 篇**。前四篇讲的是"逻辑上怎么切"，本篇落到"物理上怎么传"——[总览](/blog/megatron-lm-01-overview/) / [TP](/blog/megatron-lm-02-tensor-parallel/) / [PP](/blog/megatron-lm-03-pipeline-parallel/) / [DP](/blog/megatron-lm-04-data-parallel-zero/)。

## 背景：互联、RDMA 与 GDR

前面反复强调"TP 绑 NVLink、DP/PP 走跨机"，这里补齐物理层概念：

- **NVLink**：NVIDIA GPU 之间的**机内**高速互联，带宽数百 GB/s 到 TB/s，远高于 PCIe。TP 这种高频通信靠它。
- **RoCE / InfiniBand（IB）**：**机间**网络。RoCE（RDMA over Converged Ethernet）让以太网也能做 RDMA。带宽几十到 100+ GB/s，用于 DP/PP 这类低频跨机通信。
- **RDMA（远程直接内存访问）**：网卡直接读写对端内存，**绕过 CPU**，低延迟高带宽。
- **GPUDirect RDMA（GDR）**：更进一步，网卡直接 DMA 读写**显存**，连 CPU 内存中转都省了——这是 RoCE 跑出高带宽的命门。

衡量通信性能看 **busbw（总线带宽）**（见第 4 篇），它可直接对标硬件线速。下面以一套真实的 64 卡 RoCE 集群为例。

## 网络家底（从日志读出）

```
每节点 8× mlx5_bond_0..7      # 8 张 Mellanox RoCE 网卡（每卡配一张，双口 bond）
NCCL version 2.27.3+cuda12.9
IP 网段 10.16.x.x
```

8 卡配 8 网卡 → 跨机不是一根管子，而是 **8 条 RoCE 链路并行**。假设每口 200Gb/s（25GB/s），则单节点理论聚合 ≈ **200 GB/s** 单向。

## NCCL 在"机内 NVLink + 机间 8×RoCE"上怎么跑

```
一次 64 卡 all-reduce:
  ① 机内 reduce-scatter: 8 卡 NVLink（超快）
  ② 机间传输:           8 节点经 8×mlx5_bond RoCE 组环（~200GB/s 聚合）
  ③ 机内 all-gather:    NVLink 回每张卡
```

NCCL 会自动把 8 张网卡分配给 8 个本地 rank（rank0→bond_0…），让每张卡的跨机流量走专属网卡，充分利用聚合带宽。这就是"分层集合通信"：先在机内用 NVLink 归约，再用 RoCE 跨机，最后机内分发。

## RoCE 调优旋钮

基础项（通常脚本里已设对）：`NCCL_IB_DISABLE=0`（开 RoCE）、`NCCL_SOCKET_IFNAME='^lo,docker0,...'`（排除虚拟网卡）、`NCCL_DEBUG_SUBSYS=INIT,NET,ENV`、`CUDA_DEVICE_MAX_CONNECTIONS=1`。

生产环境建议补：

| 变量 | 作用 | 典型值 |
|---|---|---|
| `NCCL_IB_HCA=mlx5_bond` | 指定用哪些 HCA | 前缀 `mlx5_bond` |
| `NCCL_IB_GID_INDEX` | RoCE v2 选对 GID | 常见 3（视环境而定） |
| `NCCL_IB_TC` / `NCCL_IB_SL` | 流量分类 / 服务等级，配合 QoS/PFC | 看网管 |
| `NCCL_NET_GDR_LEVEL=PHB` | 开 GPUDirect RDMA（网卡直读显存） | `PHB` / `SYS` |

> **GDR（GPUDirect RDMA）是 RoCE 性能命门**：网卡直接 DMA 读写显存、不经 CPU。用 `NCCL_DEBUG=INFO` 看日志里有没有 `GDRDMA` / `GPU Direct RDMA` 字样确认生效。

## busbw 健康标准

```
单节点理论 ≈ 8 × 25 GB/s = 200 GB/s
目标:  健康 ≳ 160~180 GB/s(80-90%) | 可接受 ~120-160 | 有问题 <100(查 GDR/PFC/GID)
```

只看大消息（512/1024MB）的平台值，小消息受延迟影响不代表带宽上限。

## 本篇落点：带宽够不够"藏住"通信

一切并行最终的问题是：跨机通信能否被反向计算**隐藏**（overlap）？判据很简单：

$$\text{能否 overlap} \iff \frac{\text{梯度字节数}}{\text{busbw}}\ (\text{通信耗时}) < \text{反向计算耗时}\ (\text{可隐藏窗口})$$

举例：每卡 DP 同步梯度 4GB，busbw=180GB/s → 通信 ≈ 22ms，若反向 ≈ 100ms 则**完全藏住**；若 busbw 掉到 40GB/s → 通信 ≈ 100ms **藏不住**，暴露在关键路径上 → 每步都变慢。

> **这就是死磕网络实测的意义**：busbw 决定 DP/PP 跨机通信能否被计算隐藏。busbw 高 → 跨机几乎无感；busbw 低 → 3D 并行退化成"假并行"，加卡不加速。

## 小结

- 物理层：**NVLink**（机内，TP）、**RoCE/IB**（机间，DP/PP），**GDR** 让网卡直读显存、是 RoCE 高带宽的关键。
- NCCL 在多网卡集群上做**分层集合通信**：机内 NVLink 归约 → 机间多网卡 RoCE 组环 → 机内分发；每卡绑一张专属网卡。
- 看 **busbw 大消息平台值**判断健康度（目标 80-90% 理论值），异常先查 GDR / PFC / GID。
- 带宽的终极意义：通信耗时 < 反向计算耗时，才能 overlap；否则跨机通信拖慢每一步。

下一篇进入**MoE 专家并行 EP 与上下文并行 CP**，把稠密模型的 3D 并行扩展到稀疏模型和超长序列。
