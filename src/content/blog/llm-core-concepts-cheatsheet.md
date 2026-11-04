---
title: "大模型知识点梳理：从 Attention 到分布式训练"
description: "一份按层次组织的大模型知识点速查：模型结构（Attention / RoPE / 归一化 / MoE）、长序列效率（稀疏 Attention / FlashAttention）、推理（KV Cache / MLA / PagedAttention / 量化）、训练（显存估算 / 损失函数 / 优化器 / LoRA）、分布式（ZeRO / 六种并行）。每个知识点配最小可读代码。"
pubDate: 2024-12-18
tags: ["大模型", "Transformer", "面试", "分布式训练", "推理优化"]
---

> 这是一篇**给自己复习用的结构化笔记**，不是教程。每个知识点的写法统一为：**一句话定义 → 关键结论 → 最小代码**。代码都是示意实现（省略 batch、mask、dtype 细节），目标是看清机制，不是为了能跑。
>
> 组织方式按**层次**而非按字母：模型结构 → 长序列效率 → 推理 → 训练 → 分布式与框架。这样排的好处是能看出「哪个技术是为了解决什么问题而存在的」——比如 KV Cache 是长序列推理的产物，而 PagedAttention 和 GQA/MLA 又都是 KV Cache 的产物。

## 一、模型结构层

### 1.1 Attention 与它的 O(N²)

$$
\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

两个必答的点：

- **为什么要除 $\sqrt{d}$**：$QK^\top$ 的方差随 $d$ 线性增长，不缩放会把 softmax 推到饱和区，梯度消失。
- **复杂度**：时间 **O(N²d)**，空间 **O(N²)**。随序列长度**二次方增长**，这是 Transformer 的根本瓶颈——第二节整节都在解决它。

```python
def attention(Q, K, V, mask=None):
    # Q, K, V: (B, N, d)
    d = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)      # (B, N, N) ← O(N²d) 算、O(N²) 存
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return torch.softmax(scores, dim=-1) @ V             # (B, N, d)
```

**多头（MHA）** 是在 $d$ 维上切成 $h$ 份各自算完再 concat，让不同头关注不同子空间：

```python
q = q.view(B, N, h, d // h).transpose(1, 2)              # (B, h, N, d/h)
o = attention(q, k, v).transpose(1, 2).reshape(B, N, d)  # concat 回来
o = W_o @ o
```

### 1.2 Cross Attention

计算方式和 self-attention **完全一样**，区别只在 Q 和 K/V 来自哪个序列：

```python
self_attn  = attention(x, x, x)          # Q/K/V 同源
cross_attn = attention(x_q, y_kv, y_kv)  # Q 来自序列 A，K/V 来自序列 B
```

典型用途是**模态/序列融合**：Encoder-Decoder（翻译）、Stable Diffusion（图像 latent 查询文本 embedding）、多模态 LLM（文本 query 查询视觉 token）。

### 1.3 位置编码：RoPE

RoPE 用旋转矩阵把**绝对位置**编码进 Q/K，但内积结果只依赖**相对位置**：

$$
\langle \text{RoPE}(q, m),\ \text{RoPE}(k, n) \rangle = f(q, k, m-n)
$$

直觉：相邻两维当作复平面上的点，位置 $m$ 就是转 $m\theta$ 角。两 token 距离越近，旋转角差越小，内积衰减越少——相当于在原内积上叠了一层**距离权重**。

流程：先算 q/k → 按位置算旋转角 → 对 q/k **两两一组**做旋转 → 再算内积。

```python
def rope(x, pos, theta=10000.0):
    # x: (..., N, d)，d 为偶数；pos: (N,)
    d = x.size(-1)
    freq = theta ** (-torch.arange(0, d, 2).float() / d)   # (d/2,) 每维一个频率
    ang  = pos[:, None] * freq[None, :]                    # (N, d/2) 旋转角
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]                    # 两两一组：实部 / 虚部
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

频率按维度递减：**低维高频管局部，高维低频管全局**——这是理解长度外推的前提。

**长度外推**（推理超出训练长度）：没见过的旋转角会让注意力分布崩掉。

| 方法 | 做法 | 问题 |
| :--- | :--- | :--- |
| 位置插值（PI） | 所有频率等比压缩 `pos/s` | 高频维被压坏，局部分辨率下降 |
| NTK-aware | 改 `theta` 而非 `pos`，低频压得多 | 不需微调但仍有损 |
| **YaRN** | **分维度斜坡插值 + 注意力温度缩放** | 当前主流 |

```python
def yarn_freq(d, s, theta=10000.0, alpha=1, beta=32):
    """简化示意（非论文原式）：真实 YaRN 按每维旋转圈数 r=L·freq/2π 与 α/β 设阈值。
    直觉：高频维不插值（保局部），低频维按 1/s 插值（保全局）。"""
    freq = theta ** (-torch.arange(0, d, 2).float() / d)
    wavelen = 2 * math.pi / freq
    ramp = ((wavelen / (2 * math.pi) - alpha) / (beta - alpha)).clamp(0, 1)  # 斜坡过渡
    return freq * ((1 - ramp) + ramp / s)

# 第二招：序列变长→熵升高，给 q/k（或 scores）乘上这个因子来锐化分布
attn_scale = 0.1 * math.log(s) + 1.0
```

### 1.4 归一化：BN / LN / RMSNorm 与 Pre/Post-Norm

**BatchNorm 适合 CV，LayerNorm 适合 NLP**，根本原因是统计维度不同：

```python
# BatchNorm（CV）：对 B×H×W 求统计量，输出 1×C×1×1
#   → 依赖 batch size、变长序列无法对齐、训练/推理行为不一致、多机需同步（SyncBN）
mu = x.mean(dim=(0, 2, 3), keepdim=True)     # x: (B, C, H, W)

# LayerNorm（NLP）：沿特征维，每个样本（每个 token）独立 → 不依赖 batch，多机无需同步
y = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True, unbiased=False) + eps).sqrt()
y = y * g + b

# RMSNorm：去掉减均值和 bias，少一次统计 → LLaMA / Qwen 都用它
y = x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt() * g
```

一句话：**LayerNorm 在单个样本内部归一化**，所以与 batch 大小、序列长度解耦。

**Pre-Norm 还是 Post-Norm** 是一组真实的 trade-off，别只记一半：

```python
# Post-LN（原始 Transformer / Swin-V2）：梯度必须穿过 LN
x = LN(x + Attn(x));  x = LN(x + FFN(x))

# Pre-LN（现代 LLM / ViT / Swin-T 标配）：残差是干净的恒等通路
x = x + Attn(LN(x));  x = x + FFN(LN(x))
```

| | Pre-Norm | Post-Norm |
| :--- | :--- | :--- |
| 训练稳定性 | **好**，梯度直达底层，深层不易爆 | 差，需要 warmup、小心调参 |
| 同配置效果 | 略弱 | **略强** |
| 原因 | $x_{t+1} \approx x_t$，残差主导 → 等于**增加宽度而降低了有效深度** | 每层被强制重新归一化，层间差异大，深度是"真"的 |

大模型因为训不稳的代价太高，普遍选 Pre-Norm；Swin-V2 则反向改回 res-post-norm 来提效果。折中方案有 Sandwich-LN、DeepNorm。

### 1.5 FFN、激活函数与 MoE

FFN 的三个作用：① **逐位置**特征变换，不跨 token 交互（那是 attention 的事）；② **引入非线性**，否则堆层近似线性；③ **参数量主体**——$d_{ff}=4d$ 时 FFN 参数 $8d^2$，attention 仅 $4d^2$，FFN 约占 2/3。

```python
h = W2 @ gelu(W1 @ x)                        # 标准 FFN：d → 4d → d
h = W3 @ (F.silu(W1 @ x) * (W2 @ x))         # SwiGLU（LLaMA 系）：三矩阵门控，d_ff ≈ 8/3·d
```

**GELU = ReLU 的平滑版本**，用标准正态 CDF 做软门控而非硬截断（erf 即高斯误差函数）：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \tfrac{1}{2}\left[1 + \mathrm{erf}\!\left(\tfrac{x}{\sqrt{2}}\right)\right]
$$

```python
gelu_exact  = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
gelu_approx = 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
silu        = x * torch.sigmoid(x)           # = Swish，SwiGLU 用的就是它
```

**MoE**：把一个大 FFN 换成 E 个专家 FFN，每个 token 只路由 Top-K 个（K=1~2）。

```python
logits     = x @ W_gate                       # (N, E) 路由打分
topv, topi = logits.topk(k, dim=-1)           # 每个 token 选 K 个专家
w = topv.softmax(-1)
y = sum(w[:, j:j+1] * experts[topi[:, j]](x) for j in range(k))

# 负载均衡 loss：避免所有 token 挤向同几个专家（专家塌缩）
f = count_per_expert / N                      # 实际分配比例
P = logits.softmax(-1).mean(0)                # 平均路由概率
aux_loss = E * (f * P).sum()
```

核心性价比：**总参数量随 E 线性增长，单 token FLOPs 几乎不变**。难点在负载均衡、capacity factor（溢出 token 丢弃或直通）、专家并行的 all-to-all 通信（详见 [DeepSeek-V3 训练解读](/blog/deepseek-v3-training-explained/)）。

### 1.6 结构层速记

- **架构三分**：Encoder-only（BERT，双向，判别）/ Encoder-Decoder（T5，翻译摘要）/ **Decoder-only**（GPT，causal mask，主流）。LLM 收敛到 Decoder-only，因为训练并行度高、天然适配自回归与 in-context learning。

```python
causal = torch.tril(torch.ones(N, N))             # 下三角：第 i 个 token 只能看 ≤ i
scores = scores.masked_fill(causal == 0, float("-inf"))
```

- **Tokenizer**：BPE（按频率合并子词）→ BBPE（在字节上做 BPE，天然覆盖任意 Unicode，中文/代码/emoji 不 OOV）→ SentencePiece（直接在原始文本上训练，无需预分词）。
- **Weight tying**：输入 embedding 与输出 `lm_head` 共享同一权重矩阵，省一份 `vocab×d` 参数。
- **ALiBi**：不旋转，而是给 attention 分数直接**加一个与距离成正比的负偏置**，天然可外推。和 RoPE 是两条外推路线。

```python
scores = scores - m_head * (i - j).clamp(min=0)   # m_head: 每个头一个固定斜率
```

- **QK-Norm**：算 attention 前先对 Q、K 各做一次 RMSNorm，抑制 logits 爆炸，近年大模型用来稳训练。

## 二、长序列效率层

### 2.1 稀疏 Attention 谱系

思路：$N^2$ 的注意力里大部分连接无用，只算**重要的 token 对**。本质是把 full mask 换成稀疏 mask：

```python
i = torch.arange(N)[:, None]
j = torch.arange(N)[None, :]

sliding = (i - j).abs() <= L                   # 滑动窗口：只看邻近 ±L
dilated = ((i - j) % r == 0) & ((i - j).abs() <= L * r)   # 空洞：等间距跳采样
global_ = (j < g) | (i < g)                    # 全局 token：少数位置与所有位置互连
strided = ((i - j) % stride == 0)              # 跨步：每隔 stride 取一个

longformer = sliding | dilated | global_
scores = scores.masked_fill(~longformer, float("-inf"))
```

| 方法 | 机制 | 复杂度 | 适用 |
| :--- | :--- | :--- | :--- |
| 原始 softmax attention | 全连接 | O(N²d) | 短序列 |
| **Sparse Transformer**（Strided） | 一头看局部行、一头看跨步列 | O(N√N) | **图像、音频** |
| **Sparse Transformer**（Fixed） | 每隔 L 取「摘要位置」 | O(N√N) | **文本** |
| **Blockwise** | 切成 n 块，$Q_i$ 只配对某个 $K_j,V_j$ | O(N²/n) | 通用 |
| **Longformer** | SW + DSW + Global | **O(N)** | 长文档 |

**Factorized Self-Attention**（Sparse Transformer）把一次全连接 attention 分解成两步稀疏 attention，合起来保证任意两 token 间有路径可达：

- **Strided**：head1 看前 `stride` 个邻居（局部行），head2 看 `i-stride, i-2·stride...`（跨步列）。要求数据有周期结构，所以适合图像/音频。
- **Fixed**：每隔 L 选一个「摘要 token」负责被其他位置读取，普通 token attend 到块内 + 最近的摘要位置。文本没有固定周期，用 Fixed 更稳。

**Blockwise** 思想极简：长度 N 平均切成 n 个短序列，每个 $Q_i$ 只和被分配到的 $K_j, V_j$ 算。

```python
n = N // block_size
perm = [(i + 1) % n for i in range(n)]         # shifting one position：单头的一种分配方式
out = [attention(Q[i], K[perm[i]], V[perm[i]]) for i in range(n)]   # O(N²/n)
```

**Longformer** 的三种 attention：SW（只看左边 L 个）、DSW（等间距跳着看左边，同样预算换更大感受野）、Global（少数固定位置既 attend 全部、也被全部 attend）。局部 + 全局兼顾，复杂度降到 O(N)。

> **实践提示**：**block-sparse**（按 16×16 块稀疏）比 token 级稀疏对 GPU 友好得多，否则理论 FLOPs 降了、实测速度不一定降。近期工作开始用「可学习的块级选择」来决定哪些块值得算。

### 2.2 FlashAttention

> *Fast and Memory-Efficient **Exact** Attention with IO-Awareness*

和稀疏 attention 的根本区别：**稀疏是近似，FlashAttention 是精确**——数学上等价于标准 attention，只改执行方式。

关键认知：attention 是 **memory-bound** 而非 compute-bound，真正的开销是把 $N \times N$ 的 scores 写进 HBM 再读回来。做法是**永不实例化完整 scores**：按块载入 SRAM，用 online softmax 增量合并。

```python
# 核心：外层遍历 K/V 分块，用 (m, l) 维护"到目前为止"的 softmax 状态
o = torch.zeros_like(q)                      # 累积输出
l = torch.zeros(N, 1)                        # 累积分母 Σexp
m = torch.full((N, 1), float("-inf"))        # 当前行最大值（数值稳定）

for Kj, Vj in blocks(K, V):                  # 每块小到能放进 SRAM
    s = q @ Kj.transpose(-2, -1) / math.sqrt(d)          # 只有 (N, block_size)
    m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
    p     = torch.exp(s - m_new)
    scale = torch.exp(m - m_new)             # 旧的 o/l 按新最大值重新缩放
    l = scale * l + p.sum(-1, keepdim=True)
    o = scale * o + p @ Vj
    m = m_new

o = o / l                                    # 最后统一除分母
```

激活显存从 O(N²) 降到 O(N)；反向不存 scores 而是**重算**（用时间换显存）。这直接缓解了后面 4.1 节「激活值占显存大头」的问题。

## 三、推理层

### 3.1 KV Cache

自回归解码时前 n−1 个 token 的 K/V 不会变，缓存即可避免重复计算：

```python
def decode_step(x_new, cache):
    q, k, v = proj_q(x_new), proj_k(x_new), proj_v(x_new)   # 只投影 1 个 token
    cache.k = torch.cat([cache.k, k], dim=1)                # (B, n, d)
    cache.v = torch.cat([cache.v, v], dim=1)
    return torch.softmax(q @ cache.k.transpose(-2, -1) / math.sqrt(d), -1) @ cache.v
```

代价是显存，随 batch × 长度线性膨胀：

```text
KV cache 字节数 = 2(K&V) × layers × seq_len × n_kv_heads × d_head × bytes

例：LLaMA-7B（32 层, 32 头, d_head=128, fp16），4K 上下文
  2 × 32 × 4096 × 32 × 128 × 2 ≈ 2 GB / 请求
```

### 3.2 MHA / MQA / GQA / MLA

四者只差一件事：**K/V 头数**。

```python
# MHA: n_kv = n_heads   每个 Q 头独享 KV，质量最好、cache 最大
# GQA: n_kv = G         分 G 组，组内共享（LLaMA-2 70B: 64 Q / 8 KV）← 主流折中
# MQA: n_kv = 1         所有头共享一组，cache 最小、质量略降

def gqa(q, k, v, n_heads, n_kv):
    rep = n_heads // n_kv
    k = k.repeat_interleave(rep, dim=2)       # KV 头广播到 Q 头
    v = v.repeat_interleave(rep, dim=2)
    return attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
```

| | n_kv | KV cache | 质量 |
| :--- | :--- | :--- | :--- |
| MHA | n_heads | 1× | 基准 |
| GQA | G | n_kv / n_heads | 接近 MHA |
| MQA | 1 | 1 / n_heads | 略降 |
| **MLA**（DeepSeek-V2/V3） | — | 比 GQA 更小 | ≥ MHA |

MLA 换了个思路：不减头数，而是把 KV **低秩压缩**成小隐向量存进 cache，用时上投影恢复。

```python
c = W_dkv @ x                 # 下投影 → 压缩隐向量，cache 里只存这个
k, v = W_uk @ c, W_uv @ c     # 用时上投影恢复（升维矩阵可吸收进 Q 的投影，不增加 decode 计算）
```

### 3.3 PagedAttention 与推理框架

KV cache 长度不可预知，按最大长度预分配会浪费 60%~80% 显存。借用 OS 虚拟内存思路：**逻辑连续、物理分散**。

```python
BLOCK = 16                                   # 每块存 16 个 token 的 KV
kv_pool     = alloc(num_blocks, BLOCK, ...)  # 全局物理块池
block_table = {seq_id: [7, 3, 12]}           # 逻辑块号 → 物理块号

def read_kv(seq_id, t):
    phys = block_table[seq_id][t // BLOCK]
    return kv_pool[phys][t % BLOCK]

def append(seq_id, seq_len):                 # 写满才申请新块 → 浪费 < 1 块
    if seq_len % BLOCK == 0:
        block_table[seq_id].append(free_list.pop())
```

额外收益是**共享**：beam search 的多个候选、多请求的相同 system prompt，可让块表指向同一物理块，写时复制。

**别把 PagedAttention 和 FlashAttention 混为一谈**：

| | FlashAttention | PagedAttention |
| :--- | :--- | :--- |
| 场景 | 训练 + 推理 prefill | 推理 decode |
| 省什么 | HBM 读写（IO）、$N^2$ 中间矩阵 | KV cache 的显存碎片 |
| 手段 | 分块进 SRAM + online softmax | 固定 block + 块表间接寻址 |

### 3.4 量化：PTQ / QAT

```python
# 对称 MinMax 线性量化（PTQ 最基础做法）
scale = x.abs().max() / 127
q = (x / scale).round().clamp(-127, 127).to(torch.int8)
x_hat = q.float() * scale                    # 反量化
```

| | **PTQ** | **QAT** |
| :--- | :--- | :--- |
| 时机 | 训完直接量化 | 训练中模拟量化 |
| 数据 | 无 / 少量校准集 | 需要训练集 |
| 成本 | 分钟级 | 一次完整训练 |
| 精度 | 低比特时掉点明显 | 更好，4-bit 以下优势大 |

QAT 的关键技巧：**前向模拟量化、反向照常走浮点**（weights/bias 仍以浮点存储，才能接收微小梯度更新）。取整不可导，用 STE（直通估计器）近似：

```python
def fake_quant(x, scale):
    q = (x / scale).round().clamp(-127, 127) * scale
    return x + (q - x).detach()              # STE：前向取 q，反向梯度视为恒等
```

LLM 侧常见：权重 only 的 **GPTQ / AWQ**（PTQ，4-bit）、训练侧的 **NF4**（QLoRA，见 4.4）、**SmoothQuant**（把激活离群值迁到权重，让 W8A8 可行）。

量化**粒度**也决定精度：per-tensor（一个 scale，最快最糙）< per-channel（每列一个 scale）< per-group（每 128 个元素一个 scale，GPTQ/AWQ 常用）；对称（零点固定为 0）比非对称省一次零点加法，但对分布偏移的激活不如非对称准。

### 3.5 推理层速记

- **采样策略**：`greedy`（取 argmax，确定但单调）→ `temperature`（除 T 调随机性）→ `top-k`（只保留概率最高的 k 个）→ `top-p`（nucleus，保留累积概率到 p 的最小集合）→ `repetition penalty`（压低已出现 token，防复读）。

```python
logits = logits / T
logits[logits < torch.topk(logits, k).values[-1]] = -float("inf")   # top-k
probs = logits.softmax(-1)
next_id = torch.multinomial(probs, 1)
```

- **Prefill vs Decode**：prefill 一次性处理整个 prompt，能并行、**compute-bound**（打满算力）；decode 逐 token 生成，每步只算 1 个 token、**memory-bound**（受显存带宽限制）。这正好解释了 FlashAttention 管 prefill、PagedAttention 管 decode。
- **投机解码**：小模型（或额外头）先猜 k 个 token，大模型一次前向并行校验，接受前缀、拒绝处回退——等价一次前向出多个 token。代表：Speculative Decoding、Medusa（多头）、EAGLE（特征级草稿）。
- **continuous batching**：请求到达即插入、完成即退出，不等整批对齐，是 vLLM 吞吐高的关键。
- **PPL = exp(CE)**：困惑度就是交叉熵取指数，CE 降 ⇔ PPL 降。

## 四、训练层

### 4.1 参数量、算力与显存

```text
存储字节 = 参数量 × 每参数字节数        （fp32=4, fp16/bf16=2, int8=1, int4=0.5）

1B(10 亿) 参数 @ fp32 ≈ 3.73 GiB   → 口算：1B ≈ 4GB
1M(100 万) 参数 @ fp32 = 4e6/1024/1024 ≈ 3.81 MiB
7B @ fp16 ≈ 14 GB（推理权重）
```

厂商说的 "GB" 是 $10^9$，显存显示的是 GiB（$2^{30}$），差约 7%。

**FLOPs ≠ FLOPS**：**GFLOPs** 是十亿次浮点**运算量**（模型复杂度，论文常用），**FLOPS** 是每秒浮点**运算次数**（硬件算力，如 A100 BF16 ≈ 312 TFLOPS）。粗估训练算力 $\text{FLOPs} \approx 6 \times N_{params} \times N_{tokens}$（前向 2 + 反向 4）。

**训练显存四大块**：

```text
显存 = 模型参数 + 优化器状态 + 梯度 + 激活值
                                        ↑ 通常占大头
```

以 Ψ = 参数量、Adam + 混合精度为例：

```text
参数 fp16                      2Ψ
梯度 fp16                      2Ψ
Adam: fp32 参数副本 + m + v   12Ψ   ← 3 份 fp32（主副本+m+v），是 fp16 权重的 6 倍
────────────────────────────────
合计（不含激活）              16Ψ   → 7B ≈ 112 GB，单卡放不下 → 第五节的 ZeRO
```

激活值才是真正的大头，且 attention 部分是 O(N²)（用 FlashAttention 降到 O(N)）。两个常规手段：

```python
h = torch.utils.checkpoint.checkpoint(layer, h)   # ① 梯度检查点：显存换算力(+约 30%)
# ② 梯度累积：小 micro-batch 多次累积，等效大 batch 而不增显存
```

**推理显存**简单得多：权重 + KV cache + 少量激活，**没有优化器和梯度**。

### 4.2 损失函数：CE / Focal

**二分类 CE** 两种写法等价，都同时考虑正负类：

$$
L = -[\,y\log p + (1-y)\log(1-p)\,]
$$

**多分类 CE** 只需看正确类别的概率 $p_y$（其他类信息已通过 softmax 分母进来），所以 **CE 必须和 softmax 配套**：

```python
loss = F.cross_entropy(logits, target)             # 内部 = log_softmax + nll_loss，数值稳定
# 等价但危险的写法：torch.log(logits.softmax(-1)) → 概率接近 0 时 log 溢出
```

LLM 的 next-token 目标就是错位一格的多分类 CE：

```python
loss = F.cross_entropy(logits[:, :-1].flatten(0, 1),
                       labels[:, 1:].flatten(),
                       ignore_index=-100)          # mask 掉 prompt / padding
```

**Focal Loss** 在 CE 前乘系数，**让已经学好的样本少贡献梯度**：

$$
L = -\alpha_t (1 - p_t)^{\gamma} \log p_t
$$

```python
p_t = probs.gather(-1, target[..., None]).squeeze(-1)
loss = -alpha * (1 - p_t) ** gamma * torch.log(p_t)    # p_t→1（好学）时权重→0
```

### 4.3 优化器：SGD → Adam → AdamW

```python
p -= lr * g                                    # ① 朴素梯度下降

m = beta1 * m + (1 - beta1) * g                # ② + Momentum：一阶动量
p -= lr * m

v = beta2 * v + (1 - beta2) * g * g            # ③ Adam：再加二阶（自适应步长）
m_hat, v_hat = m / (1-beta1**t), v / (1-beta2**t)      # 偏差校正（t 小时很关键）
p -= lr * m_hat / (v_hat.sqrt() + eps)

p -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * p)      # ④ AdamW：weight decay 解耦
```

关键区别：Adam 的 L2 正则加在**梯度里**，会被 $\sqrt{v}$ 自适应缩放掉——梯度大的参数反而被正则得更弱；AdamW 解耦后正则强度一致，泛化更好。LLM 训练统一用 AdamW（典型 `betas=(0.9, 0.95)`、`wd=0.1`），代价就是那 12Ψ。

### 4.4 高效微调：LoRA / QLoRA

> PET 家族还有 Adapter、Prefix-Tuning、P-Tuning、IA³，工业界基本收敛到 LoRA。

LoRA 冻结原权重 $W$，只训练低秩旁路 $\Delta W = \frac{\alpha}{r}BA$，可训练参数降到 $r(d+k)$（通常 0.1%~1%）：

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)                      # 冻结原权重
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))    # 零初始化 → 起点 = 原模型
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + self.s * (x @ self.A.T @ self.B.T)

    def merge(self):                                                # 推理期合并，零额外延迟
        self.base.weight.data += self.s * self.B @ self.A
```

要点：`B` 必须零初始化；挂所有线性层（含 FFN）效果好于只挂 Q/V；省的主要是**优化器状态和梯度**（12Ψ → 近乎 0），**激活值并不省**。

**QLoRA** 把冻结的基座量化到 4-bit，单卡 48G 可微调 65B：① **NF4**（按正态分布分位点设计的 4-bit 类型）；② **双重量化**（对量化常数再量化，每参数再省约 0.37 bit）；③ **分页优化器**（扛长序列显存尖峰）。

```python
base  = load_in_4bit(model, quant_type="nf4", double_quant=True)   # 基座 4-bit 冻结
model = add_lora(base, r=64, alpha=16, target="all-linear")        # 适配器 bf16 训练
```

**LoRA 变体**：rsLoRA（把缩放改成 $\alpha/\sqrt{r}$，大 rank 更稳）、LoRA+（给 B 更大学习率）、PiSSA（用 W 的主奇异向量初始化 A/B，收敛更快）、DoRA（权重拆成幅度 + 方向，只对方向做低秩）。

### 4.5 训练稳定性速记

- **学习率调度**：warmup（线性升到峰值）+ cosine decay。**必须 warmup** 的原因：训练初期 Adam 的二阶动量 $v$ 估计不准，直接大步长容易发散。
- **混合精度**：fp16 动态范围小，梯度易下溢 → 需 **loss scaling**（放大 loss、回传后再缩回）；bf16 指数位和 fp32 一样宽，通常**免 loss scaling**。无论哪种都保留一份 fp32 主权重。
- **梯度裁剪**：按全局范数裁剪，防 loss spike。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

- **对齐链路**：SFT（模仿）→ RM（训奖励模型）→ RLHF（PPO）；近年简化为 **DPO**（无需显式 RM）、Agent 场景用 **GRPO / GiGPO**（见 [GiGPO 信用分配](/blog/agent-rl-credit-assignment-gigpo/)）。
- **Scaling Law**：Chinchilla 结论——给定算力，参数量与训练 token 数应**同比例增长**，最优配比约 **1 : 20**（1B 参数配 ~20B token）。

## 五、分布式与框架

### 5.1 ZeRO

ZeRO 的洞察：DP 下每张卡都存了一份**完全相同**的模型状态，纯冗余，切开即可。

```text
          参数(fp16)  梯度(fp16)  Adam(fp32 参数+m+v)   每卡总计
baseline      2Ψ          2Ψ            12Ψ               16Ψ
ZeRO-1        2Ψ          2Ψ           12Ψ/N        4Ψ + 12Ψ/N   ← 切 optimizer 状态
ZeRO-2        2Ψ         2Ψ/N          12Ψ/N        2Ψ + 14Ψ/N   ← 再切 gradient
ZeRO-3       2Ψ/N        2Ψ/N          12Ψ/N           16Ψ/N     ← 再切 parameter
```

```python
# ZeRO-3 / FSDP 的核心动作：用完即释放
def forward(layer, x):
    full_w = all_gather(layer.shard)      # 临时凑出完整权重
    y = layer.compute(x, full_w)
    del full_w                            # 立刻释放，只留自己那一份 shard
    return y
# 反向用 reduce_scatter（而非 all_reduce）→ 每卡只保留自己负责的梯度分片
```

ZeRO-3 通信量约为前两级的 1.5 倍。PyTorch **FSDP ≈ ZeRO-3**；再叠 CPU/NVMe offload 可换更大模型，代价是带宽。细节见 [Megatron 笔记（四）：数据并行与 ZeRO](/blog/megatron-lm-04-data-parallel-zero/)。

### 5.2 六种并行怎么配

| 并行 | 切什么 | 通信 | 适用 |
| :--- | :--- | :--- | :--- |
| **DP** / ZeRO | batch（+ 模型状态分片） | all-reduce / all-gather | 通用基础 |
| **TP**（Tensor） | 单层权重的行/列 | **每层** all-reduce，带宽敏感 | 节点内（NVLink） |
| **SP**（Sequence） | 序列维上切 LayerNorm/Dropout 的激活 | all-gather + reduce-scatter | 与 TP 搭配省激活显存 |
| **PP**（Pipeline） | 按层切成 stage | 相邻 stage P2P | 跨节点，注意 bubble |
| **EP**（Expert） | MoE 专家 | all-to-all | MoE 模型 |
| **CP**（Context） | 超长序列的序列维（含 attention） | ring 通信 | 长上下文 |

```python
# TP 示例：FFN 两个矩阵分别按列、按行切 → 一次 all_reduce 就够
h = act(x @ W1_col_shard)          # 列切：各卡算一部分隐藏维，无需通信
y = all_reduce(h @ W2_row_shard)   # 行切：各卡算部分和，相加得完整结果
```

经验顺序：**TP 不跨节点**（放进 NVLink 域）→ PP 跨节点 → 最外层 DP/ZeRO；长序列加 CP + SP，MoE 加 EP。拆开讲见 [张量并行](/blog/megatron-lm-02-tensor-parallel/) 与 [流水并行](/blog/megatron-lm-03-pipeline-parallel/)。

**几个常被追问的细节**：

- **Ring All-Reduce**：DP 梯度同步 = reduce-scatter + all-gather 两段环形传输，每卡通信量约 $2\Psi$，**与卡数无关**（NCCL 默认实现）。
- **Pipeline bubble**：朴素流水的空泡占比 $\frac{p-1}{m+p-1}$（p=stage 数，m=micro-batch 数）→ 增大 m、用 **1F1B / interleaved** 调度来摊薄。
- **MoE 进阶**：DeepSeek 的**共享专家**（恒激活，兜底通用知识）+ **细粒度专家**（切更小、组合更多）；训练追求 **dropless**（不丢 token）；EP 的瓶颈是 all-to-all 通信。

### 5.3 框架速查

| 方向 | 框架 | 核心卖点 |
| :--- | :--- | :--- |
| 推理 | **vLLM** | PagedAttention + continuous batching，吞吐优先 |
| 推理 | **SGLang** | RadixAttention（前缀树复用 KV）+ 结构化输出，多轮/Agent 场景强 |
| 训练 | **Megatron-LM** | TP/SP/PP/EP/CP 多维并行，千卡级预训练首选 |
| 训练 | **FSDP** | PyTorch 原生参数分片（≈ ZeRO-3），接入成本低 |
| 训练 | **DeepSpeed** | ZeRO 1/2/3 + offload，易接入 HF 生态 |

一句话区分：**推理框架优化显存管理与调度，训练框架优化并行切分与通信**。

## 附：视觉方向常考点

不属于大模型主线，但面试常一起问，集中放这里。

**评估指标**

| 指标 | 衡量什么 | 方向 |
| :--- | :--- | :--- |
| **FID** | 生成图集与真实图集在 Inception 特征空间的分布距离 | 越小越好 |
| **PSNR** | 像素级误差（峰值信噪比） | 越大越好 |
| **SSIM** | 亮度 + 对比度 + 结构 三方面相似度 | 越接近 1 越好 |

```python
# FID：两图集分别过 Inception（取 pool3 的 2048 维特征）→ 均值与协方差 → Fréchet 距离
fid = ((mu1-mu2)**2).sum() + np.trace(S1 + S2 - 2*sqrtm(S1 @ S2))
psnr = 10 * np.log10(MAX**2 / mse)                        # 只看逐像素 MSE
ssim = ((2*mu_x*mu_y+C1) * (2*cov_xy+C2)) / ((mu_x**2+mu_y**2+C1) * (var_x+var_y+C2))
```

注意 PSNR 高不代表视觉好（过度平滑的结果往往 PSNR 更高），所以超分论文会同时报感知类指标。

**MobileNet v1/v2**：v1 是**深度可分离卷积** = Depthwise（`groups=C`，逐通道空间滤波）+ Pointwise（1×1 混通道），计算量比值约 $1/C_{out} + 1/k^2$。v2 是**倒残差 + 线性瓶颈**：PW 先升维 t 倍 → DW → PW 降维（DW 表达能力弱，要在高维做），且**降维后不加 ReLU**——ReLU 在低维特征上会不可逆丢信息。

```python
nn.Conv2d(C, C, k, groups=C)   # Depthwise
nn.Conv2d(C, C_out, 1)         # Pointwise
```

**Anchor-free 检测**：FCOS 逐像素预测——落在 GT 框内即正样本，回归目标是到**四条边的距离** $(l,t,r,b)$，再用 `centerness` 压低偏心预测：

```python
l, t, r, b = x - x1, y - y1, x2 - x, y2 - y
centerness = math.sqrt(min(l, r)/max(l, r) * min(t, b)/max(t, b))
score = cls_score * centerness
```

DETR 把检测变成**集合预测**：CNN 特征 → Transformer Encoder-Decoder → N 个 object query 各输出一个 (类别, 框) → 匈牙利算法做一对一匹配 → 算 loss。一对一匹配天然避免重复框，所以**不需要 NMS**；代价是收敛慢（需 500 epoch），后续 Deformable DETR、DINO 主要解决这点。

**Matting** 求连续 alpha 而非二值 mask，核心是合成方程 $I = \alpha F + (1-\alpha)B$。一个方程三个未知量是欠定的，所以必须引入先验（trimap / 语义 / 纯背景图）。关键在**半透明区域**（发丝、烟雾、玻璃），这是它与语义分割的本质差别：

```python
loss = l1(alpha_pred, alpha_gt) \
     + l1(grad(alpha_pred), grad(alpha_gt)) \                    # 梯度损失：保边缘锐度
     + l1(alpha_pred * F + (1 - alpha_pred) * B, image)          # 合成损失：回代原方程
```

## 小结

回头看，这套知识点其实是一条**因果链**，而不是一堆孤立名词：

```text
Attention 是 O(N²) 
  → 长序列上算不起 → 稀疏 Attention（近似）/ FlashAttention（精确，省 IO）
  → 推理时重复算 K/V → KV Cache（省算力，但吃显存）
  → KV Cache 吃显存 → GQA/MLA（减 cache 体积）+ PagedAttention（减碎片）
  → 模型太大装不进一张卡 → 显存四大块拆解 → ZeRO 切状态、TP/PP/EP 切计算
  → 全量微调更装不下 → LoRA（省优化器和梯度）+ QLoRA（基座 4-bit）
```
