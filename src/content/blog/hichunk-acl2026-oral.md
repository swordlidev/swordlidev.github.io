---
title: "HiChunk：用层次化分块提升 RAG（ACL 2026 Oral 演讲稿）"
description: "我在 ACL 2026 的 oral presentation 整理：面向 RAG 的层次化分块框架 HiChunk 与评测基准 HiCBench。逐页给出幻灯片要点，并附上当时的英文演讲稿（讲稿备注）。"
pubDate: 2026-07-07
tags: ["RAG", "ACL 2026", "HiChunk", "Chunking", "LLM", "演讲"]
---

> 这是我在 **ACL 2026**（2026 年 5 月 28 日）做的一次 **oral presentation** 的整理。论文题为 *HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking*，来自 **Tencent Youtu Lab**。
>
> 下面按幻灯片顺序给出每页要点，并附上我当时的**英文演讲稿**（即 PPT 的讲稿备注），方便回顾与分享。每页的 `演讲稿` 引用块即为现场口述内容的原文。

![我在 ACL 2026 做 oral presentation 的现场](/images/acl2026/talk.jpg)

*ACL 2026 oral presentation 现场。*

![ACL 2026 会议现场](/images/acl2026/venue.jpg)

*ACL 2026 会议现场。*

## 一分钟了解 HiChunk

RAG 系统里，文档先要被切成小块（chunking）再进检索。分块质量直接决定“检索器能找到什么、生成器能产出什么”。我们指出两个被忽视的问题——**证据稀疏（evidence sparsity）**让现有基准无法区分分块好坏；**线性分块**破坏了文档本身的层次结构——并给出两样东西：

- **HiCBench**：一个带人工层次标注、证据密集型问答、支持多维度评测的新基准；
- **HiChunk**：用微调小模型（Qwen3-4B）把文档预测成**层次树**，用 **Residual Lines** 解决长文档的“层级漂移”，再用 **Auto-Merge** 在检索时动态合并粒度。

结果：一级分块点 F1 **48.41**（LumberChunk 28.49），证据召回 **+4.95%**、事实覆盖 **+4.74%**，且比强基线 LumberChunk **快约 3 倍**。

---

## 1. 开场（Title）

**幻灯片**：HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking · Tencent Youtu Lab · ACL 2026（May 28, 2026）。

> **演讲稿：** Good morning, everyone, and thank you for attending this session. It is my pleasure to present our work titled "HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking."
>
> This research is a product of the Tencent Youtu Lab. In this presentation, I will explain how we address a fundamental yet often overlooked problem in retrieval-augmented generation systems, namely how documents are structured before they enter the retrieval pipeline.
>
> First, allow me to provide a brief overview of what I will cover today.

## 2. 目录（Agenda）

**幻灯片**：① Background & Motivation ② HiCBench Dataset ③ HiChunk Framework ④ Experimental Results ⑤ Conclusion & Future Work。

> **演讲稿：** This slide outlines the overall structure of the next ten minutes or so. I have divided the presentation into five main parts.
>
> First, I will establish context by discussing the background and motivation, focusing on why document chunking plays a critical role in RAG systems.
>
> Second, I will introduce our methodology, including our newly proposed benchmark HiCBench and the HiChunk framework we designed.
>
> Third, we will examine the experimental results, where I will share key findings that demonstrate the effectiveness of our approach.
>
> Fourth, I will candidly assess the limitations of our work and point to promising directions for future research.
>
> Finally, I will provide a concise summary of our contributions. Let us now move into the first part.

## 3. 背景与动机（Background & Motivation）

**幻灯片要点**：分块是连接长文档与 LLM 上下文窗口的关键桥梁，直接决定检索准确率与生成质量。两大核心挑战：

1. **证据稀疏（Evidence Sparsity）**：关键证据往往分散在不同块中，或被无关信息淹没，检索模型难以定位完整上下文。
2. **线性分块的局限**：传统定长/规则式线性分块破坏了文档固有的语义连贯性与层次结构。

> **演讲稿：** I would like to begin by discussing a foundational step in building RAG systems. When constructing a retrieval-augmented generation pipeline, the initial step involves dividing documents into smaller segments, a process known as chunking. This seemingly straightforward operation has profound implications for downstream performance. The quality of the generated chunks directly determines what the retriever can locate and what the generator can ultimately produce.
>
> Our research is driven by two core challenges in this field.
>
> The first challenge is what we term the Evidence Sparsity problem. In many existing evaluation benchmarks, the answer to a query depends on only a small number of sentences within a lengthy document, typically one or two sentences. This evidence sparsity means that when evaluating RAG systems, traditional chunking methods can still answer correctly even with poor chunking quality, making it impossible to effectively assess segmentation quality.
>
> The second challenge concerns the limitations of linear chunking. Most existing methods treat documents as flat sequences of text, completely ignoring the hierarchical structure that authors intentionally embed through sections, subsections, and paragraphs during the writing process. If properly captured, this structural information can significantly enhance retrieval quality. These two challenges together form the foundation of our research exploration.

## 4. 为什么需要 HiCBench

**幻灯片要点**：现有基准不是为“评测分块”设计的——大多只看端到端表现、无法隔离分块影响，且受证据稀疏问题困扰（无论怎么切都能答对）。HiCBench 的三个特点：**人工层次标注**、**证据密集型问答**（需跨多块综合信息）、**多维度评测**（分块准确率 + 检索召回 + 生成质量）。

> **演讲稿：** Having identified these challenges, we encountered a follow-up question: how can one rigorously evaluate whether a chunking method is effective? Our examination of existing benchmarks revealed that they were not designed to adequately answer this question. Most existing benchmarks focus on end-to-end RAG performance without isolating the specific impact of chunking. Furthermore, many of them suffer from the evidence sparsity problem I just described, where the evidence relevant to a question can be fully retrieved regardless of how the document is chunked.
>
> These limitations motivated us to develop HiCBench, which features three distinctive characteristics.
>
> First, it provides human-annotated hierarchical structure. Trained annotators examine documents and mark where semantic boundaries occur at different levels, including section boundaries, subsection boundaries, and paragraph boundaries.
>
> Second, it contains evidence-intensive question-answer pairs. Unlike existing benchmarks, our questions require synthesizing information from multiple chunks, which makes chunking quality truly decisive.
>
> Third, it supports multi-dimensional evaluation, enabling researchers to directly assess chunking accuracy, measure retrieval recall, and evaluate generation quality within the same unified benchmark.
>
> This comprehensive design offers a complete picture of how chunking influences the entire RAG pipeline.

## 5. HiChunk 框架

**幻灯片要点**：三大组件协同工作——

1. **基于微调 LLM 的层次化分块**：用紧凑模型（Qwen3-4B）微调，预测分块边界及其层级标签（L1 章、L2 节……），把扁平文档变成语义连贯的树。
2. **带 Residual Lines 的迭代推理**：长文档用滑动窗口处理，窗口孤立处理会丢失全局结构（“层级漂移”）；Residual Lines 把此前窗口的压缩结构信息带入当前窗口，保持模型对自身在层次中位置的感知。
3. **Auto-Merge 检索算法**：从细粒度块出发，按三个条件动态向上合并到父块——**coherence（一致性）**、**substantiality（充实度）**、**feasibility（可行性/预算）**。

> **演讲稿：** I will now introduce our proposed solution, the HiChunk framework. At its core, HiChunk consists of three main components that work together to produce high-quality hierarchical chunks.
>
> The first component is hierarchical structuring driven by a fine-tuned language model. We adopt a relatively compact large language model, specifically Qwen3 with 4 billion parameters, and fine-tune it to predict chunk boundaries and their corresponding hierarchical levels. Given a document, the model identifies appropriate split points and assigns a level label to each segment, for example, level one for top-level sections, level two for subsections, and so forth. This process transforms a flat document into a structured tree of semantically coherent chunks.
>
> The second component is iterative inference with residual lines. Real-world documents can be quite long, often exceeding the context window of our model. To address this, we employ a sliding window approach. However, a subtle problem arises: when the model processes a window in isolation, it may lose awareness of the global document structure. We term this phenomenon hierarchical drift. To mitigate this, we introduce residual lines, which carry compressed structural information from previously processed windows into the current window. This mechanism preserves the model's sense of its position within the overall document hierarchy.
>
> The third component is the Auto-Merge algorithm, which serves as our retrieval strategy. Rather than always retrieving at a fixed granularity, Auto-Merge dynamically adjusts based on query characteristics. It begins with fine-grained chunks and then determines whether to merge them upward into coarser parent chunks. This decision is governed by three conditions. First, the consistency condition requires that at least two sibling chunks be retrieved. Second, the sufficiency condition verifies that the retrieved content covers a significant portion of the parent chunk. Third, the feasibility condition ensures that sufficient token budget exists to accommodate the merged chunk. This adaptive mechanism combines the advantages of both approaches: the semantic richness of fine-grained chunks and the structural integrity of coarse-grained chunks.

## 6. 实验结果总览

**三句话结论**：分块更准（一级点 F1 48.41 vs LumberChunk 28.49；全层级 54.5 vs 48.58）、检索/生成更强、速度约快 3 倍。

**检索与端到端质量**（ERec=证据召回，Fact-Cov=事实覆盖，T1 为证据密集任务）：

| Chunk Method | Qasper ERec | Qasper F1 | HiCBench ERec (T1) | HiCBench Fact-Cov (T1) |
|---|---|---|---|---|
| Fixed Chunk (200) | 84.32 | 46.49 | 74.06 | 63.20 |
| Semantic Chunk | 82.22 | 46.39 | 71.26 | 61.09 |
| LumberChunk | 87.43 | 46.82 | 75.53 | 64.76 |
| HiChunk (200) | 86.49 | 46.99 | 77.68 | 63.93 |
| **HiChunk (200)+Auto Merge** | **87.85** | **47.25** | **81.03** | **68.12** |

**时间开销（s/doc）**：HiChunk+Auto-Merge 在质量最好的同时，远快于 LumberChunk。

| Chunk Method | Qasper Time | Qasper F1 | HiCBench Time | HiCBench Fact-Cov |
|---|---|---|---|---|
| Semantic Chunk | 0.4867 | 46.39 | 1.0169 | 61.09 |
| LumberChunk | 5.4991 | 46.82 | 13.4414 | 64.76 |
| **HiChunk (200)+Auto Merge** | **1.4993** | **47.25** | **5.7506** | **68.12** |

> **演讲稿：** I will now present the key experimental results, highlighting three core metrics that summarize HiChunk's overall performance.
>
> Regarding chunking accuracy, at the first-level split points, our method achieved an F1 score of 48.41 on HiCBench, representing a 20 percent improvement over the next best method, LumberChunk. It also showed a 6 percent improvement across all split points. The Semantic Chunk method, by contrast, performed poorly in accuracy. We therefore provide significantly more accurate chunk boundaries.
>
> Regarding end-to-end RAG performance, compared to the strongest baseline LumberChunk, HiChunk with Auto-Merge improved evidence recall by 4.95 percentage points on HiCBench. The end-to-end factual coverage of responses improved by 4.74 percentage points. On evidence-intensive tasks, where chunking quality truly matters, HiChunk's gains were particularly pronounced. On the Qasper dataset, where evidence is relatively sparse, HiChunk also achieved the best performance.
>
> Regarding time efficiency, our method runs approximately three times faster than LumberChunker while achieving comparable or superior quality. For instance, on the Qasper dataset, HiChunk requires 1.49 seconds, whereas LumberChunker requires 5.49 seconds. This efficiency advantage makes HiChunk practical for real-world deployment scenarios where processing documents one by one under strict time constraints is essential.
>
> In summary, we achieved better chunking, stronger retrieval, and higher efficiency. That represents the central narrative of our results.

## 7. 消融实验

**微调 vs 少样本提示**（证明微调的必要性）：

| Method | F1 (L1) | F1 (all) | ERec | Fact-Cov |
|---|---|---|---|---|
| HC (1-shot) | 17.84 | 23.28 | 73.47 | 46.62 |
| HC (3-shot) | 25.00 | 21.99 | 73.05 | 46.55 |
| **HC (fine-tuned)** | **48.41** | **54.50** | **80.86** | **49.97** |

**Auto-Merge 三条件消融**（三条件组合最稳健）：

| 条件组合 | ERec | Fact-Cov | LongBench |
|---|---|---|---|
| Cond 3 only | 81.43 | 51.35 | 43.25 |
| Cond 1+3 | 80.55 | 50.70 | 43.80 |
| **Cond 1+2+3** | 80.86 | 49.97 | **44.41** |

**Auto-Merge 对“层次 vs 扁平”的依赖**（只有配合层次结构才显著增益）：

| Method | ERec | Fact-Cov |
|---|---|---|
| Semantic Chunk + AM | 73.35 (+0.60) | 45.92 (−0.19) |
| LumberChunk + AM | 76.86 (+0.46) | 50.21 (+0.16) |
| HiChunk (L1) + AM | 77.66 (+0.96) | 49.68 (+0.42) |
| **HiChunk + AM** | **81.30 (+2.67)** | **54.70 (+5.17)** |

> **演讲稿：** To understand the sources of these performance gains, we conducted a series of ablation studies. I will highlight three key findings from this analysis.
>
> Our first study examined the necessity of fine-tuning. We investigated whether few-shot prompting could serve as a viable alternative. The results indicate unequivocally that fine-tuning is essential. We compared the fine-tuned model against one-shot and three-shot prompting baselines. The fine-tuned model achieved an F1 score of 54.50 on HiCBench, while three-shot prompting reached only 21.99, revealing a substantial performance gap. This gap also propagated to end-to-end RAG performance. The model therefore genuinely needs to learn this task through training rather than through instruction prompting alone.
>
> Our second study examined the Auto-Merge algorithm, specifically testing whether each of its three conditions contributes substantively. When we removed the consistency condition, performance declined. When we removed the sufficiency condition, performance declined. When we removed the feasibility condition, performance also declined. All three conditions are necessary for the algorithm to operate robustly across different task types.
>
> Notably, we also found that applying the Auto-Merge retrieval strategy on top of flat chunking methods such as Semantic Chunk and LumberChunk produced almost no improvement. Only when Auto-Merge was combined with HiChunk did the metrics of evidence recall and response factual coverage show meaningful gains. The algorithm only demonstrates its value when paired with hierarchical structure. This finding confirms that the synergy between structural organization and retrieval strategy drives our results.

## 8. 增强验证（鲁棒性）

**幻灯片要点**：① 不同**上下文预算**（2k–4k tokens）下，HiChunk+Auto-Merge 始终最优，预算越大全体提升但优势保持；② 不同**最大层级**：从 1 层到 3 层稳定提升，超过 3 层收益递减——三层已能捕获大部分有用的结构信息。

> **演讲稿：** We further investigated how HiChunk performs under varying operational conditions. To this end, we conducted two additional sets of experiments.
>
> The first set varied the context budget available to the retriever, testing budgets ranging from 2,000 tokens to 4,000 tokens. The results show that HiChunk with Auto-Merge consistently delivers the best response quality across all budget levels. Even under tight budget constraints, our method outperforms the baselines. As the budget increases, all methods show improvement, but our performance advantage persists. This indicates that HiChunk remains robust under different resource constraints that practitioners may face.
>
> The second set varied the maximum hierarchical level the model is allowed to predict, ranging from only one level, corresponding to essentially flat chunking, to four levels, capturing very fine-grained substructures. The results show that performance steadily improves when moving from one level to three levels. However, beyond three levels, the gains begin to diminish. Evidence recall plateaus, and response quality shows only marginal improvement. This suggests that three hierarchical levels capture most of the useful structural information, while deeper levels add complexity without corresponding benefits.

## 9. 学术贡献

**幻灯片要点**：① **新基准 HiCBench**——填补证据密集型分块评测空白；② **新框架 HiChunk**——语义完整的层次化分块；③ **新算法 Auto-Merge**——动态调整检索粒度，平衡召回与噪声；④ **长文档优化**——迭代推理解决层级漂移。

> **演讲稿：** I will now step back and summarize what we consider to be the four main contributions of this work.
>
> First, we introduced HiCBench, a new benchmark specifically designed for evaluating document chunking in RAG systems. It provides human-annotated hierarchical structure, evidence-intensive question-answer pairs, and multi-dimensional evaluation metrics. We hope this will become a standard resource for the community to rigorously evaluate chunking methods.
>
> Second, we proposed the HiChunk framework, a novel approach that leverages fine-tuned language models to generate hierarchical document structure. It handles long documents through iterative inference with residual lines and achieves state-of-the-art chunking accuracy at a practical time cost.
>
> Third, we designed the Auto-Merge algorithm, a retrieval strategy that dynamically adjusts chunk granularity based on query requirements. It balances semantic richness and structural integrity through three principled conditions. We demonstrated that it only functions effectively when combined with hierarchical structure.
>
> Fourth, we provided a comprehensive optimization study for long-document RAG. We analyzed how chunking interacts with retrieval budget, hierarchical depth, and different task types. Our findings offer practical guidance for building more effective RAG systems in real-world applications.

## 10. 局限与未来工作

**幻灯片要点**：三个局限——① **收益递减**：极深层次结构收益有限、开销上升；② **数据依赖**：依赖高质量层次标注，构建成本高、低资源域受限；③ **漏斗效应**：自顶向下检索易累积误差，上层检索失败会永久丢失下层关键信息。未来方向：无监督/弱监督层次构建；全局-局部联合打分缓解漏斗效应；扩展到更多长上下文与跨模态检索场景。

> **演讲稿：** I wish to be candid about the limitations of our approach. We have identified three main constraints.
>
> The first limitation concerns diminishing returns. As noted earlier, performance gains tend to plateau beyond three hierarchical levels. While the framework technically supports deeper hierarchies, the practical benefits remain limited. This suggests that more sophisticated modeling techniques may be needed to extract value from very deep document structures.
>
> The second limitation relates to data dependency. Our method requires fine-tuning data with hierarchical annotations. Creating such data is labor-intensive and currently available only for a limited set of domains. Extending to new domains would require additional annotation effort, which poses a practical barrier to widespread adoption.
>
> The third limitation is what we call the funnel effect. Errors at lower levels propagate upward through the structure. If the model misclassifies a level-two boundary, this affects all higher-level structures built upon it. While our residual line mechanism partially mitigates this issue, it remains an inherent challenge in hierarchical modeling.
>
> Regarding future work, we identify two promising directions. One direction involves developing more efficient high-level modeling techniques to overcome the challenges of diminishing returns and data scarcity. The other direction involves combining HiChunk with complementary RAG techniques, such as deferred chunking or context extension methods, to further enhance end-to-end performance. We view hierarchical chunking not as a standalone solution but as a foundation upon which other techniques can be built.

## 11. 结论

**幻灯片要点**：HiChunk 通过**层次化分块 + 动态检索**，同时解决证据稀疏与线性分块两大难题；在多个基准上显著超越传统检索基线，为 LLM 提供更准确、更具上下文的 prompt，减少幻觉。

> **演讲稿：** I will now synthesize the key takeaways from this presentation. We began with two core challenges in RAG document chunking: evidence sparsity, where answers depend on scattered text fragments, and the limitations of linear chunking, which ignores document structure. Our innovative approach addresses both challenges simultaneously through the HiChunk framework, which employs a fine-tuned language model to predict hierarchical chunk boundaries, utilizes iterative inference with residual lines to handle long documents, and dynamically adjusts retrieval granularity through the Auto-Merge algorithm. The results are significant. HiChunk achieves substantial gains on evidence-intensive tasks and does so with relatively acceptable time cost. Our ablation studies confirm that each component of the framework contributes substantively, and our augmented validation demonstrates robustness across varying operational conditions. We believe this work provides both practical tools and conceptual foundations for advancing document structuring in RAG systems.

## 12. 致谢

> **演讲稿：** Thank you once again for your time and attention.

---

最后放一张花絮：ACL 2026 的录用论文作者国家/地区分布，中国学者占比排在最前。

![ACL 2026 录用论文作者的国家/地区分布，中国学者占比最高](/images/acl2026/authors-by-country.jpg)

*ACL 2026 录用作者的国家/地区分布，中国学者数量最多。*

以上就是这次 ACL 2026 oral 的完整内容与讲稿。如果你对 RAG、长文档检索或层次化分块有想法，欢迎在[关于页](/about/)的邮箱与我交流。
