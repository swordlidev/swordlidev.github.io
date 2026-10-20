// 论文列表。重点为近 3 年（2024-2026）工作，并保留早期代表作。
// 作者署名中会自动加粗 PROFILE.name（见 PubItem 组件）。
// note 可标注作者角色 / 荣誉；links 为相关链接。

export type PubType = 'preprint' | 'conference' | 'journal';

export interface Publication {
  title: string;
  authors: string;
  venue: string;
  year: number;
  type: PubType;
  links?: { label: string; href: string }[];
  topics?: string[];
  selected?: boolean;
  note?: string;
  citations?: number;
}

export const PUBLICATIONS: Publication[] = [
  // ---------------- 2026 ----------------
  {
    title: 'ArtSociety: Multi-Agent Multimodal Collaboration for Art Emotion Understanding',
    authors:
      'Jian Li, Fan Ji, Jun Lai, Ying Tai, Jian Yang, Xiaotong Yuan, Chengjie Wang, Yabiao Wang',
    venue: 'arXiv preprint',
    year: 2026,
    type: 'preprint',
    note: 'First author',
    topics: ['Agent', 'MLLM'],
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2609.13240' }],
  },
  {
    title: 'SE-Search: Self-Evolving Search Agent via Memory and Dense Reward',
    authors:
      'Jian Li, Yizhang Jin, Dongqi Liu, Hang Ding, Jiafu Wu, Dongsheng Chen, Yunhang Shen, Yulei Qin, Ying Tai, Chengjie Wang, Xiaotong Yuan, Yabiao Wang',
    venue: 'arXiv preprint',
    year: 2026,
    type: 'preprint',
    selected: true,
    note: 'First author',
    topics: ['Agent', 'RL', 'RAG'],
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2603.03293' }],
  },
  {
    title: 'Improving Search Agent with One Line of Code',
    authors: 'Jian Li, Dongsheng Chen, Zhenhua Xu, Yizhang Jin, Jiafu Wu, Chengjie Wang, Xiaotong Yuan, Yabiao Wang',
    venue: 'arXiv preprint',
    year: 2026,
    type: 'preprint',
    note: 'First author',
    topics: ['Agent', 'RAG'],
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2603.10069' }],
  },
  {
    title: 'TGPO: Efficient Policy Optimization through Sequence Anchor and Information Gating',
    authors:
      'Hang Ding, Dongqi Liu, Qiming Feng, Jian Li, Tong Lei, Jiafu Wu, Shuo Wang, Jiangning Zhang, Chengjie Wang, Yabiao Wang',
    venue: 'ICML 2026',
    year: 2026,
    type: 'conference',
    selected: true,
    topics: ['RL', 'LLM'],
    links: [{ label: 'PMLR', href: 'https://proceedings.mlr.press/v306/ding26i.html' }],
  },
  {
    title: 'RoleArena: A Multi-Agent Role-Playing Environment for Long Multi-Turn Dialogues with Autonomous Plot Progression',
    authors: 'Qi Zhao, Jian Li, Yizhang Jin, Dongqi Liu, Hang Ding, Qi Song, Xiangyang Li',
    venue: 'EMNLP 2026',
    year: 2026,
    type: 'conference',
    topics: ['Agent', 'LLM'],
    links: [{ label: 'PDF', href: 'https://openreview.net/pdf/9cf1d49dde06535d284333a09ff7513ace91e35d.pdf' }],
  },
  {
    title: 'AdaMARP: An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing',
    authors: 'Zhenhua Xu, Dongsheng Chen, Shuo Wang, Jian Li, Chengjie Wang, Meng Han, Yabiao Wang',
    venue: 'Findings of ACL 2026',
    year: 2026,
    type: 'conference',
    topics: ['Agent', 'LLM'],
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2601.11007' }],
  },
  {
    title: 'Disco-RAG: Discourse-Aware Retrieval-Augmented Generation',
    authors:
      'Dongqi Liu, Hang Ding, Qiming Feng, Xurong Xie, Zhucun Xue, Chengjie Wang, Jian Li, Jiangning Zhang, Yabiao Wang',
    venue: 'ACL 2026',
    year: 2026,
    type: 'conference',
    topics: ['RAG', 'LLM'],
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2601.04377' }],
  },
  {
    title: 'VITA-Audio: Fast Interleaved Audio-Text Token Generation for Efficient Large Speech-Language Model',
    authors:
      'Zuwei Long, Yunhang Shen, Chaoyou Fu, Heting Gao, Lijiang Li, Peixian Chen, Mengdan Zhang, Hang Shao, Jian Li, Jinlong Peng, et al.',
    venue: 'NeurIPS 2026',
    year: 2026,
    type: 'conference',
    topics: ['MLLM', 'Speech'],
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2505.03739' }],
  },
  {
    title: 'LLM-Oriented Token-Adaptive Knowledge Distillation',
    authors: 'Xurong Xie, Zhucun Xue, Jiafu Wu, Jian Li, Yabiao Wang, Xiaobin Hu, Yong Liu, Jiangning Zhang',
    venue: 'AAAI 2026',
    year: 2026,
    type: 'conference',
    topics: ['LLM', 'Efficient'],
    citations: 16,
  },
  {
    title: 'LLaVA-MR: Large Language-and-Vision Assistant for Video Moment Retrieval',
    authors: 'Weiheng Lu, Jian Li, An Yu, Ming-Ching Chang, Shengpeng Ji, Min Xia',
    venue: 'ICPR 2026',
    year: 2026,
    type: 'conference',
    selected: true,
    note: 'Corresponding author',
    topics: ['MLLM', 'Video'],
    citations: 22,
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2411.14505' }],
  },
  {
    title: 'Dr.V: A Hierarchical Perception-Temporal-Cognition Framework to Diagnose Video Hallucination by Fine-Grained Spatial-Temporal Grounding',
    authors:
      'Mingyi Luo, Shuo Wu, Longbiao Jing, Tianshui Ju, Lei Zheng, Jun Lai, Tao Wu, Xinyi Du, Jian Li, Shanshan Yan, Jiebo Luo, et al.',
    venue: 'IJCV 2026',
    year: 2026,
    type: 'journal',
    topics: ['MLLM', 'Video'],
  },
  {
    title: 'SMART: Shot-Aware Multimodal Video Moment Retrieval with Audio-Enhanced MLLM',
    authors: 'An Yu, Weiheng Lu, Jian Li, Zenghui Zhang, Yixuan Shen, Francis X. F. Ye, Ming-Ching Chang',
    venue: 'IEEE TMM 2026',
    year: 2026,
    type: 'journal',
    topics: ['MLLM', 'Video'],
  },
  {
    title: 'Active Dataset Distillation via Dual-Space Informative Matching',
    authors: 'Ding Qi, Jian Li, Shuguang Dou, Jingyu Gao, Yabiao Wang, Bo Zhao, Jun Wan, Chunzhi Zhao',
    venue: 'IEEE TIP 2026',
    year: 2026,
    type: 'journal',
    topics: ['Dataset Distillation'],
  },
  {
    title: 'Towards Fine-Grained Vision-Language Alignment for Few-Shot Anomaly Detection',
    authors: 'Yilin Fan, Jun Liu, Xu Chen, Bin-Bin Gao, Jian Li, Yong Liu, Jinlong Peng, Chengjie Wang',
    venue: 'Pattern Recognition 2026',
    year: 2026,
    type: 'journal',
    topics: ['MLLM', 'Anomaly Detection'],
  },

  // ---------------- 2025 ----------------
  {
    title: 'Efficient Multimodal Large Language Models: A Survey',
    authors:
      'Yizhang Jin, Jian Li, Yexin Liu, Tianjun Gu, Kai Wu, Zhengkai Jiang, Muyang He, Bo Zhao, Xin Tan, Zhenye Gan, Yabiao Wang, Chengjie Wang, Lizhuang Ma',
    venue: 'Visual Intelligence 2025',
    year: 2025,
    type: 'journal',
    selected: true,
    note: 'Co-first author',
    topics: ['MLLM', 'Efficient', 'Survey'],
    citations: 254,
    links: [
      { label: 'arXiv', href: 'https://arxiv.org/abs/2405.10739' },
      { label: 'GitHub', href: 'https://github.com/swordlidev/Efficient-Multimodal-LLMs-Survey' },
    ],
  },
  {
    title: 'MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection',
    authors: 'Xi Jiang, Jian Li, Hanqiu Deng, Yong Liu, Bin-Bin Gao, Yifeng Zhou, Jialin Li, Chengjie Wang, Feng Zheng',
    venue: 'ICLR 2025',
    year: 2025,
    type: 'conference',
    selected: true,
    note: 'Corresponding author',
    topics: ['MLLM', 'Benchmark', 'Anomaly Detection'],
    citations: 148,
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2410.09453' }],
  },
  {
    title: 'A Survey on AI Search with Large Language Models',
    authors: 'Jian Li, Xiaoxi Li, Yan Zheng, Yizhang Jin, Shuo Wang, Jiafu Wu, Yabiao Wang, Chengjie Wang, Xiaotong Yuan',
    venue: 'Preprints 2025',
    year: 2025,
    type: 'preprint',
    note: 'First author',
    topics: ['Agent', 'RAG', 'Survey'],
    citations: 4,
    links: [{ label: 'Awesome-AI-Search', href: 'https://github.com/swordlidev/Awesome-AI-Search' }],
  },
  {
    title: 'Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly',
    authors:
      'Yexin Liu, Zhengyang Liang, Yueze Wang, Xianfeng Wu, Feilong Tang, Muyang He, Jian Li, Zheng Liu, Harry Yang, Ser-Nam Lim, et al.',
    venue: 'CVPR 2025',
    year: 2025,
    type: 'conference',
    topics: ['MLLM', 'Benchmark'],
    citations: 48,
  },
  {
    title: 'Towards Universal Dataset Distillation via Task-Driven Diffusion',
    authors: 'Ding Qi, Jian Li, Jingyu Gao, Shuguang Dou, Ying Tai, Jianlong Hu, Bo Zhao, Yabiao Wang, Chengjie Wang, Chunzhi Zhao',
    venue: 'CVPR 2025',
    year: 2025,
    type: 'conference',
    topics: ['Dataset Distillation'],
    citations: 7,
  },
  {
    title: 'From Zero to Detail: Deconstructing Ultra-High-Definition Image Restoration from Progressive Spectral Perspective',
    authors: 'Chen Zhao, Zhizhou Chen, Yunzhe Xu, Enxuan Gu, Jian Li, Zili Yi, Qian Wang, Jian Yang, Ying Tai',
    venue: 'CVPR 2025',
    year: 2025,
    type: 'conference',
    topics: ['Low-level Vision'],
    citations: 38,
  },
  {
    title: 'Multi-Task Visual Grounding with Coarse-to-Fine Consistency Constraints',
    authors: 'Ming Dai, Jian Li, Jiedong Zhuang, Xian Zhang, Wankou Yang',
    venue: 'AAAI 2025',
    year: 2025,
    type: 'conference',
    topics: ['MLLM', 'Visual Grounding'],
    citations: 43,
  },
  {
    title: 'MAP: Parameter-Efficient Tuning for Referring Expression Comprehension via Multi-Modal Adaptive Positional Encoding',
    authors: 'Ruilin Yao, Yu Rong, Teng Zou, Bo Zhang, Jian Li, Shenrong Xiong, Shuang Xiong',
    venue: 'ACM MM 2025',
    year: 2025,
    type: 'conference',
    topics: ['MLLM', 'Efficient'],
    citations: 5,
  },
  {
    title: 'BoxSeg: Quality-Aware and Peer-Assisted Learning for Box-Supervised Instance Segmentation',
    authors: 'Jun Lai, Wenjia Wu, Jian Zhan, Jian Li, Bin-Bin Gao, Jun Liu, Jiangning Zhang, Shanyang Guo',
    venue: 'ACM MM 2025',
    year: 2025,
    type: 'conference',
    topics: ['Segmentation'],
    citations: 2,
  },
  {
    title: 'Swin DiT: Diffusion Transformer using Pseudo Shifted Windows',
    authors: 'Jiafu Wu, Yabiao Wang, Jian Li, Jinlong Peng, Yun Cao, Chengjie Wang, Jiangning Zhang',
    venue: 'arXiv preprint',
    year: 2025,
    type: 'preprint',
    topics: ['Generation', 'Efficient'],
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2505.13219' }],
  },

  // ---------------- 2024 ----------------
  {
    title: 'A Survey on Benchmarks of Multimodal Large Language Models',
    authors:
      'Jian Li, Weiheng Lu, Hao Fei, Meng Luo, Ming Dai, Min Xia, Yizhang Jin, Zhenye Gan, Ding Qi, Chaoyou Fu, Ying Tai, et al.',
    venue: 'arXiv preprint',
    year: 2024,
    type: 'preprint',
    selected: true,
    note: 'First author',
    topics: ['MLLM', 'Benchmark', 'Survey'],
    citations: 235,
    links: [
      { label: 'arXiv', href: 'https://arxiv.org/abs/2408.08632' },
      { label: 'GitHub', href: 'https://github.com/swordlidev/Evaluation-Multimodal-LLMs-Survey' },
    ],
  },
  {
    title: 'WavChat: A Survey of Spoken Dialogue Models',
    authors:
      'Shengpeng Ji, Yifu Chen, Minghui Fang, Jialong Zuo, Jingyu Lu, Hanting Wang, Zhengdong Jiang, Long Zhou, Shujie Liu, ... Jian Li, et al.',
    venue: 'arXiv preprint',
    year: 2024,
    type: 'preprint',
    topics: ['Speech', 'Survey'],
    citations: 137,
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2411.13577' }],
  },
  {
    title: 'Fetch and Forge: Efficient Dataset Condensation for Object Detection',
    authors: 'Ding Qi, Jian Li, Jinlong Peng, Bo Zhao, Shuguang Dou, Jialin Li, Jiangning Zhang, Yabiao Wang, Chengjie Wang, Chunzhi Zhao',
    venue: 'NeurIPS 2024',
    year: 2024,
    type: 'conference',
    topics: ['Dataset Distillation', 'Detection'],
    citations: 18,
  },
  {
    title: 'Spider: Any-to-Many Multimodal LLM',
    authors: 'Jun Lai, Jiangning Zhang, Jun Liu, Jian Li, Xiangtai Lu, Shanyang Guo',
    venue: 'arXiv preprint',
    year: 2024,
    type: 'preprint',
    topics: ['MLLM'],
    citations: 10,
    links: [{ label: 'arXiv', href: 'https://arxiv.org/abs/2411.09439' }],
  },
  {
    title: 'LLaVA-VSD: Large Language-and-Vision Assistant for Visual Spatial Description',
    authors: 'Yizhang Jin, Jian Li, Jiangning Zhang, Jianlong Hu, Zhenye Gan, Xin Tan, Yong Liu, Yabiao Wang, Chengjie Wang, Lizhuang Ma',
    venue: 'ACM MM 2024',
    year: 2024,
    type: 'conference',
    topics: ['MLLM'],
    citations: 12,
  },

  // ---------------- 早期代表作（Selected Earlier Work） ----------------
  {
    title: 'Rethinking Mobile Block for Efficient Attention-based Models (EMO)',
    authors:
      'Jiangning Zhang, Xiangtai Li, Jian Li, Liang Liu, Zhucun Xue, Boshen Zhang, Zhengkai Jiang, Tianxin Huang, Yabiao Wang, Chengjie Wang',
    venue: 'ICCV 2023',
    year: 2023,
    type: 'conference',
    selected: true,
    topics: ['Efficient', 'Backbone'],
    citations: 473,
  },
  {
    title: 'DSFD: Dual Shot Face Detector',
    authors: 'Jian Li, Yabiao Wang, Changan Wang, Ying Tai, Jianjun Qian, Jian Yang, Chengjie Wang, Jilin Li, Feiyue Huang',
    venue: 'CVPR 2019',
    year: 2019,
    type: 'conference',
    selected: true,
    note: 'First author · Most cited',
    topics: ['Face Detection'],
    citations: 711,
    links: [{ label: 'GitHub', href: 'https://github.com/swordlidev/lightDSFD' }],
  },
  {
    title: 'Fast Learning of Temporal Action Proposal via Dense Boundary Generator (DBG)',
    authors: 'Chuming Lin, Jian Li, Yabiao Wang, Ying Tai, Donghao Luo, Zhipeng Cui, Chengjie Wang, Jilin Li, Feiyue Huang, Rongrong Ji',
    venue: 'AAAI 2020',
    year: 2020,
    type: 'conference',
    selected: true,
    topics: ['Temporal Action Detection'],
    citations: 274,
  },
  {
    title: 'ASFD: Automatic and Scalable Face Detector',
    authors: 'Jian Li, Bin Zhang, Yabiao Wang, Ying Tai, Zhenyu Zhang, Chengjie Wang, Jilin Li, Xiaoming Huang, Yili Xia',
    venue: 'ACM MM 2021',
    year: 2021,
    type: 'conference',
    note: 'First author',
    topics: ['Face Detection', 'NAS'],
    citations: 65,
  },
];
