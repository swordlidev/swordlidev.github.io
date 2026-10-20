// 站点全局配置：修改这里即可更新网站上的个人信息

export const SITE = {
  title: 'Jian Li',
  description:
    'Jian Li (李剑) 的学术主页 —— 腾讯优图实验室研究员，研究方向包括大语言模型（LLM）、多模态大模型（MLLM）、智能体（Agent）与强化学习（RL）。',
  author: 'Jian Li',
  url: 'https://swordlidev.github.io',
  lang: 'zh-CN',
};

export const PROFILE = {
  name: 'Jian Li',
  nameCn: '李剑',
  avatar: 'https://github.com/swordlidev.png',
  title: 'Researcher · 研究员',
  affiliation: 'Youtu Lab, Tencent（腾讯优图实验室）',
  location: 'Shanghai, China',
  email: 'swordli@tencent.com',
  bio: '我是腾讯优图实验室的研究员，研究方向聚焦大语言模型（LLM）、多模态大模型（MLLM）、智能体（Agent）与强化学习（RL）。此前在计算机视觉（目标/人脸检测、时序动作检测、高效网络结构）方向有较多工作。这里记录我的研究、论文与一些技术思考。',
};

// Google Scholar 指标（随时间变化，仅供参考）
export const SCHOLAR = {
  citations: '2900+',
  hIndex: 20,
  i10Index: 28,
  profile: 'https://scholar.google.com/citations?user=ACb5C40AAAAJ',
};

export const RESEARCH_INTERESTS: string[] = [
  'Large Language Models (LLM)',
  'Multimodal LLM (MLLM)',
  'LLM Agents',
  'Reinforcement Learning (RL)',
  'Retrieval-Augmented Generation (RAG)',
  'Efficient / Multimodal Understanding',
];

export type SocialIcon = 'github' | 'scholar' | 'mail' | 'rss';

export const SOCIALS: { name: string; href: string; icon: SocialIcon }[] = [
  { name: 'Google Scholar', href: SCHOLAR.profile, icon: 'scholar' },
  { name: 'GitHub', href: 'https://github.com/swordlidev', icon: 'github' },
  { name: 'Email', href: 'mailto:swordli@tencent.com', icon: 'mail' },
  { name: 'RSS', href: '/rss.xml', icon: 'rss' },
];

// 论文 / 工作动态（倒序）。首页只展示最新若干条，关于页展示全部。
export const NEWS: { date: string; html: string }[] = [
  { date: '2026-06', html: '«TGPO»（高效策略优化）被 <strong>ICML 2026</strong> 接收。' },
  { date: '2026-05', html: '«RoleArena»（多智能体角色扮演环境）被 <strong>EMNLP 2026</strong> 接收。' },
  { date: '2026-05', html: '«AdaMARP» 被 <strong>ACL 2026 Findings</strong> 接收；«Disco-RAG» 被 <strong>ACL 2026</strong> 接收。' },
  { date: '2026-02', html: '推出搜索智能体工作 «SE-Search» 与 «Improving Search Agent with One Line of Code»。' },
  { date: '2025-10', html: '«VITA-Audio»（高效语音语言模型）被 <strong>NeurIPS 2025</strong> 接收。' },
  { date: '2025-06', html: '综述 «Efficient Multimodal Large Language Models» 发表于 <strong>Visual Intelligence</strong>。' },
  { date: '2025-02', html: '«Unveiling the Ignorance of MLLMs» 等多篇被 <strong>CVPR 2025</strong> 接收。' },
  { date: '2025-01', html: '«MMAD»（工业异常检测多模态基准）被 <strong>ICLR 2025</strong> 接收。' },
  { date: '2024-09', html: '«Fetch and Forge»（目标检测数据集压缩）被 <strong>NeurIPS 2024</strong> 接收。' },
  { date: '2024-08', html: '发布综述 «A Survey on Benchmarks of Multimodal Large Language Models»。' },
  { date: '2023-07', html: '高效骨干网络 «EMO»（Rethinking Mobile Block）被 <strong>ICCV 2023</strong> 接收。' },
  { date: '2020-01', html: '时序动作提名 «DBG» 被 <strong>AAAI 2020</strong> 接收。' },
  { date: '2019-03', html: '人脸检测 «DSFD» 被 <strong>CVPR 2019</strong> 接收（代表作，被引 700+）。' },
];

// 个人经历（倒序）
export const TIMELINE: { date: string; html: string }[] = [
  { date: '2019.03', html: '入职 <strong>腾讯优图实验室</strong>（Tencent Youtu Lab），任研究员至今。' },
  { date: '2017.09', html: '在 <strong>腾讯微信</strong> 实习。' },
  { date: '2017.09', html: '所在团队 <strong>abccba</strong> 获 2017 Didi-Udacity 自动驾驶挑战赛（Self-Driving Car Challenge）<strong>第一名</strong>。' },
  { date: '2016.09', html: '加入南京理工大学 <strong>PCALab</strong> 攻读硕士。' },
  { date: '2016.07', html: '在 <strong>ArcSoft（虹软）</strong> 实习。' },
  { date: '2015.12', html: '在 <strong>北京大学计算所</strong> 担任访问学生。' },
];
