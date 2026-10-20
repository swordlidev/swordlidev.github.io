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

// 近期动态（倒序）
export const NEWS: { date: string; html: string }[] = [
  { date: '2026-06', html: '«TGPO»（高效策略优化）被 <strong>ICML 2026</strong> 接收。' },
  { date: '2026-05', html: '«RoleArena»（多智能体角色扮演环境）被 <strong>EMNLP 2026</strong> 接收。' },
  { date: '2026-05', html: '«AdaMARP» 被 <strong>ACL 2026 Findings</strong> 接收；«Disco-RAG» 被 <strong>ACL 2026</strong> 接收。' },
  { date: '2026-02', html: '推出搜索智能体工作 «SE-Search» 与 «Improving Search Agent with One Line of Code»。' },
  { date: '2025-01', html: '«MMAD»（工业异常检测多模态基准）被 <strong>ICLR 2025</strong> 接收。' },
];
