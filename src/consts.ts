// Global site config — edit here to update the personal info shown across the site.

export const SITE = {
  title: 'Jian Li',
  description:
    "Academic homepage of Jian Li — Researcher at Youtu Lab, Tencent. Research on large language models (LLM), multimodal LLMs (MLLM), agents, and reinforcement learning (RL).",
  author: 'Jian Li',
  url: 'https://swordlidev.github.io',
  lang: 'en',
};

export const PROFILE = {
  name: 'Jian Li',
  nameCn: '李剑',
  avatar: 'https://github.com/swordlidev.png',
  title: 'Researcher',
  affiliation: 'Youtu Lab, Tencent',
  location: 'Shanghai, China',
  email: 'swordli@tencent.com',
  bio: "I am a researcher at Youtu Lab, Tencent. My research focuses on large language models (LLMs), multimodal LLMs (MLLMs), agents, and reinforcement learning (RL). Earlier, I worked extensively on computer vision — face/object detection, temporal action detection, and efficient network architectures. This site collects my research, publications, and technical notes.",
};

// Google Scholar metrics (change over time; for reference only)
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

// Publication / work news (reverse chronological). Homepage shows the latest few; About shows all.
export const NEWS: { date: string; html: string }[] = [
  { date: '2026-06', html: '<strong>TGPO</strong> (efficient policy optimization) accepted to <strong>ICML 2026</strong>.' },
  { date: '2026-05', html: '<strong>RoleArena</strong> (multi-agent role-playing environment) accepted to <strong>EMNLP 2026</strong>.' },
  { date: '2026-05', html: '<strong>AdaMARP</strong> accepted to <strong>ACL 2026 Findings</strong>; <strong>Disco-RAG</strong> accepted to <strong>ACL 2026</strong>.' },
  { date: '2026-02', html: 'Released search-agent works <strong>SE-Search</strong> and <em>Improving Search Agent with One Line of Code</em>.' },
  { date: '2025-10', html: '<strong>VITA-Audio</strong> (efficient speech-language model) accepted to <strong>NeurIPS 2025</strong>.' },
  { date: '2025-06', html: 'Survey <em>Efficient Multimodal Large Language Models</em> published in <strong>Visual Intelligence</strong>.' },
  { date: '2025-02', html: 'Multiple papers incl. <em>Unveiling the Ignorance of MLLMs</em> accepted to <strong>CVPR 2025</strong>.' },
  { date: '2025-01', html: '<strong>MMAD</strong> (industrial anomaly detection benchmark for MLLMs) accepted to <strong>ICLR 2025</strong>.' },
  { date: '2024-09', html: '<strong>Fetch and Forge</strong> (dataset condensation for detection) accepted to <strong>NeurIPS 2024</strong>.' },
  { date: '2024-08', html: 'Released survey <em>A Survey on Benchmarks of Multimodal Large Language Models</em>.' },
  { date: '2023-07', html: 'Efficient backbone <strong>EMO</strong> (Rethinking Mobile Block) accepted to <strong>ICCV 2023</strong>.' },
  { date: '2020-01', html: 'Temporal action proposal <strong>DBG</strong> accepted to <strong>AAAI 2020</strong>.' },
  { date: '2019-03', html: 'Face detector <strong>DSFD</strong> accepted to <strong>CVPR 2019</strong> (700+ citations).' },
];

// Experience timeline (reverse chronological)
export const TIMELINE: { date: string; html: string }[] = [
  { date: '2019.03', html: 'Joined <strong>Youtu Lab, Tencent</strong> as a Researcher.' },
  { date: '2017.09', html: 'Research intern at <strong>Tencent WeChat</strong>.' },
  { date: '2017.09', html: 'Our team <strong>abccba</strong> won <strong>1st place</strong> in the 2017 Didi-Udacity Self-Driving Car Challenge.' },
  { date: '2016.09', html: 'Started M.S. studies at <strong>PCALab, NJUST</strong> (Nanjing University of Science and Technology).' },
  { date: '2016.07', html: 'Research intern at <strong>ArcSoft</strong>.' },
  { date: '2015.12', html: 'Visiting student at the <strong>Institute of Computing, Peking University</strong>.' },
];
