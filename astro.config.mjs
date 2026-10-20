// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import { satteri } from '@astrojs/markdown-satteri';
import { mathKatex } from './src/plugins/math-katex.mjs';

export default defineConfig({
  site: 'https://swordlidev.github.io',
  integrations: [sitemap()],
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
    },
    // 开启数学公式解析，并在构建期用 KaTeX 渲染成 HTML（前端只需加载 CSS）
    processor: satteri({
      features: { math: true },
      mdastPlugins: [mathKatex],
    }),
  },
});
