# swordlidev.github.io

学术主页 + 技术博客，基于 [Astro](https://astro.build) 构建，托管于 GitHub Pages。

线上地址：<https://swordlidev.github.io/>

## 功能

- 首页（个人简介、Scholar 指标、研究方向、近期动态、代表论文、最新博客）
- 论文列表页（按年份归档，自动加粗本人署名，重点展示近 3 年工作）
- 博客归档、标签、关于、404
- Markdown 写作，Shiki 代码高亮（随明暗主题切换）、代码一键复制
- **数学公式**：`$...$` / `$$...$$` 在构建期用 KaTeX 渲染为 HTML（前端只加载 CSS，无运行时 JS）
- 文章目录（宽屏右侧显示，滚动高亮）、阅读时长、上一篇 / 下一篇
- 暗色模式（跟随系统，可手动切换并记忆）
- RSS（`/rss.xml`）、Sitemap、SEO / Open Graph 元信息
- 草稿机制：`draft: true` 的文章只在本地开发时可见
- 推送到 `main` 分支后由 GitHub Actions 自动构建部署

## 本地开发

需要 Node.js 22+。

```bash
npm install       # 安装依赖
npm run dev       # 本地开发 http://localhost:4321
npm run build     # 构建到 dist/
npm run preview   # 预览构建结果
```

## 写文章

```bash
npm run new -- my-post-slug "文章标题"
```

会在 `src/content/blog/` 下生成 `my-post-slug.md`，文章地址为 `/blog/my-post-slug/`。

Frontmatter 字段：

```yaml
---
title: "文章标题"          # 必填
description: "文章摘要"    # 必填，用于列表、SEO 和 RSS
pubDate: 2026-10-20        # 必填，发布日期
updatedDate: 2026-10-21    # 可选，更新日期
tags: ["Astro", "前端"]    # 可选
draft: false               # 可选，true 时不会发布到线上
---
```

图片放到 `public/images/`，在文章中用 `![描述](/images/xxx.png)` 引用。

数学公式示例：行内 `$E=mc^2$`，块级：

```md
$$
\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a\mid s)\, A\big]
$$
```

## 修改个人信息与论文

- 个人信息、Scholar 指标、研究方向、社交链接、近期动态：`src/consts.ts`
- 论文列表：`src/data/publications.ts`（`selected: true` 会出现在首页"代表性论文"）
- 关于页正文：`src/pages/about.astro`

## 目录结构

```
├── .github/workflows/deploy.yml   # GitHub Pages 自动部署
├── public/
│   └── katex/                     # 自托管的 KaTeX CSS 与字体
├── scripts/new-post.mjs           # 新建文章脚本
└── src/
    ├── components/                # 组件（含 PubItem 论文条目）
    ├── content/blog/              # 博客文章（Markdown）
    ├── data/publications.ts       # 论文数据
    ├── layouts/                   # 页面布局
    ├── pages/                     # 路由页面（含 publications.astro）
    ├── plugins/math-katex.mjs     # 构建期 KaTeX 渲染插件
    ├── styles/global.css          # 全局样式
    ├── utils/posts.ts             # 文章工具函数
    ├── consts.ts                  # 站点配置
    └── content.config.ts          # 内容集合 Schema
```

## 部署到 GitHub Pages

1. 在 GitHub 新建仓库，名称必须为 **`swordlidev.github.io`**（公开仓库）。
2. 推送代码：

   ```bash
   git remote add origin https://github.com/swordlidev/swordlidev.github.io.git
   git push -u origin main
   ```

3. 打开仓库 **Settings → Pages**，将 **Source** 设置为 **GitHub Actions**。
4. 等待 Actions 中的 `Deploy to GitHub Pages` 工作流完成，访问 <https://swordlidev.github.io/>。

之后每次推送到 `main` 分支都会自动重新部署。
