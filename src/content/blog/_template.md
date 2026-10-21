---
title: "在这里写标题"
description: "一句话摘要，会显示在文章列表、SEO 和 RSS 里"
pubDate: 2025-01-01
tags: ["标签1", "标签2"]
draft: true
badge: "模板"
---

> 使用说明（写完后可删除本段）：
> 1. 复制本文件，重命名为你的文章 slug，例如 `my-post.md`，网址即 `/blog/my-post/`。
> 2. 文件名用小写字母、数字、连字符。
> 3. 必填字段：title、description、pubDate；不需要 updatedDate 时删掉该行。
> 4. 写完把 `draft: true` 改成 `false` 才会发布到线上。

开篇用一两句话交代这篇文章要解决什么问题、读者能获得什么。

## 背景

说明动机、相关工作或需要的前置知识。

## 方法 / 正文

正文内容。常用语法示例如下。

行内公式：$\mathrm{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$。

块级公式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\!\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t\mid s_t)\, A_t\right]
$$

代码块（带语言名会自动高亮、支持一键复制）：

```python
def example(x: int) -> int:
    return x * 2
```

列表：

- 要点一
- 要点二
  - 子要点

表格：

| 方案 | 优点 | 缺点 |
| ---- | ---- | ---- |
| A    | 简单 | 慢   |
| B    | 快   | 复杂 |

引用与图片（图片放在 `public/images/` 下）：

> 这是一段引用。

![图片描述](/images/example.png)

## 实验 / 分析

贴结果、对比或分析。

## 小结

总结要点，可附上参考链接，例如相关[论文](/publications/)。
