// 用法: npm run new -- <slug> [标题]
// 例如: npm run new -- my-first-post "我的第一篇文章"
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const [slug, ...titleParts] = process.argv.slice(2);

if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
  console.error(
    '用法: npm run new -- <slug> [标题]\n' +
      '  slug 只能包含小写字母、数字和连字符，例如:\n' +
      '  npm run new -- my-first-post "我的第一篇文章"',
  );
  process.exit(1);
}

const title = titleParts.join(' ').trim() || slug;
const dir = fileURLToPath(new URL('../src/content/blog/', import.meta.url));
const file = path.join(dir, `${slug}.md`);

if (existsSync(file)) {
  console.error(`文件已存在: ${path.relative(process.cwd(), file)}`);
  process.exit(1);
}

const d = new Date();
const today = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

const content = `---
title: ${JSON.stringify(title)}
description: ""
pubDate: ${today}
tags: []
draft: true
---

在这里开始写作...
`;

await mkdir(dir, { recursive: true });
await writeFile(file, content, 'utf8');
console.log(`已创建: ${path.relative(process.cwd(), file)}`);
console.log('提示: 写完后把 draft 改为 false 才会发布到线上。');
