import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'blog'>;

export interface TagInfo {
  slug: string;
  name: string;
  posts: Post[];
}

/** 获取已发布文章（开发模式下包含草稿），按发布时间倒序 */
export async function getPublishedPosts(): Promise<Post[]> {
  const posts = await getCollection('blog', ({ data }) => import.meta.env.DEV || !data.draft);
  return posts.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

export function tagSlug(tag: string): string {
  return tag.trim().toLowerCase().replace(/[\s/\\#?%]+/g, '-');
}

export function groupByTag(posts: Post[]): TagInfo[] {
  const map = new Map<string, TagInfo>();
  for (const post of posts) {
    for (const name of new Set(post.data.tags)) {
      const slug = tagSlug(name);
      if (!slug) continue;
      const info = map.get(slug) ?? { slug, name, posts: [] };
      info.posts.push(post);
      map.set(slug, info);
    }
  }
  return [...map.values()].sort(
    (a, b) => b.posts.length - a.posts.length || a.name.localeCompare(b.name, 'zh-CN'),
  );
}

export function formatDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

const CJK_RE = /[\u3400-\u9fff\uf900-\ufaff]/g;

/** 估算阅读时长（分钟）：中文约 400 字/分钟，英文约 200 词/分钟 */
export function readingTime(text: string = ''): number {
  const cjk = text.match(CJK_RE)?.length ?? 0;
  const words = text.replace(CJK_RE, ' ').split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.ceil(cjk / 400 + words / 200));
}
