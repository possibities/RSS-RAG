# 数据摄入管道

完整流程：RSS 抓取 → 三层去重 → 内容清洗 → 批量摘要 → 批量 Embedding → 向量分类 → Chunk 切分 → 写入数据库

---

## 完整数据流

```
[RSS 抓取调度]（Celery Beat 定时触发，每30分钟）
       ↓
[层级1 去重] URL Hash 精确匹配 → 命中则跳过
       ↓
[层级2 去重] 标题 trigram 相似度 > 0.85 → 命中则跳过
       ↓
[内容清洗] HTML 剥离 / 正文提取 / 编码修正
       ↓
[批量摘要] vLLM Summary Service，batch_size=32
       ↓
[批量 Embedding] vLLM Embedding Service，batch_size=64
       ↓
[层级3 去重] 文章 embedding 余弦相似度 > 0.95 → 跳过（语义重复）
       ↓
[向量分类] 相似度匹配 + 三级阈值 + 分层逻辑
       ↓
[Chunk 切分] 句子边界感知，300-500 tokens，overlap 50
       ↓
[Chunk Embedding] 批量计算
       ↓
[批量写入 PostgreSQL] articles + chunks + article_tags
       ↓
[异步触发] 推送任务 / 缓存失效 / 日报生成
```

---

## 1. RSS 抓取

```python
import asyncio
import hashlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx
from bs4 import BeautifulSoup

from config import VLLM_EMBED_URL, VLLM_SUMMARIZE_URL, EMBED_MODEL, SUMMARIZE_MODEL
```

---

## 1. RSS 抓取

```python
RSS_SOURCES = [
    {"name": "Hugging Face Blog", "url": "https://huggingface.co/blog/feed.xml"},
    {"name": "ArXiv AI",          "url": "https://arxiv.org/rss/cs.AI"},
    # 添加更多 RSS 源
]

async def fetch_rss(source: dict) -> list[dict]:
    """
    抓取单个 RSS 源，返回原始文章列表。
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(source["url"])
        resp.raise_for_status()
        feed = await asyncio.to_thread(feedparser.parse, resp.text)

    articles = []
    for entry in feed.entries:
        articles.append({
            "title":        entry.get("title", "").strip(),
            "url":          entry.get("link", ""),
            "content":      (entry.get("content") or [{}])[0].get("value", "")
                            or entry.get("summary", ""),
            "source":       source["name"],
            "publish_time": parse_time(entry.get("published")),
        })
    return articles

def parse_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        return None
```

---

## 2. 三层去重

```python
async def deduplicate(article: dict, db) -> bool:
    """
    返回 True 表示重复，应跳过。
    """
    # 层级 1：URL Hash 精确匹配（最快，O(1)）
    url_hash = hashlib.sha256(article["url"].encode("utf-8")).hexdigest()
    if await db.fetchval("SELECT 1 FROM articles WHERE url_hash = $1", url_hash):
        return True

    # 层级 2：标题 trigram 相似度（快，适合标题改写的重复）
    row = await db.fetchrow("""
        SELECT id FROM articles
        WHERE similarity(title, $1) > 0.85
          AND publish_time > NOW() - INTERVAL '7 days'
        LIMIT 1
    """, article["title"])
    if row:
        return True

    # 层级 3：语义重复（慢，仅在前两层通过后执行）
    # 注意：此时 article 尚未生成 embedding，需在 embedding 生成后单独检查
    # 见 deduplicate_by_embedding() 函数

    return False

async def deduplicate_by_embedding(embedding: list[float], db, threshold=0.95) -> bool:
    """
    语义去重：在 embedding 生成后调用。
    """
    row = await db.fetchrow("""
        SELECT id FROM articles
        WHERE 1 - (embedding <=> $1) > $2
          AND created_at > NOW() - INTERVAL '30 days'
        LIMIT 1
    """, embedding, threshold)
    return row is not None
```

---

## 3. 内容清洗

```python
def clean_content(raw_html: str) -> str:
    """
    剥离 HTML 标签，提取纯文本正文。
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # 移除无用标签
    for tag in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # 清理多余空行
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    # 截断超长正文（避免 LLM 上下文溢出）
    return text[:8000]
```

---

## 4. 批量摘要

```python
SUMMARIZE_PROMPT = """请对以下文章生成一段200字以内的中文摘要，重点提取：
1. 核心技术或方法
2. 主要结论或进展
3. 应用场景或影响

文章标题：{title}
文章内容：{content}

摘要："""

async def batch_summarize(articles: list[dict], batch_size=32) -> list[str]:
    """
    批量调用 vLLM 生成摘要。
    """
    results = []
    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            tasks = [
                client.post(VLLM_SUMMARIZE_URL, json={
                    "model": SUMMARIZE_MODEL,
                    "messages": [{"role": "user", "content": SUMMARIZE_PROMPT.format(
                        title=a["title"],
                        content=a["content"][:3000]
                    )}],
                    "max_tokens": 300,
                    "temperature": 0.3,
                })
                for a in batch
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for resp in responses:
                if isinstance(resp, Exception):
                    results.append("")  # 单条失败不影响整批
                    continue
                data = resp.json()
                summary = data["choices"][0]["message"]["content"].strip()
                results.append(summary)
    return results
```

---

## 5. 批量 Embedding 生成

```python
async def batch_embed(texts: list[str], batch_size=64) -> list[list[float]]:
    """
    批量生成 embedding，返回向量列表。
    texts 应为 "title\n\nsummary" 格式。
    """
    results = []
    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = await client.post(VLLM_EMBED_URL, json={
                "model": EMBED_MODEL,
                "input": batch,
            })
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            results.extend(embeddings)
    return results

def make_embed_text(title: str, summary: str | None) -> str:
    """文章 embedding 的输入文本格式。防御 summary 为 None 的情况。"""
    return f"{title}\n\n{summary}" if summary else title
```

---

## 6. 向量分类

```python
THRESHOLD_PRIMARY = 0.72   # 主分类最低置信度
THRESHOLD_TAG     = 0.60   # 多标签最低置信度
THRESHOLD_PRECISE = 0.80   # 触发二级分类的置信度

async def classify_article(
    article_embedding: list[float],
    db,
) -> dict:
    """
    两阶段向量分类：
    1. 一级分类（parent_id IS NULL）
    2. 若置信度 >= THRESHOLD_PRECISE，进一步做二级分类
    返回：{"category_id": uuid, "score": float, "tags": [{"tag": str, "score": float}]}
    """
    # 阶段一：一级分类
    rows = await db.fetch("""
        SELECT id, name, 1 - (embedding <=> $1) AS score
        FROM categories
        WHERE parent_id IS NULL
        ORDER BY score DESC
        LIMIT 5
    """, article_embedding)

    if not rows or rows[0]["score"] < THRESHOLD_PRIMARY:
        # 置信度不足，归入未分类队列
        return {
            "category_id": None,
            "score": rows[0]["score"] if rows else 0.0,
            "tags": [],
            "status": "uncategorized",
        }

    primary = rows[0]
    tags = [r for r in rows if r["score"] >= THRESHOLD_TAG]

    # 阶段二：二级分类（仅在高置信度时触发）
    if primary["score"] >= THRESHOLD_PRECISE:
        sub_rows = await db.fetch("""
            SELECT id, name, 1 - (embedding <=> $1) AS score
            FROM categories
            WHERE parent_id = $2
            ORDER BY score DESC
            LIMIT 3
        """, article_embedding, primary["id"])

        if sub_rows and sub_rows[0]["score"] >= THRESHOLD_PRIMARY:
            primary = sub_rows[0]

    return {
        "category_id": primary["id"],
        "score":        primary["score"],
        "tags":         [{"name": r["name"], "score": r["score"]} for r in tags],
        "status":       "classified",
    }
```

---

## 7. Chunk 切分

```python
def split_sentences(text: str) -> list[str]:
    """按中英文句子边界切分。"""
    return re.split(r'(?<=[。！？.!?\n])\s*', text)

def count_tokens(text: str) -> int:
    """简单估算 token 数（中文按字符数 / 1.5，英文按空格分词）。"""
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    other   = len(text) - chinese
    return int(chinese / 1.5 + other / 4)

def chunk_article(
    text: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[str]:
    """
    句子边界感知切分。
    overlap：保留上一个 chunk 末尾约 overlap_tokens 个 token 的句子，作为上下文延续。
    """
    sentences = [s for s in split_sentences(text) if s.strip()]
    chunks = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            # 保留末尾句子直到达到 overlap_tokens
            overlap: list[str] = []
            overlap_count = 0
            for s in reversed(current):
                t = count_tokens(s)
                if overlap_count + t > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_count += t
            current = overlap
            current_tokens = overlap_count

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks
```

---

## 8. 批量写入数据库

```python
import asyncpg
from pgvector.asyncpg import register_vector

async def create_db_pool(dsn: str) -> asyncpg.Pool:
    """创建连接池，并在每个连接上注册 pgvector 类型。"""
    async def init_connection(conn):
        await register_vector(conn)
    return await asyncpg.create_pool(dsn, init=init_connection)

async def ingest_article(article: dict, db: asyncpg.Connection):
    """
    原子性写入一篇文章及其 chunk。
    """
    async with db.transaction():
        # 写入主文章
        article_id = await db.fetchval("""
            INSERT INTO articles
                (title, content, summary, url, source, publish_time, embedding, category_id, category_score)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (url_hash) DO NOTHING
            RETURNING id
        """,
            article["title"],
            article["content"],
            article["summary"],
            article["url"],
            article["source"],
            article["publish_time"],
            article["embedding"],
            article["category_id"],
            article["category_score"],
        )

        if not article_id:
            return  # URL 重复，跳过

        # 批量写入 chunk
        await db.executemany("""
            INSERT INTO chunks (article_id, chunk_index, chunk_text, embedding, token_count)
            VALUES ($1, $2, $3, $4, $5)
        """, [
            (article_id, i, chunk["text"], chunk["embedding"], chunk["token_count"])
            for i, chunk in enumerate(article["chunks"])
        ])

        # 写入标签
        for tag_info in article.get("tags", []):
            tag_id = await db.fetchval("""
                INSERT INTO tags (name) VALUES ($1)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
            """, tag_info["name"])
            await db.execute("""
                INSERT INTO article_tags (article_id, tag_id, score)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
            """, article_id, tag_id, tag_info["score"])


async def run_pipeline(source: dict, db):
    """
    单个 RSS 源的完整处理流程。
    """
    raw_articles = await fetch_rss(source)

    new_articles = []
    for art in raw_articles:
        if await deduplicate(art, db):
            continue
        art["content"] = clean_content(art.get("content", ""))
        new_articles.append(art)

    if not new_articles:
        return

    # 批量摘要
    summaries = await batch_summarize(new_articles)
    for art, summary in zip(new_articles, summaries):
        art["summary"] = summary

    # 批量 embedding（文章级）
    embed_texts = [make_embed_text(a["title"], a["summary"]) for a in new_articles]
    embeddings  = await batch_embed(embed_texts)

    for art, art_emb in zip(new_articles, embeddings):
        # 语义去重（第三层）
        if await deduplicate_by_embedding(art_emb, db):
            continue
        art["embedding"] = art_emb

        # 分类
        cls = await classify_article(art_emb, db)
        art["category_id"]    = cls["category_id"]
        art["category_score"] = cls["score"]
        art["tags"]           = cls["tags"]

        # Chunk 切分 + embedding
        raw_chunks = chunk_article(art["content"])
        chunk_embeddings = await batch_embed(raw_chunks)
        art["chunks"] = [
            {"text": text, "embedding": chunk_emb, "token_count": count_tokens(text)}
            for text, chunk_emb in zip(raw_chunks, chunk_embeddings)
        ]

        await ingest_article(art, db)
```
