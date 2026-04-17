# RAG 问答系统

基于私有 PostgreSQL 知识库的检索增强生成（RAG）问答系统。
核心隔离机制：Prompt 约束模型仅基于检索到的上下文回答，不使用通用知识。

---

## 系统流程

```
用户提问
    ↓
[Query 改写]（可选，提升稀疏词召回率）
    ↓
[并行检索]
    ├── 向量检索（pgvector cosine）
    └── BM25 全文检索（pg_trgm / tsvector）
    ↓
[RRF 融合排序]
    ↓
[空结果检测] → 置信度不足时提前返回兜底回复
    ↓
[拼装上下文 + 调用 vLLM]
    ↓
返回答案 + 引用来源
```

---

## 统一 import

```python
import asyncio
import json
import hashlib

import asyncpg
import httpx
import redis.asyncio as aioredis

from config import VLLM_QA_URL, VLLM_EMBED_URL, QA_MODEL, EMBED_MODEL
```

---

## 1. 混合检索

### 1.1 向量检索

```python
async def vector_search(
    query_embedding: list[float],
    db: asyncpg.Connection,
    category_ids: list[str] | None = None,
    days: int = 90,
    top_k: int = 10,
) -> list[dict]:
    """
    向量相似度检索，结合时间衰减评分。
    """
    base_sql = """
        SELECT
            c.id,
            c.chunk_text,
            c.article_id,
            a.title,
            a.url,
            a.publish_time,
            (1 - (c.embedding <=> $1))
                * EXP(-EXTRACT(EPOCH FROM (NOW() - a.publish_time)) / 86400 / 30)
                AS score
        FROM chunks c
        JOIN articles a ON c.article_id = a.id
        WHERE a.publish_time > NOW() - make_interval(days => $3)
    """
    params = [query_embedding, top_k, days]

    if category_ids:
        base_sql += " AND a.category_id = ANY($4::uuid[])"
        params.append(category_ids)

    base_sql += " ORDER BY score DESC LIMIT $2"

    rows = await db.fetch(base_sql, *params)
    return [dict(r) for r in rows]
```

### 1.2 全文检索（BM25 近似）

> `tsv` 列和 GIN 索引已在 `02_database_schema.md` 的建表 SQL 中定义。

```python
async def fulltext_search(
    query: str,
    db: asyncpg.Connection,
    top_k: int = 10,
) -> list[dict]:
    """
    全文检索（BM25 近似），用于补充向量检索在稀疏词上的不足。
    按 ts_rank 排序文章，再按 chunk_index 选取最靠前的 chunk 代表该文章。
    """
    rows = await db.fetch("""
        SELECT DISTINCT ON (c.article_id)
            c.id,
            c.chunk_text,
            c.article_id,
            a.title,
            a.url,
            a.publish_time,
            ts_rank(a.tsv, plainto_tsquery('simple', $1)) AS score
        FROM articles a
        JOIN chunks c ON c.article_id = a.id
        WHERE a.tsv @@ plainto_tsquery('simple', $1)
        ORDER BY c.article_id, c.chunk_index ASC
    """, query)
    # 按分数降序重排
    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    return [dict(r) for r in rows[:top_k]]
```

### 1.3 RRF 融合排序

```python
def rrf_merge(
    vector_results: list[dict],
    bm25_results:   list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion：融合两路检索结果，无需归一化分数。
    将 RRF 分数写入 rrf_score 字段，用于统一的空结果判断。
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for rank, item in enumerate(vector_results):
        cid = item["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        meta[cid] = item

    for rank, item in enumerate(bm25_results):
        cid = item["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in meta:
            meta[cid] = item

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    results = []
    for cid in ranked_ids:
        item = meta[cid].copy()
        item["rrf_score"] = scores[cid]
        results.append(item)
    return results
```

---

## 2. Query 改写（可选）

提升召回率，尤其适合口语化、模糊问题。

```python
REWRITE_PROMPT = """将以下用户问题改写为适合文档检索的搜索查询，输出3个不同角度的查询，每行一个，不要编号：

原问题：{question}

检索查询："""

async def rewrite_query(question: str) -> list[str]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(VLLM_QA_URL, json={
            "model": QA_MODEL,
            "messages": [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
            "max_tokens": 200,
            "temperature": 0.5,
        })
    text = resp.json()["choices"][0]["message"]["content"].strip()
    queries = [q.strip() for q in text.split("\n") if q.strip()]
    return queries[:3] if queries else [question]
```

---

## 3. Prompt 设计

```python
SYSTEM_PROMPT = """你是一个技术情报分析助手，专门基于提供的上下文回答问题。

规则：
1. 仅基于 <context> 标签内的信息作答
2. 如上下文不足以回答问题，明确说明"当前知识库中暂无足够信息回答此问题"，不要编造内容
3. 引用内容时注明来源文章标题
4. 回答使用中文，专业术语保留英文原文
5. 回答长度与问题复杂度匹配，不要过度展开"""

USER_PROMPT = """<context>
{context}
</context>

问题：{question}"""

def build_context(chunks: list[dict], max_chars: int = 4000) -> str:
    """将检索到的 chunk 拼装为上下文字符串。"""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        pt = chunk['publish_time']
        date_str = pt.strftime('%Y-%m-%d') if hasattr(pt, 'strftime') else str(pt)[:10]
        header = f"[来源 {i+1}] {chunk['title']} ({date_str})\n"
        body   = chunk["chunk_text"].strip()
        part   = header + body + "\n"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n---\n".join(parts)
```

---

## 4. 空结果检测与兜底

```python
# RRF 分数阈值：单路命中约 1/(60+1)=0.0164，双路命中约 0.033
# 低于 0.01 说明两路检索均未有效召回
MIN_RRF_THRESHOLD = 0.01

def is_sufficient(results: list[dict]) -> bool:
    if not results:
        return False
    top_rrf = results[0].get("rrf_score", 0)
    return top_rrf >= MIN_RRF_THRESHOLD

FALLBACK_REPLY = "当前知识库中暂无相关内容，建议扩充数据源或换一种提问方式。"
```

---

## 5. 完整 RAG 问答入口

```python
async def rag_query(
    question: str,
    db: asyncpg.Connection,
    category_ids: list[str] | None = None,
    use_rewrite: bool = True,
) -> dict:
    """
    完整 RAG 流程：检索 → 排序 → 生成。
    返回：{"answer": str, "sources": list[dict], "query_used": str}
    """
    # Step 1: Query 改写（可选）
    queries = await rewrite_query(question) if use_rewrite else [question]
    primary_query = queries[0]

    # Step 2: 生成 query embedding（带 Redis 缓存）
    query_embedding = await get_or_embed(primary_query)

    # Step 3: 并行执行两路检索
    vector_task  = vector_search(query_embedding, db, category_ids=category_ids)
    bm25_task    = fulltext_search(primary_query, db)
    vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)

    # Step 4: RRF 融合
    merged = rrf_merge(vector_results, bm25_results)

    # Step 5: 空结果检测
    if not is_sufficient(merged):
        return {"answer": FALLBACK_REPLY, "sources": [], "query_used": primary_query}

    top_chunks = merged[:6]

    # Step 6: 拼装 Prompt 并调用 vLLM
    context = build_context(top_chunks)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(VLLM_QA_URL, json={
            "model": QA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT.format(
                    context=context,
                    question=question,
                )},
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        })
    answer = resp.json()["choices"][0]["message"]["content"].strip()

    # Step 7: 整理引用来源
    sources = [
        {
            "title":        c["title"],
            "url":          c["url"],
            "publish_time": c["publish_time"].isoformat(),
        }
        for c in top_chunks
    ]

    return {
        "answer":     answer,
        "sources":    sources,
        "query_used": primary_query,
    }
```

---

## 6. FastAPI 接口

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RAG 知识库问答 API")

class QueryRequest(BaseModel):
    question:    str
    categories:  list[str] | None = None  # 分类名称（非 ID）
    use_rewrite: bool = True

class QueryResponse(BaseModel):
    answer:     str
    sources:    list[dict]
    query_used: str

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    # 将分类名称转换为 ID
    category_ids = None
    if req.categories:
        rows = await db.fetch(
            "SELECT id FROM categories WHERE name = ANY($1::text[])",
            req.categories
        )
        category_ids = [str(r["id"]) for r in rows]

    result = await rag_query(
        question=req.question,
        db=db,
        category_ids=category_ids,
        use_rewrite=req.use_rewrite,
    )
    return result

@app.get("/search")
async def search_endpoint(q: str, category: str | None = None, days: int = 90):
    """
    结构化检索接口（不经过 LLM，直接返回相关文章列表）。
    """
    embed = (await batch_embed([q]))[0]
    category_ids = None
    if category:
        row = await db.fetchrow("SELECT id FROM categories WHERE name = $1", category)
        category_ids = [str(row["id"])] if row else None

    results = await vector_search(embed, db, category_ids=category_ids, days=days, top_k=20)

    # 去重（同一文章多个 chunk 只保留最高分）
    seen = {}
    for r in results:
        aid = r["article_id"]
        if aid not in seen or r["score"] > seen[aid]["score"]:
            seen[aid] = r

    return [
        {"title": r["title"], "url": r["url"], "score": r["score"], "publish_time": r["publish_time"]}
        for r in sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:10]
    ]
```

---

## 7. Redis 缓存集成

```python
redis_client = aioredis.from_url("redis://localhost:6379")

async def cached_rag_query(question: str, **kwargs) -> dict:
    """
    带缓存的 RAG 问答。热门问题 15 分钟内复用结果。
    cache key 包含所有影响结果的参数，避免不同参数命中同一缓存。
    """
    key_data = json.dumps({
        "q": question,
        "cat": sorted(kwargs.get("category_ids") or []),
        "rw": kwargs.get("use_rewrite", True),
    }, sort_keys=True)
    cache_key = "rag:" + hashlib.md5(key_data.encode()).hexdigest()
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    result = await rag_query(question, **kwargs)
    await redis_client.setex(cache_key, 900, json.dumps(result, ensure_ascii=False, default=str))
    return result

async def get_or_embed(text: str) -> list[float]:
    """
    带缓存的 embedding 生成。同义问题复用 embedding，TTL 1小时。
    """
    cache_key = "emb:" + hashlib.md5(text.encode()).hexdigest()
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    embedding = (await batch_embed([text]))[0]
    await redis_client.setex(cache_key, 3600, json.dumps(embedding))
    return embedding
```
