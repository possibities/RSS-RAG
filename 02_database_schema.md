# 数据库 Schema 设计

> 所有表统一使用 PostgreSQL 15+ + pgvector 0.7+
> Embedding 维度统一锁定为 **1024**（bge-m3）

---

## 初始化扩展

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- gen_random_uuid() 是 PG13+ 内置函数，无需 uuid-ossp 扩展
```

---

## 分类表：`categories`

分类体系支持两级层次结构（parent_id IS NULL 为一级分类）。

```sql
CREATE TABLE categories (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    description TEXT NOT NULL,   -- 用于生成分类 embedding 的完整文本，应包含同义词与领域术语
                                 -- 例如："大语言模型、GPT、推理优化、instruction tuning、RLHF"
    parent_id   UUID REFERENCES categories(id) ON DELETE SET NULL,
    embedding   vector(1024),    -- 由 description 字段生成，离线批量计算
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (name, parent_id)  -- PG15+：确保 parent_id IS NULL 时 name 也唯一
);

-- 分类 embedding 索引（分类数量少，用 HNSW 更适合精确检索）
CREATE INDEX ON categories USING hnsw (embedding vector_cosine_ops);
```

### 初始化示例数据

```sql
INSERT INTO categories (name, description) VALUES
('AI',  '人工智能、机器学习、深度学习、神经网络、计算机视觉'),
('LLM', '大语言模型、GPT、推理优化、instruction tuning、RLHF、对话系统'),
('GPU', 'GPU硬件、CUDA、显存优化、推理加速、芯片'),
('MLOps', '模型部署、训练框架、分布式训练、监控、CI/CD'),
('数据工程', '数据管道、ETL、向量数据库、数据治理');
```

---

## 主表：`articles`

```sql
CREATE TABLE articles (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title        TEXT NOT NULL,
    content      TEXT,                  -- 原始正文（清洗后）
    summary      TEXT,                  -- vLLM 生成的摘要（200字以内）
    url          TEXT NOT NULL,          -- 唯一性由 url_hash 唯一索引保证，避免双重索引
    url_hash     CHAR(64) GENERATED ALWAYS AS (
                     encode(sha256(convert_to(url, 'UTF8')), 'hex')
                 ) STORED,             -- 用于快速精确去重，显式 UTF8 编码避免多字节 URL 不一致
    source       TEXT,                  -- RSS 来源名称
    publish_time TIMESTAMPTZ,
    embedding    vector(1024),          -- 由 title + summary 生成
    category_id  UUID REFERENCES categories(id) ON DELETE SET NULL,
    category_score FLOAT,              -- 分类时的相似度置信度分数
    tsv          TSVECTOR GENERATED ALWAYS AS (
                     to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(summary,''))
                 ) STORED,             -- 全文检索用，BM25 近似
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- 精确去重索引
CREATE UNIQUE INDEX ON articles(url_hash);

-- 向量相似度索引（HNSW 支持空表创建和增量插入，IVFFlat 要求建索引时表内已有数据）
CREATE INDEX ON articles
    USING hnsw (embedding vector_cosine_ops);

-- 时间过滤索引
CREATE INDEX ON articles(publish_time DESC);
CREATE INDEX ON articles(category_id);

-- 标题 trigram 索引（用于去重第二层）
CREATE INDEX ON articles USING gin(title gin_trgm_ops);

-- 全文检索索引（BM25 近似，tsv 列为 GENERATED STORED 列，已在建表时定义）
CREATE INDEX ON articles USING gin(tsv);
```

---

## Chunk 表：`chunks`

用于 RAG 检索的文章切片，每篇文章对应多个 chunk。

```sql
CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id  UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,           -- 在文章中的序号（0-based）
    chunk_text  TEXT NOT NULL,
    embedding   vector(1024),
    token_count INT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(article_id, chunk_index)
);

-- 向量检索索引（HNSW，支持空表创建；数据量超过 50 万后可评估切换 IVFFlat）
CREATE INDEX ON chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX ON chunks(article_id);
```

---

## 标签表：`tags` + `article_tags`

```sql
CREATE TABLE tags (
    id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE article_tags (
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    tag_id     UUID REFERENCES tags(id) ON DELETE CASCADE,
    score      FLOAT NOT NULL,    -- 分类时的相似度分数，用于排序
    PRIMARY KEY (article_id, tag_id)
);

CREATE INDEX ON article_tags(article_id);
CREATE INDEX ON article_tags(tag_id);
```

---

## 完整 ERD 关系说明

```
categories (id) ←── articles.category_id
     ↑
categories.parent_id（自引用，支持两级层次）

articles (id) ──→ chunks.article_id （1:N，CASCADE DELETE）
articles (id) ──→ article_tags.article_id （M:N）
tags (id)     ──→ article_tags.tag_id （M:N）
```

---

## 关键查询示例

### 按分类检索最新文章

```sql
SELECT a.id, a.title, a.summary, a.url, a.publish_time
FROM articles a
JOIN categories c ON a.category_id = c.id
WHERE c.name = 'LLM'
  AND a.publish_time > NOW() - INTERVAL '7 days'
ORDER BY a.publish_time DESC
LIMIT 20;
```

### 向量相似度检索文章

```sql
SELECT id, title, 1 - (embedding <=> $1) AS similarity
FROM articles
WHERE publish_time > NOW() - INTERVAL '90 days'
ORDER BY embedding <=> $1
LIMIT 10;
```

### 混合检索（向量 + 分类 + 时间衰减）

```sql
SELECT
    c.chunk_text,
    a.title,
    a.publish_time,
    a.url,
    (1 - (c.embedding <=> $1))
        * EXP(-EXTRACT(EPOCH FROM (NOW() - a.publish_time)) / 86400 / 30)
        AS final_score
FROM chunks c
JOIN articles a ON c.article_id = a.id
WHERE
    a.category_id = ANY($2::uuid[])
    AND a.publish_time > NOW() - INTERVAL '90 days'
ORDER BY final_score DESC
LIMIT 8;
```

> **时间衰减说明**：衰减常数 τ=30天，30天前文章得分衰减至约37%（e⁻¹），90天前约5%（e⁻³）。
> 可通过调整 `/ 30` 参数修改衰减常数（单位：天）。若需真正的半衰期（50%衰减），应使用 `POWER(0.5, days/T)`。

### 标题 trigram 去重查询

```sql
SELECT id, title, similarity(title, $1) AS sim
FROM articles
WHERE similarity(title, $1) > 0.85
  AND publish_time > NOW() - INTERVAL '7 days'
ORDER BY sim DESC
LIMIT 1;
```

---

## 索引参数调优建议

初始部署使用 HNSW 索引（支持空表创建、增量插入）。当数据量增长到一定规模后，
可评估切换为 IVFFlat 以获得更好的内存/性能比。IVFFlat 要求建索引时表内已有数据。

| 表 | 数据量级 | 建议索引 | ivfflat lists |
|----|---------|---------|---------------|
| articles | < 100K | HNSW（默认） | — |
| articles | 100K–1M | 可切换 IVFFlat | 300 |
| chunks | < 500K | HNSW（默认） | — |
| chunks | 500K–5M | 可切换 IVFFlat | 500 |

```sql
-- 数据量增长后，切换为 IVFFlat 示例（需先有数据）
DROP INDEX CONCURRENTLY IF EXISTS chunks_embedding_idx;
CREATE INDEX CONCURRENTLY ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 500);
```
