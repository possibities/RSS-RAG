# RSS → RAG 知识库系统：架构总览

## 项目目标

构建一套基于 RSS 数据源的私有化知识库与 AI 问答系统，最终交付三项核心能力：

1. **知识库**：可检索、可分类、可结构化的文章数据库
2. **AI 问答系统**：基于私有数据的 RAG 问答，非通用 GPT
3. **信息分析系统**：自动生成技术趋势分析与热点报告

---

## 系统架构

```
RSS抓取 → 去重/清洗 → 摘要(vLLM) → Embedding → 向量分类 + Chunk切分
                                                        ↓
                                              PostgreSQL + pgvector
                                                        ↓
                                          混合检索（向量 + SQL过滤 + 时间衰减）
                                                        ↓
                                              RAG问答（vLLM）
                                                        ↓
                                          邮件 / API / UI / 趋势报告
```

**核心原则**：
- PostgreSQL 作为统一存储层（关系型 + 向量），无需引入独立向量数据库
- vLLM 作为统一 AI 推理层（LLM + Embedding），通过标准 OpenAI 兼容 HTTP API 解耦

---

## 技术选型

### 数据库

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| PostgreSQL | ≥ 15 | 主数据库 |
| pgvector | ≥ 0.7 | 向量存储与相似度检索 |
| pg_trgm | 内置扩展 | 标题模糊去重 |

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### AI 推理层（vLLM）

按职责拆分三个独立服务，通过 Nginx Gateway 路由：

```
vLLM Gateway（Nginx 路由层）
  ├── /v1/embeddings                  → Embedding Service（bge-m3，独立 GPU）
  ├── /v1/chat/completions/summarize  → Summary Service（Qwen2.5-7B，批处理模式）
  └── /v1/chat/completions/qa         → QA Service（Qwen2.5-14B，在线推理模式）
```

### Embedding 模型选型

> ⚠️ 必须在部署前锁定维度，向量维度不一致将导致索引失效。

| 模型 | 维度 | 推荐场景 |
|------|------|----------|
| `bge-m3` | 1024 | 中英文混合内容（**首选**） |
| `bge-large-zh` | 1024 | 纯中文内容 |
| `e5-mistral-7b` | 4096 | 英文高精度，资源开销大 |

**本方案统一使用 `bge-m3`，维度 `1024`。**

### 任务调度

| 组件 | 用途 |
|------|------|
| Celery + Redis | 异步任务队列（RSS 抓取、批量 embedding、日报生成） |
| APScheduler | 定时触发调度 |

### 缓存层

| 缓存项 | 存储 | TTL |
|--------|------|-----|
| Query Embedding | Redis | 1 小时 |
| 热门问题完整 RAG 结果 | Redis | 15 分钟 |
| 分类 Embedding | 内存（启动时全量加载） | 进程生命周期 |

---

## 目录结构建议

```
rss-rag/
├── ingestion/
│   ├── fetcher.py          # RSS 抓取
│   ├── deduplicator.py     # 三层去重
│   └── cleaner.py          # HTML 清洗 / 正文提取
├── processing/
│   ├── summarizer.py       # 批量摘要
│   ├── embedder.py         # 批量 Embedding
│   ├── classifier.py       # 向量分类
│   └── chunker.py          # Chunk 切分
├── storage/
│   ├── schema.sql          # 完整数据库 Schema（见 02_database_schema.md）
│   └── db.py               # 数据库操作封装
├── retrieval/
│   ├── searcher.py         # 混合检索
│   └── reranker.py         # RRF 融合排序
├── rag/
│   ├── pipeline.py         # RAG 主流程
│   └── prompt.py           # Prompt 模板
├── analysis/
│   ├── digest.py           # 日报生成
│   └── trend.py            # 趋势分析
├── api/
│   └── main.py             # FastAPI 对外接口
└── config.py               # 统一配置（模型名、阈值、TTL 等）
```

---

## 监控指标

| 指标 | 告警阈值 | 说明 |
|------|---------|------|
| 分类置信度均值 | < 0.65 | 分类 embedding 需重新生成 |
| RAG 检索延迟 P99 | > 2s | 索引参数需调优 |
| 未分类文章比例 | > 15% | 分类体系覆盖不足 |
| chunk embedding 入库失败率 | > 1% | vLLM embedding 服务异常 |

---

## 文件索引

| 文件 | 内容 |
|------|------|
| `01_architecture_overview.md` | 本文件：架构总览、技术选型 |
| `02_database_schema.md` | 完整数据库 Schema（SQL） |
| `03_data_pipeline.md` | 数据摄入全流程（抓取→去重→摘要→embedding→分类→入库） |
| `04_rag_system.md` | RAG 问答系统（检索→生成→回退策略） |
| `05_analysis_system.md` | 信息分析系统（日报→趋势分析→热点提取） |
