# Repository Guidelines

## Source of Truth

实现本仓库前，必须先读 `CLAUDE.md`。它定义了架构、目录结构、阈值常量和编码约束，优先级高于其他文档。推荐阅读顺序固定为：

1. `CLAUDE.md`
2. `01_architecture_overview.md`
3. `03_data_pipeline.md`
4. `02_database_schema.md`
5. `04_rag_system.md`
6. `05_analysis_system.md`

其中 `01` 仅作架构参考；`03 → 02 → 04 → 05` 是实际实现顺序。

## Required Build Order

按以下顺序交付代码，不要跳步：

1. `config.py`：集中定义 vLLM 三个服务 URL、模型名、阈值、`DATABASE_DSN`、`REDIS_URL`、邮件配置。
2. `ingestion/` + `processing/`：RSS 抓取、三层去重、HTML 清洗、摘要、embedding、分类、chunk 切分。
3. `storage/schema.sql` + `storage/db.py`：PostgreSQL schema 与连接池初始化。
4. `retrieval/` + `rag/` + `api/main.py`：混合检索、RAG 问答、FastAPI、Redis 缓存。
5. `analysis/`：日报、趋势、热点、邮件推送、定时调度。

文档中的代码是片段式的；你需要补全模块 import、共享 `FastAPI` app、lifespan、Celery task 注册和调度胶水代码。

## Project Layout

目标目录结构以 `CLAUDE.md` 为准：

- `config.py`
- `ingestion/`
- `processing/`
- `storage/`
- `retrieval/`
- `rag/`
- `analysis/`
- `api/`
- `tests/`

测试目录应镜像源码结构，例如 `tests/retrieval/test_searcher.py`。

## Non-Negotiable Rules

- 所有 embedding 统一为 `vector(1024)`，模型固定 `bge-m3`。
- vLLM URL 和模型名只能从 `config.py` 导入，禁止硬编码。
- `asyncpg` 连接池必须在 `init` 回调中调用 `pgvector.asyncpg.register_vector(conn)`。
- 向量索引初始实现只用 HNSW，不用 IVFFlat。
- SQL 时间区间参数统一使用 `make_interval(days => $N)`。
- 邮件 HTML 中所有 RSS 来源内容必须先做 `html.escape()`。
- `async` 函数内不得直接执行阻塞式同步调用；`feedparser`、`smtplib` 等必须走 `asyncio.to_thread()`。
- 所有 import 放在模块顶部。

## Development & Testing

常用命令：

- `psql "$DATABASE_DSN" -f storage/schema.sql`
- `uvicorn api.main:app --reload`
- `pytest`

测试文件命名使用 `test_<module>.py`。重点覆盖：去重、分类阈值、RRF 融合、空结果兜底、Redis 缓存键、邮件 HTML 转义。

## Change Guidance

提交变更时，说明对应设计文档章节；若实现与 `CLAUDE.md` 或 `01-05` 不一致，必须在变更说明中显式解释原因。默认保持小步提交，提交信息用祈使句或 Conventional Commits，例如 `feat: implement vector search pipeline`。
