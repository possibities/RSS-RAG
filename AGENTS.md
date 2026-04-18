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

## Current Status

截至 2026-04-18，仓库主体实现已完成，当前状态如下：

- `config.py` 已集中管理 vLLM 三个服务 URL、模型名、阈值、`DATABASE_DSN`、`REDIS_URL`、Celery 和邮件配置。
- `ingestion/` + `processing/` 已实现 RSS 抓取、三层去重、HTML 清洗、摘要、embedding、分类、chunk 切分与入库前处理。
- `storage/schema.sql` + `storage/db.py` 已实现 PostgreSQL schema、HNSW 索引、连接池初始化，以及 `pgvector.asyncpg.register_vector(conn)` 的 `init` 回调注册。
- `retrieval/` + `rag/` + `api/main.py` 已实现向量检索、全文检索、RRF 融合、RAG 问答、FastAPI 共享 `app`、lifespan 和 Redis 缓存。
- `analysis/` 已实现日报、趋势分析、热点提取、邮件推送、APScheduler 调度和 Celery 任务注册。

近期已补充的规范修正：

- `analysis/digest.py`、`analysis/hot_topics.py`、`analysis/trend.py` 中的聚类和质心计算已通过 `asyncio.to_thread()` 执行，避免在 `async` 流程中直接阻塞事件循环。
- `processing/classifier.py`、`analysis/trend.py`、`retrieval/searcher.py` 已补充 `embedding IS NOT NULL` 过滤，避免空向量参与相似度计算。
- `retrieval/searcher.py` 的时间衰减查询已改为参数化 SQL，并继续使用 `make_interval(days => $N)`。

测试基线：

- 当前 `pytest` 结果为 `15 passed`。
- 已覆盖去重、分类阈值、RRF 融合、空结果兜底、Redis 缓存键、邮件 HTML 转义。
- 已新增回归测试，锁定 `analysis/*` 中的 `to_thread` 约束，以及 `retrieval/searcher.py` 中的 `make_interval` / 参数化检索约束。

剩余风险：

- 尚未在真实 `PostgreSQL + Redis + vLLM` 环境中完成端到端联调。
- 当前验证范围以代码审计和单元测试为主；如果继续推进，应优先补集成测试和实际服务连通性验证。
