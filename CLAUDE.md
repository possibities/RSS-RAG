# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RSS→RAG private knowledge base system. Three capabilities: searchable article database, RAG Q&A over private data, automated trend analysis and daily digests. Currently in design phase (5 specification docs); implementation pending.

## Implementation Order

**03 (data pipeline) → 02 (database schema) → 04 (RAG system) → 05 (analysis system)**, with 01 as architecture reference.

## Architecture

```
RSS Fetch → 3-Layer Dedup → HTML Clean → vLLM Summary → vLLM Embedding → Vector Classification → Chunk Split → PostgreSQL+pgvector
                                                                                                                        ↓
                                                                                              Hybrid Search (vector + BM25 + time decay) → RAG Q&A (vLLM)
                                                                                                                        ↓
                                                                                              Daily Digest / Weekly Trend / Hot Topics → Email / API
```

- **Single storage layer**: PostgreSQL 15+ with pgvector (no separate vector DB)
- **Single AI inference layer**: vLLM serving all models via OpenAI-compatible API
- **Async throughout**: asyncpg + httpx + asyncio

## Critical Constraints

- **Embedding dimension**: ALL vectors are `vector(1024)` using `bge-m3`. Dimension mismatch breaks indexes.
- **vLLM endpoints**: Must be centralized in `config.py`, never hardcoded. Three services:
  - Embedding: bge-m3
  - Summary: Qwen2.5-7B-Instruct
  - QA: Qwen2.5-14B-Instruct
- **pgvector + asyncpg**: Must call `pgvector.asyncpg.register_vector(conn)` on every connection before passing vector parameters.
- **Index strategy**: Use HNSW (not IVFFlat) for initial deployment — IVFFlat errors on empty tables.

## Key Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Primary classification | 0.72 | Min cosine similarity for main category |
| Multi-tag | 0.60 | Min cosine similarity for tag assignment |
| Sub-category trigger | 0.80 | Triggers second-level classification |
| RAG min RRF score | 0.01 | Below this → fallback reply (RRF scale, not cosine) |
| Semantic dedup | 0.95 | Cosine similarity → treat as duplicate |
| Title dedup | 0.85 | Trigram similarity → treat as duplicate |
| Time decay τ | 30 days | Exponential decay constant (37% at 30d, not half-life) |

## Target Directory Structure

```
rss-rag/
├── config.py               # All vLLM URLs, model names, thresholds, DB DSN
├── ingestion/               # RSS fetch, dedup, HTML cleaning
├── processing/              # Summarize, embed, classify, chunk
├── storage/                 # schema.sql, db.py (pool init, vector codec)
├── retrieval/               # Vector search, BM25, RRF fusion
├── rag/                     # RAG pipeline, prompts
├── analysis/                # Digest, trend, hot topics
└── api/                     # FastAPI endpoints
```

## Database Tables

Five tables: `categories` (2-level hierarchy with embeddings), `articles` (main, with url_hash dedup + embedding), `chunks` (article slices for RAG retrieval), `tags`, `article_tags` (M:N with scores). Full schema in `02_database_schema.md`.

## Coding Constraints (Enforced in Design Docs)

The following rules are baked into the design docs (02–05) and must be followed during implementation:

### Database (02)
- `url_hash` uses `convert_to(url, 'UTF8')`, not `url::bytea`
- `UNIQUE NULLS NOT DISTINCT (name, parent_id)` on categories — standard `UNIQUE` doesn't deduplicate NULLs
- `articles.tsv` is a `GENERATED ALWAYS AS ... STORED` column defined in the CREATE TABLE, with a GIN index
- All vector indexes use **HNSW** (not IVFFlat) — IVFFlat requires existing data at index creation time
- No `uuid-ossp` extension needed — `gen_random_uuid()` is PG13+ built-in
- Time decay formula is `EXP(-days/30)` (exponential decay constant τ=30, **not** half-life)

### Data Pipeline (03)
- All imports at module top level — no `import` inside function bodies
- All vLLM URLs/model names imported from `config.py`, never hardcoded
- `feedparser.parse()` wrapped in `asyncio.to_thread()` to avoid blocking the event loop
- `resp.raise_for_status()` after every HTTP fetch
- RSS entry content access: `(entry.get("content") or [{}])[0]` — guards against empty list
- `make_embed_text()` handles `summary=None` gracefully (falls back to title only)
- `batch_summarize` uses `asyncio.gather(*tasks, return_exceptions=True)` — single failures don't crash the batch
- Variable naming: outer loop uses `art_emb`, inner list comprehension uses `chunk_emb` — no shadowing
- DB connection pool created with `pgvector.asyncpg.register_vector` in the `init` callback
- URL hash computed with explicit `encode("utf-8")`

### RAG System (04)
- SQL interval parameters use `make_interval(days => $N)`, not string concatenation
- RRF merge writes `rrf_score` into each result dict — `is_sufficient()` checks `rrf_score` (threshold 0.01), not the heterogeneous source-side `score`
- `build_context` handles `publish_time` as both datetime and string (cache deserialization safety)
- `rag_query` Step 2 calls `get_or_embed()` (Redis-cached), not raw `batch_embed()`
- `cached_rag_query` cache key is a hash of `{question, category_ids, use_rewrite}`, not just the question
- Fulltext search uses `DISTINCT ON (article_id)` + `chunk_index ASC` to pick one representative chunk per article

### Analysis System (05)
- All RSS-sourced content in email HTML must be passed through `html.escape()` (titles, URLs, digests, category names)
- `smtplib.SMTP` operations run via `asyncio.to_thread()` — never block the async event loop
- SQL interval parameters use `make_interval(days => $N)`
- Analysis API routes mount on the shared `app` instance from 04, no duplicate `FastAPI()` creation
