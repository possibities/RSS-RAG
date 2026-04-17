from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg
from pgvector.asyncpg import register_vector

try:
    import redis.asyncio as aioredis
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    aioredis = None

from config import DATABASE_DSN, REDIS_URL, require_setting


PoolOrConn = asyncpg.Pool | asyncpg.Connection
RedisClient = Any


async def init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def create_db_pool(
    dsn: str = DATABASE_DSN,
    min_size: int = 1,
    max_size: int = 10,
) -> asyncpg.Pool:
    database_dsn = require_setting("DATABASE_DSN", dsn)
    return await asyncpg.create_pool(
        dsn=database_dsn,
        min_size=min_size,
        max_size=max_size,
        init=init_connection,
    )


async def close_db_pool(pool: asyncpg.Pool | None) -> None:
    if pool is not None:
        await pool.close()


def create_redis_client(redis_url: str = REDIS_URL) -> RedisClient:
    if not redis_url.strip():
        return None
    if aioredis is None:
        raise ModuleNotFoundError("redis is required to create a Redis client")
    return aioredis.from_url(redis_url, decode_responses=True)


async def close_redis_client(redis_client: RedisClient | None) -> None:
    if redis_client is not None:
        await redis_client.aclose()


@asynccontextmanager
async def get_connection(db: PoolOrConn) -> AsyncIterator[asyncpg.Connection]:
    if isinstance(db, asyncpg.Connection):
        yield db
        return

    async with db.acquire() as connection:
        yield connection


async def resolve_category_ids(db: PoolOrConn, names: list[str]) -> list[str]:
    if not names:
        return []
    async with get_connection(db) as connection:
        rows = await connection.fetch(
            "SELECT id FROM categories WHERE name = ANY($1::text[])",
            names,
        )
    return [str(row["id"]) for row in rows]


async def insert_article_graph(article: dict[str, object], db: PoolOrConn) -> str | None:
    async with get_connection(db) as connection:
        async with connection.transaction():
            article_id = await connection.fetchval(
                """
                INSERT INTO articles (
                    title,
                    content,
                    summary,
                    url,
                    source,
                    publish_time,
                    embedding,
                    category_id,
                    category_score
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (url_hash) DO NOTHING
                RETURNING id
                """,
                article["title"],
                article.get("content"),
                article.get("summary"),
                article["url"],
                article.get("source"),
                article.get("publish_time"),
                article.get("embedding"),
                article.get("category_id"),
                article.get("category_score"),
            )

            if not article_id:
                return None

            await connection.executemany(
                """
                INSERT INTO chunks (article_id, chunk_index, chunk_text, embedding, token_count)
                VALUES ($1, $2, $3, $4, $5)
                """,
                [
                    (
                        article_id,
                        index,
                        chunk["text"],
                        chunk["embedding"],
                        chunk["token_count"],
                    )
                    for index, chunk in enumerate(article.get("chunks", []))
                ],
            )

            for tag_info in article.get("tags", []):
                tag_id = await connection.fetchval(
                    """
                    INSERT INTO tags (name)
                    VALUES ($1)
                    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                    RETURNING id
                    """,
                    tag_info["name"],
                )
                await connection.execute(
                    """
                    INSERT INTO article_tags (article_id, tag_id, score)
                    VALUES ($1, $2, $3)
                    ON CONFLICT DO NOTHING
                    """,
                    article_id,
                    tag_id,
                    tag_info["score"],
                )

    return str(article_id) if article_id is not None else None
