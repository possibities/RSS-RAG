from __future__ import annotations

from config import DEFAULT_TOP_K, DEFAULT_VECTOR_SEARCH_DAYS, TIME_DECAY_DAYS


async def vector_search(
    query_embedding: list[float],
    db,
    category_ids: list[str] | None = None,
    days: int = DEFAULT_VECTOR_SEARCH_DAYS,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, object]]:
    if category_ids:
        rows = await db.fetch(
            f"""
            SELECT
                c.id,
                c.chunk_text,
                c.article_id,
                a.title,
                a.url,
                a.publish_time,
                (1 - (c.embedding <=> $1))
                    * EXP(-EXTRACT(EPOCH FROM (NOW() - a.publish_time)) / 86400 / {TIME_DECAY_DAYS})
                    AS score
            FROM chunks c
            JOIN articles a ON c.article_id = a.id
            WHERE a.publish_time > NOW() - make_interval(days => $3)
              AND a.category_id = ANY($4::uuid[])
            ORDER BY score DESC
            LIMIT $2
            """,
            query_embedding,
            top_k,
            days,
            category_ids,
        )
    else:
        rows = await db.fetch(
            f"""
            SELECT
                c.id,
                c.chunk_text,
                c.article_id,
                a.title,
                a.url,
                a.publish_time,
                (1 - (c.embedding <=> $1))
                    * EXP(-EXTRACT(EPOCH FROM (NOW() - a.publish_time)) / 86400 / {TIME_DECAY_DAYS})
                    AS score
            FROM chunks c
            JOIN articles a ON c.article_id = a.id
            WHERE a.publish_time > NOW() - make_interval(days => $3)
            ORDER BY score DESC
            LIMIT $2
            """,
            query_embedding,
            top_k,
            days,
        )
    return [dict(row) for row in rows]


async def fulltext_search(
    query: str,
    db,
    category_ids: list[str] | None = None,
    days: int = DEFAULT_VECTOR_SEARCH_DAYS,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, object]]:
    if category_ids:
        rows = await db.fetch(
            """
            SELECT *
            FROM (
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
                  AND a.publish_time > NOW() - make_interval(days => $2)
                  AND a.category_id = ANY($4::uuid[])
                ORDER BY c.article_id, c.chunk_index ASC
            ) ranked
            ORDER BY score DESC
            LIMIT $3
            """,
            query,
            days,
            top_k,
            category_ids,
        )
    else:
        rows = await db.fetch(
            """
            SELECT *
            FROM (
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
                  AND a.publish_time > NOW() - make_interval(days => $2)
                ORDER BY c.article_id, c.chunk_index ASC
            ) ranked
            ORDER BY score DESC
            LIMIT $3
            """,
            query,
            days,
            top_k,
        )

    return [dict(row) for row in rows]


def collapse_results_by_article(results: list[dict[str, object]], top_k: int = DEFAULT_TOP_K) -> list[dict[str, object]]:
    deduped: dict[object, dict[str, object]] = {}
    for result in results:
        article_id = result["article_id"]
        if article_id not in deduped or result["score"] > deduped[article_id]["score"]:
            deduped[article_id] = result
    return sorted(deduped.values(), key=lambda item: item["score"], reverse=True)[:top_k]
