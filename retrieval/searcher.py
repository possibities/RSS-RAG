from __future__ import annotations

from config import DEFAULT_TOP_K, DEFAULT_VECTOR_SEARCH_DAYS, TIME_DECAY_DAYS


async def vector_search(
    query_embedding: list[float],
    db,
    category_ids: list[str] | None = None,
    days: int = DEFAULT_VECTOR_SEARCH_DAYS,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, object]]:
    base_sql = """
        SELECT
            c.id,
            c.chunk_text,
            c.article_id,
            a.title,
            a.url,
            a.publish_time,
            (1 - (c.embedding <=> $1))
                * EXP(-EXTRACT(EPOCH FROM (NOW() - a.publish_time)) / 86400 / CAST($4 AS double precision))
                AS score
        FROM chunks c
        JOIN articles a ON c.article_id = a.id
        WHERE c.embedding IS NOT NULL
          AND a.publish_time > NOW() - make_interval(days => $3)
    """
    params: list[object] = [query_embedding, top_k, days, TIME_DECAY_DAYS]

    if category_ids:
        base_sql += " AND a.category_id = ANY($5::uuid[])"
        params.append(category_ids)

    base_sql += " ORDER BY score DESC LIMIT $2"
    rows = await db.fetch(base_sql, *params)
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
