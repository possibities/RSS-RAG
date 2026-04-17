from __future__ import annotations

import hashlib

from config import SEMANTIC_DEDUP_THRESHOLD, TITLE_DEDUP_THRESHOLD


def compute_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


async def deduplicate(
    article: dict[str, object],
    db,
    title_days: int = 7,
    title_threshold: float = TITLE_DEDUP_THRESHOLD,
) -> bool:
    """
    Return True when the article should be skipped before expensive processing.
    """
    url_hash = compute_url_hash(str(article["url"]))
    if await db.fetchval("SELECT 1 FROM articles WHERE url_hash = $1", url_hash):
        return True

    row = await db.fetchrow(
        """
        SELECT id
        FROM articles
        WHERE similarity(title, $1) > $2
          AND publish_time > NOW() - make_interval(days => $3)
        LIMIT 1
        """,
        article["title"],
        title_threshold,
        title_days,
    )
    return row is not None


async def deduplicate_by_embedding(
    embedding: list[float],
    db,
    threshold: float = SEMANTIC_DEDUP_THRESHOLD,
    days: int = 30,
) -> bool:
    row = await db.fetchrow(
        """
        SELECT id
        FROM articles
        WHERE 1 - (embedding <=> $1) > $2
          AND created_at > NOW() - make_interval(days => $3)
        LIMIT 1
        """,
        embedding,
        threshold,
        days,
    )
    return row is not None
