from __future__ import annotations

from config import (
    MULTI_TAG_THRESHOLD,
    PRIMARY_CLASSIFICATION_THRESHOLD,
    SUBCATEGORY_TRIGGER_THRESHOLD,
)


async def classify_article(article_embedding: list[float], db) -> dict[str, object]:
    rows = await db.fetch(
        """
        SELECT id, name, 1 - (embedding <=> $1) AS score
        FROM categories
        WHERE parent_id IS NULL
        ORDER BY score DESC
        LIMIT 5
        """,
        article_embedding,
    )

    if not rows or rows[0]["score"] < PRIMARY_CLASSIFICATION_THRESHOLD:
        return {
            "category_id": None,
            "score": rows[0]["score"] if rows else 0.0,
            "tags": [],
            "status": "uncategorized",
        }

    primary = rows[0]
    tags = [
        {"name": row["name"], "score": row["score"]}
        for row in rows
        if row["score"] >= MULTI_TAG_THRESHOLD
    ]

    if primary["score"] >= SUBCATEGORY_TRIGGER_THRESHOLD:
        sub_rows = await db.fetch(
            """
            SELECT id, name, 1 - (embedding <=> $1) AS score
            FROM categories
            WHERE parent_id = $2
            ORDER BY score DESC
            LIMIT 3
            """,
            article_embedding,
            primary["id"],
        )
        if sub_rows and sub_rows[0]["score"] >= PRIMARY_CLASSIFICATION_THRESHOLD:
            primary = sub_rows[0]

    return {
        "category_id": primary["id"],
        "score": primary["score"],
        "tags": tags,
        "status": "classified",
    }
