from __future__ import annotations

import httpx
import numpy as np

from config import QA_MODEL, QA_TIMEOUT_SECONDS, VLLM_QA_URL, require_setting


async def get_week_embeddings(
    category_id: str,
    days_start: int,
    days_end: int,
    db,
) -> np.ndarray | None:
    rows = await db.fetch(
        """
        SELECT embedding
        FROM articles
        WHERE category_id = $1
          AND publish_time BETWEEN NOW() - make_interval(days => $2)
                               AND NOW() - make_interval(days => $3)
          AND embedding IS NOT NULL
        """,
        category_id,
        days_start,
        days_end,
    )
    if not rows:
        return None
    return np.array([row["embedding"] for row in rows], dtype=float)


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    return embeddings.mean(axis=0)


async def weekly_trend_report(category_name: str, category_id: str, db) -> dict[str, object]:
    qa_url = require_setting("VLLM_QA_URL", VLLM_QA_URL)
    this_week_embeddings = await get_week_embeddings(category_id, days_start=7, days_end=0, db=db)
    last_week_embeddings = await get_week_embeddings(category_id, days_start=14, days_end=7, db=db)

    if this_week_embeddings is None:
        return {"category": category_name, "report": "本周暂无数据。", "hot_articles": []}

    this_centroid = compute_centroid(this_week_embeddings)
    if last_week_embeddings is not None:
        last_centroid = compute_centroid(last_week_embeddings)
        trend_vector = this_centroid - last_centroid
        has_comparison = True
    else:
        last_centroid = None
        trend_vector = this_centroid
        has_comparison = False

    hot_rows = await db.fetch(
        """
        SELECT title, url, summary, publish_time
        FROM articles
        WHERE category_id = $1
          AND publish_time > NOW() - make_interval(days => $3)
        ORDER BY embedding <=> $2
        LIMIT 5
        """,
        category_id,
        trend_vector.tolist(),
        7,
    )
    hot_articles = [dict(row) for row in hot_rows]

    declining_articles: list[dict[str, object]] = []
    if last_centroid is not None:
        declining_rows = await db.fetch(
            """
            SELECT title, url,
                   1 - (embedding <=> $1) AS last_week_sim,
                   1 - (embedding <=> $2) AS this_week_sim
            FROM articles
            WHERE category_id = $3
              AND publish_time > NOW() - make_interval(days => $4)
            ORDER BY (1 - (embedding <=> $1)) - (1 - (embedding <=> $2)) DESC
            LIMIT 3
            """,
            last_centroid.tolist(),
            this_centroid.tolist(),
            category_id,
            14,
        )
        declining_articles = [dict(row) for row in declining_rows]

    hot_text = "\n".join(
        f"- {article['title']}: {str(article.get('summary') or '')[:100]}"
        for article in hot_articles
    )
    declining_text = (
        "\n".join(f"- {article['title']}" for article in declining_articles)
        if declining_articles
        else "（无明显衰退话题）"
    )
    comparison_note = "（注：本周为首次分析，无上周数据对比）" if not has_comparison else ""
    prompt = f"""请基于以下数据，生成 {category_name} 领域本周技术趋势分析报告 {comparison_note}。
新兴热点文章（本周增长最显著的话题）：
{hot_text}

相对降温的话题：
{declining_text}

要求：
1. 指出本周新兴的技术方向或话题，2-3 点。
2. 点评降温话题的原因，如有。
3. 给出下周值得关注的方向预测，1-2 点。
4. 总字数 400 字以内，专业简洁。

趋势报告："""

    async with httpx.AsyncClient(timeout=QA_TIMEOUT_SECONDS) as client:
        response = await client.post(
            qa_url,
            json={
                "model": QA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.5,
            },
        )
        response.raise_for_status()
    report = response.json()["choices"][0]["message"]["content"].strip()

    return {
        "category": category_name,
        "has_comparison": has_comparison,
        "report": report,
        "hot_articles": [{"title": article["title"], "url": article["url"]} for article in hot_articles],
        "declining_articles": [
            {"title": article["title"], "url": article["url"]}
            for article in declining_articles
        ],
    }


async def weekly_trend_report_all_categories(db) -> list[dict[str, object]]:
    categories = await db.fetch("SELECT id, name FROM categories WHERE parent_id IS NULL")
    reports: list[dict[str, object]] = []
    for category in categories:
        reports.append(await weekly_trend_report(category["name"], str(category["id"]), db))
    return reports
