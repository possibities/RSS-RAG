from __future__ import annotations

import asyncio

import httpx
import numpy as np
from sklearn.cluster import KMeans

from config import QA_MODEL, QA_TIMEOUT_SECONDS, VLLM_QA_URL, require_setting


DIGEST_PROMPT = """你是一位技术领域编辑，请基于以下 {count} 篇文章生成一份 {category} 日报。
要求：
1. 总体概述 2-3 句，说明今日该领域整体动态。
2. 逐条列出重要进展，每条 80 字以内，并附文章标题。
3. 结尾用一句话点评今日最值得关注的内容。
4. 总字数控制在 500 字以内，使用中文。

文章列表：
{articles_text}

日报内容："""


async def get_daily_articles(category_id: str, date_str: str, db) -> list[dict[str, object]]:
    rows = await db.fetch(
        """
        SELECT id, title, summary, url, embedding, publish_time
        FROM articles
        WHERE category_id = $1
          AND DATE(publish_time AT TIME ZONE 'Asia/Shanghai') = $2::date
          AND embedding IS NOT NULL
        ORDER BY publish_time DESC
        LIMIT 50
        """,
        category_id,
        date_str,
    )
    return [dict(row) for row in rows]


def cluster_articles(articles: list[dict[str, object]], k: int = 5) -> list[dict[str, object]]:
    if len(articles) <= k:
        return articles

    embeddings = np.array([article["embedding"] for article in articles], dtype=float)
    cluster_count = min(k, len(articles))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    representatives: list[dict[str, object]] = []
    for cluster_id in range(cluster_count):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        cluster_embeddings = embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        best_index = cluster_indices[int(np.argmin(distances))]
        representative = dict(articles[int(best_index)])
        representative["cluster_size"] = int(len(cluster_indices))
        representatives.append(representative)

    return sorted(representatives, key=lambda item: item["cluster_size"], reverse=True)


async def _cluster_articles_async(
    articles: list[dict[str, object]],
    k: int,
) -> list[dict[str, object]]:
    return await asyncio.to_thread(cluster_articles, articles, k)


async def generate_daily_digest(
    category_name: str,
    category_id: str,
    date_str: str,
    db,
) -> dict[str, object]:
    qa_url = require_setting("VLLM_QA_URL", VLLM_QA_URL)
    articles = await get_daily_articles(category_id, date_str, db)
    if not articles:
        return {
            "category": category_name,
            "date": date_str,
            "digest": "今日暂无相关内容。",
            "sources": [],
        }

    k = min(5, max(1, len(articles) // 3))
    representatives = await _cluster_articles_async(articles, k=k)
    articles_text = "\n\n".join(
        f"{index + 1}. {article['title']}\n{article.get('summary') or ''}"
        for index, article in enumerate(representatives)
    )

    async with httpx.AsyncClient(timeout=QA_TIMEOUT_SECONDS) as client:
        response = await client.post(
            qa_url,
            json={
                "model": QA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": DIGEST_PROMPT.format(
                            count=len(representatives),
                            category=category_name,
                            articles_text=articles_text,
                        ),
                    }
                ],
                "max_tokens": 800,
                "temperature": 0.4,
            },
        )
        response.raise_for_status()
    digest = response.json()["choices"][0]["message"]["content"].strip()

    return {
        "category": category_name,
        "date": date_str,
        "digest": digest,
        "sources": [{"title": article["title"], "url": article["url"]} for article in representatives],
    }
