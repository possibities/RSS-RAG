from __future__ import annotations

import asyncio

import httpx
import numpy as np
from sklearn.cluster import KMeans

from config import HOT_TOPICS_TOP_N, QA_MODEL, QA_TIMEOUT_SECONDS, VLLM_QA_URL, require_setting


def cluster_hot_topics(
    articles: list[dict[str, object]],
    top_n: int = HOT_TOPICS_TOP_N,
) -> list[dict[str, object]]:
    if not articles:
        return []

    if len(articles) < top_n:
        return [
            {
                "title": article["title"],
                "url": article["url"],
                "category": article.get("category_name") or "未分类",
                "article_count": 1,
                "summary": article.get("summary") or "",
            }
            for article in articles
        ]

    embeddings = np.array([article["embedding"] for article in articles], dtype=float)
    cluster_count = min(top_n, max(3, len(articles) // 5))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    topics: list[dict[str, object]] = []
    for cluster_id in range(cluster_count):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
        best_index = cluster_indices[int(np.argmin(distances))]
        representative = articles[int(best_index)]
        topics.append(
            {
                "title": representative["title"],
                "url": representative["url"],
                "category": representative.get("category_name") or "未分类",
                "article_count": int(len(cluster_indices)),
                "summary": representative.get("summary") or "",
            }
        )

    return sorted(topics, key=lambda item: item["article_count"], reverse=True)


async def _cluster_hot_topics_async(
    articles: list[dict[str, object]],
    top_n: int,
) -> list[dict[str, object]]:
    return await asyncio.to_thread(cluster_hot_topics, articles, top_n)


async def extract_hot_topics(db, top_n: int = HOT_TOPICS_TOP_N, days: int = 7) -> list[dict[str, object]]:
    rows = await db.fetch(
        """
        SELECT a.id, a.title, a.summary, a.url, a.embedding, c.name AS category_name
        FROM articles a
        LEFT JOIN categories c ON a.category_id = c.id
        WHERE a.publish_time > NOW() - make_interval(days => $1)
          AND a.embedding IS NOT NULL
        """,
        days,
    )

    if not rows:
        return []

    articles = [dict(row) for row in rows]
    return await _cluster_hot_topics_async(articles, top_n)


async def generate_hot_topics_report(db, days: int = 7) -> str:
    qa_url = require_setting("VLLM_QA_URL", VLLM_QA_URL)
    topics = await extract_hot_topics(db, top_n=HOT_TOPICS_TOP_N, days=days)
    if not topics:
        return "本周暂无足够数据生成热点报告。"

    topics_text = "\n".join(
        f"{index + 1}. [{topic['category']}] {topic['title']}（涉及 {topic['article_count']} 篇文章）\n"
        f"   {str(topic.get('summary') or '')[:80]}"
        for index, topic in enumerate(topics)
    )
    prompt = f"""请基于以下本周技术热点话题，生成一份简洁的热点综述报告。
热点话题（按热度排序）：
{topics_text}

要求：
1. 标题：本周 AI 技术热点 TOP{len(topics)}
2. 每个热点用 2-3 句话点评其技术意义。
3. 末尾给出总体判断：本周技术社区最关注的核心议题是什么。
4. 总字数 700 字以内。

报告："""

    async with httpx.AsyncClient(timeout=QA_TIMEOUT_SECONDS) as client:
        response = await client.post(
            qa_url,
            json={
                "model": QA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 700,
                "temperature": 0.4,
            },
        )
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
