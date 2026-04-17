from __future__ import annotations

import asyncio

import httpx

from config import (
    MAX_SUMMARY_INPUT_CHARS,
    MAX_SUMMARY_TOKENS,
    SUMMARY_BATCH_SIZE,
    SUMMARIZE_MODEL,
    VLLM_TIMEOUT_SECONDS,
    VLLM_SUMMARIZE_URL,
    require_setting,
)

SUMMARIZE_PROMPT = """请对以下文章生成一段 200 字以内的中文摘要，重点提取：
1. 核心技术或方法
2. 主要结论或进展
3. 应用场景或影响

文章标题：{title}
文章内容：{content}

摘要："""


async def _summarize_one(client: httpx.AsyncClient, article: dict[str, object]) -> str:
    summarize_url = require_setting("VLLM_SUMMARIZE_URL", VLLM_SUMMARIZE_URL)
    response = await client.post(
        summarize_url,
        json={
            "model": SUMMARIZE_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": SUMMARIZE_PROMPT.format(
                        title=article["title"],
                        content=str(article["content"])[:MAX_SUMMARY_INPUT_CHARS],
                    ),
                }
            ],
            "max_tokens": MAX_SUMMARY_TOKENS,
            "temperature": 0.3,
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


async def batch_summarize(
    articles: list[dict[str, object]],
    batch_size: int = SUMMARY_BATCH_SIZE,
) -> list[str]:
    if not articles:
        return []

    results: list[str] = []
    async with httpx.AsyncClient(timeout=VLLM_TIMEOUT_SECONDS) as client:
        for start in range(0, len(articles), batch_size):
            batch = articles[start : start + batch_size]
            tasks = [_summarize_one(client, article) for article in batch]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for response in responses:
                results.append("" if isinstance(response, Exception) else response)
    return results
