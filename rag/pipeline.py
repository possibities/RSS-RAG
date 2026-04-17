from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import httpx

try:
    import redis.asyncio as aioredis
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    aioredis = None

from config import (
    MAX_QA_TOKENS,
    QA_MODEL,
    QA_TIMEOUT_SECONDS,
    RAG_CONTEXT_MAX_CHARS,
    RAG_MIN_RRF_SCORE,
    REDIS_EMBED_TTL_SECONDS,
    REDIS_RAG_TTL_SECONDS,
    VLLM_QA_URL,
    require_setting,
)
from processing.embedder import batch_embed
from rag.prompt import FALLBACK_REPLY, REWRITE_PROMPT, SYSTEM_PROMPT, USER_PROMPT
from retrieval.reranker import rrf_merge
from retrieval.searcher import fulltext_search, vector_search

RedisClient = Any


def build_context(chunks: list[dict[str, object]], max_chars: int = RAG_CONTEXT_MAX_CHARS) -> str:
    parts: list[str] = []
    total = 0
    for index, chunk in enumerate(chunks, start=1):
        publish_time = chunk["publish_time"]
        date_str = publish_time.strftime("%Y-%m-%d") if hasattr(publish_time, "strftime") else str(publish_time)[:10]
        part = f"[来源 {index}] {chunk['title']} ({date_str})\n{str(chunk['chunk_text']).strip()}\n"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n---\n".join(parts)


def is_sufficient(results: list[dict[str, object]]) -> bool:
    if not results:
        return False
    return float(results[0].get("rrf_score", 0.0)) >= RAG_MIN_RRF_SCORE


async def rewrite_query(question: str) -> list[str]:
    qa_url = require_setting("VLLM_QA_URL", VLLM_QA_URL)
    async with httpx.AsyncClient(timeout=QA_TIMEOUT_SECONDS) as client:
        response = await client.post(
            qa_url,
            json={
                "model": QA_MODEL,
                "messages": [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
                "max_tokens": 200,
                "temperature": 0.5,
            },
        )
        response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()

    queries: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        query = line.strip()
        if query and query not in seen:
            queries.append(query)
            seen.add(query)
    return queries[:3] if queries else [question]


async def get_or_embed(text: str, redis_client: RedisClient | None) -> list[float]:
    cache_key = f"emb:{hashlib.md5(text.encode('utf-8')).hexdigest()}"
    if redis_client is not None:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

    embedding = (await batch_embed([text]))[0]
    if redis_client is not None:
        await redis_client.setex(cache_key, REDIS_EMBED_TTL_SECONDS, json.dumps(embedding))
    return embedding


async def rag_query(
    question: str,
    db,
    redis_client: RedisClient | None = None,
    category_ids: list[str] | None = None,
    use_rewrite: bool = True,
) -> dict[str, object]:
    queries = await rewrite_query(question) if use_rewrite else [question]
    primary_query = queries[0]
    query_embedding = await get_or_embed(primary_query, redis_client)

    vector_task = vector_search(query_embedding, db, category_ids=category_ids)
    bm25_task = fulltext_search(primary_query, db, category_ids=category_ids)
    vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)

    merged = rrf_merge(vector_results, bm25_results)
    if not is_sufficient(merged):
        return {"answer": FALLBACK_REPLY, "sources": [], "query_used": primary_query}

    top_chunks = merged[:6]
    context = build_context(top_chunks)
    qa_url = require_setting("VLLM_QA_URL", VLLM_QA_URL)

    async with httpx.AsyncClient(timeout=QA_TIMEOUT_SECONDS) as client:
        response = await client.post(
            qa_url,
            json={
                "model": QA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(context=context, question=question),
                    },
                ],
                "max_tokens": MAX_QA_TOKENS,
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()

    sources = [
        {
            "title": chunk["title"],
            "url": chunk["url"],
            "publish_time": chunk["publish_time"].isoformat()
            if hasattr(chunk["publish_time"], "isoformat")
            else str(chunk["publish_time"]),
        }
        for chunk in top_chunks
    ]
    return {"answer": answer, "sources": sources, "query_used": primary_query}


async def cached_rag_query(
    question: str,
    db,
    redis_client: RedisClient | None = None,
    category_ids: list[str] | None = None,
    use_rewrite: bool = True,
) -> dict[str, object]:
    if redis_client is None:
        return await rag_query(
            question=question,
            db=db,
            redis_client=redis_client,
            category_ids=category_ids,
            use_rewrite=use_rewrite,
        )

    key_data = json.dumps(
        {
            "q": question,
            "cat": sorted(category_ids or []),
            "rw": use_rewrite,
        },
        sort_keys=True,
    )
    cache_key = f"rag:{hashlib.md5(key_data.encode('utf-8')).hexdigest()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    result = await rag_query(
        question=question,
        db=db,
        redis_client=redis_client,
        category_ids=category_ids,
        use_rewrite=use_rewrite,
    )
    await redis_client.setex(
        cache_key,
        REDIS_RAG_TTL_SECONDS,
        json.dumps(result, ensure_ascii=False, default=str),
    )
    return result
