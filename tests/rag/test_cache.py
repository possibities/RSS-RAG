from __future__ import annotations

import asyncio

import pytest

from rag.pipeline import cached_rag_query


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.setex_calls: list[tuple[str, int, str]] = []

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self.store[key] = value
        self.setex_calls.append((key, ttl, value))


def test_cached_rag_query_key_includes_categories_and_rewrite_flag(monkeypatch) -> None:
    redis_client = FakeRedis()
    calls: list[tuple[tuple[str, ...] | None, bool]] = []

    async def fake_rag_query(question, db, redis_client, category_ids=None, use_rewrite=True):
        calls.append((tuple(category_ids) if category_ids else None, use_rewrite))
        return {"answer": question, "sources": [], "query_used": question}

    monkeypatch.setattr("rag.pipeline.rag_query", fake_rag_query)

    asyncio.run(
        cached_rag_query(
            "hello",
            db=object(),
            redis_client=redis_client,
            category_ids=["b", "a"],
            use_rewrite=True,
        )
    )
    asyncio.run(
        cached_rag_query(
            "hello",
            db=object(),
            redis_client=redis_client,
            category_ids=["a", "b"],
            use_rewrite=True,
        )
    )
    asyncio.run(
        cached_rag_query(
            "hello",
            db=object(),
            redis_client=redis_client,
            category_ids=["a", "b"],
            use_rewrite=False,
        )
    )

    assert len(calls) == 2
    assert len(redis_client.setex_calls) == 2
    assert redis_client.setex_calls[0][0] != redis_client.setex_calls[1][0]
