from __future__ import annotations

import asyncio

import pytest

from rag.pipeline import is_sufficient, rag_query
from rag.prompt import FALLBACK_REPLY


def test_is_sufficient_uses_rrf_score() -> None:
    assert is_sufficient([{"rrf_score": 0.02}]) is True
    assert is_sufficient([{"score": 999, "rrf_score": 0.0}]) is False
    assert is_sufficient([]) is False


def test_rag_query_returns_fallback_when_rrf_is_too_low(monkeypatch) -> None:
    async def fake_get_or_embed(text, redis_client):
        return [0.1, 0.2]

    async def fake_vector_search(*args, **kwargs):
        return [{"id": "chunk-1", "score": 0.99}]

    async def fake_fulltext_search(*args, **kwargs):
        return []

    def fake_rrf_merge(vector_results, bm25_results, k=60):
        return [{"id": "chunk-1", "rrf_score": 0.009}]

    monkeypatch.setattr("rag.pipeline.get_or_embed", fake_get_or_embed)
    monkeypatch.setattr("rag.pipeline.vector_search", fake_vector_search)
    monkeypatch.setattr("rag.pipeline.fulltext_search", fake_fulltext_search)
    monkeypatch.setattr("rag.pipeline.rrf_merge", fake_rrf_merge)

    result = asyncio.run(rag_query("what happened?", db=object(), redis_client=None, use_rewrite=False))

    assert result == {"answer": FALLBACK_REPLY, "sources": [], "query_used": "what happened?"}
