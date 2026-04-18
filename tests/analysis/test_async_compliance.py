from __future__ import annotations

import asyncio

import numpy as np

from analysis.digest import _cluster_articles_async
from analysis.hot_topics import _cluster_hot_topics_async
from analysis.trend import _compute_centroid_async


def test_cluster_articles_async_uses_to_thread(monkeypatch) -> None:
    calls: list[tuple[object, tuple[object, ...]]] = []

    async def fake_to_thread(func, *args):
        calls.append((func, args))
        return [{"title": "representative"}]

    monkeypatch.setattr("analysis.digest.asyncio.to_thread", fake_to_thread)

    result = asyncio.run(_cluster_articles_async([{"title": "article"}], k=2))

    assert result == [{"title": "representative"}]
    assert calls[0][0].__name__ == "cluster_articles"
    assert calls[0][1] == ([{"title": "article"}], 2)


def test_cluster_hot_topics_async_uses_to_thread(monkeypatch) -> None:
    calls: list[tuple[object, tuple[object, ...]]] = []

    async def fake_to_thread(func, *args):
        calls.append((func, args))
        return [{"title": "topic"}]

    monkeypatch.setattr("analysis.hot_topics.asyncio.to_thread", fake_to_thread)

    result = asyncio.run(_cluster_hot_topics_async([{"title": "article"}], top_n=4))

    assert result == [{"title": "topic"}]
    assert calls[0][0].__name__ == "cluster_hot_topics"
    assert calls[0][1] == ([{"title": "article"}], 4)


def test_compute_centroid_async_uses_to_thread(monkeypatch) -> None:
    calls: list[tuple[object, tuple[object, ...]]] = []
    sentinel = np.array([0.5, 0.5])

    async def fake_to_thread(func, *args):
        calls.append((func, args))
        return sentinel

    monkeypatch.setattr("analysis.trend.asyncio.to_thread", fake_to_thread)
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0]])

    result = asyncio.run(_compute_centroid_async(embeddings))

    assert np.array_equal(result, sentinel)
    assert calls[0][0].__name__ == "compute_centroid"
    assert np.array_equal(calls[0][1][0], embeddings)
