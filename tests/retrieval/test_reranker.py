from __future__ import annotations

import pytest

from retrieval.reranker import rrf_merge


def test_rrf_merge_combines_ranks_and_writes_rrf_score() -> None:
    vector_results = [
        {"id": "chunk-1", "score": 0.9, "title": "A"},
        {"id": "chunk-2", "score": 0.8, "title": "B"},
    ]
    bm25_results = [
        {"id": "chunk-2", "score": 10.0, "title": "B"},
        {"id": "chunk-3", "score": 9.0, "title": "C"},
    ]

    merged = rrf_merge(vector_results, bm25_results, k=60)

    assert merged[0]["id"] == "chunk-2"
    assert merged[0]["rrf_score"] == pytest.approx((1 / 62) + (1 / 61))
    assert all("rrf_score" in item for item in merged)
