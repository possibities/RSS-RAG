from __future__ import annotations

import asyncio

import pytest

from processing.classifier import classify_article


class FakeDB:
    def __init__(self, primary_rows, sub_rows):
        self.primary_rows = primary_rows
        self.sub_rows = sub_rows

    async def fetch(self, query: str, *args):
        if "parent_id IS NULL" in query:
            return self.primary_rows
        return self.sub_rows


def test_classify_article_returns_uncategorized_below_primary_threshold() -> None:
    db = FakeDB(primary_rows=[{"id": "cat-1", "name": "AI", "score": 0.71}], sub_rows=[])

    result = asyncio.run(classify_article([0.1, 0.2], db))

    assert result["status"] == "uncategorized"
    assert result["category_id"] is None
    assert result["tags"] == []


def test_classify_article_promotes_to_subcategory_when_triggered() -> None:
    db = FakeDB(
        primary_rows=[
            {"id": "root-1", "name": "AI", "score": 0.91},
            {"id": "root-2", "name": "GPU", "score": 0.62},
        ],
        sub_rows=[{"id": "sub-1", "name": "LLM", "score": 0.83}],
    )

    result = asyncio.run(classify_article([0.1, 0.2], db))

    assert result["status"] == "classified"
    assert result["category_id"] == "sub-1"
    assert result["score"] == pytest.approx(0.83)
    assert result["tags"] == [
        {"name": "AI", "score": 0.91},
        {"name": "GPU", "score": 0.62},
    ]
