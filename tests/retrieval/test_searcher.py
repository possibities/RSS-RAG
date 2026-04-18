from __future__ import annotations

import asyncio

from config import TIME_DECAY_DAYS
from retrieval.searcher import vector_search


class FakeDB:
    def __init__(self) -> None:
        self.query = ""
        self.args: tuple[object, ...] = ()

    async def fetch(self, query: str, *args):
        self.query = query
        self.args = args
        return []


def test_vector_search_uses_make_interval_and_parameterized_decay() -> None:
    db = FakeDB()

    result = asyncio.run(vector_search([0.1, 0.2], db, days=30, top_k=5))

    assert result == []
    assert "make_interval(days => $3)" in db.query
    assert "INTERVAL '" not in db.query
    assert "CAST($4 AS double precision)" in db.query
    assert "c.embedding IS NOT NULL" in db.query
    assert db.args == ([0.1, 0.2], 5, 30, TIME_DECAY_DAYS)


def test_vector_search_adds_category_filter_as_fifth_parameter() -> None:
    db = FakeDB()

    asyncio.run(vector_search([0.1, 0.2], db, category_ids=["cat-1"], days=7, top_k=3))

    assert "a.category_id = ANY($5::uuid[])" in db.query
    assert db.args == ([0.1, 0.2], 3, 7, TIME_DECAY_DAYS, ["cat-1"])
