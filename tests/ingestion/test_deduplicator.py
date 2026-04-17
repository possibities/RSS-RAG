from __future__ import annotations

import asyncio
import hashlib

import pytest

from ingestion.deduplicator import compute_url_hash, deduplicate


class FakeDB:
    def __init__(self, fetchval_result=None, fetchrow_result=None):
        self.fetchval_result = fetchval_result
        self.fetchrow_result = fetchrow_result
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchval(self, query: str, *args):
        self.fetchval_calls.append((query, args))
        return self.fetchval_result

    async def fetchrow(self, query: str, *args):
        self.fetchrow_calls.append((query, args))
        return self.fetchrow_result


def test_compute_url_hash_uses_utf8_encoding() -> None:
    url = "https://example.com/中文"
    expected = hashlib.sha256(url.encode("utf-8")).hexdigest()
    assert compute_url_hash(url) == expected


def test_deduplicate_short_circuits_on_existing_url_hash() -> None:
    article = {"title": "Title", "url": "https://example.com/a"}
    db = FakeDB(fetchval_result=1)

    assert asyncio.run(deduplicate(article, db)) is True
    assert len(db.fetchrow_calls) == 0


def test_deduplicate_checks_recent_title_similarity_when_url_is_new() -> None:
    article = {"title": "New title", "url": "https://example.com/b"}
    db = FakeDB(fetchval_result=None, fetchrow_result={"id": "article-1"})

    assert asyncio.run(deduplicate(article, db)) is True
    assert db.fetchrow_calls[0][1][1] == pytest.approx(0.85)
    assert db.fetchrow_calls[0][1][2] == 7
