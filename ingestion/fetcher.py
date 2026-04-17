from __future__ import annotations

import asyncio
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
import httpx

from config import RSS_FETCH_TIMEOUT_SECONDS


def parse_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        return None


async def fetch_rss(
    source: dict[str, str],
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, object]]:
    """
    Fetch a single RSS source and normalize entries into article payloads.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=RSS_FETCH_TIMEOUT_SECONDS)

    try:
        response = await client.get(source["url"])
        response.raise_for_status()
        feed = await asyncio.to_thread(feedparser.parse, response.text)
    finally:
        if owns_client:
            await client.aclose()

    articles: list[dict[str, object]] = []
    for entry in feed.entries:
        content = (entry.get("content") or [{}])[0].get("value", "") or entry.get("summary", "")
        articles.append(
            {
                "title": entry.get("title", "").strip(),
                "url": entry.get("link", "").strip(),
                "content": content,
                "source": source["name"],
                "publish_time": parse_time(entry.get("published")),
            }
        )
    return articles


async def fetch_all_sources(sources: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    async with httpx.AsyncClient(timeout=RSS_FETCH_TIMEOUT_SECONDS) as client:
        results = await asyncio.gather(
            *(fetch_rss(source, client=client) for source in sources),
            return_exceptions=True,
        )

    payload: dict[str, list[dict[str, object]]] = {}
    for source, result in zip(sources, results):
        payload[source["name"]] = [] if isinstance(result, Exception) else result
    return payload
