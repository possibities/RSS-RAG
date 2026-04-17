from __future__ import annotations

import asyncio

from celery_app import celery_app
from config import RSS_SOURCES
from ingestion.pipeline import run_all_sources
from storage.db import close_db_pool, create_db_pool


async def _run_ingestion() -> list[dict[str, int | str]]:
    pool = await create_db_pool()
    try:
        return await run_all_sources(RSS_SOURCES, pool)
    finally:
        await close_db_pool(pool)


@celery_app.task(name="ingestion.run_all_sources")
def run_all_sources_task() -> list[dict[str, int | str]]:
    return asyncio.run(_run_ingestion())
