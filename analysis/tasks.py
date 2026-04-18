from __future__ import annotations

import asyncio

from celery_app import celery_app
from analysis.mailer import send_daily_email
from analysis.trend import weekly_trend_report_all_categories
from storage.db import close_db_pool, create_db_pool


async def _send_daily_email() -> bool:
    pool = await create_db_pool()
    try:
        return await send_daily_email(pool)
    finally:
        await close_db_pool(pool)


async def _generate_weekly_trends() -> list[dict[str, object]]:
    pool = await create_db_pool()
    try:
        return await weekly_trend_report_all_categories(pool)
    finally:
        await close_db_pool(pool)


@celery_app.task(name="analysis.send_daily_email")
def send_daily_email_task() -> bool:
    return asyncio.run(_send_daily_email())


@celery_app.task(name="analysis.weekly_trend_report_all_categories")
def weekly_trend_report_all_categories_task() -> list[dict[str, object]]:
    return asyncio.run(_generate_weekly_trends())
