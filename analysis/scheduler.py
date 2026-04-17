from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from analysis.tasks import send_daily_email_task, weekly_trend_report_all_categories_task
from config import (
    DAILY_DIGEST_HOUR,
    DAILY_DIGEST_MINUTE,
    RSS_INGEST_INTERVAL_MINUTES,
    SCHEDULER_TIMEZONE,
    WEEKLY_TREND_DAY_OF_WEEK,
    WEEKLY_TREND_HOUR,
    WEEKLY_TREND_MINUTE,
)
from ingestion.tasks import run_all_sources_task


def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=SCHEDULER_TIMEZONE)
    scheduler.add_job(
        run_all_sources_task.delay,
        trigger=IntervalTrigger(minutes=RSS_INGEST_INTERVAL_MINUTES, timezone=SCHEDULER_TIMEZONE),
        id="rss_ingestion",
        replace_existing=True,
    )
    scheduler.add_job(
        send_daily_email_task.delay,
        trigger=CronTrigger(hour=DAILY_DIGEST_HOUR, minute=DAILY_DIGEST_MINUTE, timezone=SCHEDULER_TIMEZONE),
        id="daily_email",
        replace_existing=True,
    )
    scheduler.add_job(
        weekly_trend_report_all_categories_task.delay,
        trigger=CronTrigger(
            day_of_week=WEEKLY_TREND_DAY_OF_WEEK,
            hour=WEEKLY_TREND_HOUR,
            minute=WEEKLY_TREND_MINUTE,
            timezone=SCHEDULER_TIMEZONE,
        ),
        id="weekly_trend",
        replace_existing=True,
    )
    return scheduler
