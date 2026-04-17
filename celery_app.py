from __future__ import annotations

from celery import Celery

from config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CELERY_TASK_ALWAYS_EAGER,
    SCHEDULER_TIMEZONE,
)


celery_app = Celery(
    "rss_rag",
    broker=CELERY_BROKER_URL or None,
    backend=CELERY_RESULT_BACKEND or None,
)
celery_app.conf.update(
    accept_content=["json"],
    result_serializer="json",
    task_serializer="json",
    timezone=SCHEDULER_TIMEZONE,
    task_always_eager=CELERY_TASK_ALWAYS_EAGER,
)
celery_app.autodiscover_tasks(["ingestion", "analysis"])
