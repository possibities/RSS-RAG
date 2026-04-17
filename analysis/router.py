from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request

from analysis.digest import generate_daily_digest
from analysis.hot_topics import extract_hot_topics, generate_hot_topics_report
from analysis.trend import weekly_trend_report
from config import HOT_TOPICS_DAYS, SCHEDULER_TIMEZONE


router = APIRouter(tags=["analysis"])


@router.get("/digest/{category_name}")
async def get_digest(category_name: str, request: Request, date: str | None = None) -> dict[str, object]:
    date_str = date
    if date_str is None:
        date_str = datetime.now(ZoneInfo(SCHEDULER_TIMEZONE)).date().isoformat()

    db_pool = request.app.state.db_pool
    row = await db_pool.fetchrow("SELECT id FROM categories WHERE name = $1", category_name)
    if not row:
        return {"error": f"分类 '{category_name}' 不存在"}
    return await generate_daily_digest(category_name, str(row["id"]), date_str, db_pool)


@router.get("/trend/{category_name}")
async def get_trend(category_name: str, request: Request) -> dict[str, object]:
    db_pool = request.app.state.db_pool
    row = await db_pool.fetchrow("SELECT id FROM categories WHERE name = $1", category_name)
    if not row:
        return {"error": f"分类 '{category_name}' 不存在"}
    return await weekly_trend_report(category_name, str(row["id"]), db_pool)


@router.get("/hot-topics")
async def get_hot_topics(request: Request, days: int = HOT_TOPICS_DAYS) -> dict[str, object]:
    db_pool = request.app.state.db_pool
    topics = await extract_hot_topics(db_pool, days=days)
    report = await generate_hot_topics_report(db_pool, days=days)
    return {"topics": topics, "report": report}
