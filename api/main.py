from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from analysis.router import router as analysis_router
from analysis.scheduler import create_scheduler
from config import ENABLE_SCHEDULER
from rag.pipeline import cached_rag_query, get_or_embed
from retrieval.searcher import collapse_results_by_article, vector_search
from storage.db import (
    close_db_pool,
    close_redis_client,
    create_db_pool,
    create_redis_client,
    resolve_category_ids,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_pool = await create_db_pool()
    app.state.redis_client = create_redis_client()
    app.state.scheduler = None

    if ENABLE_SCHEDULER:
        scheduler = create_scheduler()
        scheduler.start()
        app.state.scheduler = scheduler

    try:
        yield
    finally:
        scheduler = app.state.scheduler
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        await close_redis_client(app.state.redis_client)
        await close_db_pool(app.state.db_pool)


app = FastAPI(title="RSS-RAG API", lifespan=lifespan)
app.include_router(analysis_router)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    categories: list[str] | None = None
    use_rewrite: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, object]]
    query_used: str


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, request: Request) -> dict[str, object]:
    db_pool = request.app.state.db_pool
    redis_client = request.app.state.redis_client
    category_ids = await resolve_category_ids(db_pool, req.categories or [])
    return await cached_rag_query(
        question=req.question,
        db=db_pool,
        redis_client=redis_client,
        category_ids=category_ids or None,
        use_rewrite=req.use_rewrite,
    )


@app.get("/search")
async def search_endpoint(
    q: str,
    request: Request,
    category: str | None = None,
    days: int = 90,
) -> list[dict[str, object]]:
    db_pool = request.app.state.db_pool
    redis_client = request.app.state.redis_client
    category_ids = await resolve_category_ids(db_pool, [category] if category else [])
    query_embedding = await get_or_embed(q, redis_client)
    results = await vector_search(
        query_embedding,
        db_pool,
        category_ids=category_ids or None,
        days=days,
        top_k=20,
    )
    collapsed = collapse_results_by_article(results, top_k=10)
    return [
        {
            "title": result["title"],
            "url": result["url"],
            "score": result["score"],
            "publish_time": result["publish_time"],
        }
        for result in collapsed
    ]
