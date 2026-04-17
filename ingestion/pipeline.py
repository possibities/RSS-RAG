from __future__ import annotations

from collections.abc import Iterable

from ingestion.cleaner import clean_content
from ingestion.deduplicator import deduplicate, deduplicate_by_embedding
from ingestion.fetcher import fetch_rss
from processing.chunker import chunk_article, count_tokens
from processing.classifier import classify_article
from processing.embedder import batch_embed, make_embed_text
from processing.summarizer import batch_summarize
from storage.db import insert_article_graph


async def prepare_articles(raw_articles: Iterable[dict[str, object]], db) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for article in raw_articles:
        if not article.get("title") or not article.get("url"):
            continue
        if await deduplicate(article, db):
            continue
        article = dict(article)
        article["content"] = clean_content(str(article.get("content", "")))
        prepared.append(article)
    return prepared


async def process_articles(articles: list[dict[str, object]], db) -> dict[str, int]:
    if not articles:
        return {"fetched": 0, "processed": 0, "inserted": 0}

    inserted = 0
    summaries = await batch_summarize(articles)
    for article, summary in zip(articles, summaries):
        article["summary"] = summary

    embed_texts = [make_embed_text(str(article["title"]), article.get("summary")) for article in articles]
    embeddings = await batch_embed(embed_texts)

    for article, art_emb in zip(articles, embeddings):
        if await deduplicate_by_embedding(art_emb, db):
            continue

        article["embedding"] = art_emb
        classification = await classify_article(art_emb, db)
        article["category_id"] = classification["category_id"]
        article["category_score"] = classification["score"]
        article["tags"] = classification["tags"]

        raw_chunks = chunk_article(str(article.get("content", "")))
        chunk_embeddings = await batch_embed(raw_chunks)
        article["chunks"] = [
            {
                "text": text,
                "embedding": chunk_emb,
                "token_count": count_tokens(text),
            }
            for text, chunk_emb in zip(raw_chunks, chunk_embeddings)
        ]

        article_id = await insert_article_graph(article, db)
        if article_id is not None:
            inserted += 1

    return {"fetched": len(articles), "processed": len(embeddings), "inserted": inserted}


async def run_pipeline(source: dict[str, str], db) -> dict[str, int | str]:
    raw_articles = await fetch_rss(source)
    prepared = await prepare_articles(raw_articles, db)
    stats = await process_articles(prepared, db)
    return {"source": source["name"], **stats}


async def run_all_sources(sources: list[dict[str, str]], db) -> list[dict[str, int | str]]:
    results: list[dict[str, int | str]] = []
    for source in sources:
        results.append(await run_pipeline(source, db))
    return results
