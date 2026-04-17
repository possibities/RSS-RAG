from __future__ import annotations

import httpx

from config import EMBED_BATCH_SIZE, EMBED_MODEL, VLLM_EMBED_URL, VLLM_TIMEOUT_SECONDS, require_setting


def make_embed_text(title: str, summary: str | None) -> str:
    return f"{title}\n\n{summary}" if summary else title


async def batch_embed(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    if not texts:
        return []

    embeddings: list[list[float]] = []
    embed_url = require_setting("VLLM_EMBED_URL", VLLM_EMBED_URL)
    async with httpx.AsyncClient(timeout=VLLM_TIMEOUT_SECONDS) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = await client.post(
                embed_url,
                json={
                    "model": EMBED_MODEL,
                    "input": batch,
                },
            )
            response.raise_for_status()
            data = response.json()
            ordered = sorted(data["data"], key=lambda item: item["index"])
            embeddings.extend(item["embedding"] for item in ordered)
    return embeddings
