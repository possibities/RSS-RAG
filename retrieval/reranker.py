from __future__ import annotations


def rrf_merge(
    vector_results: list[dict[str, object]],
    bm25_results: list[dict[str, object]],
    k: int = 60,
) -> list[dict[str, object]]:
    scores: dict[str, float] = {}
    meta: dict[str, dict[str, object]] = {}

    for rank, item in enumerate(vector_results):
        chunk_id = str(item["id"])
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank + 1)
        meta[chunk_id] = item

    for rank, item in enumerate(bm25_results):
        chunk_id = str(item["id"])
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank + 1)
        if chunk_id not in meta:
            meta[chunk_id] = item

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    results: list[dict[str, object]] = []
    for chunk_id in ranked_ids:
        merged = dict(meta[chunk_id])
        merged["rrf_score"] = scores[chunk_id]
        results.append(merged)
    return results
