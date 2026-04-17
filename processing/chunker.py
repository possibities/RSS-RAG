from __future__ import annotations

import re

from config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS


def split_sentences(text: str) -> list[str]:
    segments = re.split(r"(?<=[。！？!?\.])\s+|\n+", text)
    return [segment.strip() for segment in segments if segment and segment.strip()]


def count_tokens(text: str) -> int:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = max(len(text) - chinese, 0)
    return int(chinese / 1.5 + other / 4)


def chunk_article(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))

            overlap: list[str] = []
            overlap_count = 0
            for existing in reversed(current):
                token_count = count_tokens(existing)
                if overlap_count + token_count > overlap_tokens:
                    break
                overlap.insert(0, existing)
                overlap_count += token_count

            current = overlap
            current_tokens = overlap_count

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks
