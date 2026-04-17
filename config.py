from __future__ import annotations

import json
import os
from typing import Final


def _get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default


def _get_int(name: str, default: int) -> int:
    return int(_get_env(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(_get_env(name, str(default)))


def _get_bool(name: str, default: bool) -> bool:
    raw = _get_env(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return []
    if value.startswith("["):
        return [str(item) for item in json.loads(value)]
    return [item.strip() for item in value.split(",") if item.strip()]


def require_setting(name: str, value: str) -> str:
    if value.strip():
        return value
    raise RuntimeError(f"{name} is not configured")


EMBEDDING_DIMENSION: Final[int] = 1024
EMBED_MODEL: Final[str] = _get_env("EMBED_MODEL", "bge-m3")
SUMMARIZE_MODEL: Final[str] = _get_env("SUMMARIZE_MODEL", "Qwen2.5-7B-Instruct")
QA_MODEL: Final[str] = _get_env("QA_MODEL", "Qwen2.5-14B-Instruct")

VLLM_EMBED_URL: Final[str] = _get_env("VLLM_EMBED_URL")
VLLM_SUMMARIZE_URL: Final[str] = _get_env("VLLM_SUMMARIZE_URL")
VLLM_QA_URL: Final[str] = _get_env("VLLM_QA_URL")

PRIMARY_CLASSIFICATION_THRESHOLD: Final[float] = _get_float("PRIMARY_CLASSIFICATION_THRESHOLD", 0.72)
MULTI_TAG_THRESHOLD: Final[float] = _get_float("MULTI_TAG_THRESHOLD", 0.60)
SUBCATEGORY_TRIGGER_THRESHOLD: Final[float] = _get_float("SUBCATEGORY_TRIGGER_THRESHOLD", 0.80)
RAG_MIN_RRF_SCORE: Final[float] = _get_float("RAG_MIN_RRF_SCORE", 0.01)
SEMANTIC_DEDUP_THRESHOLD: Final[float] = _get_float("SEMANTIC_DEDUP_THRESHOLD", 0.95)
TITLE_DEDUP_THRESHOLD: Final[float] = _get_float("TITLE_DEDUP_THRESHOLD", 0.85)
TIME_DECAY_DAYS: Final[int] = _get_int("TIME_DECAY_DAYS", 30)

RSS_FETCH_TIMEOUT_SECONDS: Final[int] = _get_int("RSS_FETCH_TIMEOUT_SECONDS", 30)
VLLM_TIMEOUT_SECONDS: Final[int] = _get_int("VLLM_TIMEOUT_SECONDS", 120)
QA_TIMEOUT_SECONDS: Final[int] = _get_int("QA_TIMEOUT_SECONDS", 60)
SUMMARY_BATCH_SIZE: Final[int] = _get_int("SUMMARY_BATCH_SIZE", 32)
EMBED_BATCH_SIZE: Final[int] = _get_int("EMBED_BATCH_SIZE", 64)
MAX_CONTENT_CHARS: Final[int] = _get_int("MAX_CONTENT_CHARS", 8000)
MAX_SUMMARY_INPUT_CHARS: Final[int] = _get_int("MAX_SUMMARY_INPUT_CHARS", 3000)
MAX_SUMMARY_TOKENS: Final[int] = _get_int("MAX_SUMMARY_TOKENS", 300)
MAX_QA_TOKENS: Final[int] = _get_int("MAX_QA_TOKENS", 1024)
CHUNK_MAX_TOKENS: Final[int] = _get_int("CHUNK_MAX_TOKENS", 400)
CHUNK_OVERLAP_TOKENS: Final[int] = _get_int("CHUNK_OVERLAP_TOKENS", 50)
DEFAULT_VECTOR_SEARCH_DAYS: Final[int] = _get_int("DEFAULT_VECTOR_SEARCH_DAYS", 90)
DEFAULT_TOP_K: Final[int] = _get_int("DEFAULT_TOP_K", 10)
RAG_CONTEXT_MAX_CHARS: Final[int] = _get_int("RAG_CONTEXT_MAX_CHARS", 4000)
HOT_TOPICS_DAYS: Final[int] = _get_int("HOT_TOPICS_DAYS", 7)
HOT_TOPICS_TOP_N: Final[int] = _get_int("HOT_TOPICS_TOP_N", 8)
RSS_INGEST_INTERVAL_MINUTES: Final[int] = _get_int("RSS_INGEST_INTERVAL_MINUTES", 30)

DATABASE_DSN: Final[str] = _get_env("DATABASE_DSN")
REDIS_URL: Final[str] = _get_env("REDIS_URL")
REDIS_EMBED_TTL_SECONDS: Final[int] = _get_int("REDIS_EMBED_TTL_SECONDS", 3600)
REDIS_RAG_TTL_SECONDS: Final[int] = _get_int("REDIS_RAG_TTL_SECONDS", 900)

CELERY_BROKER_URL: Final[str] = _get_env("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND: Final[str] = _get_env("CELERY_RESULT_BACKEND", REDIS_URL)
CELERY_TASK_ALWAYS_EAGER: Final[bool] = _get_bool("CELERY_TASK_ALWAYS_EAGER", False)

EMAIL_CONFIG: Final[dict[str, object]] = {
    "smtp_host": _get_env("SMTP_HOST"),
    "smtp_port": _get_int("SMTP_PORT", 587),
    "username": _get_env("SMTP_USERNAME", ""),
    "password": _get_env("SMTP_PASSWORD", ""),
    "from_addr": _get_env("SMTP_FROM_ADDR"),
    "to_addrs": _get_list("SMTP_TO_ADDRS", []),
    "use_starttls": _get_bool("SMTP_USE_STARTTLS", True),
}

SCHEDULER_TIMEZONE: Final[str] = _get_env("SCHEDULER_TIMEZONE", "Asia/Shanghai")
ENABLE_SCHEDULER: Final[bool] = _get_bool("ENABLE_SCHEDULER", True)
DAILY_DIGEST_HOUR: Final[int] = _get_int("DAILY_DIGEST_HOUR", 7)
DAILY_DIGEST_MINUTE: Final[int] = _get_int("DAILY_DIGEST_MINUTE", 0)
WEEKLY_TREND_DAY_OF_WEEK: Final[str] = _get_env("WEEKLY_TREND_DAY_OF_WEEK", "mon")
WEEKLY_TREND_HOUR: Final[int] = _get_int("WEEKLY_TREND_HOUR", 8)
WEEKLY_TREND_MINUTE: Final[int] = _get_int("WEEKLY_TREND_MINUTE", 0)

RSS_SOURCES: Final[list[dict[str, str]]] = [
    {"name": "Hugging Face Blog", "url": "https://huggingface.co/blog/feed.xml"},
    {"name": "ArXiv AI", "url": "https://arxiv.org/rss/cs.AI"},
]
