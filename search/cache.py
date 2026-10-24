"""Redis-backed query result cache."""
import hashlib
import json
from typing import List, Tuple, Optional

import redis

from config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_CACHE_TTL
)


_client: Optional[redis.Redis] = None


def _get_client() -> redis.Redis:
    global _client

    if _client is None:
        _client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )

    return _client


def _cache_key(
    query: str,
    top_k: int,
    alpha: float
) -> str:
    raw = f"{query}|{top_k}|{alpha}"

    return "search:" + hashlib.sha256(
        raw.encode()
    ).hexdigest()


def get(
    query: str,
    top_k: int,
    alpha: float
) -> Optional[List[Tuple[str, float]]]:

    key = _cache_key(query, top_k, alpha)
    data = _get_client().get(key)

    if data:
        return json.loads(data)

    return None


def set(
    query: str,
    top_k: int,
    alpha: float,
    results: List[Tuple[str, float]]
) -> None:

    key = _cache_key(query, top_k, alpha)

    _get_client().setex(
        key,
        REDIS_CACHE_TTL,
        json.dumps(results)
    )


def invalidate_all() -> None:
    client = _get_client()

    for key in client.scan_iter("search:*"):
        client.delete(key)