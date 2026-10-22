"""
Redis Connection Helper

Module-level singleton so it works from any thread (including background
threads that have no Flask application context).
"""

import os
import redis

_client: redis.Redis | None = None
_binary_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """Return the shared Redis connection (decode_responses=True), creating it on first call."""
    global _client
    if _client is None:
        url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        _client = redis.Redis.from_url(url, decode_responses=True)
    return _client


_vectors_client: redis.Redis | None = None


def get_redis_vectors() -> redis.Redis:
    """Return the connection to the Redis vector search experiment (raw bytes).

    A separate server (REDIS_VECTOR_URL, port 6380 by default) from REDIS_URL,
    which Celery, rate limiting and the BM25 cache use.
    """
    global _vectors_client
    if _vectors_client is None:
        url = os.environ.get('REDIS_VECTOR_URL', 'redis://localhost:6380/0')
        # Timeouts so a hung or paused Redis makes search fall back instead of
        # blocking job matching forever.
        _vectors_client = redis.Redis.from_url(
            url, decode_responses=False, socket_timeout=2, socket_connect_timeout=1,
        )
    return _vectors_client


def get_redis_binary() -> redis.Redis:
    """Return a Redis connection that returns raw bytes (for pickle data)."""
    global _binary_client
    if _binary_client is None:
        url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        _binary_client = redis.Redis.from_url(url, decode_responses=False)
    return _binary_client
