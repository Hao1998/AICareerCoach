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


def get_redis_binary() -> redis.Redis:
    """Return a Redis connection that returns raw bytes (for pickle data)."""
    global _binary_client
    if _binary_client is None:
        url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        _binary_client = redis.Redis.from_url(url, decode_responses=False)
    return _binary_client
