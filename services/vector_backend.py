"""
Vector backend switch.

VECTOR_BACKEND=redis turns on the Redis vector search experiment: dense job
search is answered by the RediSearch index on REDIS_VECTOR_URL, and job
changes are written to the vector_outbox table for the relay to apply.

Read from os.environ on every call (not current_app.config) because dense
search also runs from the scout and from threads with no app context.
"""
import os


def is_redis_vectors() -> bool:
    return os.environ.get("VECTOR_BACKEND", "").strip().lower() == "redis"
