"""
tests/test_redis_vector_store.py
================================
RedisJobVectorStore — the RediSearch HNSW index for the Redis vector search
experiment.

The unit tests need nothing. The integration tests need a Redis with the query
engine on REDIS_VECTOR_URL (`make redis-up`) and are skipped otherwise. They
use their own index/prefix (test:idx:jobs / test:job:) so the persisted
idx:jobs data in .data/redis is never touched.
"""
import os
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from jobs.redis_vector_store import (
    RedisJobVectorStore,
    distance_to_similarity,
    encode_vector,
)

DIM = 768


def _onehot(pos):
    v = [0.0] * DIM
    v[pos] = 1.0
    return v


def _job(job_id, hot=0, embedding="default", source="test"):
    return SimpleNamespace(
        id=job_id,
        embedding=_onehot(hot) if embedding == "default" else embedding,
        source=source,
        embedding_updated_at=datetime(2026, 10, 20, 12, 0, 0),
    )


# ── unit ──────────────────────────────────────────────────────────────────────

def test_encode_vector_is_float32_bytes():
    b = encode_vector([0.5] * DIM)
    assert len(b) == DIM * 4
    assert np.frombuffer(b, dtype=np.float32)[0] == 0.5


def test_distance_to_similarity_clamps():
    assert distance_to_similarity(0.0) == 1.0
    assert distance_to_similarity(0.25) == 0.75
    assert distance_to_similarity(1.5) == 0.0
    assert distance_to_similarity(-0.1) == 1.0


def test_key_uses_prefix():
    assert RedisJobVectorStore(client=None, prefix="t:job:").key(7) == "t:job:7"


def test_search_with_non_positive_k_returns_empty_without_redis():
    assert RedisJobVectorStore(client=None).search(_onehot(0), k=0) == []


# ── integration ───────────────────────────────────────────────────────────────

@pytest.fixture
def live_store():
    import redis

    url = os.environ.get("REDIS_VECTOR_URL", "redis://localhost:6380/0")
    client = redis.Redis.from_url(url, decode_responses=False, socket_connect_timeout=1)
    try:
        client.execute_command("FT._LIST")
    except Exception as e:
        pytest.skip(f"no Redis query engine at {url}: {e}")

    store = RedisJobVectorStore(client, index_name="test:idx:jobs", prefix="test:job:")
    store.drop()
    yield store
    store.drop()


def _wait_for_count(store, n, timeout=2.0):
    deadline = time.monotonic() + timeout
    while store.count() != n and time.monotonic() < deadline:
        time.sleep(0.05)
    return store.count()


def test_rebuild_then_search_returns_nearest(live_store):
    a, b = _job(1, hot=0), _job(2, hot=1)
    assert live_store.rebuild([a, b, _job(3, embedding=None)]) == 2
    assert _wait_for_count(live_store, 2) == 2
    hits = live_store.search(_onehot(0), k=2)
    assert hits[0][0] == 1 and hits[0][1] > 0.99
    assert [jid for jid, _ in hits] == [1, 2]


def test_remove_job_drops_from_results(live_store):
    live_store.rebuild([_job(1, hot=0), _job(2, hot=1)])
    live_store.remove_job(1)
    assert _wait_for_count(live_store, 1) == 1
    assert [jid for jid, _ in live_store.search(_onehot(0), k=5)] == [2]


def test_upsert_is_idempotent(live_store):
    live_store.ensure_index()
    live_store.ensure_index()  # second call must not raise
    j = _job(5, hot=3)
    live_store.upsert_job(j)
    live_store.upsert_job(j)
    assert _wait_for_count(live_store, 1) == 1


def test_rebuild_does_not_touch_other_indexes(live_store):
    other = RedisJobVectorStore(live_store.client, index_name="test:idx:other", prefix="test:other:")
    other.drop()
    try:
        other.rebuild([_job(9, hot=2)])
        live_store.rebuild([_job(1, hot=0)])
        assert _wait_for_count(other, 1) == 1
    finally:
        other.drop()


def test_index_is_not_built_until_rebuild(live_store):
    live_store.ensure_index()
    assert live_store.is_built() is False      # e.g. the relay recreated it after a reset
    live_store.rebuild([_job(1, hot=0)])
    assert live_store.is_built() is True
    live_store.drop()
    assert live_store.is_built() is False


def test_vector_client_has_socket_timeouts(monkeypatch):
    import services.redis_client as rc
    monkeypatch.setattr(rc, "_vectors_client", None)
    client = rc.get_redis_vectors()
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs.get("socket_timeout") and kwargs.get("socket_connect_timeout")
    monkeypatch.setattr(rc, "_vectors_client", None)
