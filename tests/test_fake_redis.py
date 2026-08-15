"""
tests/test_fake_redis.py
========================
Guards the FakeRedis test double itself. Every rate-limit and pending-action
test depends on this behaving like redis.Redis(decode_responses=True), so a
silent divergence here would make those suites pass against fiction.
"""

from tests.conftest import FakeRedis


def test_incr_starts_at_one_and_accumulates():
    r = FakeRedis()
    assert r.incr("k") == 1
    assert r.incr("k") == 2


def test_ttl_is_minus_two_for_missing_key():
    r = FakeRedis()
    assert r.ttl("nope") == -2


def test_ttl_is_minus_one_for_key_without_expiry():
    r = FakeRedis()
    r.incr("k")
    assert r.ttl("k") == -1


def test_expire_sets_ttl_and_returns_true():
    r = FakeRedis()
    r.incr("k")
    assert r.expire("k", 60) is True
    assert r.ttl("k") == 60


def test_set_nx_does_not_overwrite():
    r = FakeRedis()
    assert r.set("k", "first", nx=True) is True
    assert r.set("k", "second", nx=True) is None
    assert r.get("k") == "first"


def test_delete_returns_count_removed():
    r = FakeRedis()
    r.set("a", "1")
    assert r.delete("a", "missing") == 1
    assert r.get("a") is None


def test_pipeline_executes_ops_in_order():
    r = FakeRedis()
    pipe = r.pipeline()
    pipe.incr("k")
    pipe.ttl("k")
    count, ttl = pipe.execute()
    assert count == 1
    assert ttl == -1
