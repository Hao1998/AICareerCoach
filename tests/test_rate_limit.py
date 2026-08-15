"""
tests/test_rate_limit.py
========================
Feature tested: per-user, per-capability rate limiting (spec §2.2, C1/C2).

Why this exists
---------------
flask-limiter only sees HTTP routes. The WebSocket chat path and the
background planner both invoke capabilities without passing through a route,
so the budget has to live in a service both can call. These tests pin the
fixed-window semantics that service provides.

What each test covers
---------------------
test_allow_permits_calls_up_to_the_limit
    A budget of (3, 3600) lets exactly 3 calls through.
test_allow_denies_the_call_past_the_limit
    The 4th call returns False.
test_allow_sets_a_ttl_on_first_increment
    The window is bounded — the counter key gets the budget's TTL.
test_allow_repairs_a_missing_ttl
    A key that somehow lost its TTL gets one re-applied, so a counter can
    never become permanent and lock a user out forever.
test_allow_isolates_users
    User 1 exhausting the budget does not affect user 2.
test_allow_isolates_scopes
    The scout budget and the plan budget are independent counters.
test_allow_requires_every_budget_to_pass
    With [(2, 60), (100, 86400)], the tighter budget is what denies.
test_would_allow_does_not_increment
    The read-only check can be called repeatedly without consuming budget.
test_would_allow_reflects_exhaustion
    After the budget is spent, would_allow returns False.
"""

from services.rate_limit import allow, would_allow


def test_allow_permits_calls_up_to_the_limit(fake_redis):
    assert [allow("scout", 1, [(3, 3600)]) for _ in range(3)] == [True, True, True]


def test_allow_denies_the_call_past_the_limit(fake_redis):
    for _ in range(3):
        allow("scout", 1, [(3, 3600)])
    assert allow("scout", 1, [(3, 3600)]) is False


def test_allow_sets_a_ttl_on_first_increment(fake_redis):
    allow("scout", 1, [(3, 3600)])
    key = "ratelimit:scout:1:3600"
    assert fake_redis.ttl(key) == 3600


def test_allow_repairs_a_missing_ttl(fake_redis):
    key = "ratelimit:scout:1:3600"
    fake_redis.store[key] = 1          # counter with no TTL
    allow("scout", 1, [(3, 3600)])
    assert fake_redis.ttl(key) == 3600


def test_allow_isolates_users(fake_redis):
    for _ in range(3):
        allow("scout", 1, [(3, 3600)])
    assert allow("scout", 2, [(3, 3600)]) is True


def test_allow_isolates_scopes(fake_redis):
    for _ in range(3):
        allow("scout", 1, [(3, 3600)])
    assert allow("plan", 1, [(3, 3600)]) is True


def test_allow_requires_every_budget_to_pass(fake_redis):
    budgets = [(2, 60), (100, 86400)]
    assert allow("chat", 1, budgets) is True
    assert allow("chat", 1, budgets) is True
    assert allow("chat", 1, budgets) is False


def test_would_allow_does_not_increment(fake_redis):
    for _ in range(10):
        assert would_allow("scout", 1, [(3, 3600)]) is True
    assert allow("scout", 1, [(3, 3600)]) is True


def test_would_allow_reflects_exhaustion(fake_redis):
    for _ in range(3):
        allow("scout", 1, [(3, 3600)])
    assert would_allow("scout", 1, [(3, 3600)]) is False
