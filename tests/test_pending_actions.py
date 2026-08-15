"""
tests/test_pending_actions.py
=============================
Feature tested: the single-use nonce store backing the confirmation gate
(spec §3.2, §3.3).

The property under test
-----------------------
The client transmits ONLY a nonce. It cannot supply the capability name or
the arguments — those are fixed server-side at proposal time. Nothing
downstream (the model, injected text, a tampered client) can alter what
actually runs. These tests pin that property.

What each test covers
---------------------
test_propose_returns_a_nonce_and_stores_the_payload
    The capability and args are recoverable by nonce.
test_propose_sets_the_ttl
    Unconfirmed actions expire rather than lingering.
test_propose_generates_distinct_nonces
    Two proposals never collide.
test_claim_returns_the_stored_payload
    The round trip preserves capability and args exactly.
test_claim_is_single_use
    A second claim of the same nonce returns None — defeats replay.
test_claim_rejects_another_users_nonce
    User B cannot execute an action proposed for user A.
test_claim_does_not_consume_another_users_nonce
    Critically: a rejected cross-user claim must NOT delete the nonce, or
    user B could grief user A by burning every pending action.
test_claim_returns_none_for_expired_nonce
test_claim_returns_none_for_unknown_nonce
"""

from services.pending_actions import propose, claim, TTL_SECONDS


def test_propose_returns_a_nonce_and_stores_the_payload(fake_redis):
    nonce = propose(1, "abandon_career_plan", {"reason": "restarting"}, "Abandon plan?")
    assert isinstance(nonce, str) and nonce
    assert fake_redis.get(f"pending_action:{nonce}") is not None


def test_propose_sets_the_ttl(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    assert fake_redis.ttl(f"pending_action:{nonce}") == TTL_SECONDS


def test_propose_generates_distinct_nonces(fake_redis):
    a = propose(1, "abandon_career_plan", {}, "x")
    b = propose(1, "abandon_career_plan", {}, "x")
    assert a != b


def test_claim_returns_the_stored_payload(fake_redis):
    nonce = propose(1, "trigger_job_scout_agent", {"reason": "user asked"}, "Run scout?")
    claimed = claim(1, nonce)
    assert claimed["capability"] == "trigger_job_scout_agent"
    assert claimed["args"] == {"reason": "user asked"}
    assert claimed["label"] == "Run scout?"


def test_claim_is_single_use(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    assert claim(1, nonce) is not None
    assert claim(1, nonce) is None


def test_claim_rejects_another_users_nonce(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    assert claim(2, nonce) is None


def test_claim_does_not_consume_another_users_nonce(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    claim(2, nonce)
    assert claim(1, nonce) is not None


def test_claim_returns_none_for_expired_nonce(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    fake_redis.expire_now(f"pending_action:{nonce}")
    assert claim(1, nonce) is None


def test_claim_returns_none_for_unknown_nonce(fake_redis):
    assert claim(1, "never-issued") is None
