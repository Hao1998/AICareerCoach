"""
tests/test_ws_chat_rate_limit.py
================================
Feature tested: per-user rate limiting on the WebSocket chat entrypoint
(spec §2.2, C1).

Background
----------
flask-limiter decorates /api/chat but cannot see socket.io events, so the
primary chat path shipped with no frequency limit at all. acquire_stream_slot
bounds concurrency to one in-flight message but not rate: send, await done,
send again, forever.

These tests exercise the guard function directly rather than standing up a
socket.io client, so they stay fast and have no server dependency. The handler
calls this same function before acquiring the stream slot.

What each test covers
---------------------
test_chat_rate_guard_allows_within_budget
    The first 20 messages in a minute pass.
test_chat_rate_guard_denies_past_budget
    The 21st is refused.
test_chat_rate_guard_is_checked_before_the_stream_slot
    A refused message must not consume the single concurrency slot, or a
    rate-limited user would be locked out for the slot's 300s TTL.
"""

from controllers.ws_chat_controller import chat_rate_guard
from services.streaming import acquire_stream_slot


def test_chat_rate_guard_allows_within_budget(fake_redis):
    assert all(chat_rate_guard(7) for _ in range(20))


def test_chat_rate_guard_denies_past_budget(fake_redis):
    for _ in range(20):
        chat_rate_guard(7)
    assert chat_rate_guard(7) is False


def test_chat_rate_guard_is_checked_before_the_stream_slot(fake_redis):
    for _ in range(20):
        chat_rate_guard(7)
    assert chat_rate_guard(7) is False
    # The slot must still be free — the guard runs first and returns early.
    assert acquire_stream_slot(7) is True
