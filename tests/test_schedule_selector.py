"""
tests/test_schedule_selector.py
================================
Feature tested: per-user timezone-aware schedule selection for the Celery Beat
job scout dispatcher (jobs/schedule_selector.py — due_user_ids).

Background
----------
With APScheduler, each user's cron ran as a separate in-process job.  With
Celery Beat there is a single periodic task (scout.dispatch_due) that fires
every minute.  That task calls due_user_ids() to decide which users to enqueue
right now.

due_user_ids(configs, now_utc) takes a list of AgentConfig-like objects and the
current UTC time, converts each user's schedule_time to UTC using their stored
timezone, and returns the IDs of users whose local HH:MM matches now.

Because it is a pure function with no I/O, it is easy to unit-test by passing
a fixed NOW and hand-crafted config objects (SimpleNamespace).

Reference time for all tests
-----------------------------
NOW = 2026-01-15 14:00 UTC
  -> UTC users due at 14:00
  -> America/New_York (EST = UTC-5) users due at 09:00 local

What each test covers
---------------------
test_includes_user_whose_local_time_matches
    A UTC user scheduled for 14:00 is included when UTC is 14:00.

test_excludes_user_whose_time_does_not_match
    A UTC user scheduled for 09:00 is not included when UTC is 14:00.

test_respects_user_timezone
    A New York user scheduled for 09:00 local is included (09:00 EST = 14:00
    UTC), while a UTC user scheduled for 09:00 is not.

test_excludes_disabled_users
    A user with is_enabled=False is never enqueued even if their time matches.

test_unknown_timezone_falls_back_to_utc
    An unrecognised timezone string falls back to UTC so the user is still
    dispatched (no crash, no silent drop).

test_handles_missing_schedule_time
    Users with None or empty schedule_time are skipped gracefully.

No Celery, DB, or network needed.
"""

from datetime import datetime
from types import SimpleNamespace

from jobs.schedule_selector import due_user_ids


def _cfg(user_id, schedule_time, timezone="UTC", is_enabled=True):
    return SimpleNamespace(
        user_id=user_id, schedule_time=schedule_time,
        timezone=timezone, is_enabled=is_enabled,
    )


# 2026-01-15 14:00 UTC  ->  New York (EST, UTC-5) local time is 09:00
NOW = datetime(2026, 1, 15, 14, 0)


def test_includes_user_whose_local_time_matches():
    configs = [_cfg(1, "14:00", "UTC")]
    assert due_user_ids(configs, NOW) == [1]


def test_excludes_user_whose_time_does_not_match():
    configs = [_cfg(1, "09:00", "UTC")]  # local 14:00 != 09:00
    assert due_user_ids(configs, NOW) == []


def test_respects_user_timezone():
    # NY user scheduled for 09:00 local is due when UTC is 14:00 (EST).
    configs = [_cfg(1, "09:00", "America/New_York"), _cfg(2, "09:00", "UTC")]
    assert due_user_ids(configs, NOW) == [1]


def test_excludes_disabled_users():
    configs = [_cfg(1, "14:00", "UTC", is_enabled=False)]
    assert due_user_ids(configs, NOW) == []


def test_unknown_timezone_falls_back_to_utc():
    configs = [_cfg(1, "14:00", "Not/AZone")]
    assert due_user_ids(configs, NOW) == [1]


def test_handles_missing_schedule_time():
    configs = [_cfg(1, None, "UTC"), _cfg(2, "", "UTC")]
    assert due_user_ids(configs, NOW) == []
