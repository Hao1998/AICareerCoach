"""
tests/test_scout_tasks.py
=========================
Feature tested: Job Scout task logic that executes inside a Celery worker
(tasks/scout_tasks.py — run_scout_for_user_logic, dispatch_due_scouts_logic).

Background
----------
The Celery tasks themselves are thin decorators.  The real work lives in two
plain functions so they can be tested without a broker or worker process:

  run_scout_for_user_logic(flask_app, user_id, run_type, run_id=None)
      Creates a JobScoutAgent and calls run_for_user.  Runs inside an
      app context so the agent can access the DB.

  dispatch_due_scouts_logic(flask_app, enqueue, fetch_configs, now_utc)
      Calls due_user_ids() with the live AgentConfigs, then calls enqueue()
      for each due user.  enqueue() is injected so tests can capture calls
      without touching a real broker.

monkeypatch replaces JobScoutAgent with a FakeAgent that records its call
arguments and returns a scripted dict — no LLM or DB required.

What each test covers
---------------------
test_run_scout_for_user_logic_invokes_agent
    run_scout_for_user_logic creates a FakeAgent, calls run_for_user with the
    correct (uid, run_type, run_id) arguments, and returns the agent's result
    dict.  Confirms the task logic wires app context and agent correctly.

test_dispatch_enqueues_only_due_users
    Given three configs (user 1 due now, user 2 wrong time, user 3 disabled),
    dispatch_due_scouts_logic calls enqueue only for user 1.  Confirms the
    integration between due_user_ids filtering and the enqueue callback.

No broker, worker, DB, or API key needed.
"""

from datetime import datetime
from types import SimpleNamespace

import pytz
from flask import Flask

import jobs.scout_agent as scout_agent_module
from tasks.scout_tasks import run_scout_for_user_logic, dispatch_due_scouts_logic


def test_run_scout_for_user_logic_invokes_agent(monkeypatch):
    calls = {}

    class FakeAgent:
        def __init__(self, app):
            calls["app"] = app

        def run_for_user(self, uid, run_type, run_id=None):
            calls["args"] = (uid, run_type, run_id)
            return {"status": "success", "matches_found": 3}

    monkeypatch.setattr(scout_agent_module, "JobScoutAgent", FakeAgent)
    app = Flask(__name__)

    result = run_scout_for_user_logic(app, 7, "scheduled")

    assert calls["args"] == (7, "scheduled", None)
    assert result["matches_found"] == 3


def test_dispatch_enqueues_only_due_users():
    app = Flask(__name__)
    configs = [
        SimpleNamespace(user_id=1, schedule_time="14:00", timezone="UTC", is_enabled=True),
        SimpleNamespace(user_id=2, schedule_time="09:00", timezone="UTC", is_enabled=True),
        SimpleNamespace(user_id=3, schedule_time="14:00", timezone="UTC", is_enabled=False),
    ]
    enqueued = []
    now = datetime(2026, 1, 15, 14, 0, tzinfo=pytz.UTC)

    due = dispatch_due_scouts_logic(
        app, enqueue=enqueued.append, fetch_configs=lambda: configs, now_utc=now,
    )

    assert due == [1]
    assert enqueued == [1]
