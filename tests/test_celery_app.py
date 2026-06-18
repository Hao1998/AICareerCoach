"""
tests/test_celery_app.py
========================
Feature tested: Celery application wiring — that make_celery() correctly
registers tasks, Beat schedule, and the Flask app extension
(tasks/celery_app.py — make_celery).

Background
----------
make_celery(flask_app) creates the Celery instance, registers the scout tasks
(scout.run_for_user, scout.dispatch_due), sets up the Beat schedule to fire
dispatch every minute, and stores the Celery instance on app.extensions["celery"].

Tests run in Celery's "eager" mode: task_always_eager=True makes .delay().get()
execute the task synchronously in the current process, so no broker or worker
process is needed.

_make_app() builds a minimal Flask app with an in-memory broker so the Celery
instance can be configured without Redis.

What each test covers
---------------------
test_run_for_user_task_executes_agent_eager
    Calls celery.run_scout_for_user.delay(5, "scheduled").get() in eager mode
    with FakeAgent monkeypatched in.  Confirms the Celery task wrapper correctly
    passes user_id and run_type into run_scout_for_user_logic and returns the
    result dict.  This is the closest to an end-to-end test of the Celery path
    without a real broker.

test_tasks_and_beat_schedule_registered
    After make_celery(), both task names exist in celery.tasks and the Beat
    schedule entry "dispatch-due-scouts-every-minute" points to scout.dispatch_due.
    Guards against typos or missing _register_scout_tasks() calls.

test_celery_stored_on_app_extensions
    make_celery() stores the Celery instance on app.extensions["celery"] so
    factory.py and other app code can retrieve it via current_app.extensions.

No broker, worker, or API key needed.
"""

import pytest
from flask import Flask

import jobs.scout_agent as scout_agent_module
from tasks.celery_app import make_celery


def _make_app():
    app = Flask(__name__)
    app.config["REDIS_URL"] = "memory://"
    app.config["CELERY_CONFIG"] = {
        "task_always_eager": True,
        "task_eager_propagates": True,
        "broker_url": "memory://",
        "result_backend": "cache+memory://",
    }
    return app


def test_run_for_user_task_executes_agent_eager(monkeypatch):
    class FakeAgent:
        def __init__(self, app):
            pass

        def run_for_user(self, uid, run_type, run_id=None):
            return {"status": "success", "uid": uid, "run_type": run_type}

    monkeypatch.setattr(scout_agent_module, "JobScoutAgent", FakeAgent)

    celery = make_celery(_make_app())
    result = celery.run_scout_for_user.delay(5, "scheduled").get()

    assert result["uid"] == 5
    assert result["run_type"] == "scheduled"


def test_tasks_and_beat_schedule_registered():
    celery = make_celery(_make_app())

    assert "scout.run_for_user" in celery.tasks
    assert "scout.dispatch_due" in celery.tasks
    assert "dispatch-due-scouts-every-minute" in celery.conf.beat_schedule
    assert celery.conf.beat_schedule["dispatch-due-scouts-every-minute"]["task"] == "scout.dispatch_due"


def test_celery_stored_on_app_extensions():
    app = _make_app()
    celery = make_celery(app)
    assert app.extensions["celery"] is celery
