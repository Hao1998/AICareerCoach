"""
tests/test_gated_actions.py
===========================
Feature tested: the real implementations behind the two gated capabilities,
and the dispatcher the confirm endpoint calls (spec §3).

Background
----------
After the confirmation gate lands, the @tool wrappers for these capabilities
only propose. These functions are the sole path that actually runs the work,
reached exclusively from POST /api/chat/confirm.

What each test covers
---------------------
test_gated_tool_names_lists_both_capabilities
test_abandon_plan_marks_the_active_plan_abandoned
test_abandon_plan_reports_failure_when_no_active_plan
test_run_job_scout_enforces_the_hourly_budget
    The budget is consumed at execution, not at proposal, so proposing
    repeatedly without confirming cannot exhaust it.
test_execute_confirmed_dispatches_to_the_named_capability
test_execute_confirmed_rejects_an_unknown_capability
    A capability name that is not in the registry must never be called, even
    though the name can only come from server-side storage.
"""

import pytest

from chatbot import gated_actions
from models import TaskPlan, db


def test_gated_tool_names_lists_both_capabilities():
    assert gated_actions.GATED_TOOL_NAMES == frozenset({
        "trigger_job_scout_agent",
        "abandon_career_plan",
    })


def test_abandon_plan_marks_the_active_plan_abandoned(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        db.session.add(TaskPlan(user_id=1, goal="become an ML engineer", status="active"))
        db.session.commit()

        result = gated_actions.abandon_plan(app, 1, "changed my mind")

        assert result["success"] is True
        assert TaskPlan.query.filter_by(user_id=1).first().status == "abandoned"


def test_abandon_plan_reports_failure_when_no_active_plan(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        result = gated_actions.abandon_plan(app, 1, "")
        assert result["success"] is False


def test_run_job_scout_enforces_the_hourly_budget(app_sqlite, fake_redis, monkeypatch):
    app = app_sqlite

    class _FakeScheduler:
        def trigger_manual_run(self, user_id):
            return {"status": "success", "matches_found": 2, "jobs_analyzed": 5, "jobs_fetched": 9}

    app.extensions['scheduler'] = _FakeScheduler()

    with app.app_context():
        results = [gated_actions.run_job_scout(app, 1, "test") for _ in range(4)]

    assert [r["success"] for r in results[:3]] == [True, True, True]
    assert results[3]["success"] is False
    assert "limit" in results[3]["message"].lower()


def test_execute_confirmed_dispatches_to_the_named_capability(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        db.session.add(TaskPlan(user_id=1, goal="g", status="active"))
        db.session.commit()

        result = gated_actions.execute_confirmed(app, 1, {
            "capability": "abandon_career_plan",
            "args": {"reason": "done"},
        })

        assert result["success"] is True


def test_execute_confirmed_rejects_an_unknown_capability(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        result = gated_actions.execute_confirmed(app, 1, {
            "capability": "delete_everything",
            "args": {},
        })
        assert result["success"] is False
