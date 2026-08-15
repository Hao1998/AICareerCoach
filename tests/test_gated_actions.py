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
from models import TaskPlan, User, db


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


def test_abandon_plan_by_id_abandons_the_intended_plan(app_sqlite, fake_redis):
    """When a plan_id is supplied (the normal path since the fix), that exact
    plan is abandoned regardless of what is currently "the active plan"."""
    app = app_sqlite
    with app.app_context():
        plan = TaskPlan(user_id=1, goal="become an ML engineer", status="active")
        db.session.add(plan)
        db.session.commit()
        plan_id = plan.id

        result = gated_actions.abandon_plan(app, 1, "changed my mind", plan_id=plan_id)

        assert result["success"] is True
        db.session.expire_all()
        assert TaskPlan.query.get(plan_id).status == "abandoned"


def test_abandon_plan_stale_plan_id_is_a_noop_and_spares_the_newer_plan(app_sqlite, fake_redis):
    """Regression test for the stale-nonce finding: a confirmation button
    minted for plan X (now no longer 'active' — already abandoned via a
    different, later button click) must not destroy whatever plan is active
    now (plan Y), even though it is well within the nonce's 300s TTL. This
    must fail against the pre-fix abandon_plan, which ignored plan_id
    entirely and always re-resolved get_active_plan(user_id) — so clicking
    the stale button for X would have abandoned Y instead.
    """
    app = app_sqlite
    with app.app_context():
        plan_x = TaskPlan(user_id=1, goal="plan X", status="active")
        db.session.add(plan_x)
        db.session.commit()
        plan_x_id = plan_x.id

        # Turn 2: a second button proposed and clicked for the same plan X,
        # abandoning it for real.
        plan_x.status = "abandoned"
        db.session.commit()

        # Turn 3: user creates a new plan Y, which becomes the active plan.
        plan_y = TaskPlan(user_id=1, goal="plan Y", status="active")
        db.session.add(plan_y)
        db.session.commit()
        plan_y_id = plan_y.id

        # The stale turn-1 button for plan X is now clicked.
        result = gated_actions.abandon_plan(app, 1, "reason", plan_id=plan_x_id)

        assert result["success"] is False
        db.session.expire_all()
        assert TaskPlan.query.get(plan_y_id).status == "active", (
            "stale confirmation destroyed the wrong (newer) plan"
        )
        assert TaskPlan.query.get(plan_x_id).status == "abandoned"


def test_abandon_plan_refuses_a_plan_id_belonging_to_a_different_user(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        owner = User(username="owner", email="owner@example.com")
        owner.set_password("x")
        attacker = User(username="attacker", email="attacker@example.com")
        attacker.set_password("x")
        db.session.add_all([owner, attacker])
        db.session.commit()

        plan = TaskPlan(user_id=owner.id, goal="owner's plan", status="active")
        db.session.add(plan)
        db.session.commit()
        plan_id = plan.id

        result = gated_actions.abandon_plan(app, attacker.id, "reason", plan_id=plan_id)

        assert result["success"] is False
        db.session.expire_all()
        assert TaskPlan.query.get(plan_id).status == "active"


def test_abandon_plan_without_plan_id_falls_back_to_active_plan(app_sqlite, fake_redis):
    """Backward compatibility: a pending action minted before this field
    existed (no plan_id in its stored args) must still work by falling back
    to resolving the active plan at execution time."""
    app = app_sqlite
    with app.app_context():
        plan = TaskPlan(user_id=1, goal="legacy pending action", status="active")
        db.session.add(plan)
        db.session.commit()
        plan_id = plan.id

        result = gated_actions.abandon_plan(app, 1, "reason", plan_id=None)

        assert result["success"] is True
        db.session.expire_all()
        assert TaskPlan.query.get(plan_id).status == "abandoned"
