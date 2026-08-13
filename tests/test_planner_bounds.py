"""
tests/test_planner_bounds.py
============================
Feature tested: the hard iteration cap on the autonomous plan execution loop
(spec §2.2, C3).

Background
----------
create_career_plan spawns a daemon thread running execute_plan's
plan -> execute -> replan loop. Without a cap, a replanner that keeps
producing new steps consumes LLM budget until the process dies. Nobody is
watching that thread.

What each test covers
---------------------
test_max_plan_iterations_is_defined
    The cap exists as a module constant at its current value, so it is one
    reviewable line rather than a magic number buried in the loop.
test_exhausted_plan_gets_a_terminal_status
    THE LOAD-BEARING TEST. A plan that consumes every iteration without
    completing must not be left at status='active'. If it is, get_active_plan
    keeps returning it and create_career_plan refuses forever — the user is
    permanently locked out of creating plans.
test_create_career_plan_enforces_the_daily_budget
    5 plans per day per user. Unlike the gated capabilities, this budget is
    consumed in the tool itself — create_career_plan needs no confirmation,
    so there is no later step at which to charge it.
"""

import json

import chatbot.planner as planner_module
from chatbot.tools import build_tools
from models import TaskPlan, PlanStep, User, db


def test_max_plan_iterations_is_defined():
    assert planner_module.MAX_PLAN_ITERATIONS == 15


def test_exhausted_plan_gets_a_terminal_status(app_sqlite, monkeypatch):
    """A plan that burns every iteration must not stay status='active'.

    get_active_plan filters on status='active' and create_career_plan refuses
    whenever it returns a plan, so an 'active' plan that can never progress
    locks the user out of creating any new plan, permanently.
    """
    app = app_sqlite
    monkeypatch.setattr(planner_module, "MAX_PLAN_ITERATIONS", 3)
    # execute_plan does `from services.llm_service import get_llm` INSIDE the
    # function body (not at module scope), so patching planner_module.get_llm
    # has no effect — that local import always re-resolves against
    # services.llm_service at call time. Patch it there instead.
    monkeypatch.setattr("services.llm_service.get_llm", lambda: None)

    with app.app_context():
        # execute_plan's context-building step (_load_context) requires a
        # real User row for user_id=1 — without one it raises AttributeError
        # on user.full_name before ever reaching the iteration loop.
        user = User(email="loopforever@example.com", username="loopforever", password_hash="x")
        db.session.add(user)
        db.session.commit()

        plan = TaskPlan(user_id=user.id, goal="loop forever", status="active")
        db.session.add(plan)
        db.session.commit()
        plan_id = plan.id

        # Seed more pending steps than the cap allows. Each names a tool that
        # is NOT in tools_by_name, so it hits the loop's
        # "Unknown tool ... Skipping." branch — that consumes an iteration and
        # makes no LLM call, keeping this test offline.
        for i in range(1, 11):
            db.session.add(PlanStep(
                plan_id=plan_id,
                step_order=i,
                description=f"step {i}",
                tool_name="no_such_tool",
                status="pending",
            ))
        db.session.commit()

        planner_module.execute_plan(app, 1, plan_id)

        refreshed = db.session.get(TaskPlan, plan_id)
        assert refreshed.status != "active", (
            "exhausted plan left active — user is now locked out of new plans"
        )
        assert refreshed.status == "failed"

        # And the user can create plans again.
        from chatbot.planner import get_active_plan
        assert get_active_plan(1) is None


def test_create_career_plan_enforces_the_daily_budget(app_sqlite, fake_redis, monkeypatch):
    app = app_sqlite

    created = {"n": 0}

    def _fake_generate_plan(app_arg, uid, goal, llm):
        created["n"] += 1
        plan = TaskPlan(user_id=uid, goal=goal, status="running")
        db.session.add(plan)
        db.session.commit()
        return plan

    # create_career_plan does `from chatbot.planner import generate_plan,
    # execute_plan` INSIDE the function body, so patching the planner module's
    # attributes works — the lookup happens at call time.
    monkeypatch.setattr(planner_module, "generate_plan", _fake_generate_plan)
    monkeypatch.setattr(planner_module, "execute_plan", lambda *a, **k: None)
    # The tool imports get_llm from services.llm_service, not from planner.
    monkeypatch.setattr("services.llm_service.get_llm", lambda: None)

    with app.app_context():
        tools = {t.name: t for t in build_tools(app, 1)}
        create = tools["create_career_plan"]

        outcomes = []
        for _ in range(6):
            outcomes.append(json.loads(create.invoke("become an ML engineer")))
            # Clear the active plan between calls. Without this, calls 2-6
            # short-circuit on "you already have an active plan" and never
            # reach the budget check at all.
            TaskPlan.query.filter_by(user_id=1).update({"status": "abandoned"})
            db.session.commit()

        assert created["n"] == 5, "5 plans should be created before the budget refuses"
        assert outcomes[5]["success"] is False
        assert "limit" in outcomes[5]["error"].lower()
