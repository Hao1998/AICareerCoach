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
test_unhandled_exception_gets_a_terminal_status
    An unhandled exception anywhere in execute_plan (LLM timeout, DB error,
    network failure — considerably more likely in production than exhausting
    15 iterations) hits the outer except block. That block used to only log
    and return {} without touching plan.status, leaving the identical
    permanent-lockout bug open on the exception path. It must now mark the
    plan 'failed'.
test_unhandled_exception_does_not_overwrite_a_completed_plan
    Guards the `status == 'active'` check in the exception handler: if the
    loop already reached a legitimate terminal status ('completed') before a
    later line threw, the handler must not clobber it with 'failed'.
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
        tools = {t.name: t for t in build_tools(app, 1, surface="chat")}
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


def test_unhandled_exception_gets_a_terminal_status(app_sqlite, monkeypatch):
    """An unhandled exception in execute_plan must not leave status='active'.

    This is the same permanent-lockout bug as the exhaustion path, but
    reached via the outer except block instead of the for/else. An LLM
    timeout, DB error, or network failure is far more likely in production
    than burning all 15 iterations, so this path matters more in practice.
    """
    app = app_sqlite
    monkeypatch.setattr("services.llm_service.get_llm", lambda: None)

    # Force an unhandled exception early in execute_plan (build_tools is
    # called right after get_llm(), before the loop or _load_context run).
    def _boom(*a, **k):
        raise RuntimeError("simulated LLM/tool-building failure")

    monkeypatch.setattr("chatbot.tools.build_tools", _boom)

    with app.app_context():
        user = User(email="exc-user@example.com", username="exc-user", password_hash="x")
        db.session.add(user)
        db.session.commit()

        plan = TaskPlan(user_id=user.id, goal="explode", status="active")
        db.session.add(plan)
        db.session.commit()
        plan_id = plan.id

        result = planner_module.execute_plan(app, user.id, plan_id)

        assert result == {}

        # execute_plan's exception handler re-enters its own app_context
        # (production has none active at that point — the outer `with
        # app.app_context():` already unwound before the except runs). That
        # gives it a distinct scoped session from this test's outer context,
        # so this session's identity map still holds the pre-failure 'active'
        # object. Expire it to force a fresh read of what was actually
        # committed.
        db.session.expire_all()
        refreshed = db.session.get(TaskPlan, plan_id)
        assert refreshed.status == "failed", (
            "unhandled exception left plan active — user is now locked out of new plans"
        )

        from chatbot.planner import get_active_plan
        assert get_active_plan(user.id) is None


def test_unhandled_exception_does_not_overwrite_a_completed_plan(app_sqlite, monkeypatch):
    """The exception handler's status=='active' guard must not clobber a
    legitimate terminal status reached before the exception fired."""
    app = app_sqlite
    monkeypatch.setattr("services.llm_service.get_llm", lambda: None)

    def _boom(*a, **k):
        raise RuntimeError("simulated failure after the plan already completed")

    monkeypatch.setattr("chatbot.tools.build_tools", _boom)

    with app.app_context():
        user = User(email="completed-user@example.com", username="completed-user", password_hash="x")
        db.session.add(user)
        db.session.commit()

        # Plan is already in a legitimate terminal state when the exception fires.
        plan = TaskPlan(user_id=user.id, goal="already done", status="completed")
        db.session.add(plan)
        db.session.commit()
        plan_id = plan.id

        planner_module.execute_plan(app, user.id, plan_id)

        db.session.expire_all()  # see comment in the sibling test above
        refreshed = db.session.get(TaskPlan, plan_id)
        assert refreshed.status == "completed", (
            "exception handler overwrote a legitimate terminal status"
        )
