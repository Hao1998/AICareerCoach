"""
tests/test_tool_surfaces.py
===========================
Feature tested: surface-scoped tool construction (spec §3.5, G1 and G3).

The hole this closes
--------------------
chatbot/planner.py builds the full tool list and dispatches by LLM-chosen
name inside an unsupervised daemon thread. A confirmation gate is meaningless
there — nobody is present to click. So the planner surface must not receive
gated capabilities at all.

What each test covers
---------------------
test_chat_surface_includes_gated_tools
    The model must still be able to PROPOSE them, so they stay in the chat list.
test_planner_surface_excludes_gated_tools
    The planner physically cannot invoke what it never receives.
test_surface_is_required
    Omitting it raises TypeError rather than defaulting to the permissive
    value. This is the G3 defence: a future caller that forgets fails loudly.
test_gated_tool_proposes_without_executing
    Calling the chat-surface wrapper stores a pending action and returns
    confirm_required — it must NOT perform the work.
test_gated_tool_does_not_consume_rate_budget_on_propose
    Budget is spent at confirm time (Task 6), not here.
"""

import json

import pytest

from chatbot.tools import build_tools
from chatbot.gated_actions import GATED_TOOL_NAMES
from services.rate_limit import would_allow, SCOUT_BUDGETS


def _names(tools):
    return {t.name for t in tools}


def test_chat_surface_includes_gated_tools(app_sqlite):
    tools = build_tools(app_sqlite, 1, surface="chat")
    assert GATED_TOOL_NAMES <= _names(tools)


def test_planner_surface_excludes_gated_tools(app_sqlite):
    tools = build_tools(app_sqlite, 1, surface="planner")
    assert _names(tools).isdisjoint(GATED_TOOL_NAMES)


def test_surface_is_required(app_sqlite):
    with pytest.raises(TypeError):
        build_tools(app_sqlite, 1)


def test_gated_tool_proposes_without_executing(app_sqlite, fake_redis):
    app = app_sqlite
    from models import TaskPlan, db

    with app.app_context():
        db.session.add(TaskPlan(user_id=1, goal="become an ML engineer", status="active"))
        db.session.commit()

        tools = {t.name: t for t in build_tools(app, 1, surface="chat")}
        raw = tools["abandon_career_plan"].invoke("changed my mind")
        payload = json.loads(raw)

        assert payload["action"] == "confirm_required"
        assert payload["nonce"]
        # The plan must be untouched — proposing is not doing.
        assert TaskPlan.query.filter_by(user_id=1).first().status == "active"


def test_gated_tool_does_not_consume_rate_budget_on_propose(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        tools = {t.name: t for t in build_tools(app, 1, surface="chat")}
        for _ in range(5):
            tools["trigger_job_scout_agent"].invoke("please run it")
        assert would_allow("scout_manual", 1, SCOUT_BUDGETS) is True
