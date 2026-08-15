"""
tests/test_tool_registry.py
===========================
Feature tested: the consolidated tool set (spec §4).

Why a registry test
-------------------
Tool-selection accuracy degrades with confusable, overlapping tools. This
test pins the exact set so a future addition is a deliberate, reviewed change
rather than drift.

What each test covers
---------------------
test_chat_surface_exposes_exactly_the_expected_tools
test_removed_tools_are_gone
test_get_job_history_returns_matches_and_preferences_together
    The merge must not lose either half of what the two old tools returned.
"""

import json

from chatbot.tools import build_tools

EXPECTED_CHAT_TOOLS = {
    "find_jobs_matching_resume",
    "lookup_job_by_title",
    "get_resume_info",
    "search_memory",
    "get_job_history",
    "tailor_resume_to_job",
    "create_career_plan",
    "trigger_job_scout_agent",
    "abandon_career_plan",
}


def test_chat_surface_exposes_exactly_the_expected_tools(app_sqlite):
    tools = {t.name for t in build_tools(app_sqlite, 1, surface="chat")}
    assert tools == EXPECTED_CHAT_TOOLS


def test_removed_tools_are_gone(app_sqlite):
    tools = {t.name for t in build_tools(app_sqlite, 1, surface="chat")}
    assert "explain_feature" not in tools
    assert "get_career_plan_status" not in tools
    assert "get_recent_matches" not in tools
    assert "get_user_preferences" not in tools


def test_get_job_history_returns_matches_and_preferences_together(app_sqlite):
    app = app_sqlite
    with app.app_context():
        tools = {t.name: t for t in build_tools(app, 1, surface="chat")}
        payload = json.loads(tools["get_job_history"].invoke({"limit": 5}))

        assert payload["success"] is True
        # Both halves of the two merged tools are present.
        assert "matches" in payload
        assert "personalization_active" in payload
        assert "liked_jobs" in payload
        assert "disliked_jobs" in payload
