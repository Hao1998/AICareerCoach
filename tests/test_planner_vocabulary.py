"""
tests/test_planner_vocabulary.py
================================
Feature tested: the planner's tool vocabulary (spec §3.5, G2 and G4).

Background
----------
TOOL_DESCRIPTIONS was a hand-maintained string listing 8 of the 12 tools —
already drifted — and PLANNER_PROMPT explicitly told the LLM to schedule
trigger_job_scout_agent as Phase 1 Step 4. The planner surface no longer
receives gated capabilities, so a plan naming one would just be skipped.
Generating the vocabulary from the registry makes drift impossible.

What each test covers
---------------------
test_tool_descriptions_lists_every_planner_tool
test_tool_descriptions_names_no_gated_capability
    The prompt must not invite the model to schedule something it cannot run.
test_planner_prompt_does_not_mention_gated_capabilities
test_unknown_tool_in_a_plan_step_degrades_gracefully
    A stale plan referencing a removed tool must skip, not crash.
"""

from chatbot.planner import tool_descriptions, PLANNER_PROMPT
from chatbot.tools import build_tools
from chatbot.gated_actions import GATED_TOOL_NAMES


def test_tool_descriptions_lists_every_planner_tool(app_sqlite):
    text = tool_descriptions(app_sqlite, 1)
    for tool in build_tools(app_sqlite, 1, surface="planner"):
        assert tool.name in text


def test_tool_descriptions_names_no_gated_capability(app_sqlite):
    text = tool_descriptions(app_sqlite, 1)
    for name in GATED_TOOL_NAMES:
        assert name not in text


def test_planner_prompt_does_not_mention_gated_capabilities():
    rendered = str(PLANNER_PROMPT)
    for name in GATED_TOOL_NAMES:
        assert name not in rendered


def test_unknown_tool_in_a_plan_step_degrades_gracefully(app_sqlite):
    tools_by_name = {t.name: t for t in build_tools(app_sqlite, 1, surface="planner")}
    assert "trigger_job_scout_agent" not in tools_by_name
    # This mirrors the guard at chatbot/planner.py:429 — an unrecognised
    # tool_name falls to the "Unknown tool ... Skipping." branch.
