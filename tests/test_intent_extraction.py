"""
tests/test_intent_extraction.py
================================
Feature tested: chat UI intent signalling — the mechanism that tells the frontend
to open the "tailor resume" modal or redirect to the jobs list after the AI
responds.

Background
----------
When the chat agent calls a tool (e.g. find_top_jobs, tailor_resume_to_job) the
tool returns a JSON payload that may contain an "action" key.  The chat controller
reads that action and sends an intent + action_data to the frontend so it can
react (redirect, open modal, etc.).

After migrating from AgentExecutor to LangGraph, tool outputs now arrive as
ToolMessage objects inside a message list rather than AgentAction tuples.  Two
functions bridge the old contract to the new:

  _tool_steps_from_messages(messages)
      Extracts (tool_name, tool_output_str) pairs from a LangGraph message list.

  _extract_intent(steps)
      Reads the pairs and returns (intent_str | None, action_data_json | None).

What each test covers
---------------------
test_extract_intent_redirect_from_find_top_jobs
    find_top_jobs output with action=redirect_to_jobs produces the correct intent
    and preserves the job_ids list.

test_extract_intent_open_tailor_modal
    tailor_resume_to_job output with action=open_tailor_modal produces the correct
    intent and preserves ats_before / ats_after scores.

test_extract_intent_none_when_no_actionable_tool
    Tools that return plain text (not an action JSON) produce intent=None so the
    UI does nothing special.

test_extract_intent_ignores_malformed_tool_output
    Non-JSON tool output does not crash — intent stays None.

test_tool_steps_from_messages_extracts_tool_calls
    _tool_steps_from_messages skips HumanMessage / AIMessage and only returns
    the ToolMessage entries, in order.

test_extract_intent_end_to_end_from_messages
    Full pipeline: message list -> _tool_steps_from_messages -> _extract_intent
    produces the correct intent in one shot.

No API key or LLM needed — all inputs are hand-crafted message lists.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chatbot.agent import _extract_intent, _tool_steps_from_messages


def test_extract_intent_redirect_from_find_top_jobs():
    steps = [(
        "find_top_jobs",
        json.dumps({"success": True, "action": "redirect_to_jobs", "job_ids": [1, 2, 3]}),
    )]
    intent, action_data = _extract_intent(steps)
    assert intent == "redirect_to_jobs"
    assert json.loads(action_data)["job_ids"] == [1, 2, 3]


def test_extract_intent_open_tailor_modal():
    steps = [(
        "tailor_resume_to_job",
        json.dumps({
            "action": "open_tailor_modal", "job_id": 42,
            "job": {"id": 42, "title": "ML Eng", "company": "ACME"},
            "ats_before": 55, "ats_after": 80,
        }),
    )]
    intent, action_data = _extract_intent(steps)
    assert intent == "open_tailor_modal"
    data = json.loads(action_data)
    assert data["job_id"] == 42
    assert data["ats_after"] == 80


def test_extract_intent_none_when_no_actionable_tool():
    steps = [("get_resume_info", "You have 5 years of Python experience.")]
    intent, action_data = _extract_intent(steps)
    assert intent is None
    assert action_data is None


def test_extract_intent_ignores_malformed_tool_output():
    steps = [("find_top_jobs", "not-json")]
    intent, action_data = _extract_intent(steps)
    assert intent is None
    assert action_data is None


def test_tool_steps_from_messages_extracts_tool_calls():
    messages = [
        HumanMessage(content="find me jobs"),
        AIMessage(content="", tool_calls=[
            {"name": "find_top_jobs", "args": {"query": "jobs"}, "id": "call_1"}
        ]),
        ToolMessage(content=json.dumps({"success": True, "action": "redirect_to_jobs",
                                        "job_ids": [7]}), name="find_top_jobs",
                    tool_call_id="call_1"),
        AIMessage(content="Here are your matches."),
    ]
    steps = _tool_steps_from_messages(messages)
    assert steps == [(
        "find_top_jobs",
        json.dumps({"success": True, "action": "redirect_to_jobs", "job_ids": [7]}),
    )]


def test_extract_intent_end_to_end_from_messages():
    messages = [
        ToolMessage(content=json.dumps({"action": "open_tailor_modal", "job_id": 9,
                                        "job": {"id": 9}, "ats_before": 40, "ats_after": 75}),
                    name="tailor_resume_to_job", tool_call_id="c1"),
    ]
    intent, action_data = _extract_intent(_tool_steps_from_messages(messages))
    assert intent == "open_tailor_modal"
    assert json.loads(action_data)["job_id"] == 9


def test_extract_intent_confirm_required():
    """A gated tool's propose payload surfaces as a confirm_required intent
    carrying the nonce and the human-readable label for the button."""
    steps = [("abandon_career_plan", json.dumps({
        "success": True,
        "action": "confirm_required",
        "nonce": "abc123",
        "label": "Abandon your plan 'Become an ML engineer'?",
    }))]
    intent, action_data = _extract_intent(steps)
    assert intent == "confirm_required"
    parsed = json.loads(action_data)
    assert parsed["nonce"] == "abc123"
    assert parsed["label"] == "Abandon your plan 'Become an ML engineer'?"


def test_extract_intent_ignores_confirm_payload_without_nonce():
    """A malformed propose payload must not produce a button with no nonce."""
    steps = [("abandon_career_plan", json.dumps({
        "success": True,
        "action": "confirm_required",
        "label": "Abandon?",
    }))]
    intent, _ = _extract_intent(steps)
    assert intent is None
