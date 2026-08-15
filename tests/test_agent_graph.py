"""
tests/test_agent_graph.py
=========================
Feature tested: the LangGraph ReAct chat agent that powers the career coaching
chat interface (chatbot/agent.py — _build_agent, _invoke_agent).

Background
----------
The chat agent was migrated from LangChain's AgentExecutor to LangGraph's
create_react_agent.  The agent loop is:

  user message -> LLM decides (answer directly OR call a tool)
               -> if tool: execute tool, feed ToolMessage back to LLM
               -> LLM produces final AIMessage -> return text + message list

ScriptedToolModel is a minimal BaseChatModel that pops pre-baked AIMessages
(with or without tool_calls) from a list instead of calling xAI.  This lets
the full build -> invoke -> tool-execution -> final-answer cycle run offline.

stream_fallback_text is also tested here: it is the safety net that pushes the
final text as a streaming token when a cache hit (or any other reason) causes
the agent to skip token emission, which would otherwise show "(no response)"
in the chat UI.

What each test covers
---------------------
test_stream_fallback_none_when_tokens_streamed
    If tokens were captured during streaming, fallback returns None (nothing
    extra to emit — the UI already has the content).

test_stream_fallback_returns_response_when_nothing_streamed
    If nothing was captured (e.g. a cache hit bypassed the stream) and the
    agent produced a final text, fallback returns that text so the UI shows it.

test_stream_fallback_none_when_no_content_at_all
    If both captured_text and response_text are empty, fallback returns None
    (no point emitting an empty token).

test_agent_runs_tool_then_returns_final_text
    The scripted model first returns an AIMessage with a tool_call, then a
    final AIMessage.  Confirms the full ReAct loop runs: tool executes,
    ToolMessage is fed back, final answer is returned.  Also confirms intent
    extraction works end-to-end (redirect_to_jobs intent + job_ids).

test_agent_direct_answer_no_tool
    The model answers directly without calling any tool.  Confirms the agent
    exits cleanly and intent is None.

test_agent_passes_chat_history_and_current_message
    Chat history (prior Human + AI messages) is prepended before the current
    message.  Confirms the message list passed to the LLM contains both prior
    turns and the new question.

No API key needed — ChatXAI is never instantiated.
"""

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool

from chatbot.agent import (
    _build_agent,
    _invoke_agent,
    _extract_intent,
    _tool_steps_from_messages,
    stream_fallback_text,
)


def test_stream_fallback_none_when_tokens_streamed():
    assert stream_fallback_text("streamed reply", "final reply") is None


def test_stream_fallback_returns_response_when_nothing_streamed():
    assert stream_fallback_text("", "final reply") == "final reply"


def test_stream_fallback_none_when_no_content_at_all():
    assert stream_fallback_text("", "") is None


class ScriptedToolModel(BaseChatModel):
    """Returns scripted AIMessages in order; supports bind_tools (no-op)."""
    responses: list

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self):
        return "scripted-tool"

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def find_jobs_matching_resume(query: str) -> str:
    """Find jobs for the user."""
    return json.dumps({"success": True, "action": "redirect_to_jobs", "job_ids": [11, 22]})


def test_agent_runs_tool_then_returns_final_text():
    model = ScriptedToolModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "find_jobs_matching_resume", "args": {"query": "jobs"}, "id": "c1"}
        ]),
        AIMessage(content="Here are your matches."),
    ])
    agent = _build_agent(model, [find_jobs_matching_resume], "You are a career coach.")
    response_text, messages = _invoke_agent(agent, "find me jobs", [])

    assert response_text == "Here are your matches."
    intent, action_data = _extract_intent(_tool_steps_from_messages(messages))
    assert intent == "redirect_to_jobs"
    assert json.loads(action_data)["job_ids"] == [11, 22]


def test_agent_direct_answer_no_tool():
    model = ScriptedToolModel(responses=[AIMessage(content="Hello, how can I help?")])
    agent = _build_agent(model, [find_jobs_matching_resume], "You are a career coach.")
    response_text, messages = _invoke_agent(agent, "hi", [])

    assert response_text == "Hello, how can I help?"
    intent, _ = _extract_intent(_tool_steps_from_messages(messages))
    assert intent is None


def test_agent_passes_chat_history_and_current_message():
    model = ScriptedToolModel(responses=[AIMessage(content="Recalled.")])
    agent = _build_agent(model, [find_jobs_matching_resume], "sys prompt")
    history = [HumanMessage(content="earlier message"), AIMessage(content="earlier reply")]
    response_text, messages = _invoke_agent(agent, "now", history)

    assert response_text == "Recalled."
    contents = [m.content for m in messages]
    assert "earlier message" in contents
    assert "now" in contents
