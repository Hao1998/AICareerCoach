"""
Streaming utilities for the chat endpoint.

Bridges LangChain's callback system to a thread-safe queue that the SSE
response generator drains. The agent runs in a background thread and pushes
events; the request thread yields them as Server-Sent Events.

Why a queue (not async)?
    Flask is sync. AgentExecutor.invoke() blocks. A queue + thread combo gives
    us token-by-token streaming on Flask without rewriting to async.

Event types pushed onto the queue:
    {"type": "token",      "content": "..."}                     per token from the LLM
    {"type": "tool_start", "tool": "find_top_jobs", "label": "Searching jobs…"}
    {"type": "progress",   "label": "Ranking matches…"}          mid-tool sub-step
    {"type": "tool_end",   "tool": "find_top_jobs"}
    {"type": "done",       "intent":  "...", "action_data": {...}}
    {"type": "error",      "error":   "..."}
    None                                              sentinel — stop streaming
"""

from __future__ import annotations

import queue
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler


# ── Concurrent-stream guard ───────────────────────────────────────────────────
# Tracks which user_ids currently have a stream in flight. Prevents abuse
# (one user holding many concurrent streams), accidental double-clicks on the
# Send button, and runaway costs.
#
# In production with multiple chat pods, replace this dict with Redis SETNX +
# TTL. For a single-pod deployment, the in-process dict is fine.
import threading
_active_streams: set[int] = set()
_active_streams_lock = threading.Lock()


def acquire_stream_slot(user_id: int) -> bool:
    """Try to mark a stream as active for this user. Returns False if already streaming."""
    with _active_streams_lock:
        if user_id in _active_streams:
            return False
        _active_streams.add(user_id)
        return True


def release_stream_slot(user_id: int) -> None:
    """Mark the user's stream as finished."""
    with _active_streams_lock:
        _active_streams.discard(user_id)


# ── Tool label map ────────────────────────────────────────────────────────────

_TOOL_LABELS: dict[str, str] = {
    "find_top_jobs":           "Searching jobs…",
    "get_resume_info":         "Reading your resume…",
    "tailor_resume_to_job":    "Tailoring your resume…",
    "get_recent_matches":      "Fetching recent matches…",
    "search_job_by_title":     "Looking up job…",
    "trigger_job_scout_agent": "Running job scout agent…",
    "get_user_preferences":    "Checking your preferences…",
    "explain_feature":         "Looking up help…",
    "search_memory":           "Searching memory…",
}


# ── Callback handler ──────────────────────────────────────────────────────────

class TokenStreamHandler(BaseCallbackHandler):
    """
    LangChain callback that pushes token + tool events onto a queue.

    Attach via:
        agent_executor.invoke(inputs, config={"callbacks": [handler]})
    """

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        self._captured_text: list[str] = []
        self._current_tool: str | None = None

    # Token-level streaming from the LLM.
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            self._captured_text.append(token)
            self.q.put({"type": "token", "content": token})

    def push_progress(self, label: str) -> None:
        """Called by tools mid-execution to report sub-step progress."""
        self.q.put({"type": "progress", "label": label})

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        name = (serialized or {}).get("name", "tool")
        self._current_tool = name
        label = _TOOL_LABELS.get(name, f"Running {name}…")
        self.q.put({"type": "tool_start", "tool": name, "label": label})

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self.q.put({"type": "tool_end", "tool": self._current_tool or ""})
        self._current_tool = None

    # If the LLM call errors, surface it cleanly to the client.
    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.q.put({"type": "error", "error": str(error)})

    @property
    def captured_text(self) -> str:
        """The full assistant response after streaming is done."""
        return "".join(self._captured_text)
