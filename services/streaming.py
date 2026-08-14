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
import threading
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler


# ── Concurrent-stream guard ───────────────────────────────────────────────────
# Uses Redis SETNX with a 5-minute TTL so slots auto-expire if a process
# crashes. Works across multiple workers/pods.

def acquire_stream_slot(user_id: int) -> bool:
    """Try to mark a stream as active for this user. Returns False if already streaming."""
    from services.redis_client import get_redis
    r = get_redis()
    key = f"stream:active:{user_id}"
    acquired = r.set(key, "1", nx=True, ex=300)
    return bool(acquired)


def release_stream_slot(user_id: int) -> None:
    """Mark the user's stream as finished."""
    from services.redis_client import get_redis
    r = get_redis()
    r.delete(f"stream:active:{user_id}")


# ── Tool label map ────────────────────────────────────────────────────────────

_TOOL_LABELS: dict[str, str] = {
    "find_top_jobs":           "Searching jobs…",
    "get_resume_info":         "Reading your resume…",
    "tailor_resume_to_job":    "Tailoring your resume…",
    "get_job_history":         "Reviewing your match history…",
    "search_job_by_title":     "Looking up job…",
    "trigger_job_scout_agent": "Preparing job scout run…",
    "search_memory":           "Searching memory…",
    "create_career_plan":      "Creating your career plan…",
    "abandon_career_plan":     "Checking your active plan…",
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


class WebSocketStreamHandler(BaseCallbackHandler):
    """LangChain callback that emits events via Socket.IO with cancel support."""

    def __init__(self, room: str, cancel_event: threading.Event):
        super().__init__()
        self.room = room
        self._cancel_event = cancel_event
        self._captured_text: list[str] = []
        self._current_tool: str | None = None

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self._cancel_event.is_set():
            raise KeyboardInterrupt("User cancelled")
        if token:
            self._captured_text.append(token)
            from factory import socketio
            socketio.emit('token', {'content': token}, room=self.room)

    def push_progress(self, label: str) -> None:
        from factory import socketio
        socketio.emit('progress', {'label': label}, room=self.room)

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        if self._cancel_event.is_set():
            raise KeyboardInterrupt("User cancelled")
        name = (serialized or {}).get("name", "tool")
        self._current_tool = name
        label = _TOOL_LABELS.get(name, f"Running {name}...")
        from factory import socketio
        socketio.emit('tool_start', {'tool': name, 'label': label}, room=self.room)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        from factory import socketio
        socketio.emit('tool_end', {'tool': self._current_tool or ''}, room=self.room)
        self._current_tool = None

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        from factory import socketio
        socketio.emit('error', {'error': str(error)}, room=self.room)

    @property
    def captured_text(self) -> str:
        return "".join(self._captured_text)
