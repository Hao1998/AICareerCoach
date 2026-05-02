"""
Job Scout LangGraph

Converts the sequential Job Scout Agent pipeline into a LangGraph StateGraph,
enabling conditional branching and autonomous flow control.

Graph nodes (in order):
  load_user        → load user profile + config, check if agent is enabled
  load_resume      → load active resume + extract text from PDF
  extract_keywords → LLM decides which job titles to search for
  fetch_jobs       → call Adzuna API to retrieve fresh postings
  rebuild_index    → rebuild FAISS vector index (only if new jobs were stored)
  find_matches     → FAISS + LLM hybrid matching, save results to DB
  finalize         → persist stats to AgentRunHistory, emit "done"

Conditional edges:
  load_user    → END                 if agent disabled or user not found
  load_resume  → END                 if no active resume found
  fetch_jobs   → rebuild_index       if new jobs were stored
             → find_matches         if no new jobs (skip rebuild)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import json

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from models import db, User, Resume, AgentConfig, AgentRunHistory
from job_utils import build_job_faiss_index

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class JobScoutState(TypedDict):
    """Shared state that flows through every node of the graph."""

    # ---- inputs (set before graph.invoke) ----
    user_id: int
    run_id: int
    run_type: str

    # ---- populated by load_user ----
    is_enabled: Optional[bool]
    match_threshold: Optional[float]
    max_results_per_run: Optional[int]
    adzuna_location: Optional[str]
    adzuna_max_jobs: Optional[int]
    adzuna_max_days_old: Optional[int]
    explicit_preferences: Optional[dict]  # preferences expressed via chat

    # ---- populated by load_resume ----
    resume_id: Optional[int]
    resume_filename: Optional[str]
    resume_text: Optional[str]

    # ---- populated by extract_keywords ----
    keywords: Optional[str]

    # ---- populated by fetch_jobs ----
    job_stats: Optional[Dict[str, Any]]

    # ---- populated by find_matches ----
    matches_analyzed: Optional[List[Dict]]
    matches_saved: Optional[List[Dict]]

    # ---- control flow ----
    status: str          # "running" | "completed" | "failed" | "disabled"
    error: Optional[str]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_job_scout_graph(agent, emit_fn: Callable, mark_done_fn: Callable, checkpointer=None):
    """
    Build and compile the LangGraph StateGraph for the Job Scout Agent.

    Args:
        agent:        JobScoutAgent instance — provides .llm, .app,
                      .matching_prompt, .extract_text_from_pdf(),
                      ._extract_keywords_from_resume(), ._fetch_new_jobs(),
                      ._find_and_save_matches()
        emit_fn:      _emit_progress(run_id, message) callable
        mark_done_fn: _mark_done(run_id) callable

    Returns:
        Compiled LangGraph graph ready for .invoke()

    Note on checkpointer:
        When a checkpointer is supplied the graph serialises the full
        JobScoutState to the backing store after every node.  Each run must be
        given a unique thread_id via the invocation config:

            graph.invoke(state, config={"configurable": {"thread_id": "run-42"}})

        This enables crash recovery: if the process dies mid-run, calling
        invoke() again with the same thread_id resumes from the last
        completed node rather than restarting from scratch.
    """

    # ------------------------------------------------------------------ #
    # NODE FUNCTIONS
    # Each returns a dict of state keys to update (partial state update).
    # ------------------------------------------------------------------ #

    def node_load_user(state: JobScoutState) -> dict:
        """Load user and AgentConfig. Decide whether to continue or exit."""
        user_id = state["user_id"]
        run_id = state["run_id"]

        emit_fn(run_id, "Loading user profile and configuration...")

        user = User.query.get(user_id)
        if not user:
            _fail_run(run_id, f"User {user_id} not found")
            return {"status": "failed", "error": f"User {user_id} not found"}

        config = AgentConfig.query.filter_by(user_id=user_id).first()
        if not config:
            config = AgentConfig(user_id=user_id)
            db.session.add(config)
            db.session.commit()

        if not config.is_enabled:
            emit_fn(run_id, "Agent is disabled — skipping.")
            _finish_run(run_id, status="completed", summary={"message": "Agent is disabled"})
            return {"status": "disabled", "is_enabled": False}

        return {
            "is_enabled": True,
            "match_threshold": config.match_threshold,
            "max_results_per_run": config.max_results_per_run,
            "adzuna_location": config.adzuna_location,
            "adzuna_max_jobs": config.adzuna_max_jobs or 20,
            "adzuna_max_days_old": config.adzuna_max_days_old or 30,
            "explicit_preferences": config.explicit_preferences,
        }

    def node_load_resume(state: JobScoutState) -> dict:
        """Load the user's latest active resume and extract text from it."""
        user_id = state["user_id"]
        run_id = state["run_id"]

        emit_fn(run_id, "Reading your resume...")

        latest_resume = Resume.query.filter_by(
            user_id=user_id, is_active=True
        ).order_by(Resume.uploaded_at.desc()).first()

        if not latest_resume:
            _fail_run(run_id, "No resume found for user")
            return {"status": "failed", "error": "No resume found for user"}

        try:
            resume_text = agent.extract_text_from_pdf(latest_resume.file_path)
        except Exception as exc:
            msg = f"Failed to read resume: {exc}"
            _fail_run(run_id, msg)
            return {"status": "failed", "error": msg}

        return {
            "resume_id": latest_resume.id,
            "resume_filename": latest_resume.original_filename,
            "resume_text": resume_text,
        }

    def node_extract_keywords(state: JobScoutState) -> dict:
        """Ask the LLM which job titles to search for based on the resume."""
        run_id = state["run_id"]
        emit_fn(run_id, "Extracting job search keywords from your resume...")

        keywords = agent._extract_keywords_from_resume(
            state["resume_text"],
            explicit_preferences=state.get("explicit_preferences"),
        )
        emit_fn(run_id, f"Searching for: {keywords}")

        return {"keywords": keywords}

    def node_fetch_jobs(state: JobScoutState) -> dict:
        """Use Adzuna API to fetch fresh job postings matching the keywords."""
        run_id = state["run_id"]
        emit_fn(run_id, "Fetching new jobs from Adzuna...")

        # Build a lightweight proxy so _fetch_new_jobs sees the expected interface
        class _ConfigProxy:
            adzuna_location = state["adzuna_location"]
            adzuna_max_jobs = state["adzuna_max_jobs"]
            adzuna_max_days_old = state["adzuna_max_days_old"]

        job_stats = agent._fetch_new_jobs(state["keywords"], _ConfigProxy())
        emit_fn(
            run_id,
            f"Fetched {job_stats['stored']} new jobs "
            f"(skipped {job_stats['duplicates']} duplicates)"
        )

        return {"job_stats": job_stats}

    def node_rebuild_index(state: JobScoutState) -> dict:
        """Rebuild the FAISS vector index to include newly stored jobs."""
        run_id = state["run_id"]
        emit_fn(run_id, "Rebuilding job search index...")
        build_job_faiss_index()
        return {}  # no state change needed

    def node_find_matches(state: JobScoutState) -> dict:
        """Run FAISS + LLM hybrid matching and persist results to DB."""
        run_id = state["run_id"]
        emit_fn(run_id, "Analysing job matches against your resume...")

        matches = agent._find_and_save_matches(
            user_id=state["user_id"],
            resume_id=state["resume_id"],
            resume_text=state["resume_text"],
            resume_filename=state["resume_filename"],
            threshold=state["match_threshold"],
            max_results=state["max_results_per_run"],
            run_history_id=run_id,
            explicit_preferences=state.get("explicit_preferences"),
        )

        return {
            "matches_analyzed": matches["analyzed"],
            "matches_saved": matches["saved"],
        }

    def node_finalize(state: JobScoutState) -> dict:
        """Write final stats to AgentRunHistory, update config, and signal done."""
        db_run_id = state["run_id"]
        user_id = state["user_id"]
        job_stats = state.get("job_stats") or {}
        matches_saved = state.get("matches_saved") or []
        matches_analyzed = state.get("matches_analyzed") or []

        run_history = AgentRunHistory.query.get(db_run_id)
        if run_history:
            run_history.jobs_fetched = job_stats.get("stored", 0)
            run_history.jobs_analyzed = len(matches_analyzed)
            run_history.matches_found = len(matches_saved)
            run_history.status = "completed"
            run_history.completed_at = datetime.utcnow()
            run_history.results_summary = json.dumps({
                "jobs_fetched": job_stats.get("stored", 0),
                "jobs_analyzed": len(matches_analyzed),
                "matches_found": len(matches_saved),
                "top_match_score": matches_saved[0]["match_score"] if matches_saved else 0,
                "keywords_used": state.get("keywords", ""),
            })

        config = AgentConfig.query.filter_by(user_id=user_id).first()
        if config:
            config.last_run_at = datetime.utcnow()

        db.session.commit()

        emit_fn(
            db_run_id,
            f"Done! Found {len(matches_saved)} matches "
            f"from {len(matches_analyzed)} jobs analysed."
        )
        mark_done_fn(db_run_id)

        return {"status": "completed"}

    # ------------------------------------------------------------------ #
    # HELPER: update AgentRunHistory for early exits
    # ------------------------------------------------------------------ #

    def _fail_run(run_id: int, error_msg: str):
        run_history = AgentRunHistory.query.get(run_id)
        if run_history:
            run_history.status = "failed"
            run_history.completed_at = datetime.utcnow()
            run_history.error_message = error_msg
            db.session.commit()
        emit_fn(run_id, f"Error: {error_msg}")
        mark_done_fn(run_id)

    def _finish_run(run_id: int, status: str, summary: dict):
        run_history = AgentRunHistory.query.get(run_id)
        if run_history:
            run_history.status = status
            run_history.completed_at = datetime.utcnow()
            run_history.results_summary = json.dumps(summary)
            db.session.commit()
        mark_done_fn(run_id)

    # ------------------------------------------------------------------ #
    # ROUTING (conditional edges)
    # ------------------------------------------------------------------ #

    def route_after_load_user(state: JobScoutState) -> str:
        """Continue normally, or exit early if disabled/not found."""
        if state.get("status") in ("failed", "disabled"):
            return END
        return "load_resume"

    def route_after_load_resume(state: JobScoutState) -> str:
        """Continue normally, or exit early if resume is missing."""
        if state.get("status") == "failed":
            return END
        return "extract_keywords"

    def route_after_fetch_jobs(state: JobScoutState) -> str:
        """
        If new jobs were stored, rebuild the FAISS index first.
        Otherwise skip straight to matching (existing index is still valid).
        """
        job_stats = state.get("job_stats") or {}
        if job_stats.get("stored", 0) > 0:
            return "rebuild_index"
        return "find_matches"

    # ------------------------------------------------------------------ #
    # GRAPH ASSEMBLY
    # ------------------------------------------------------------------ #

    graph = StateGraph(JobScoutState)

    graph.add_node("load_user", node_load_user)
    graph.add_node("load_resume", node_load_resume)
    graph.add_node("extract_keywords", node_extract_keywords)
    graph.add_node("fetch_jobs", node_fetch_jobs)
    graph.add_node("rebuild_index", node_rebuild_index)
    graph.add_node("find_matches", node_find_matches)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("load_user")

    graph.add_conditional_edges(
        "load_user",
        route_after_load_user,
        {"load_resume": "load_resume", END: END},
    )
    graph.add_conditional_edges(
        "load_resume",
        route_after_load_resume,
        {"extract_keywords": "extract_keywords", END: END},
    )
    graph.add_edge("extract_keywords", "fetch_jobs")
    graph.add_conditional_edges(
        "fetch_jobs",
        route_after_fetch_jobs,
        {"rebuild_index": "rebuild_index", "find_matches": "find_matches"},
    )
    graph.add_edge("rebuild_index", "find_matches")
    graph.add_edge("find_matches", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
