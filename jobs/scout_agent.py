"""
Job Scout Agent - Autonomous AI agent for job searching and matching

This agent runs autonomously to:
1. Fetch new jobs from external sources (Adzuna API)
2. Analyze jobs against user's resume
3. Find and save high-quality matches
4. Learn from user feedback
"""

import os
import json
import sqlite3
import threading
from datetime import datetime
from models import db, User, Resume, JobPosting, JobMatch, AgentConfig, AgentRunHistory
from services.db_lock import safe_commit
from jobs.utils import get_job_faiss_index, build_job_faiss_index, cosine_similarity
from pypdf import PdfReader
import numpy as np
from jobs.scout_graph import build_job_scout_graph

# ---------------------------------------------------------------------------
# Checkpointer — single instance per process, initialised lazily on first use.
# Dev: SQLite file.  Prod: PostgreSQL (set CHECKPOINT_DB_PATH to a psycopg3 DSN).
# ---------------------------------------------------------------------------
_checkpointer = None
_checkpointer_lock = threading.Lock()


def _get_checkpointer(app):
    """Return (and lazily create) the process-level LangGraph checkpointer."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    with _checkpointer_lock:
        if _checkpointer is not None:
            return _checkpointer

        path_or_dsn = app.config.get('CHECKPOINT_DB_PATH', '')

        if path_or_dsn.startswith('postgresql'):
            from langgraph.checkpoint.postgres import PostgresSaver
            # Strip SQLAlchemy dialect prefix if present: postgresql+psycopg2:// → postgresql://
            dsn = path_or_dsn
            if '+' in dsn.split('://')[0]:
                scheme, rest = dsn.split('://', 1)
                dsn = f"{scheme.split('+')[0]}://{rest}"
            cp = PostgresSaver.from_conn_string(dsn)
            cp.setup()  # creates checkpoint tables once (idempotent)
        else:
            from langgraph.checkpoint.sqlite import SqliteSaver
            os.makedirs(os.path.dirname(path_or_dsn), exist_ok=True)
            conn = sqlite3.connect(path_or_dsn, check_same_thread=False)
            cp = SqliteSaver(conn)

        _checkpointer = cp
        return _checkpointer


# ---------------------------------------------------------------------------
# In-memory progress store
# Each manual run writes events here; the SSE endpoint reads them.
# Key: run_id (int)  Value: {"events": [...], "done": bool}
# ---------------------------------------------------------------------------
_run_progress: dict = {}
_progress_lock = threading.Lock()


def _init_run_progress(run_id: int):
    with _progress_lock:
        _run_progress[run_id] = {"events": [], "done": False}


def _emit_progress(run_id: int, message: str):
    with _progress_lock:
        if run_id in _run_progress:
            _run_progress[run_id]["events"].append({
                "message": message,
                "ts": datetime.utcnow().strftime("%H:%M:%S")
            })


def _mark_done(run_id: int):
    with _progress_lock:
        if run_id in _run_progress:
            _run_progress[run_id]["done"] = True


def get_run_events(run_id: int, since: int = 0):
    """Return (new_events_list, is_done). Thread-safe."""
    with _progress_lock:
        store = _run_progress.get(run_id)
        if not store:
            return [], True
        return store["events"][since:], store["done"]


def cleanup_run_progress(run_id: int):
    with _progress_lock:
        _run_progress.pop(run_id, None)


class JobScoutAgent:
    """
    Autonomous agent that scouts for jobs and matches them to users

    The agent demonstrates agentic AI behavior by:
    - Autonomous decision-making (what jobs to fetch, what to analyze)
    - Tool use (Adzuna API, FAISS search, LLM analysis)
    - Goal-oriented behavior (find best matches for user)
    - Learning from feedback (adjusts based on user preferences)
    """

    def __init__(self, app):
        self.app = app
        # Specialist agents (JobAnalystAgent, KeywordResearchAgent) are accessed
        # via the coordinator inside each method — no local LLM needed here.
        checkpointer = _get_checkpointer(app)
        self.graph = build_job_scout_graph(self, _emit_progress, _mark_done, checkpointer)

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF resume"""
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def run_for_user(self, user_id, run_type='manual', existing_run_id=None):
        """
        Run the job scout agent for a specific user via the LangGraph pipeline.

        The graph drives the full flow autonomously:
          load_user → load_resume → extract_keywords → fetch_jobs
          → [rebuild_index] → find_matches → finalize

        Conditional edges skip the index rebuild when no new jobs were fetched,
        and exit early when the agent is disabled or no resume is found.

        Args:
            user_id: User ID to run agent for
            run_type: 'manual' or 'scheduled'
            existing_run_id: If provided, update this AgentRunHistory record
                             instead of creating a new one (used by async trigger)

        Returns:
            dict: Results summary with stats
        """
        with self.app.app_context():
            # Create or fetch the AgentRunHistory record
            if existing_run_id:
                run_history = AgentRunHistory.query.get(existing_run_id)
                if not run_history:
                    return {'status': 'failed', 'error': 'Run record not found'}
            else:
                run_history = AgentRunHistory(
                    user_id=user_id,
                    run_type=run_type,
                    status='running',
                    started_at=datetime.utcnow()
                )
                db.session.add(run_history)
                safe_commit()

            run_id = run_history.id
            _init_run_progress(run_id)
            _emit_progress(run_id, "Job Scout Agent starting...")

            # Build the initial state and invoke the LangGraph pipeline
            initial_state = {
                "user_id": user_id,
                "run_id": run_id,
                "run_type": run_type,
                # optional fields initialised to None
                "is_enabled": None,
                "match_threshold": None,
                "max_results_per_run": None,
                "adzuna_location": None,
                "adzuna_max_jobs": None,
                "adzuna_max_days_old": None,
                "enabled_sources": None,
                "resume_id": None,
                "resume_filename": None,
                "resume_text": None,
                "keywords": None,
                "job_stats": None,
                "matches_analyzed": None,
                "matches_saved": None,
                "explicit_preferences": None,
                "status": "running",
                "error": None,
            }

            # thread_id uniquely identifies this run to the checkpointer.
            # If the process crashes mid-run, invoking with the same thread_id
            # and initial_state=None resumes from the last completed node.
            invoke_config = {"configurable": {"thread_id": f"run-{run_id}"}}

            try:
                final_state = self.graph.invoke(initial_state, config=invoke_config)
            except Exception as e:
                # Unexpected graph-level exception
                run_history.status = 'failed'
                run_history.completed_at = datetime.utcnow()
                run_history.error_message = str(e)
                safe_commit()

                _emit_progress(run_id, f"Error: {str(e)}")
                _mark_done(run_id)

                return {'status': 'failed', 'error': str(e), 'run_id': run_id}

            # Surface the final state as the return value
            if final_state.get("status") == "failed":
                return {
                    'status': 'failed',
                    'error': final_state.get("error", "Unknown error"),
                    'run_id': run_id,
                }
            if final_state.get("status") == "disabled":
                return {'status': 'disabled', 'message': 'Agent is disabled'}

            matches_saved = final_state.get("matches_saved") or []
            job_stats = final_state.get("job_stats") or {}
            return {
                'status': 'success',
                'run_id': run_id,
                'jobs_fetched': job_stats.get('stored', 0),
                'jobs_analyzed': len(final_state.get("matches_analyzed") or []),
                'matches_found': len(matches_saved),
                'matches': matches_saved,
            }

    def _extract_keywords_from_resume(self, resume_text, explicit_preferences=None):
        """
        Keyword Research Agent: extract job search titles from resume.
        Uses structured output — no free-text parsing, injection stays in typed fields.
        explicit_preferences (from chat) constrain the search when provided.
        """
        try:
            from services.llm_service import run_keyword_extraction
            result = run_keyword_extraction(resume_text, explicit_preferences)
            keywords = ", ".join(result.job_titles)
            return keywords if keywords else "software engineer"
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return "software engineer"

    def _fetch_new_jobs(self, keywords, config):
        """Fetch new jobs from all sources enabled in the user's config."""
        from jobs.fetchers.registry import fetch_from_sources
        from jobs.fetcher import fetch_jobs_from_adzuna

        location = config.adzuna_location
        max_jobs = config.adzuna_max_jobs or 20
        max_days_old = config.adzuna_max_days_old or 30
        sources = config.enabled_sources or ['adzuna']

        print(f"[DEBUG] Fetching jobs — keywords: {keywords}, location: {location}, "
              f"max_jobs: {max_jobs}, max_days_old: {max_days_old}, sources: {sources}")

        try:
            if sources == ['adzuna']:
                stats = fetch_jobs_from_adzuna(
                    keywords=keywords,
                    location=location,
                    max_jobs=max_jobs,
                    max_days_old=max_days_old,
                )
            else:
                result = fetch_from_sources(
                    sources=sources,
                    keywords=keywords,
                    location=location,
                    max_jobs_per_source=max_jobs,
                    max_days_old=max_days_old,
                )
                stats = result['totals']

            print(f"[DEBUG] Fetch stats — fetched: {stats.get('fetched')}, stored: {stats.get('stored')}, "
                  f"duplicates: {stats.get('duplicates')}, errors: {stats.get('errors')}")
            return stats

        except Exception as e:
            print(f"Error fetching jobs: {e}")
            return {'fetched': 0, 'stored': 0, 'duplicates': 0, 'errors': 1, 'error_messages': [str(e)]}

    def _find_and_save_matches(self, user_id, resume_id, resume_text, resume_filename,
                               threshold, max_results, run_history_id, explicit_preferences=None):
        """
        AUTONOMOUS ANALYSIS: Find matches and decide which to save

        Agent analyzes jobs, evaluates matches, and decides which are worth showing to user.
        Uses hybrid scoring: 70% resume match + 30% user preference match (if available)
        """
        matches_analyzed = []
        matches_saved = []

        try:
            # Get job index
            job_index = get_job_faiss_index()
            if job_index is None:
                return {'analyzed': [], 'saved': []}

            # Get user's learned preferences
            user_config = AgentConfig.query.filter_by(user_id=user_id).first()
            user_preference_vector = np.array(user_config.preference_embedding) if (user_config and user_config.preference_embedding is not None) else None

            # Track if we're using preference-based personalization
            using_preferences = user_preference_vector is not None
            if using_preferences:
                print(f"Using personalized matching for user {user_id} (preferences learned from feedback)")
            else:
                print(f"Using resume-only matching for user {user_id} (no feedback history yet)")

            # Search for similar jobs using FAISS (Stage 1: Fast retrieval)
            docs_with_scores = job_index.similarity_search_with_score(
                resume_text,
                k=min(20, job_index.index.ntotal)  # Get top 20 candidates
            )

            if not docs_with_scores:
                return {'analyzed': [], 'saved': []}

            # Stage 2: Detailed LLM analysis for candidates
            candidates = docs_with_scores[:max_results * 2]
            total_candidates = len(candidates)
            for idx, (doc, distance) in enumerate(candidates, 1):
                job_id = doc.metadata.get("job_id")
                job = JobPosting.query.get(job_id)

                if not job or not job.is_active:
                    continue

                _emit_progress(run_history_id, f"Analysing job {idx}/{total_candidates}: {job.title} at {job.company}")

                # Check if user already has this match
                existing_match = JobMatch.query.filter_by(
                    user_id=user_id,
                    job_id=job_id,
                    resume_id=resume_id
                ).first()

                if existing_match:
                    continue  # Skip duplicates

                # Job Analyst Agent: structured output, no json.loads needed
                try:
                    resume_input = resume_text[:3000]
                    if explicit_preferences:
                        pref_summary = explicit_preferences.get("summary", "")
                        if pref_summary:
                            resume_input += f"\n\nUser preference context: {pref_summary}"

                    from services.llm_service import run_job_matching
                    analysis = run_job_matching(
                        resume=resume_input,
                        job_title=job.title,
                        company=job.company,
                        job_description=job.description[:1000],
                        job_requirements=job.requirements[:1000] if job.requirements else "Not specified",
                    )
                    # analysis is a JobMatchResult — typed fields, no parsing risk

                    base_match_score = float(analysis.match_score)

                    # Hybrid scoring: 70% resume match + 30% preference match
                    if using_preferences and job.embedding is not None:
                        preference_similarity = cosine_similarity(user_preference_vector, np.array(job.embedding))
                        preference_score = float((preference_similarity + 1) * 50)
                        final_score = float(0.7 * base_match_score + 0.3 * preference_score)
                        print(f"Job {job.id}: Resume={base_match_score:.1f}, Pref={preference_score:.1f}, Final={final_score:.1f}")
                    else:
                        final_score = base_match_score

                    matches_analyzed.append({
                        'job_id': job.id,
                        'match_score': final_score,
                        'base_score': base_match_score,
                        'preference_adjusted': using_preferences,
                    })

                    if final_score >= threshold:
                        job_match = JobMatch(
                            user_id=user_id,
                            resume_id=resume_id,
                            resume_filename=resume_filename,
                            job_id=job.id,
                            match_score=float(final_score),
                            matched_skills=json.dumps(analysis.matched_skills),
                            gaps=json.dumps(analysis.skill_gaps),
                            recommendation=analysis.recommendation,
                            agent_generated=True,
                            agent_run_id=run_history_id,
                            created_at=datetime.utcnow()
                        )
                        db.session.add(job_match)
                        matches_saved.append({
                            'job_id': job.id,
                            'job_title': job.title,
                            'company': job.company,
                            'match_score': float(final_score),
                            'matched_skills': analysis.matched_skills,
                            'skill_gaps': analysis.skill_gaps,
                        })

                        if len(matches_saved) >= max_results:
                            break

                except Exception as e:
                    print(f"Error analyzing job {job.id}: {e}")
                    continue

            safe_commit()

        except Exception as e:
            print(f"Error in find_and_save_matches: {e}")
            db.session.rollback()

        return {
            'analyzed': matches_analyzed,
            'saved': matches_saved
        }

    def run_for_all_users(self, run_type='scheduled'):
        """
        Run agent for all users with enabled agents

        This would be called by the scheduler daily
        """
        with self.app.app_context():
            # Get all users with enabled agents
            configs = AgentConfig.query.filter_by(is_enabled=True).all()

            results = []
            for config in configs:
                result = self.run_for_user(config.user_id, run_type=run_type)
                results.append({
                    'user_id': config.user_id,
                    'result': result
                })

            return results