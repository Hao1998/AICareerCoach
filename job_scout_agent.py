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
import threading
from datetime import datetime
from models import db, User, Resume, JobPosting, JobMatch, AgentConfig, AgentRunHistory
from job_fetcher import AdzunaJobFetcher
from job_utils import get_job_faiss_index, build_job_faiss_index, cosine_similarity
from langchain.prompts import PromptTemplate
from langchain_xai import ChatXAI
import PyPDF2
import numpy as np
from job_scout_graph import build_job_scout_graph

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
        """
        Initialize the Job Scout Agent

        Args:
            app: Flask application instance
        """
        self.app = app

        # Initialize LLM for job analysis
        xai_api_key = os.getenv("XAI_API_KEY")
        if not xai_api_key:
            raise RuntimeError("XAI_API_KEY environment variable is not set")

        self.llm = ChatXAI(
            xai_api_key=xai_api_key,
            model="grok-3",
            temperature=0.7,
        )

        # Build the LangGraph pipeline (nodes close over `self`)
        self.graph = build_job_scout_graph(self, _emit_progress, _mark_done)

        # Job matching prompt template
        self.matching_prompt = PromptTemplate(
            input_variables=["resume", "job_title", "company", "job_description", "job_requirements"],
            template="""You are an expert career coach analyzing if a job matches a candidate's resume.
 
Resume Summary:
{resume}
 
Job Details:
Title: {job_title}
Company: {company}
Description: {job_description}
Requirements: {job_requirements}
 
Analyze the match and provide a JSON response with:
1. match_score: 0-100 (higher is better match)
2. matched_skills: List of candidate's skills that match job requirements
3. skill_gaps: List of skills the candidate needs to develop
4. recommendation: Brief personalized recommendation (2-3 sentences)
 
Return ONLY valid JSON in this format:
{{"match_score": 85, "matched_skills": ["Python", "SQL"], "skill_gaps": ["AWS", "Docker"], "recommendation": "Strong match..."}}
"""
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF resume"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
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
                db.session.commit()

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
                "resume_id": None,
                "resume_filename": None,
                "resume_text": None,
                "keywords": None,
                "job_stats": None,
                "matches_analyzed": None,
                "matches_saved": None,
                "status": "running",
                "error": None,
            }

            try:
                final_state = self.graph.invoke(initial_state)
            except Exception as e:
                # Unexpected graph-level exception
                run_history.status = 'failed'
                run_history.completed_at = datetime.utcnow()
                run_history.error_message = str(e)
                db.session.commit()

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

    def _extract_keywords_from_resume(self, resume_text):
        """
        AUTONOMOUS DECISION: Extract job search keywords from resume

        Agent analyzes resume and decides what jobs to search for
        """
        try:
            # Use LLM to extract key job titles/roles from resume
            prompt = f"""Analyze this resume and extract 2-3 key job titles or roles the person is qualified for.
Return only the job titles, comma-separated, no extra text.
 
Resume:
{resume_text[:1500]}
 
Job titles:"""

            response = self.llm.invoke(prompt)
            keywords = response.content.strip()

            # If LLM fails, fall back to generic search
            if not keywords or len(keywords) > 100:
                keywords = "software engineer"

            return keywords

        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return "software engineer"  # Safe fallback

    def _fetch_new_jobs(self, keywords, config):
        """
        TOOL USE: Fetch new jobs from external API

        Agent uses Adzuna API to get fresh job postings
        Uses user-specific preferences for location and max_jobs
        """
        try:
            fetcher = AdzunaJobFetcher()

            # Get user-specific preferences from config
            location = config.adzuna_location  # User's preferred location
            max_jobs = config.adzuna_max_jobs if config.adzuna_max_jobs else 20  # Default to 20 if not set
            max_days_old = config.adzuna_max_days_old if config.adzuna_max_days_old else 30  # Default to 30 if not set

            # Fetch recent jobs using user preferences
            print(f"[DEBUG] Fetching jobs — keywords: {keywords}, location: {location}, max_jobs: {max_jobs}, max_days_old: {max_days_old}")
            stats = fetcher.fetch_and_store_jobs(
                keywords=keywords,
                location=location,
                max_jobs=max_jobs,
                max_days_old=max_days_old,
                skip_duplicates=True
            )

            print(f"[DEBUG] Fetch stats — fetched: {stats.get('fetched')}, stored: {stats.get('stored')}, duplicates: {stats.get('duplicates')}, errors: {stats.get('errors')}, error_messages: {stats.get('error_messages')}")
            return stats

        except Exception as e:
            print(f"Error fetching jobs: {e}")
            return {
                'fetched': 0,
                'stored': 0,
                'duplicates': 0,
                'errors': 1,
                'error_messages': [str(e)]
            }

    def _find_and_save_matches(self, user_id, resume_id, resume_text, resume_filename,
                               threshold, max_results, run_history_id):
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
            user_preference_vector = user_config.preference_embedding if user_config else None

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

                # AUTONOMOUS DECISION: Analyze job match using LLM
                try:
                    analysis_result = self.llm.invoke(
                        self.matching_prompt.format(
                            resume=resume_text[:3000],
                            job_title=job.title,
                            company=job.company,
                            job_description=job.description[:1000],
                            job_requirements=job.requirements[:1000] if job.requirements else "Not specified"
                        )
                    )

                    # Parse JSON response
                    analysis = json.loads(analysis_result.content)

                    # Get base match score from LLM analysis
                    base_match_score = analysis['match_score']

                    # HYBRID SCORING: Combine resume match with user preferences
                    if using_preferences and job.embedding is not None:
                        # Calculate preference similarity (how well job matches user's learned preferences)
                        preference_similarity = cosine_similarity(user_preference_vector, job.embedding)
                        # Convert to 0-100 scale (cosine similarity is -1 to 1)
                        preference_score = (preference_similarity + 1) * 50

                        # Weighted combination: 70% resume match + 30% preference match
                        final_score = 0.7 * base_match_score + 0.3 * preference_score

                        print(f"Job {job.id}: Resume match={base_match_score:.1f}, Preference match={preference_score:.1f}, Final={final_score:.1f}")
                    else:
                        # No preferences yet - use resume match only
                        final_score = base_match_score

                    matches_analyzed.append({
                        'job_id': job.id,
                        'match_score': final_score,
                        'base_score': base_match_score,
                        'preference_adjusted': using_preferences
                    })

                    # AUTONOMOUS DECISION: Only save matches above threshold (using final score)
                    if final_score >= threshold:
                        # Save match to database (with preference-adjusted score)
                        job_match = JobMatch(
                            user_id=user_id,
                            resume_id=resume_id,
                            resume_filename=resume_filename,
                            job_id=job.id,
                            match_score=final_score,  # Use hybrid score
                            matched_skills=json.dumps(analysis.get('matched_skills', [])),
                            gaps=json.dumps(analysis.get('skill_gaps', [])),
                            recommendation=analysis.get('recommendation', ''),
                            agent_generated=True,
                            agent_run_id=run_history_id,
                            created_at=datetime.utcnow()
                        )

                        db.session.add(job_match)
                        matches_saved.append({
                            'job_id': job.id,
                            'job_title': job.title,
                            'company': job.company,
                            'match_score': final_score,  # Use hybrid score
                            'matched_skills': analysis.get('matched_skills', []),
                            'skill_gaps': analysis.get('skill_gaps', [])
                        })

                        # Stop if we have enough good matches
                        if len(matches_saved) >= max_results:
                            break

                except json.JSONDecodeError:
                    print(f"Failed to parse LLM response for job {job.id}")
                    continue
                except Exception as e:
                    print(f"Error analyzing job {job.id}: {e}")
                    continue

            db.session.commit()

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