"""
Chatbot Tools

LangChain tool definitions for the Career Coach agent.
Each tool runs inside an app context so DB queries are safe to call
from background threads.
"""

import json
import logging

import numpy as np
from langchain_core.tools import tool
from sqlalchemy.orm import joinedload

from chatbot.gated_actions import GATED_TOOL_NAMES
from models import AgentConfig, JobMatch, Resume, db
from services.db_lock import safe_commit
from services.job_service import find_matching_jobs
from services.pending_actions import propose
from services.rate_limit import allow, would_allow, PLAN_BUDGETS, SCOUT_BUDGETS
from services.resume_service import get_resume_text, perform_qa

logger = logging.getLogger(__name__)


def build_tools(app, user_id, *, surface: str, progress_cb=None):
    """Return the list of LangChain tools bound to app + user_id.

    surface: 'chat' or 'planner'. Keyword-only with no default on purpose —
        the planner runs in a background thread with no user present, so it
        must never receive a capability that requires confirmation. A caller
        who forgets this argument gets a TypeError instead of silently
        receiving the permissive set.

    progress_cb: optional callable(label: str) pushed mid-tool to update the UI.
    """
    def _progress(label: str):
        if progress_cb:
            progress_cb(label)

    # ── Seniority and job-type keyword maps used by the intent filter ──────────
    _SENIORITY_KEYWORDS = {
        "junior":  ["junior", "entry level", "entry-level", "jr.", "jr ", "associate", "graduate", "intern"],
        "mid":     ["mid ", "mid-level", "intermediate"],
        "senior":  ["senior", "sr.", "sr ", "experienced", "principal", "staff"],
        "lead":    ["lead", "principal", "staff", "head of", "director", "manager"],
    }
    _JOB_TYPE_KEYWORDS = {
        "part-time": ["part-time", "part time"],
        "contract":  ["contract", "freelance", "temporary", "temp "],
    }

    def _intent_multiplier(job, intent) -> float:
        """
        Returns a soft-filter multiplier in [0.41, 1.0] that down-ranks jobs
        which don't match the user's stated intent.  Each unmatched filter
        applies a ×0.8 penalty, so misses compound without hard-excluding jobs.
        """
        multiplier = 1.0
        full_text = (
            f"{job.title} {job.description or ''} {job.requirements or ''}"
        ).lower()
        loc_text = (job.location or "").lower()

        # Location
        if intent.location:
            loc = intent.location.lower()
            if loc == "remote":
                if "remote" not in loc_text and "remote" not in full_text[:600]:
                    multiplier *= 0.8
            elif loc not in loc_text:
                multiplier *= 0.8

        # Seniority
        if intent.seniority_level and intent.seniority_level != "any":
            title_lower = job.title.lower()
            level_words = _SENIORITY_KEYWORDS.get(intent.seniority_level, [])
            if level_words and not any(kw in title_lower for kw in level_words):
                multiplier *= 0.8

        # Job type (full-time is the default — no penalty for it)
        if intent.job_type and intent.job_type not in ("any", "full-time"):
            type_words = _JOB_TYPE_KEYWORDS.get(intent.job_type, [])
            if type_words and not any(kw in full_text for kw in type_words):
                multiplier *= 0.8

        # Keywords
        if intent.keywords:
            kw_lower = [kw.lower() for kw in intent.keywords]
            if not any(kw in full_text for kw in kw_lower):
                multiplier *= 0.8

        return multiplier

    @tool
    def find_top_jobs(query: str) -> str:
        """Find the top matching jobs for the user based on their resume and the chat request. Use this when the user asks to find jobs, get job recommendations, or match their resume to jobs. The query parameter is exactly what the user said — include location, seniority, job type, and count if they mentioned them."""
        with app.app_context():
            from jobs.utils import cosine_similarity
            from services.llm_service import run_job_search_planning

            resume = Resume.query.filter_by(
                user_id=user_id, is_active=True
            ).order_by(Resume.uploaded_at.desc()).first()

            if not resume:
                return json.dumps({"success": False, "error": "No resume found. Please upload a resume first."})

            try:
                # ── 1. Parse the user's query into structured intent ──────────
                _progress("Understanding your request…")
                intent = run_job_search_planning(query)
                logger.info(
                    "find_top_jobs intent — keywords=%s location=%s seniority=%s "
                    "job_type=%s top_k=%d",
                    intent.keywords, intent.location, intent.seniority_level,
                    intent.job_type, intent.top_k,
                )

                # ── 2. Build an augmented FAISS search query ─────────────────
                # Steer the vector search toward what the user asked for while
                # keeping the resume as the primary signal.
                _progress("Reading your resume…")
                resume_text = get_resume_text(resume)

                search_query = resume_text
                extras = []
                if intent.keywords:
                    extras.append(f"Skills and technologies: {', '.join(intent.keywords)}")
                if intent.location and intent.location.lower() == "remote":
                    extras.append("Preference: remote work")
                if intent.seniority_level and intent.seniority_level != "any":
                    extras.append(f"Seniority level: {intent.seniority_level}")
                if extras:
                    search_query += "\n\n" + "\n".join(extras)

                # ── 3. Expand candidate pool when filters will shrink it ──────
                active_filters = sum([
                    bool(intent.keywords),
                    intent.location is not None,
                    intent.seniority_level != "any",
                    intent.job_type not in ("any", "full-time"),
                ])
                candidate_k = 40 + active_filters * 20   # up to 120

                _progress("Searching job database…")
                # Fetch more candidates (top_k) than we'll return so the
                # intent filter has a large enough pool to work with.
                fetch_k = min(candidate_k, max(intent.top_k * 4, 20))
                candidates = find_matching_jobs(search_query, top_k=fetch_k, candidate_k=candidate_k)

                # ── 4. Hybrid scoring + intent soft-filter ────────────────────
                _progress("Ranking matches…")
                config = AgentConfig.query.filter_by(user_id=user_id).first()
                preference_vector = np.array(config.preference_embedding) if (config and config.preference_embedding is not None) else None
                using_preferences = preference_vector is not None

                scored = []
                for m in candidates:
                    job = m['job']
                    base_score = m['analysis'].get('match_score', 0)

                    if using_preferences and job.embedding is not None:
                        pref_similarity = cosine_similarity(preference_vector, np.array(job.embedding))
                        pref_score = (pref_similarity + 1) * 50
                        hybrid_score = 0.7 * base_score + 0.3 * pref_score
                    else:
                        hybrid_score = base_score

                    # Intent soft-filter multiplier
                    multiplier = _intent_multiplier(job, intent)
                    final_score = hybrid_score * multiplier

                    scored.append({**m, 'final_score': final_score, 'intent_multiplier': multiplier})

                scored.sort(key=lambda x: x['final_score'], reverse=True)
                top_matches = scored[:intent.top_k]

                # ── 5. Build response ─────────────────────────────────────────
                jobs, job_ids = [], []
                for m in top_matches:
                    job = m['job']
                    jobs.append({
                        "id": job.id,
                        "title": job.title,
                        "company": job.company,
                        "location": job.location or "",
                        "match_score": round(m['final_score'], 1),
                    })
                    job_ids.append(job.id)

                # Surface which filters were actually applied so the LLM can
                # craft a response that references them ("Here are 3 senior
                # remote Python roles matching your resume…").
                filters_applied = {}
                if intent.keywords:
                    filters_applied["keywords"] = intent.keywords
                if intent.location:
                    filters_applied["location"] = intent.location
                if intent.seniority_level != "any":
                    filters_applied["seniority"] = intent.seniority_level
                if intent.job_type not in ("any", "full-time"):
                    filters_applied["job_type"] = intent.job_type

                return json.dumps({
                    "success": True,
                    "jobs": jobs,
                    "action": "redirect_to_jobs",
                    "job_ids": job_ids,
                    "personalized": using_preferences,
                    "filters_applied": filters_applied,
                    "query_interpreted_as": intent.query_summary or query,
                    "count": len(jobs),
                })
            except Exception as e:
                logger.error("find_top_jobs error: %s", e)
                return json.dumps({"success": False, "error": str(e)})

    @tool
    def get_resume_info(question: str) -> str:
        """Answer questions about the user's resume, skills, experience, or qualifications. Use this when the user asks about their resume content, skills, strengths, or weaknesses."""
        with app.app_context():
            try:
                return perform_qa(question, user_id)
            except Exception as e:
                logger.error("get_resume_info error: %s", e)
                return f"Error querying resume: {str(e)}"

    @tool
    def trigger_job_scout_agent(reason: str) -> str:
        """Ask the user to confirm running the Job Scout Agent, which searches all enabled job sources for new matches. Use this when the user asks to run the agent, scan for new jobs, or do an automatic job search. This does NOT run the scout — it shows the user a confirmation button."""
        if surface != "chat":
            raise RuntimeError(
                f"trigger_job_scout_agent requires a confirmable surface; got {surface!r}"
            )

        if not would_allow("scout_manual", user_id, SCOUT_BUDGETS):
            return json.dumps({
                "success": False,
                "error": "You've reached the limit of 3 manual scout runs per hour. "
                         "The scheduled runs will keep finding matches in the meantime.",
            })

        label = "Run the Job Scout now?"
        nonce = propose(user_id, "trigger_job_scout_agent", {"reason": reason}, label)
        return json.dumps({
            "success": True,
            "action": "confirm_required",
            "nonce": nonce,
            "label": label,
            "note": "Confirmation required. The user has been shown a button. "
                    "Tell them to click it. Do not call this tool again.",
        })

    @tool
    def get_job_history(limit: int = 5) -> str:
        """Get the user's recent job matches AND the job preferences the AI has learned from their feedback. Use this when the user asks about their matches, previous results, match history, what you've learned about them, their preferences, or how personalization works for them."""
        with app.app_context():
            try:
                matches = (JobMatch.query
                           .options(joinedload(JobMatch.job))  # type: ignore[arg-type]
                           .filter_by(user_id=user_id)
                           .order_by(JobMatch.created_at.desc())
                           .limit(limit).all())

                config = AgentConfig.query.filter_by(user_id=user_id).first()
                liked = JobMatch.query.filter(
                    JobMatch.user_id == user_id,
                    JobMatch.user_feedback.in_(['interested', 'applied'])
                ).order_by(JobMatch.feedback_at.desc()).limit(10).all()
                disliked = JobMatch.query.filter_by(
                    user_id=user_id, user_feedback='not_interested'
                ).order_by(JobMatch.feedback_at.desc()).limit(5).all()

                has_preferences = config is not None and config.preference_embedding is not None

                return json.dumps({
                    "success": True,
                    "matches": [
                        {
                            "id": m.id,
                            "job_title": m.job.title if m.job else "Unknown",
                            "company": m.job.company if m.job else "Unknown",
                            "match_score": m.match_score,
                            "feedback": m.user_feedback,
                            "created_at": m.created_at.isoformat() if m.created_at else None,
                        }
                        for m in matches
                    ],
                    "personalization_active": has_preferences,
                    "liked_count": len(liked),
                    "disliked_count": len(disliked),
                    "liked_jobs": [
                        {"title": m.job.title, "company": m.job.company, "feedback": m.user_feedback}
                        for m in liked if m.job
                    ],
                    "disliked_jobs": [
                        {"title": m.job.title, "company": m.job.company}
                        for m in disliked if m.job
                    ],
                    "message": (
                        f"Personalization is active — learned from {len(liked)} liked "
                        f"and {len(disliked)} disliked jobs."
                        if has_preferences else
                        "No preferences learned yet. Rate matches as 'interested' or "
                        "'not interested' to enable personalized recommendations."
                    ),
                })
            except Exception as e:
                logger.error("get_job_history error: %s", e)
                return json.dumps({"success": False, "error": str(e)})

    @tool
    def search_job_by_title(title: str) -> str:
        """Search for jobs in the database by job title or role name. Use this FIRST when the user wants to tailor their resume to a specific job title, so you can get the job's full description and requirements. Returns a list of matching jobs with their IDs. Accepts formats like 'AI Developer', 'AI Developer at Intellivon', or just a company name."""
        with app.app_context():
            from models import JobPosting
            from sqlalchemy import or_, and_

            title_part = title.strip()
            company_part = None
            for sep in [' at ', ' @ ']:
                if sep in title.lower():
                    idx = title.lower().index(sep)
                    title_part = title[:idx].strip()
                    company_part = title[idx + len(sep):].strip()
                    break

            title_filter = or_(
                JobPosting.title.ilike(f'%{title_part}%'),
                JobPosting.description.ilike(f'%{title_part}%'),
            )

            if company_part:
                company_filter = JobPosting.company.ilike(f'%{company_part}%')
                jobs = JobPosting.query.filter(
                    JobPosting.is_active == True, and_(title_filter, company_filter)
                ).order_by(JobPosting.posted_date.desc()).limit(5).all()
                if not jobs:
                    jobs = JobPosting.query.filter(
                        JobPosting.is_active == True, title_filter
                    ).order_by(JobPosting.posted_date.desc()).limit(5).all()
            else:
                jobs = JobPosting.query.filter(
                    JobPosting.is_active == True, title_filter
                ).order_by(JobPosting.posted_date.desc()).limit(5).all()

            if not jobs:
                return json.dumps({
                    "success": False,
                    "error": f"No jobs matching '{title}' found. Ask the user to fetch jobs from the Jobs page first.",
                })

            return json.dumps({
                "success": True,
                "jobs": [
                    {"id": j.id, "title": j.title, "company": j.company or "Unknown", "location": j.location or ""}
                    for j in jobs
                ],
            })

    @tool
    def tailor_resume_to_job(job_id: int) -> str:
        """Tailor the user's resume to ATS-optimize it for a specific job posting. Returns keyword analysis, ATS score estimate (before/after), tailored resume sections (summary, skills, experience bullets), and formatting tips. Always call search_job_by_title first to get the job_id."""
        with app.app_context():
            from models import JobPosting
            from services.llm_service import run_resume_tailoring_structured

            job = JobPosting.query.get(job_id)
            if not job:
                return json.dumps({"success": False, "error": f"Job ID {job_id} not found."})

            resume = Resume.query.filter_by(
                user_id=user_id, is_active=True
            ).order_by(Resume.uploaded_at.desc()).first()
            if not resume:
                return json.dumps({"success": False, "error": "No resume found. Please upload a resume first."})

            try:
                _progress("Reading your resume…")
                resume_text = get_resume_text(resume)
                _progress("Analyzing job requirements…")
                # Resume Tailoring Agent — structured output, no JSON parsing needed
                tailoring = run_resume_tailoring_structured(
                    resume=resume_text[:4000],
                    job_title=job.title,
                    company=job.company or "the company",
                    job_description=(job.description or "")[:2000],
                    job_requirements=(job.requirements or "")[:1500],
                )
                _progress("Rewriting resume sections…")

                existing_match = JobMatch.query.filter_by(
                    user_id=user_id, resume_id=resume.id, job_id=job.id
                ).first()
                if existing_match:
                    existing_match.tailoring_result = tailoring.model_dump_json()
                    safe_commit()

                return json.dumps({
                    "success": True,
                    "action": "open_tailor_modal",
                    "job_id": job.id,
                    "job": {"id": job.id, "title": job.title, "company": job.company},
                    "ats_before": tailoring.ats_score.before,
                    "ats_after": tailoring.ats_score.after,
                })
            except Exception as e:
                logger.error("tailor_resume_to_job error: %s", e)
                return json.dumps({
                    "success": False,
                    "error": "Tailoring failed. The ATS results page will re-run it fresh.",
                    "action": "open_tailor_modal",
                    "job_id": job_id,
                    "job": {"id": job_id, "title": getattr(job, "title", ""), "company": getattr(job, "company", "")},
                })

    @tool
    def search_memory(query: str) -> str:
        """Search long-term memory of past conversations with this user. Use this when you need to recall something the user mentioned in a previous session — their career goals, stated preferences, past experience details, or decisions made. Formulate the query as a short description of what you want to recall, e.g. 'user remote work preference' or 'user past experience at previous company'."""
        with app.app_context():
            from chatbot.memory import search_memories
            try:
                results = search_memories(user_id, query, top_k=4)
                # Wrap in untrusted_data so the agent applies the same trust model
                # as it does for resume text and conversation_summary — memory content
                # originates from user messages and must never be treated as instructions.
                return (
                    f"<untrusted_data source=\"long_term_memory\">\n"
                    f"{results}\n"
                    f"</untrusted_data>"
                )
            except Exception as e:
                logger.error("search_memory error: %s", e)
                return "Memory search failed. Proceed without recalled context."

    @tool
    def create_career_plan(goal: str) -> str:
        """Create a multi-step career plan for a complex goal. Use this when the user asks for help with a big career objective that requires multiple actions — like transitioning to a new role, preparing for interviews at a specific company, or building a career roadmap. Do NOT use this for simple questions or single-step tasks. The goal parameter should be a clear description of what the user wants to achieve."""
        import threading
        with app.app_context():
            from chatbot.planner import generate_plan, execute_plan, get_active_plan, format_plan_status
            from services.llm_service import get_llm
            from models import PlanStep

            existing = get_active_plan(user_id)
            if existing:
                return json.dumps({
                    "success": False,
                    "error": "You already have an active plan. Complete or abandon it first.",
                    "current_plan": format_plan_status(existing),
                })

            if not allow("plan_create", user_id, PLAN_BUDGETS):
                return json.dumps({
                    "success": False,
                    "error": "You've reached the limit of 5 new career plans per day. "
                             "Your existing plan's roadmap is still available.",
                })

            try:
                _progress("Creating your career plan...")
                llm = get_llm()
                # generate_plan is fast (one LLM call) — run it synchronously so
                # we can return the plan outline to the user immediately.
                plan = generate_plan(app, user_id, goal, llm)

                # Execute the steps in a background thread so the HTTP request
                # is not blocked by the full Plan → Execute → Replan loop.
                _plan_id = plan.id  # capture plain int — don't close over the ORM object

                def _bg_execute(pid=_plan_id):
                    import traceback as _tb
                    logger.info("[create_career_plan] BG thread started for plan %d", pid)
                    try:
                        execute_plan(app, user_id, pid)
                        logger.info("[create_career_plan] BG thread finished for plan %d", pid)
                    except Exception as bg_exc:
                        logger.error(
                            "[create_career_plan] BG thread CRASHED for plan %d:\n%s",
                            pid, _tb.format_exc(),
                        )

                t = threading.Thread(target=_bg_execute, daemon=True, name=f"plan-{_plan_id}")
                t.start()
                logger.info("[create_career_plan] Background thread %s launched for plan %d", t.name, _plan_id)

                steps = PlanStep.query.filter_by(plan_id=plan.id).order_by(PlanStep.step_order).all()
                return json.dumps({
                    "success": True,
                    "plan_id": plan.id,
                    "goal": plan.goal,
                    "status": "running",
                    "message": (
                        "Your plan has been created and is now executing autonomously in the "
                        "background. Progress appears in your context at the start of each "
                        "turn — just ask and I'll tell you where it's up to."
                    ),
                    "steps": [
                        {"step_order": s.step_order, "description": s.description, "status": s.status}
                        for s in steps
                    ],
                })
            except Exception as e:
                logger.error("create_career_plan error: %s", e)
                return json.dumps({"success": False, "error": str(e)})

    @tool
    def abandon_career_plan(reason: str = "") -> str:
        """Ask the user to confirm abandoning their current active career plan. Use this when the user wants to cancel, restart, or change their career plan. This does NOT abandon the plan — it shows the user a confirmation button."""
        if surface != "chat":
            raise RuntimeError(
                f"abandon_career_plan requires a confirmable surface; got {surface!r}"
            )

        with app.app_context():
            from chatbot.planner import get_active_plan

            plan = get_active_plan(user_id)
            if not plan:
                return json.dumps({"success": False, "error": "No active plan to abandon."})

            # Resolve the arguments server-side and freeze them into the pending
            # action. The client only ever echoes the nonce back.
            label = f"Abandon your plan '{plan.goal}'?"
            nonce = propose(user_id, "abandon_career_plan", {"reason": reason}, label)
            return json.dumps({
                "success": True,
                "action": "confirm_required",
                "nonce": nonce,
                "label": label,
                "note": "Confirmation required. The user has been shown a button. "
                        "Tell them to click it. Do not call this tool again.",
            })

    all_tools = [find_top_jobs, get_resume_info, trigger_job_scout_agent, get_job_history,
                 search_job_by_title, tailor_resume_to_job, search_memory,
                 create_career_plan, abandon_career_plan]

    if surface == "chat":
        return all_tools
    return [t for t in all_tools if t.name not in GATED_TOOL_NAMES]
