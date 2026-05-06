"""
Chatbot Tools

LangChain tool definitions for the Career Coach agent.
Each tool runs inside an app context so DB queries are safe to call
from background threads.
"""

import json
import logging

from langchain_core.tools import tool
from sqlalchemy.orm import joinedload

from models import AgentConfig, JobMatch, Resume, db
from services.job_service import find_matching_jobs
from services.resume_service import get_resume_text, perform_qa

logger = logging.getLogger(__name__)


def build_tools(app, user_id):
    """Return the list of LangChain tools bound to app + user_id."""

    @tool
    def find_top_jobs(query: str) -> str:
        """Find the top 5 matching jobs for the user based on their resume. Use this when the user asks to find jobs, get job recommendations, or match their resume to jobs. The query parameter can be a description of what kind of jobs they want."""
        with app.app_context():
            from job_utils import cosine_similarity
            resume = Resume.query.filter_by(
                user_id=user_id, is_active=True
            ).order_by(Resume.uploaded_at.desc()).first()

            if not resume:
                return json.dumps({"success": False, "error": "No resume found. Please upload a resume first."})

            try:
                resume_text = get_resume_text(resume)
                candidates = find_matching_jobs(resume_text, top_k=20, candidate_k=40)

                config = AgentConfig.query.filter_by(user_id=user_id).first()
                preference_vector = config.preference_embedding if config else None
                using_preferences = preference_vector is not None

                scored = []
                for m in candidates:
                    job = m['job']
                    base_score = m['analysis'].get('match_score', 0)
                    if using_preferences and job.embedding is not None:
                        pref_similarity = cosine_similarity(preference_vector, job.embedding)
                        pref_score = (pref_similarity + 1) * 50
                        final_score = 0.7 * base_score + 0.3 * pref_score
                    else:
                        final_score = base_score
                    scored.append({**m, 'final_score': final_score})

                scored.sort(key=lambda x: x['final_score'], reverse=True)
                top_matches = scored[:5]

                jobs, job_ids = [], []
                for m in top_matches:
                    job = m['job']
                    jobs.append({"id": job.id, "title": job.title, "company": job.company,
                                 "match_score": round(m['final_score'], 1)})
                    job_ids.append(job.id)

                return json.dumps({"success": True, "jobs": jobs, "action": "redirect_to_jobs",
                                   "job_ids": job_ids, "personalized": using_preferences})
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
        """Trigger the Job Scout Agent to automatically search for new jobs and find matches. Use this when the user asks to run the agent, scan for new jobs, or do an automatic job search."""
        with app.app_context():
            from flask import current_app
            try:
                result = current_app.extensions['scheduler'].trigger_manual_run(user_id)
                return json.dumps({
                    "success": result['status'] == 'success',
                    "matches_found": result.get('matches_found', 0),
                    "jobs_analyzed": result.get('jobs_analyzed', 0),
                    "jobs_fetched": result.get('jobs_fetched', 0),
                })
            except Exception as e:
                logger.error("trigger_job_scout_agent error: %s", e)
                return json.dumps({"success": False, "error": str(e)})

    @tool
    def get_recent_matches(limit: int = 5) -> str:
        """Get the user's most recent job matches with scores and details. Use this when the user asks about their matches, previous results, or match history."""
        with app.app_context():
            try:
                matches = (JobMatch.query
                           .options(joinedload(JobMatch.job))
                           .filter_by(user_id=user_id)
                           .order_by(JobMatch.created_at.desc())
                           .limit(limit).all())

                if not matches:
                    return json.dumps({"success": True, "matches": [], "message": "No matches found yet."})

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
                })
            except Exception as e:
                logger.error("get_recent_matches error: %s", e)
                return json.dumps({"success": False, "error": str(e)})

    @tool
    def explain_feature(feature_name: str) -> str:
        """Explain how a feature of the AI Career Coach app works. Use this when the user asks about how things work, what a feature does, or needs help understanding the app. Valid features: resume_upload, job_matching, job_scout_agent, resume_qa, interview_roadmap, job_feedback, fetch_jobs, agent_config."""
        features = {
            "resume_upload": "Upload your PDF resume to get an AI-powered analysis. The system extracts text, creates a vector index for Q&A, and provides a comprehensive summary of your skills, experience, and career trajectory.",
            "job_matching": "The job matching system uses a two-stage approach: first, FAISS vector search finds the most relevant jobs quickly, then the LLM analyzes your resume against each job for detailed match scores, matched skills, skill gaps, and recommendations.",
            "job_scout_agent": "The Job Scout Agent is an autonomous agent that runs on a schedule (or manually). It fetches new jobs from Adzuna, analyzes them against your resume, and saves high-quality matches for you to review. Configure it from the Agent Dashboard.",
            "resume_qa": "Ask any question about your resume and get AI-powered answers. The system uses your resume's vector index to find relevant sections and provide accurate responses about your skills, experience, and qualifications.",
            "interview_roadmap": "Generate a personalized preparation roadmap for any job. It creates a phased plan with skills to learn, resources, projects, milestones, and progressive interview questions tailored to your skill gaps.",
            "job_feedback": "Provide feedback on job matches (interested, not interested, applied) to help the system learn your preferences. Over time, the agent learns to find better matches based on your feedback patterns.",
            "fetch_jobs": "Fetch real job postings from the Adzuna API. You can filter by keywords, location, and job age. Fetched jobs are stored in the database and available for matching.",
            "agent_config": "Configure the Job Scout Agent's behavior: schedule time, timezone, match threshold (minimum score to save), max results per run, and Adzuna search preferences (location, max jobs, max age).",
            "resume_tailoring": "ATS-optimize your resume for a specific job. The system searches the job database for the target role, then uses an LLM to analyze keyword gaps, rewrite your Professional Summary, reorder your Skills section, and reframe up to 5 experience bullets using the job's language.",
        }
        result = features.get(feature_name.lower().strip(),
                              f"Unknown feature: '{feature_name}'. Available: {', '.join(features.keys())}")
        return result

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
                resume_text = get_resume_text(resume)
                # Resume Tailoring Agent — structured output, no JSON parsing needed
                tailoring = run_resume_tailoring_structured(
                    resume=resume_text[:4000],
                    job_title=job.title,
                    company=job.company or "the company",
                    job_description=(job.description or "")[:2000],
                    job_requirements=(job.requirements or "")[:1500],
                )

                existing_match = JobMatch.query.filter_by(
                    user_id=user_id, resume_id=resume.id, job_id=job.id
                ).first()
                if existing_match:
                    existing_match.tailoring_result = tailoring.model_dump_json()
                    db.session.commit()

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
    def get_user_preferences(dummy: str = "") -> str:
        """Show what job preferences the AI has learned from the user's feedback history. Use this when the user asks what you've learned about them, what their preferences are, or how personalization works for them."""
        with app.app_context():
            try:
                config = AgentConfig.query.filter_by(user_id=user_id).first()
                liked = JobMatch.query.filter(
                    JobMatch.user_id == user_id,
                    JobMatch.user_feedback.in_(['interested', 'applied'])
                ).order_by(JobMatch.feedback_at.desc()).limit(10).all()
                disliked = JobMatch.query.filter_by(
                    user_id=user_id, user_feedback='not_interested'
                ).order_by(JobMatch.feedback_at.desc()).limit(5).all()

                has_preferences = config and config.preference_embedding is not None
                return json.dumps({
                    "success": True,
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
                        f"Personalization is active. I've learned your preferences from "
                        f"{len(liked)} jobs you liked and {len(disliked)} you disliked."
                        if has_preferences else
                        "No preferences learned yet. Rate some job matches as 'interested' or "
                        "'not interested' to enable personalized recommendations."
                    ),
                })
            except Exception as e:
                logger.error("get_user_preferences error: %s", e)
                return json.dumps({"success": False, "error": str(e)})

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

    return [find_top_jobs, get_resume_info, trigger_job_scout_agent, get_recent_matches,
            explain_feature, search_job_by_title, tailor_resume_to_job, get_user_preferences,
            search_memory]
