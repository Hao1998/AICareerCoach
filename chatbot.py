"""
Career Coach AI Chatbot Module

LangChain tool-calling agent with 2-tier memory:
- Tier 1 (Hot): Last 10 raw messages from DB
- Tier 2 (Warm): Rolling LLM-generated conversation summary in AgentConfig
"""

import json
import logging
from datetime import datetime, timedelta

# Clean top-level imports — no circular dependency now that services layer exists
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from models import ChatMessage, AgentConfig, Resume, User, db
from services.llm_service import get_llm, get_resume_tailoring_chain
from services.resume_service import perform_qa, extract_text_from_pdf
from services.job_service import find_matching_jobs

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_MINUTES = 5

_UNICODE_TO_ASCII = {
    '\u201c': '"', '\u201d': '"',  # " "  smart double quotes
    '\u2018': "'", '\u2019': "'",  # ' '  smart single quotes
    '\u2014': '--',                 # —   em dash
    '\u2013': '-',                  # –   en dash
    '\u2026': '...',                # …   ellipsis
    '\u00a0': ' ',                  # non-breaking space
}


def _sanitize(text: str) -> str:
    """Replace common non-ASCII typographic characters with ASCII equivalents."""
    for char, replacement in _UNICODE_TO_ASCII.items():
        text = text.replace(char, replacement)
    return text


def build_tools(app, user_id):
    """Build LangChain tools for the career coach agent"""

    @tool
    def find_top_jobs(query: str) -> str:
        """Find the top 5 matching jobs for the user based on their resume. Use this when the user asks to find jobs, get job recommendations, or match their resume to jobs. The query parameter can be a description of what kind of jobs they want."""
        with app.app_context():
            resume = Resume.query.filter_by(
                user_id=user_id, is_active=True
            ).order_by(Resume.uploaded_at.desc()).first()

            if not resume:
                return json.dumps({"success": False, "error": "No resume found. Please upload a resume first."})

            try:
                resume_text = extract_text_from_pdf(resume.file_path)
                matches = find_matching_jobs(resume_text, top_k=5)

                jobs = []
                job_ids = []
                for m in matches:
                    job = m['job']
                    jobs.append({
                        "id": job.id,
                        "title": job.title,
                        "company": job.company,
                        "match_score": m['analysis'].get('match_score', 0)
                    })
                    job_ids.append(job.id)

                return json.dumps({
                    "success": True,
                    "jobs": jobs,
                    "action": "redirect_to_jobs",
                    "job_ids": job_ids
                })
            except Exception as e:
                logger.error(f"find_top_jobs error: {e}")
                return json.dumps({"success": False, "error": str(e)})

    @tool
    def get_resume_info(question: str) -> str:
        """Answer questions about the user's resume, skills, experience, or qualifications. Use this when the user asks about their resume content, skills, strengths, or weaknesses."""
        with app.app_context():
            try:
                result = perform_qa(question, user_id)
                return result
            except Exception as e:
                logger.error(f"get_resume_info error: {e}")
                return f"Error querying resume: {str(e)}"

    @tool
    def trigger_job_scout_agent(reason: str) -> str:
        """Trigger the Job Scout Agent to automatically search for new jobs and find matches. Use this when the user asks to run the agent, scan for new jobs, or do an automatic job search."""
        with app.app_context():
            from flask import current_app
            try:
                agent_scheduler = current_app.extensions['scheduler']
                result = agent_scheduler.trigger_manual_run(user_id)
                return json.dumps({
                    "success": result['status'] == 'success',
                    "matches_found": result.get('matches_found', 0),
                    "jobs_analyzed": result.get('jobs_analyzed', 0),
                    "jobs_fetched": result.get('jobs_fetched', 0)
                })
            except Exception as e:
                logger.error(f"trigger_job_scout_agent error: {e}")
                return json.dumps({"success": False, "error": str(e)})

    @tool
    def get_recent_matches(limit: int = 5) -> str:
        """Get the user's most recent job matches with scores and details. Use this when the user asks about their matches, previous results, or match history."""
        with app.app_context():
            from models import JobMatch
            try:
                matches = JobMatch.query.filter_by(
                    user_id=user_id
                ).order_by(JobMatch.created_at.desc()).limit(limit).all()

                if not matches:
                    return json.dumps({"success": True, "matches": [], "message": "No matches found yet."})

                results = []
                for m in matches:
                    results.append({
                        "id": m.id,
                        "job_title": m.job.title if m.job else "Unknown",
                        "company": m.job.company if m.job else "Unknown",
                        "match_score": m.match_score,
                        "feedback": m.user_feedback,
                        "created_at": m.created_at.isoformat() if m.created_at else None
                    })

                return json.dumps({"success": True, "matches": results})
            except Exception as e:
                logger.error(f"get_recent_matches error: {e}")
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
            "resume_tailoring": "ATS-optimize your resume for a specific job. The system searches the job database for the target role, then uses an LLM to analyze keyword gaps, rewrite your Professional Summary, reorder your Skills section, and reframe up to 5 experience bullets using the job's language. It also estimates your ATS keyword match score before and after the changes."
        }
        result = features.get(feature_name.lower().strip(),
                              f"Unknown feature: '{feature_name}'. Available features: {', '.join(features.keys())}")
        return result

    @tool
    def search_job_by_title(title: str) -> str:
        """Search for jobs in the database by job title or role name. Use this FIRST when the user wants to tailor their resume to a specific job title, so you can get the job's full description and requirements. Returns a list of matching jobs with their IDs."""
        with app.app_context():
            from models import JobPosting
            from sqlalchemy import or_
            jobs = JobPosting.query.filter(
                JobPosting.is_active == True,
                or_(
                    JobPosting.title.ilike(f'%{title}%'),
                    JobPosting.description.ilike(f'%{title}%')
                )
            ).order_by(JobPosting.posted_date.desc()).limit(5).all()

            if not jobs:
                return json.dumps({
                    "success": False,
                    "error": f"No jobs matching '{title}' found in the database. Ask the user to fetch jobs from the Jobs page first, or ask them to paste the job description directly."
                })

            return json.dumps({
                "success": True,
                "jobs": [
                    {"id": j.id, "title": j.title, "company": j.company or "Unknown", "location": j.location or ""}
                    for j in jobs
                ]
            })

    @tool
    def tailor_resume_to_job(job_id: int) -> str:
        """Tailor the user's resume to ATS-optimize it for a specific job posting. Returns keyword analysis, ATS score estimate (before/after), tailored resume sections (summary, skills, experience bullets), and formatting tips. Always call search_job_by_title first to get the job_id."""
        with app.app_context():
            from models import JobPosting
            job = JobPosting.query.get(job_id)
            if not job:
                return json.dumps({"success": False, "error": f"Job ID {job_id} not found."})

            resume = Resume.query.filter_by(
                user_id=user_id, is_active=True
            ).order_by(Resume.uploaded_at.desc()).first()
            if not resume:
                return json.dumps({"success": False, "error": "No resume found. Please upload a resume first."})

            try:
                resume_text = extract_text_from_pdf(resume.file_path)
                chain = get_resume_tailoring_chain()
                result = chain.invoke({
                    "resume": resume_text[:4000],
                    "job_title": job.title,
                    "company": job.company or "the company",
                    "job_description": (job.description or "")[:2000],
                    "job_requirements": (job.requirements or "")[:1500],
                })
                raw = result.get('text', str(result))
                # Strip markdown code fences if the model wraps the JSON
                raw = raw.strip()
                if raw.startswith('```'):
                    raw = raw.split('```')[1]
                    if raw.startswith('json'):
                        raw = raw[4:]
                try:
                    tailoring = json.loads(raw)
                except json.JSONDecodeError:
                    tailoring = raw
                return json.dumps({
                    "success": True,
                    "job": {"id": job.id, "title": job.title, "company": job.company},
                    "tailoring": tailoring
                })
            except Exception as e:
                logger.error(f"tailor_resume_to_job error: {e}")
                return json.dumps({"success": False, "error": str(e)})

    return [find_top_jobs, get_resume_info, trigger_job_scout_agent, get_recent_matches, explain_feature,
            search_job_by_title, tailor_resume_to_job]


def build_system_prompt(user, resume, agent_config):
    """Build the system prompt with user context and cross-session memory"""
    today = datetime.utcnow().strftime("%B %d, %Y")

    resume_summary = ""
    if resume and resume.analysis:
        analysis_text = resume.analysis
        if len(analysis_text) > 600:
            analysis_text = analysis_text[:600] + "..."
        resume_summary = f"\n\nUser's Resume Summary:\n{analysis_text}"

    conversation_summary = ""
    if agent_config and agent_config.conversation_summary:
        conversation_summary = f"\n\nPrevious sessions context:\n{agent_config.conversation_summary}"

    return f"""You are Career Coach AI, a helpful career coaching assistant for {user.full_name or user.username}.
Today's date: {today}
Username: {user.username}
{resume_summary}
{conversation_summary}

You have access to the following tools to help the user:
1. find_top_jobs - Find matching jobs based on their resume
2. get_resume_info - Answer questions about their resume
3. trigger_job_scout_agent - Run the automatic job scout agent
4. get_recent_matches - Show recent job match results
5. explain_feature - Explain app features
6. search_job_by_title - Search the job database by job title/role name (returns job IDs)
7. tailor_resume_to_job - ATS-optimize the resume for a specific job (needs job_id from search_job_by_title)

Guidelines:
1. Be friendly, professional, and encouraging.
2. When asked to find jobs, use the find_top_jobs tool and present results clearly.
3. After finding jobs, tell the user they can click the "View Matching Jobs" button to see the filtered results.
4. When asked about skills or resume content, use get_resume_info.
5. When asked to run the agent or scan for jobs, use trigger_job_scout_agent.
6. Keep responses concise but informative.
7. If the user hasn't uploaded a resume yet, guide them to do so.
8. Use the user's career context from previous sessions to give personalized advice.
9. When explaining features, use explain_feature tool for accurate information.
10. If a tool returns an error, explain the issue helpfully and suggest next steps.
11. When the user asks to tailor, adjust, or optimize their resume for a specific job title or role:
    a. ALWAYS call search_job_by_title first to find matching jobs in the database.
    b. If jobs are found, pick the best match and call tailor_resume_to_job with its ID.
    c. Present the results clearly: show the ATS score improvement, missing keywords, the tailored Professional Summary, and the top rewritten experience bullets.
    d. If no jobs are found, tell the user to fetch jobs from the Jobs page first, then try again.
    e. NEVER ask the user to paste a job description manually — always search the database first."""


def summarize_session(messages, llm):
    """Summarize a list of chat messages into a rolling session summary"""
    if not messages:
        return None

    conversation_text = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in messages
    )

    prompt = f"""Summarize the key career coaching insights from this conversation in 3-5 sentences.
Focus on: job preferences mentioned, skills discussed, career goals expressed, actions taken, and feedback given.

Conversation:
{conversation_text}

Summary:"""

    try:
        result = llm.invoke(prompt)
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        logger.error(f"Session summarization error: {e}")
        return None


class CareerCoachChatbot:
    """Main chatbot class with session management and 2-tier memory"""

    def __init__(self, app):
        self.app = app

    def detect_session_boundary(self, user_id):
        """Check if the last message was more than SESSION_TIMEOUT_MINUTES ago"""
        with self.app.app_context():
            last_msg = ChatMessage.query.filter_by(
                user_id=user_id
            ).order_by(ChatMessage.timestamp.desc()).first()

            if not last_msg or not last_msg.timestamp:
                return False

            elapsed = datetime.utcnow() - last_msg.timestamp
            return elapsed > timedelta(minutes=SESSION_TIMEOUT_MINUTES)

    def close_and_summarize_session(self, user_id):
        """Summarize old session messages and store in AgentConfig"""
        with self.app.app_context():
            messages = ChatMessage.query.filter_by(
                user_id=user_id
            ).order_by(ChatMessage.timestamp.asc()).all()

            if not messages:
                return

            llm = get_llm()
            new_summary = summarize_session(messages, llm)

            if not new_summary:
                return

            config = AgentConfig.query.filter_by(user_id=user_id).first()
            if not config:
                config = AgentConfig(user_id=user_id)
                db.session.add(config)

            if config.conversation_summary:
                merge_prompt = f"""Merge these two conversation summaries into one concise summary (3-5 sentences).
Focus on: job preferences, skills, career goals, actions taken, feedback given.

Previous summary:
{config.conversation_summary}

New session summary:
{new_summary}

Merged summary:"""
                try:
                    merged = llm.invoke(merge_prompt)
                    config.conversation_summary = merged.content if hasattr(merged, 'content') else str(merged)
                except Exception as e:
                    logger.error(f"Summary merge error: {e}")
                    config.conversation_summary = new_summary
            else:
                config.conversation_summary = new_summary

            db.session.commit()

    def get_conversation_history(self, user_id, limit=10):
        """Load last N messages as LangChain message objects"""
        from langchain_core.messages import HumanMessage, AIMessage

        messages = ChatMessage.query.filter_by(
            user_id=user_id
        ).order_by(ChatMessage.timestamp.desc()).limit(limit).all()

        messages = list(reversed(messages))

        history = []
        for msg in messages:
            if msg.role == 'user':
                history.append(HumanMessage(content=msg.content))
            else:
                history.append(AIMessage(content=msg.content))

        return history

    def chat(self, user_id, message):
        """Process a chat message and return the response"""
        with self.app.app_context():
            # Session boundary check
            if self.detect_session_boundary(user_id):
                try:
                    self.close_and_summarize_session(user_id)
                except Exception as e:
                    logger.error(f"Session summarization failed: {e}")

            # Save user message
            user_msg = ChatMessage(
                user_id=user_id,
                role='user',
                content=message,
                timestamp=datetime.utcnow()
            )
            db.session.add(user_msg)
            db.session.commit()

            # Load context
            user = User.query.get(user_id)
            resume = Resume.query.filter_by(
                user_id=user_id, is_active=True
            ).order_by(Resume.uploaded_at.desc()).first()
            config = AgentConfig.query.filter_by(user_id=user_id).first()

            # Build agent components
            llm = get_llm()
            tools = build_tools(self.app, user_id)
            system_prompt = build_system_prompt(user, resume, config)
            chat_history = self.get_conversation_history(user_id, limit=10)

            # Exclude the message we just saved from history
            if chat_history:
                chat_history = chat_history[:-1]

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            # logger.info(f"prompt: {prompt}")
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=3,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                verbose=False
            )

            try:
                result = agent_executor.invoke({
                    "input": message,
                    "chat_history": chat_history
                })
                # logger.info(f"Agent execution result: {result}")
                response_text = result.get("output", "I'm sorry, I couldn't process your request.")
            except Exception as e:
                logger.error(f"Agent execution error: {e}")
                response_text = "I encountered an error processing your request. Please try again."

            # Detect intent from intermediate steps
            intent = None
            action_data = None
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if hasattr(step, '__len__') and len(step) >= 2:
                        action = step[0]
                        tool_output = step[1]
                        if hasattr(action, 'tool') and action.tool == "find_top_jobs":
                            try:
                                parsed = json.loads(tool_output) if isinstance(tool_output, str) else tool_output
                                if parsed.get("success") and parsed.get("action") == "redirect_to_jobs":
                                    intent = "redirect_to_jobs"
                                    action_data = json.dumps({"job_ids": parsed.get("job_ids", [])})
                            except (json.JSONDecodeError, TypeError):
                                pass

            # Save assistant message
            assistant_msg = ChatMessage(
                user_id=user_id,
                role='assistant',
                content=response_text,
                timestamp=datetime.utcnow(),
                intent=intent,
                action_data=action_data
            )
            db.session.add(assistant_msg)
            db.session.commit()

            return {
                "response": response_text,
                "intent": intent,
                "action_data": json.loads(action_data) if action_data else None
            }
