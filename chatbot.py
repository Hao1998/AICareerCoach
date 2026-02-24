"""
Career Coach AI Chatbot Module

LangChain tool-calling agent with 2-tier memory:
- Tier 1 (Hot): Last 10 raw messages from DB
- Tier 2 (Warm): Rolling LLM-generated conversation summary in AgentConfig
"""

import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_MINUTES = 30


def build_tools(app, user_id):
    """Build LangChain tools for the career coach agent"""
    from langchain_core.tools import tool

    @tool
    def find_top_jobs(query: str) -> str:
        """Find the top 5 matching jobs for the user based on their resume. Use this when the user asks to find jobs, get job recommendations, or match their resume to jobs. The query parameter can be a description of what kind of jobs they want."""
        with app.app_context():
            from models import Resume
            from app import find_matching_jobs, extract_text_from_pdf

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
            from app import perform_qa
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
            from app import agent_scheduler
            try:
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
            "agent_config": "Configure the Job Scout Agent's behavior: schedule time, timezone, match threshold (minimum score to save), max results per run, and Adzuna search preferences (location, max jobs, max age)."
        }
        result = features.get(feature_name.lower().strip(),
                              f"Unknown feature: '{feature_name}'. Available features: {', '.join(features.keys())}")
        return result

    return [find_top_jobs, get_resume_info, trigger_job_scout_agent, get_recent_matches, explain_feature]


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
10. If a tool returns an error, explain the issue helpfully and suggest next steps."""


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
            from models import ChatMessage
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
            from models import ChatMessage, AgentConfig, db
            from app import get_llm

            # Get all messages (the old session)
            messages = ChatMessage.query.filter_by(
                user_id=user_id
            ).order_by(ChatMessage.timestamp.asc()).all()

            if not messages:
                return

            llm = get_llm()
            new_summary = summarize_session(messages, llm)

            if not new_summary:
                return

            # Get or create agent config
            config = AgentConfig.query.filter_by(user_id=user_id).first()
            if not config:
                config = AgentConfig(user_id=user_id)
                db.session.add(config)

            # Merge with existing summary if present
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
        from models import ChatMessage

        messages = ChatMessage.query.filter_by(
            user_id=user_id
        ).order_by(ChatMessage.timestamp.desc()).limit(limit).all()

        # Reverse to chronological order
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
            from models import ChatMessage, AgentConfig, Resume, User, db
            from app import get_llm
            from langchain.agents import create_tool_calling_agent, AgentExecutor
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

            # Remove the last message from history (it's the one we just saved)
            if chat_history and len(chat_history) > 0:
                chat_history = chat_history[:-1]

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # Create agent
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=3,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                verbose=False
            )

            # Execute
            try:
                result = agent_executor.invoke({
                    "input": message,
                    "chat_history": chat_history
                })
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