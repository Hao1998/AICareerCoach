"""
LLM Service

Provides lazy-initialized LLM instance and all LangChain chains used across the app.
No Flask routes here — pure business logic.
"""

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from schemas.output_schemas import JobMatchResult, KeywordExtractionResult, ResumeTailoringResult


def get_llm(app=None):
    """Get or create the default (non-streaming) LLM instance, cached on app.extensions."""
    from flask import current_app
    from langchain_xai import ChatXAI

    if app is None:
        app = current_app._get_current_object()

    if not hasattr(app, 'extensions'):
        app.extensions = {}

    if 'llm' not in app.extensions:
        api_key = app.config.get('XAI_API_KEY')
        if not api_key:
            raise RuntimeError("XAI_API_KEY not configured")
        app.extensions['llm'] = ChatXAI(
            model="grok-3",
            temperature=0,
            api_key=api_key
        )

    return app.extensions['llm']


def get_streaming_llm(app=None):
    """
    Streaming-enabled LLM instance for the chat endpoint.

    Cached separately from the default LLM so existing chains (resume analysis,
    job matching, roadmap) keep their non-streaming behaviour and current
    callbacks. Token-level streaming is opt-in via the callback handler passed
    at invoke time.
    """
    from flask import current_app
    from langchain_xai import ChatXAI

    if app is None:
        app = current_app._get_current_object()

    if not hasattr(app, 'extensions'):
        app.extensions = {}

    if 'llm_streaming' not in app.extensions:
        api_key = app.config.get('XAI_API_KEY')
        if not api_key:
            raise RuntimeError("XAI_API_KEY not configured")
        app.extensions['llm_streaming'] = ChatXAI(
            model="grok-3",
            temperature=0,
            api_key=api_key,
            streaming=True,   # ← emits on_llm_new_token per chunk
        )

    return app.extensions['llm_streaming']




# ── Resume Analysis Chain ──────────────────────────────────────────────────────

_resume_summary_template = """
Role: You are an AI Career Coach.

Task: Given the candidate's resume, provide a comprehensive summary that includes the following key aspects:

- Career Objective
- Skills and Expertise
- Professional Experience
- Educational Background
- Notable Achievements

Instructions:
Provide a concise summary of the resume, focusing on the candidate's skills, experience, and career trajectory. Ensure the summary is well-structured, clear, and highlights the candidate's strengths in alignment with industry standards.

Requirements:
{resume}

"""

_resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=_resume_summary_template,
)

_resume_analysis_chain = None


def get_resume_analysis_chain():
    """Get or create resume analysis chain"""
    global _resume_analysis_chain
    if _resume_analysis_chain is None:
        _resume_analysis_chain = LLMChain(llm=get_llm(), prompt=_resume_prompt)
    return _resume_analysis_chain


# ── Preparation Roadmap Chain ──────────────────────────────────────────────────

_preparation_roadmap_template = """
Role: You are an AI Career Coach creating personalized interview preparation roadmaps.

Task: Given a candidate's resume, job posting details, current skill gaps, and a preparation timeline, create a detailed month-by-month preparation roadmap with progressive technical interview questions.

Resume:
{resume}

Job Details:

Title: {job_title}
Company: {company}
Description: {job_description}
Requirements: {job_requirements}

Current Skill Gaps:
{skill_gaps}

Timeline: {timeline_months} months

Create a comprehensive preparation roadmap divided into phases. For a {timeline_months}-month timeline:
- If 3 months: Create 3 phases (Foundation, Intermediate, Advanced)
- If 6 months: Create 4-5 phases with more detailed preparation
- If 9 months: Create 5-6 phases with comprehensive coverage

For each phase, provide:

1. Phase name and duration
2. Specific skills to learn (related to the skill gaps)
3. Recommended learning resources (courses, books, documentation)
4. Hands-on projects to build (that demonstrate the required skills)
5. Milestones to achieve
6. **Technical interview questions** - Generate 4-6 role-specific questions that:
   - Are relevant to the skills being learned in this phase
   - Progress in difficulty from phase to phase (Easy → Medium → Hard)
   - Mix conceptual understanding, practical application, and problem-solving
   - Are realistic questions for this specific role ({job_title})
   - Include the difficulty level and topic area
Also provide an overview and final tips for interview success.
Provide your roadmap in the following JSON format:
{{
    "overview": "Brief overview of the preparation plan and interview strategy",
    "phases": [
        {{
            "phase_name": "Phase name (e.g., 'Foundation Building')",
            "duration": "Time period (e.g., 'Month 1-2' or 'Weeks 1-4')",
            "difficulty_level": "Easy|Medium|Hard",
            "skills": ["skill1", "skill2", "skill3"],
            "resources": ["resource1 with description", "resource2 with description"],
            "projects": ["project1 description", "project2 description"],
            "milestones": ["milestone1", "milestone2"],
            "interview_questions": [
                {{
                    "question": "The technical interview question",
                    "difficulty": "Easy|Medium|Hard",
                    "topic": "Topic area (e.g., 'Data Structures', 'System Design', 'API Design')",
                    "hint": "Brief hint or approach to tackle this question"
                }}
            ]
        }}
    ],
    "final_tips": "Important advice for interview success, including behavioral tips and common pitfalls to avoid"
}}
Make sure the roadmap is:
- Specific to the skill gaps and target role identified
- Realistic for the given timeline
- Actionable with concrete steps
- Progressive from foundational to advanced topics
- Interview questions progress in difficulty and relevance across phases
- Questions reflect real interview scenarios for {job_title} positions
"""

_preparation_roadmap_prompt = PromptTemplate(
    input_variables=["resume", "job_title", "company", "job_description", "job_requirements",
                     "skill_gaps", "timeline_months"],
    template=_preparation_roadmap_template,
)

_preparation_roadmap_chain = None


def get_preparation_roadmap_chain():
    """Get or create preparation roadmap chain"""
    global _preparation_roadmap_chain
    if _preparation_roadmap_chain is None:
        _preparation_roadmap_chain = LLMChain(llm=get_llm(), prompt=_preparation_roadmap_prompt)
    return _preparation_roadmap_chain


# ── Job Matching — delegates to JobAnalystAgent (grok-3) ──────────────────────

def run_job_matching(resume, job_title, company, job_description, job_requirements) -> JobMatchResult:
    """
    Delegates to the JobAnalystAgent (grok-3).
    Returns a typed JobMatchResult — no json.loads() needed by callers.
    """
    from agents import get_job_analyst_agent
    return get_job_analyst_agent().analyze(
        resume=resume,
        job_title=job_title,
        company=company,
        job_description=job_description,
        job_requirements=job_requirements,
    )


# ── Resume Tailoring — delegates to ResumeTailoringAgent (grok-3) ─────────────

def run_resume_tailoring_structured(
    resume, job_title, company, job_description, job_requirements
) -> ResumeTailoringResult:
    """
    Delegates to the ResumeTailoringAgent (grok-3).
    Returns a typed ResumeTailoringResult — no JSON parsing needed by callers.
    """
    from agents import get_resume_tailoring_agent
    return get_resume_tailoring_agent().tailor(
        resume=resume,
        job_title=job_title,
        company=company,
        job_description=job_description,
        job_requirements=job_requirements,
    )


# ── Keyword Extraction — delegates to KeywordResearchAgent (grok-3-mini) ───────

def run_keyword_extraction(resume_text: str, explicit_preferences: dict | None = None) -> KeywordExtractionResult:
    """
    Delegates to the KeywordResearchAgent (grok-3-mini).
    Returns a typed KeywordExtractionResult with a list of job titles.
    """
    from agents import get_keyword_research_agent
    return get_keyword_research_agent().extract_keywords(resume_text, explicit_preferences)


# ── Job Search Planning — delegates to JobSearchPlannerAgent (grok-3-mini) ────

def run_job_search_planning(query: str):
    """
    Delegates to the JobSearchPlannerAgent (grok-3-mini).
    Returns a typed JobSearchIntent parsed from the user's natural-language query.
    """
    from agents import get_job_search_planner_agent
    from schemas.output_schemas import JobSearchIntent
    try:
        return get_job_search_planner_agent().plan(query)
    except Exception:
        # Graceful fallback: treat as a generic search with no filters
        return JobSearchIntent(query_summary=query or "General job search")