"""
LLM Service

Provides lazy-initialized LLM instance and all LangChain chains used across the app.
No Flask routes here — pure business logic.
"""

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def get_llm(app=None):
    """Get or create LLM instance (lazy initialization, cached on app.extensions)"""
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


# ── Job Matching Chain ─────────────────────────────────────────────────────────

_job_matching_template = """
Role: You are an AI Career Coach analyzing job matches.

Task: Given a candidate's resume and a job posting, analyze the match and provide:
1. Match Score (0-100): Overall compatibility percentage
2. Matched Skills: List of candidate's skills that match job requirements
3. Skill Gaps: Skills required by the job that the candidate lacks
4. Recommendation: Brief advice for the candidate

Resume:
{resume}

Job Posting:
Title: {job_title}
Company: {company}
Description: {job_description}
Requirements: {job_requirements}

Provide your analysis in the following JSON format:
{{
    "match_score": <number between 0-100>,
    "matched_skills": ["skill1", "skill2", ...],
    "skill_gaps": ["gap1", "gap2", ...],
    "recommendation": "Your recommendation text here"
}}
"""

_job_matching_prompt = PromptTemplate(
    input_variables=["resume", "job_title", "company", "job_description", "job_requirements"],
    template=_job_matching_template,
)

_job_matching_chain = None


def get_job_matching_chain():
    """Get or create job matching chain"""
    global _job_matching_chain
    if _job_matching_chain is None:
        _job_matching_chain = LLMChain(llm=get_llm(), prompt=_job_matching_prompt)
    return _job_matching_chain


# ── ATS Resume Tailoring Chain ─────────────────────────────────────────────────

_resume_tailoring_template = """
Role: You are an expert ATS optimization specialist and career coach.

Task: Tailor the candidate's resume specifically for the target job, optimizing for both
Applicant Tracking Systems (ATS) keyword scanning and human recruiter review.
Your goal is to reframe the candidate's REAL experience using the language and priorities
of the target role — never fabricate skills or experience they don't have.

Resume:
{resume}

Target Job:
Title: {job_title}
Company: {company}
Description: {job_description}
Requirements: {job_requirements}

Instructions:
1. Extract the most critical ATS keywords from the job description (technical skills, tools,
   methodologies, certifications, domain terms).
2. Identify which keywords are already present in the resume vs. which are missing.
3. Rewrite the Professional Summary to position the candidate as a strong fit.
4. Reorder and expand the Skills section to front-load keywords that match the JD.
5. Rewrite up to 5 experience bullet points to mirror JD language, add metrics where implied,
   and surface accomplishments most relevant to this role.
6. Estimate ATS keyword match score before and after your suggestions (0–100).

Return ONLY a valid JSON object in this exact format:
{{
    "keyword_analysis": {{
        "critical_keywords": ["top keywords the ATS will scan for"],
        "present_in_resume": ["JD keywords already in the candidate's resume"],
        "missing_from_resume": ["important JD keywords NOT currently in the resume"]
    }},
    "ats_score": {{
        "before": <integer 0-100, estimated current keyword match>,
        "after": <integer 0-100, estimated score after applying tailoring>
    }},
    "tailored_sections": {{
        "professional_summary": "Rewritten 3-4 sentence summary optimized for this role",
        "skills": ["skill1", "skill2", "skill3 — reordered + expanded to match JD keywords"],
        "experience_bullets": [
            {{
                "original": "original bullet point from resume",
                "tailored": "rewritten bullet using JD language, stronger framing, added metrics"
            }}
        ]
    }},
    "recommendations": [
        "Specific actionable tip #1 to improve ATS ranking further",
        "Specific actionable tip #2"
    ],
    "ats_formatting_tips": "One paragraph on formatting/structure choices that help this resume pass ATS parsing."
}}
"""

_resume_tailoring_prompt = PromptTemplate(
    input_variables=["resume", "job_title", "company", "job_description", "job_requirements"],
    template=_resume_tailoring_template,
)

_resume_tailoring_chain = None


def get_resume_tailoring_chain():
    """Get or create ATS resume tailoring chain"""
    global _resume_tailoring_chain
    if _resume_tailoring_chain is None:
        _resume_tailoring_chain = LLMChain(llm=get_llm(), prompt=_resume_tailoring_prompt)
    return _resume_tailoring_chain
