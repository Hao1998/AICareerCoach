"""
Output schemas for all LLM structured outputs across every agent.

Two benefits:
  1. Type-safe exchange — no manual json.loads() or KeyError crashes.
  2. Injection firewall between agents — injected text is trapped inside
     typed string fields and cannot propagate as executable instructions
     to downstream agents. A malicious string in recommendation: str is
     data, never a command.
"""

from pydantic import BaseModel, Field


# ── Job Analyst Agent ──────────────────────────────────────────────────────────

class JobMatchResult(BaseModel):
    """Structured output from the Job Analyst Agent."""
    match_score: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    skill_gaps: list[str] = Field(default_factory=list)
    recommendation: str = Field(default="")


# ── Keyword Research Agent ─────────────────────────────────────────────────────

class KeywordExtractionResult(BaseModel):
    """Structured output from the Keyword Research Agent."""
    job_titles: list[str] = Field(
        description="2-3 job titles the candidate is most qualified for"
    )


# ── Resume Tailoring Agent ─────────────────────────────────────────────────────

class ATSScore(BaseModel):
    before: int = Field(ge=0, le=100)
    after: int = Field(ge=0, le=100)


class ExperienceBullet(BaseModel):
    original: str = Field(default="")
    tailored: str = Field(default="")


class KeywordAnalysis(BaseModel):
    critical_keywords: list[str] = Field(default_factory=list)
    present_in_resume: list[str] = Field(default_factory=list)
    missing_from_resume: list[str] = Field(default_factory=list)


class TailoredSections(BaseModel):
    professional_summary: str = Field(default="")
    skills: list[str] = Field(default_factory=list)
    experience_bullets: list[ExperienceBullet] = Field(default_factory=list)


class ResumeTailoringResult(BaseModel):
    """Structured output from the Resume Tailoring Agent."""
    keyword_analysis: KeywordAnalysis
    ats_score: ATSScore
    tailored_sections: TailoredSections
    recommendations: list[str] = Field(default_factory=list)
    ats_formatting_tips: str = Field(default="")


# ── Task Planner ──────────────────────────────────────────────────────────────

class PlannedStep(BaseModel):
    """A single step in a generated plan."""
    step_order: int = Field(ge=1)
    description: str = Field(description="What this step accomplishes")
    tool_name: str | None = Field(
        default=None,
        description="Tool to call: find_jobs_matching_resume, get_resume_info, get_job_history, "
                    "tailor_resume_to_job, lookup_job_by_title, "
                    "search_memory, or null for LLM-only reasoning steps"
    )
    tool_input: str | None = Field(
        default=None,
        description="The input argument to pass to the tool (a single string or JSON)"
    )


class PlanResult(BaseModel):
    """Structured output from the planner: a list of steps to achieve a goal."""
    reasoning: str = Field(description="Brief explanation of why this plan was chosen")
    steps: list[PlannedStep] = Field(description="Ordered steps to achieve the goal")


class ReplanResult(BaseModel):
    """Structured output from the replanner: updated remaining steps."""
    reasoning: str = Field(description="Why the plan was adjusted")
    updated_steps: list[PlannedStep] = Field(description="The new remaining steps (renumbered from 1)")
    is_complete: bool = Field(
        default=False,
        description="True if the goal is fully achieved and no more steps are needed"
    )


# ── Job Search Planner Agent ───────────────────────────────────────────────────

class JobSearchIntent(BaseModel):
    """Structured output from the Job Search Planner Agent."""
    keywords: list[str] = Field(
        default_factory=list,
        description="Specific technologies, skills, or role keywords mentioned by the user",
    )
    location: str | None = Field(
        default=None,
        description="Location preference: a city/country name, 'remote', or null if not specified.",
    )
    seniority_level: str = Field(
        default="any",
        description="One of: 'junior', 'mid', 'senior', 'lead', or 'any' if not specified.",
    )
    job_type: str = Field(
        default="any",
        description="One of: 'full-time', 'part-time', 'contract', or 'any' if not specified.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many results to return. Default 5 unless the user says otherwise (max 20).",
    )
    query_summary: str = Field(
        default="",
        description="Plain-English one-line summary of what the user is looking for.",
    )


# ── Career Roadmap ─────────────────────────────────────────────────────────────

class LearningPhase(BaseModel):
    timeframe: str = Field(description="e.g. 'Month 1-2'")
    focus: str = Field(description="Main topic or skill to develop in this phase")
    resources: list[str] = Field(
        default_factory=list,
        description="Specific courses, books, certifications, or projects"
    )
    milestone: str = Field(description="Measurable outcome to hit by end of this phase")


class CareerRoadmap(BaseModel):
    """Structured output from the roadmap synthesis step."""
    current_state: str = Field(description="Summary of where the user is now — skills, experience, strengths")
    target_state: str = Field(description="What the target role requires — skills, experience, expectations")
    skill_gaps: list[str] = Field(description="Specific skills or experience the user is missing")
    strengths: list[str] = Field(description="Existing skills that directly transfer to the target role")
    learning_path: list[LearningPhase] = Field(description="Time-boxed phases with concrete resources")
    application_strategy: str = Field(description="When and how to apply — timing, role progression, tips")
    target_companies: list[str] = Field(
        default_factory=list,
        description="Specific companies or types of companies to target, drawn from job search results"
    )
    resume_tips: list[str] = Field(description="Specific ATS and content tips for this target role")
    timeline_summary: str = Field(description="One-paragraph summary of the full timeline from now to hired")


# ── Security Guard ─────────────────────────────────────────────────────────────

class GuardCheckResult(BaseModel):
    """Output from the LLM-as-Judge security guard."""
    is_safe: bool
    reason: str = Field(default="")
