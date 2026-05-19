"""
Pydantic request schemas for API endpoints.

Each schema validates the shape, types, and bounds of an incoming JSON
request body. Used together with the `@validate_json` decorator in
`schemas/validate.py` to enforce input contracts at the controller boundary.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator('message')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('Message cannot be empty')
        return v


class JobFetchRequest(BaseModel):
    keywords: Optional[str] = Field(default=None, max_length=500)
    location: Optional[str] = Field(default=None, max_length=200)
    max_jobs: int = Field(default=50, ge=1, le=200)
    max_days_old: int = Field(default=30, ge=1, le=365)
    sources: Optional[list[str]] = None


class RoadmapRequest(BaseModel):
    job_id: int = Field(gt=0)
    timeline_months: int = Field(ge=1, le=12)


class JobRoadmapRequest(BaseModel):
    """Roadmap request for `/api/jobs/<id>/roadmap` — job_id comes from URL."""
    timeline_months: int = Field(default=3, ge=1, le=12)
    refresh: bool = False


class JobMatchRefreshRequest(BaseModel):
    """Body for `/api/jobs/<id>/match` and `/tailor` — just an optional refresh flag."""
    refresh: bool = False


class AgentConfigUpdateRequest(BaseModel):
    schedule_time: Optional[str] = Field(default=None, pattern=r'^\d{2}:\d{2}$')
    timezone: Optional[str] = Field(default=None, max_length=50)
    is_enabled: Optional[bool] = None
    match_threshold: Optional[float] = Field(default=None, ge=0, le=100)
    max_results_per_run: Optional[int] = Field(default=None, ge=1, le=50)
    adzuna_location: Optional[str] = Field(default=None, max_length=200)
    adzuna_max_jobs: Optional[int] = Field(default=None, ge=1, le=200)
    adzuna_max_days_old: Optional[int] = Field(default=None, ge=1, le=365)
    enabled_sources: Optional[list[str]] = None


class FeedbackRequest(BaseModel):
    feedback: str = Field(pattern=r'^(interested|not_interested|applied)$')
