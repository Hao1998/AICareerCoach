"""
Agent Coordinator

Single place that knows which specialist agent handles which task and which
model each uses. Factory functions cache agent instances in app.extensions
so model connections are reused across requests — same pattern as get_llm().

Model assignments (change here to swap models for any agent):
  SecurityGuardAgent    → grok-3-mini  (fast binary check, cheap)
  KeywordResearchAgent  → grok-3-mini  (simple extraction, cheap)
  JobAnalystAgent       → grok-3       (nuanced reasoning needed)
  ResumeTailoringAgent  → grok-3       (high-quality rewriting needed)
"""

import logging

logger = logging.getLogger(__name__)


def _get_api_key(app) -> str:
    key = app.config.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY not configured")
    return key


def _get_or_create(app, extension_key: str, agent_class):
    """Return a cached agent instance, creating it on first access."""
    from flask import current_app
    if app is None:
        app = current_app._get_current_object()
    if not hasattr(app, "extensions"):
        app.extensions = {}
    if extension_key not in app.extensions:
        app.extensions[extension_key] = agent_class(_get_api_key(app))
    return app.extensions[extension_key]


def get_security_guard_agent(app=None):
    from agents.security_guard_agent import SecurityGuardAgent
    return _get_or_create(app, "agent_security_guard", SecurityGuardAgent)


def get_keyword_research_agent(app=None):
    from agents.keyword_research_agent import KeywordResearchAgent
    return _get_or_create(app, "agent_keyword_research", KeywordResearchAgent)


def get_job_analyst_agent(app=None):
    from agents.job_analyst_agent import JobAnalystAgent
    return _get_or_create(app, "agent_job_analyst", JobAnalystAgent)


def get_resume_tailoring_agent(app=None):
    from agents.resume_tailoring_agent import ResumeTailoringAgent
    return _get_or_create(app, "agent_resume_tailoring", ResumeTailoringAgent)
