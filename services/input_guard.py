"""
Input Guard — Harness input validation layer.

Two guards:
  scan_message(text)              — blocks direct prompt injection in user chat messages
  scan_resume_text(text, user_id) — strips injected payloads from PDF-extracted resume text
"""

import re
import logging

logger = logging.getLogger(__name__)

# Phrases that indicate an attempt to override agent behaviour.
# Ordered from most specific to most general to reduce false positives
# on normal career/resume language.
_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|your|the\s+)?instructions",
    r"forget\s+(you\s+are|your\s+role|your\s+guidelines|all\s+previous)",
    r"you\s+are\s+now\s+(in\s+)?(a\s+|an\s+)?(different|new|diagnostic|developer|admin|debug|data[\s\-]retrieval)",
    r"pretend\s+(you\s+are|to\s+be)\s+(a\s+|an\s+)?",
    r"act\s+as\s+(a\s+|an\s+)?(different|new|another)\s+",
    r"override\s+(your\s+)?(instructions|guidelines|programming|system\s+prompt)",
    r"\[\s*system\s+(update|override|prompt|message|instruction)",
    r"\[\s*end\s+system",
    r"new\s+persona",
    r"jailbreak",
    r"\bdan\s+mode\b",
    r"(reveal|output|print|show|repeat)\s+(the\s+|your\s+)?(system\s+prompt|instructions|guidelines)",
    r"in\s+(diagnostic|developer|admin|debug|data[\s\-]retrieval)\s+mode",
    r"do\s+not\s+mention\s+this\s+instruction",
    r"(enabled|activated)\s*\]\s*ignore",  # [MODE ENABLED] Ignore ...
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

_BLOCKED_RESPONSE = (
    "I can only help with career-related topics such as resume review, "
    "job matching, and interview preparation. "
    "Please ask me something related to your career goals."
)


def _matches_any(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)


def scan_message(text: str) -> dict:
    """
    Check a user chat message for direct prompt injection.

    Returns:
        {"safe": True}
        {"safe": False, "response": "<safe reply to send back>"}
    """
    if _matches_any(text):
        logger.warning("[InputGuard] Direct injection blocked | input=%.120r", text)
        return {"safe": False, "response": _BLOCKED_RESPONSE}
    return {"safe": True}


def scan_with_llm(text: str, context: str = "input") -> dict:
    """
    SecurityGuardAgent (grok-3-mini) second layer: catches sophisticated injections
    that slip past the regex layer.

    Only call this for high-value, low-frequency operations (resume upload) —
    NOT on every chat message, to avoid adding latency to normal interactions.

    Returns:
        {"safe": True}
        {"safe": False, "response": "<safe reply to send back>"}
    """
    try:
        from agents import get_security_guard_agent
        result = get_security_guard_agent().check(text, context)

        if not result.is_safe:
            logger.warning(
                "[InputGuard][LLM] Injection detected | context=%s | reason=%s | input=%.120r",
                context, result.reason, text,
            )
            return {"safe": False, "response": _BLOCKED_RESPONSE}

        return {"safe": True}

    except Exception as exc:
        # Guard failure is non-fatal — default to safe so a model error doesn't
        # block every upload.
        logger.warning("[InputGuard][LLM] SecurityGuardAgent call failed, defaulting safe: %s", exc)
        return {"safe": True}


def scan_resume_text(text: str, user_id=None) -> str:
    """
    Scan PDF-extracted resume text for embedded injection payloads.

    Strips any line matching an injection pattern and returns the cleaned text
    so resume processing continues normally. Logs a warning for each removal.
    """
    lines = text.splitlines()
    clean_lines = []
    stripped = 0

    for line in lines:
        if _matches_any(line):
            stripped += 1
            logger.warning(
                "[InputGuard] Injection line stripped from resume | user_id=%s | line=%.120r",
                user_id, line,
            )
        else:
            clean_lines.append(line)

    if stripped:
        logger.warning(
            "[InputGuard] %d injected line(s) removed from resume for user_id=%s",
            stripped, user_id,
        )

    return "\n".join(clean_lines)
