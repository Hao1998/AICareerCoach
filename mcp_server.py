"""
AICareerCoach MCP Server

Exposes career coaching capabilities as MCP tools and resources.
Reuses the existing services/ layer — no logic duplication.

Run with:
    python mcp_server.py

Or via FastMCP CLI:
    fastmcp run mcp_server.py --transport streamable-http --port 8001

─── Security model ───────────────────────────────────────────────────────────
Layer 1 — Authentication:
    Every client must set MCP_API_KEY in their environment.
    The server refuses to start if the key is missing or unrecognised.

Layer 2 — Authorization (per-user scoping):
    MCP_API_KEYS in .env maps keys → user_ids  (format: "key1:1,key2:2")
    Tools have NO user_id parameter — the user_id is derived from the key.
    Cross-user access is architecturally impossible, not just policy.

Layer 3 — Audit logging:
    Every tool call is appended to mcp_audit.log with:
    timestamp | key fingerprint (first 8 chars) | tool | params
──────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# ── Offline mode for HuggingFace ─────────────────────────────────────────────
# Must be set BEFORE any sentence_transformers / transformers import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Suppress deprecation warnings — keeps STDIO stdout clean for JSON-RPC.
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING - 1)   # keeps ERROR+, drops WARNING and below

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on sys.path when launched with an absolute path.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

# ── Bootstrap Flask app context ───────────────────────────────────────────────
from factory import create_app
flask_app = create_app()


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 + 2 — Authentication & Authorization
# ─────────────────────────────────────────────────────────────────────────────

def _load_key_map() -> dict[str, int]:
    """
    Load the API key → user_id mapping from the appropriate secret store.

    ─ Local dev  : reads MCP_API_KEYS from .env
                   format: "key1:user_id1,key2:user_id2"

    ─ AWS prod   : reads from Secrets Manager.
                   Set SECRET_BACKEND=aws and AWS_SECRET_NAME=aicareercoach/mcp-keys
                   The secret value must be JSON: {"key1": 2, "key2": 3}
                   Access granted via the ECS task IAM role — no credentials in code.

    ─ GCP prod   : reads from Secret Manager.
                   Set SECRET_BACKEND=gcp, GCP_PROJECT_ID=your-project,
                   and GCP_SECRET_NAME=aicareercoach-mcp-keys
                   The secret value must be JSON: {"key1": 2, "key2": 3}
                   Access granted via the Cloud Run service account.

    Generate a secure key with:
        python -c "import secrets; print(secrets.token_hex(32))"
    """
    backend = os.getenv("SECRET_BACKEND", "env").lower()

    if backend == "aws":
        import boto3
        client = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION", "ap-southeast-1"))
        secret = client.get_secret_value(SecretId=os.getenv("AWS_SECRET_NAME", "aicareercoach/mcp-keys"))
        return {k: int(v) for k, v in json.loads(secret["SecretString"]).items()}

    if backend == "gcp":
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = (
            f"projects/{os.getenv('GCP_PROJECT_ID')}"
            f"/secrets/{os.getenv('GCP_SECRET_NAME', 'aicareercoach-mcp-keys')}"
            f"/versions/latest"
        )
        response = client.access_secret_version(request={"name": name})
        return {k: int(v) for k, v in json.loads(response.payload.data.decode()).items()}

    # Default: local .env
    raw = os.getenv("MCP_API_KEYS", "")
    if not raw:
        return {}
    key_map = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        key, uid = pair.rsplit(":", 1)
        try:
            key_map[key.strip()] = int(uid.strip())
        except ValueError:
            pass
    return key_map


_KEY_MAP: dict[str, int] = _load_key_map()

# Validate the key for THIS client connection at boot time.
_CLIENT_KEY: str = os.getenv("MCP_API_KEY", "")

if not _CLIENT_KEY:
    print(
        "❌ MCP_API_KEY is not set. "
        "Add it to your Claude Desktop / Cursor MCP config env block.",
        file=sys.stderr,
    )
    sys.exit(1)

if _CLIENT_KEY not in _KEY_MAP:
    print(
        f"❌ MCP_API_KEY '{_CLIENT_KEY[:8]}…' is not recognised. "
        "Add it to MCP_API_KEYS in .env  (format: key:user_id,key2:user_id2)",
        file=sys.stderr,
    )
    sys.exit(1)

# ✅ Authenticated — scope this entire server process to one user.
_AUTHENTICATED_USER_ID: int = _KEY_MAP[_CLIENT_KEY]
_KEY_FINGERPRINT: str = _CLIENT_KEY[:8] + "…"   # safe to log, never the full key

print(
    f"✅ AICareerCoach MCP server booted  "
    f"| user_id={_AUTHENTICATED_USER_ID}  "
    f"| key={_KEY_FINGERPRINT}",
    file=sys.stderr,
)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — Audit logging
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT_LOG = Path(_PROJECT_ROOT) / "mcp_audit.log"


def _audit(tool_name: str, extra: dict | None = None) -> None:
    """Append one line to mcp_audit.log for every tool invocation."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "key": _KEY_FINGERPRINT,
        "user_id": _AUTHENTICATED_USER_ID,
        "tool": tool_name,
        **(extra or {}),
    }
    with open(_AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MCP server instance
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP(name="AICareerCoach")


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS — user_id is NO LONGER a parameter.
# It is derived from the authenticated key above (_AUTHENTICATED_USER_ID).
# The model cannot access another user's data even if it tries.
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def find_matching_jobs(top_k: int = 5) -> str:
    """
    Find the top matching jobs for the authenticated user based on their active resume.
    Uses FAISS vector search + LLM scoring (two-stage pipeline).

    Args:
        top_k: Number of top matches to return (default: 5)

    Returns:
        JSON string with list of matched jobs, scores, matched skills, and skill gaps
    """
    _audit("find_matching_jobs", {"top_k": top_k})
    with flask_app.app_context():
        from models import Resume
        from services.job_service import find_matching_jobs as _find_jobs
        from services.resume_service import extract_text_from_pdf

        resume = (
            Resume.query.filter_by(user_id=_AUTHENTICATED_USER_ID, is_active=True)
            .order_by(Resume.uploaded_at.desc())
            .first()
        )
        if not resume:
            return json.dumps({
                "success": False,
                "error": "No active resume found. Upload a resume first.",
            })

        resume_text = extract_text_from_pdf(resume.file_path)
        matches = _find_jobs(resume_text, top_k=top_k)

        result = [
            {
                "job_id": m["job"].id,
                "title": m["job"].title,
                "company": m["job"].company,
                "location": m["job"].location,
                "match_score": m["analysis"]["match_score"],
                "matched_skills": m["analysis"]["matched_skills"],
                "skill_gaps": m["analysis"]["skill_gaps"],
                "recommendation": m["analysis"]["recommendation"],
            }
            for m in matches
        ]
        return json.dumps({"success": True, "matches": result})


@mcp.tool()
def analyze_resume() -> str:
    """
    Analyze the authenticated user's active resume and return a structured career summary.
    Covers: career objective, skills, experience, education, achievements.

    Returns:
        JSON string with resume analysis text
    """
    _audit("analyze_resume")
    with flask_app.app_context():
        from models import Resume
        from services.resume_service import extract_text_from_pdf
        from services.llm_service import get_resume_analysis_chain

        resume = (
            Resume.query.filter_by(user_id=_AUTHENTICATED_USER_ID, is_active=True)
            .order_by(Resume.uploaded_at.desc())
            .first()
        )
        if not resume:
            return json.dumps({"success": False, "error": "No active resume found."})

        if resume.analysis:
            return json.dumps({
                "success": True,
                "analysis": resume.analysis,
                "resume_filename": resume.original_filename,
                "source": "cached",
            })

        resume_text = extract_text_from_pdf(resume.file_path)
        chain = get_resume_analysis_chain()
        analysis = chain.run(resume=resume_text)
        return json.dumps({
            "success": True,
            "analysis": analysis,
            "resume_filename": resume.original_filename,
            "source": "fresh_analysis",
        })


@mcp.tool()
def get_skill_gaps(job_id: int) -> str:
    """
    Get the skill gaps between the authenticated user's resume and a specific job posting.
    Useful for interview prep and upskilling recommendations.

    Args:
        job_id: The job posting ID (get this from find_matching_jobs first)

    Returns:
        JSON string with matched skills, gaps, and a recommendation
    """
    _audit("get_skill_gaps", {"job_id": job_id})
    with flask_app.app_context():
        from models import Resume, JobPosting, JobMatch
        from services.resume_service import extract_text_from_pdf
        from services.llm_service import run_job_matching

        resume = (
            Resume.query.filter_by(user_id=_AUTHENTICATED_USER_ID, is_active=True)
            .order_by(Resume.uploaded_at.desc())
            .first()
        )
        job = JobPosting.query.get(job_id)

        if not resume:
            return json.dumps({"success": False, "error": "No active resume found."})
        if not job:
            return json.dumps({"success": False, "error": f"Job ID {job_id} not found."})

        existing = (
            JobMatch.query.filter_by(user_id=_AUTHENTICATED_USER_ID, job_id=job_id)
            .order_by(JobMatch.created_at.desc())
            .first()
        )
        if existing:
            return json.dumps({
                "success": True,
                "job_title": job.title,
                "company": job.company,
                "match_score": existing.match_score,
                "matched_skills": json.loads(existing.matched_skills or "[]"),
                "skill_gaps": json.loads(existing.gaps or "[]"),
                "recommendation": existing.recommendation,
                "source": "cached",
            })

        resume_text = extract_text_from_pdf(resume.file_path)
        result = json.loads(run_job_matching(
            resume=resume_text[:3000],
            job_title=job.title,
            company=job.company,
            job_description=job.description[:1000],
            job_requirements=job.requirements[:1000] if job.requirements else "Not specified",
        ))
        return json.dumps({
            "success": True,
            "job_title": job.title,
            "company": job.company,
            "source": "fresh_analysis",
            **result,
        })


@mcp.tool()
def ask_resume_question(question: str) -> str:
    """
    Ask a natural language question about your own resume.
    Uses RAG (FAISS retrieval + LLM) to answer from resume content.

    Example questions:
    - "What are my strongest technical skills?"
    - "How many years of Java experience do I have?"
    - "Do I have any cloud certifications?"

    Args:
        question: Natural language question about the resume

    Returns:
        JSON string with the answer
    """
    _audit("ask_resume_question", {"question": question[:120]})
    with flask_app.app_context():
        from services.resume_service import perform_qa
        try:
            answer = perform_qa(question, _AUTHENTICATED_USER_ID)
            return json.dumps({"success": True, "answer": answer})
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})


@mcp.tool()
def get_recent_job_matches(limit: int = 10) -> str:
    """
    Retrieve the most recent job matches for the authenticated user.
    Reads from DB cache — faster than re-running the full matching pipeline.

    Args:
        limit: Max number of matches to return (default: 10)

    Returns:
        JSON string with list of recent job matches
    """
    _audit("get_recent_job_matches", {"limit": limit})
    with flask_app.app_context():
        from models import JobMatch, JobPosting

        matches = (
            JobMatch.query.filter_by(user_id=_AUTHENTICATED_USER_ID)
            .order_by(JobMatch.created_at.desc())
            .limit(limit)
            .all()
        )
        if not matches:
            return json.dumps({"success": True, "matches": [], "message": "No matches found yet."})

        result = []
        for m in matches:
            job = JobPosting.query.get(m.job_id)
            result.append({
                "job_id": m.job_id,
                "title": job.title if job else "Unknown",
                "company": job.company if job else "Unknown",
                "match_score": m.match_score,
                "matched_skills": json.loads(m.matched_skills or "[]"),
                "skill_gaps": json.loads(m.gaps or "[]"),
                "recommendation": m.recommendation,
                "matched_at": m.created_at.isoformat() if m.created_at else None,
            })
        return json.dumps({"success": True, "matches": result})


@mcp.tool()
def trigger_job_scout() -> str:
    """
    Manually trigger the Job Scout agent for the authenticated user.
    Fetches fresh jobs from Adzuna → rebuilds FAISS index → finds matches → saves results.

    Returns:
        JSON string with run_id for tracking progress
    """
    _audit("trigger_job_scout")
    with flask_app.app_context():
        from job_scout_agent import JobScoutAgent
        try:
            agent = JobScoutAgent(flask_app)
            run_id = agent.run_for_user(_AUTHENTICATED_USER_ID, run_type="mcp_triggered")
            return json.dumps({
                "success": True,
                "run_id": run_id,
                "message": "Job Scout started. Results will be saved to database automatically.",
            })
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

@mcp.resource("resume://me/active")
def get_active_resume() -> str:
    """Raw resume text for the authenticated user."""
    _audit("resource:resume/active")
    with flask_app.app_context():
        from models import Resume
        from services.resume_service import extract_text_from_pdf

        resume = (
            Resume.query.filter_by(user_id=_AUTHENTICATED_USER_ID, is_active=True)
            .order_by(Resume.uploaded_at.desc())
            .first()
        )
        if not resume:
            return "No active resume found."
        return extract_text_from_pdf(resume.file_path)


@mcp.resource("resume://me/analysis")
def get_resume_analysis_resource() -> str:
    """Cached resume analysis for the authenticated user."""
    _audit("resource:resume/analysis")
    with flask_app.app_context():
        from models import Resume

        resume = (
            Resume.query.filter_by(user_id=_AUTHENTICATED_USER_ID, is_active=True)
            .order_by(Resume.uploaded_at.desc())
            .first()
        )
        if not resume or not resume.analysis:
            return "No analysis available. Upload a resume first."
        return resume.analysis


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    port = int(os.getenv("MCP_PORT", "8001"))

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", port=port)
