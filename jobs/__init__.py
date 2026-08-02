# Re-export the public API so callers can do `from jobs import build_job_faiss_index`
from jobs.utils import (
    get_embeddings,
    JOB_VECTOR_INDEX,
    compute_job_embedding,
    compute_all_job_embeddings,
    build_job_faiss_index,
    get_job_faiss_index,
    cosine_similarity,
    update_user_preferences,
)
from jobs.vector_store import dense_search
from jobs.fetcher import AdzunaJobFetcher, fetch_jobs_from_adzuna
from jobs.scheduler import init_scheduler
from jobs.scout_agent import JobScoutAgent, get_run_events, cleanup_run_progress

__all__ = [
    "get_embeddings", "JOB_VECTOR_INDEX",
    "compute_job_embedding", "compute_all_job_embeddings",
    "build_job_faiss_index", "get_job_faiss_index",
    "cosine_similarity", "update_user_preferences",
    "dense_search",
    "AdzunaJobFetcher", "fetch_jobs_from_adzuna",
    "init_scheduler",
    "JobScoutAgent", "get_run_events", "cleanup_run_progress",
]
