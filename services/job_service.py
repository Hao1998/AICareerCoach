"""
Job Service

Handles job matching using FAISS vector search and LLM analysis,
and the shared Adzuna fetch + config-save logic used by both the
HTML and JSON fetch endpoints.
No Flask routes here — pure business logic.
"""

import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from langsmith import traceable

from job_utils import embeddings, get_job_faiss_index
from job_fetcher import fetch_jobs_from_adzuna

# FAISS and HuggingFace embeddings are CPU-bound C extensions that gevent
# cannot patch. Spawning them on gevent's thread pool hands the blocking work
# to a real OS thread and yields the event loop to other greenlets.
try:
    from gevent import get_hub as _get_hub
    def _run_in_thread(fn, *args, **kwargs):
        return _get_hub().threadpool.spawn(fn, *args, **kwargs).get()
except ImportError:
    def _run_in_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)
from models import JobPosting
from services.llm_service import run_job_matching

logger = logging.getLogger(__name__)

# Max concurrent LLM calls per find_matching_jobs invocation.
# Grok-3 rate limits are generous; keep this ≤ top_k to avoid wasted calls.
_LLM_CONCURRENCY = 5


def calculate_embedding_similarity(resume_embedding, job_embedding):
    """Calculate cosine similarity between two embeddings"""
    similarity = np.dot(resume_embedding, job_embedding) / (
        np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
    )
    return float(similarity)


def find_matching_jobs_old(resume_text, top_k=5):
    """[DEPRECATED] Brute-force method — kept as fallback for when FAISS index is unavailable"""
    jobs = JobPosting.query.filter_by(is_active=True).all()

    if not jobs:
        return []

    resume_embedding = _run_in_thread(embeddings.embed_query, resume_text)

    matches = []
    for job in jobs:
        job_text = f"{job.title} {job.description} {job.requirements or ''}"
        job_embedding = _run_in_thread(embeddings.embed_query, job_text)
        similarity_score = calculate_embedding_similarity(resume_embedding, job_embedding)

        try:
            result = run_job_matching(
                resume=resume_text[:3000],
                job_title=job.title,
                company=job.company,
                job_description=job.description[:1000],
                job_requirements=job.requirements[:1000] if job.requirements else "Not specified",
            )
            analysis = {
                "match_score": result.match_score,
                "matched_skills": result.matched_skills,
                "skill_gaps": result.skill_gaps,
                "recommendation": result.recommendation,
            }
        except Exception:
            analysis = {
                "match_score": int(similarity_score * 100),
                "matched_skills": [],
                "skill_gaps": [],
                "recommendation": "Analysis not available"
            }

        matches.append({'job': job, 'similarity_score': similarity_score, 'analysis': analysis})

    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    return matches[:top_k]


def _analyze_job(app, resume_text: str, job: JobPosting, similarity_score: float) -> dict:
    """Run one LLM job-match analysis. Called concurrently from find_matching_jobs."""
    with app.app_context():
        try:
            result = run_job_matching(
                resume=resume_text[:3000],
                job_title=job.title,
                company=job.company,
                job_description=job.description[:1000],
                job_requirements=job.requirements[:1000] if job.requirements else "Not specified",
            )
            # result is a JobMatchResult — convert to dict to keep the downstream interface stable
            analysis = {
                "match_score": result.match_score,
                "matched_skills": result.matched_skills,
                "skill_gaps": result.skill_gaps,
                "recommendation": result.recommendation,
            }
        except Exception as e:
            logger.warning("LLM analysis failed for job %s: %s", job.id, e)
            analysis = {
                "match_score": int(similarity_score * 100),
                "matched_skills": [],
                "skill_gaps": [],
                "recommendation": "Analysis not available",
            }
    return {"job": job, "similarity_score": similarity_score, "analysis": analysis}


@traceable(run_type="chain", name="job-matching")
def find_matching_jobs(resume_text, top_k=5, candidate_k=20):
    """
    Two-stage job matching: FAISS candidate retrieval → parallel LLM analysis.

    Stage 1: Fast FAISS vector search retrieves top candidate_k jobs.
    Stage 2: LLM analysis on the top_k candidates runs concurrently
             (up to _LLM_CONCURRENCY threads), cutting wall-time from
             ~N×LLM_latency down to ~1×LLM_latency.
    """
    try:
        job_index = get_job_faiss_index()

        if job_index is None:
            logger.warning("No job index available, falling back to brute-force")
            return find_matching_jobs_old(resume_text, top_k)

        docs_with_scores = _run_in_thread(
            job_index.similarity_search_with_score,
            resume_text,
            k=min(candidate_k, job_index.index.ntotal),
        )

        if not docs_with_scores:
            return []

        # --- Stage 1: batch-load all candidate jobs in a single query ---
        candidate_meta = []
        for doc, distance in docs_with_scores[:top_k]:
            job_id = doc.metadata.get("job_id")
            similarity_score = max(0, min(1, 1 - (distance ** 2 / 2)))
            candidate_meta.append((job_id, similarity_score))

        job_ids = [jid for jid, _ in candidate_meta]
        jobs_by_id = {
            j.id: j
            for j in JobPosting.query.filter(
                JobPosting.id.in_(job_ids), JobPosting.is_active == True
            ).all()
        }

        active_candidates = [
            (jobs_by_id[jid], score)
            for jid, score in candidate_meta
            if jid in jobs_by_id
        ]

        if not active_candidates:
            return []

        # --- Stage 2: parallel LLM analysis ---
        # Preserve FAISS ranking order in the output even though futures complete
        # out of order, so the caller always gets the best-ranked jobs first.
        ordered_results = [None] * len(active_candidates)

        from flask import current_app
        app = current_app._get_current_object()
        with ThreadPoolExecutor(max_workers=_LLM_CONCURRENCY) as pool:
            future_to_idx = {
                pool.submit(_analyze_job, app, resume_text, job, score): idx
                for idx, (job, score) in enumerate(active_candidates)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as e:
                    job, score = active_candidates[idx]
                    logger.error("Unexpected error analyzing job %s: %s", job.id, e)
                    ordered_results[idx] = {
                        "job": job,
                        "similarity_score": score,
                        "analysis": {
                            "match_score": score * 100,
                            "matched_skills": [],
                            "skill_gaps": [],
                            "recommendation": "Analysis not available",
                        },
                    }

        return [r for r in ordered_results if r is not None]

    except Exception as e:
        logger.error("Error in optimized job matching: %s, falling back to brute-force", e)
        return find_matching_jobs_old(resume_text, top_k)


def fetch_and_save_jobs(user_id: int, keywords, location, max_jobs: int, max_days_old: int) -> dict:
    """
    Validate params, persist Adzuna preferences to AgentConfig, fetch jobs,
    and return the stats dict from fetch_jobs_from_adzuna.

    Raises ValueError for invalid params so the caller can surface the message.
    Used by both the HTML form endpoint and the JSON API endpoint.
    """
    from models import AgentConfig, db

    if max_jobs < 1 or max_jobs > 200:
        raise ValueError("max_jobs must be between 1 and 200")

    config = AgentConfig.query.filter_by(user_id=user_id).first()
    if not config:
        config = AgentConfig(user_id=user_id)
        db.session.add(config)

    if location is not None:
        config.adzuna_location = location if str(location).strip() else None
    config.adzuna_max_jobs = max_jobs
    config.adzuna_max_days_old = max_days_old
    db.session.commit()

    return fetch_jobs_from_adzuna(
        keywords=keywords, location=location,
        max_jobs=max_jobs, max_days_old=max_days_old,
    )
