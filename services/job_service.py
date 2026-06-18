"""
Job Service

Handles job matching using FAISS vector search and LLM analysis,
and the shared Adzuna fetch + config-save logic used by both the
HTML and JSON fetch endpoints.
No Flask routes here — pure business logic.
"""

import logging
import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from langsmith import traceable

from jobs.utils import get_embeddings, get_job_faiss_index, get_bm25_index, tokenize_for_bm25
from jobs.fetcher import fetch_jobs_from_adzuna
from jobs.fetchers.registry import fetch_from_sources

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

_LLM_CONCURRENCY = int(os.environ.get('LLM_CONCURRENCY', '5'))


def _reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge multiple ranked job-id lists using Reciprocal Rank Fusion (RRF).

    k=60 is the standard constant from the original RRF paper (Cormack 2009).
    Higher k reduces the influence of top-rank positions.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, job_id in enumerate(ranked):
            scores[job_id] = scores.get(job_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def calculate_embedding_similarity(resume_embedding, job_embedding):
    """Calculate cosine similarity between two embeddings"""
    resume_embedding = np.array(resume_embedding)
    job_embedding = np.array(job_embedding)
    similarity = np.dot(resume_embedding, job_embedding) / (
        np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
    )
    return float(similarity)


def find_matching_jobs_old(resume_text, top_k=5):
    """[DEPRECATED] Brute-force method — kept as fallback for when FAISS index is unavailable"""
    jobs = JobPosting.query.filter_by(is_active=True).all()

    if not jobs:
        return []

    resume_embedding = _run_in_thread(get_embeddings().embed_query, resume_text)

    matches = []
    for job in jobs:
        job_text = f"{job.title} {job.description} {job.requirements or ''}"
        job_embedding = _run_in_thread(get_embeddings().embed_query, job_text)
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
    Three-stage hybrid job matching: dense + sparse retrieval → RRF fusion → LLM analysis.

    Stage 1a (dense):  FAISS vector search — catches semantic similarity.
    Stage 1b (sparse): BM25 keyword search — catches exact tech-stack matches.
    Stage 1c (fusion): Reciprocal Rank Fusion merges both ranked lists.
    Stage 2 (rerank):  Parallel LLM analysis on the fused top_k candidates.
    """
    try:
        job_index = get_job_faiss_index()

        if job_index is None:
            logger.warning("No job index available, falling back to brute-force")
            return find_matching_jobs_old(resume_text, top_k)

        # --- Stage 1a: FAISS dense search ---
        docs_with_scores = _run_in_thread(
            job_index.similarity_search_with_score,
            resume_text,
            k=min(candidate_k, job_index.index.ntotal),
        )

        dense_ranked = [
            doc.metadata["job_id"]
            for doc, _ in docs_with_scores
            if doc.metadata.get("job_id") is not None
        ]
        # Build FAISS score map for use as fallback similarity score in Stage 2
        faiss_score_map = {
            doc.metadata["job_id"]: max(0.0, min(1.0, 1 - (dist ** 2 / 2)))
            for doc, dist in docs_with_scores
            if doc.metadata.get("job_id") is not None
        }

        # --- Stage 1b: BM25 sparse search ---
        sparse_ranked: list[int] = []
        bm25_result = get_bm25_index()
        if bm25_result is not None:
            bm25, bm25_job_ids = bm25_result
            query_tokens = tokenize_for_bm25(resume_text)
            bm25_scores = _run_in_thread(bm25.get_scores, query_tokens)
            top_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
            sparse_ranked = [bm25_job_ids[i] for i in top_indices]

        # --- Stage 1c: RRF fusion ---
        ranked_lists = [dense_ranked] + ([sparse_ranked] if sparse_ranked else [])
        fused = _reciprocal_rank_fusion(ranked_lists)
        top_fused_ids = [job_id for job_id, _ in fused[:top_k]]

        if not top_fused_ids:
            return []

        # --- Batch-load candidate jobs ---
        jobs_by_id = {
            j.id: j
            for j in JobPosting.query.filter(
                JobPosting.id.in_(top_fused_ids), JobPosting.is_active == True
            ).all()
        }
        active_candidates = [
            (jobs_by_id[jid], faiss_score_map.get(jid, 0.5))
            for jid in top_fused_ids
            if jid in jobs_by_id
        ]

        if not active_candidates:
            return []

        # --- Stage 2: parallel LLM analysis ---
        # Preserve fused ranking order even though futures complete out of order.
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
        logger.error("Error in hybrid job matching: %s, falling back to brute-force", e)
        return find_matching_jobs_old(resume_text, top_k)


def fetch_and_save_jobs(user_id: int, keywords, location, max_jobs: int,
                        max_days_old: int, sources: list[str] | None = None) -> dict:
    """
    Validate params, persist preferences to AgentConfig, fetch jobs from
    selected sources, and return aggregated stats.

    When sources is None, falls back to the user's enabled_sources config
    (default: ["adzuna"] for backward compat).

    Raises ValueError for invalid params so the caller can surface the message.
    Used by both the HTML form endpoint and the JSON API endpoint.
    """
    from models import AgentConfig, db
    from services.db_lock import safe_commit

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
    if sources is not None:
        config.enabled_sources = sources
    safe_commit()

    active_sources = sources or config.enabled_sources or ['adzuna']

    if active_sources == ['adzuna']:
        return fetch_jobs_from_adzuna(
            keywords=keywords, location=location,
            max_jobs=max_jobs, max_days_old=max_days_old,
        )

    result = fetch_from_sources(
        sources=active_sources,
        keywords=keywords,
        location=location,
        max_jobs_per_source=max_jobs,
        max_days_old=max_days_old,
    )
    return result['totals']
