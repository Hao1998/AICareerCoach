"""
Job Service

Handles job matching using dense vector search (pgvector on PostgreSQL,
FAISS on SQLite) and LLM analysis, and the shared Adzuna fetch + config-save
logic used by both the HTML and JSON fetch endpoints.
No Flask routes here — pure business logic.
"""

import logging
import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from langsmith import traceable

from jobs.utils import get_bm25_index, tokenize_for_bm25
from jobs import vector_store as _vector_store_module
from jobs.vector_store import dense_search
from jobs.fetcher import fetch_jobs_from_adzuna
from jobs.fetchers.registry import fetch_from_sources
from services.pgvector_support import is_postgres

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


def _rank_by_stored_embedding(query_vec, rows, top_k: int) -> list[tuple[int, float]]:
    """Exact cosine ranking of (job_id, embedding) rows against query_vec.

    Returns the top_k (job_id, similarity) pairs, best first, with similarity
    clamped to 0..1 to match what dense_search() returns.
    """
    job_ids = [row[0] for row in rows]
    matrix = np.asarray([row[1] for row in rows], dtype=np.float32)
    query = np.asarray(query_vec, dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    norms[norms == 0] = 1.0
    similarities = (matrix @ query) / norms

    top = np.argsort(similarities)[::-1][:top_k]
    return [(job_ids[i], max(0.0, min(1.0, float(similarities[i])))) for i in top]


def find_matching_jobs_from_db(resume_text, top_k=5):
    """Fallback when dense search returns nothing (e.g. no FAISS index yet).

    Ranks every active job by cosine similarity against its stored
    JobPosting.embedding — no job is re-embedded — then runs LLM analysis on
    the top_k only. Jobs without a stored embedding are skipped; they are
    embedded by compute_all_job_embeddings() on the next index build.
    """
    from flask import current_app
    from models import db
    app = current_app._get_current_object()

    # A JSON column stores Python None as JSON 'null', which isnot(None) does
    # not exclude — so drop empty embeddings here as well.
    rows = [
        row for row in (
            db.session.query(JobPosting.id, JobPosting.embedding)
            .filter(JobPosting.is_active == True, JobPosting.embedding.isnot(None))
            .all()
        )
        if row[1]
    ]
    if not rows:
        logger.warning("No stored job embeddings available for fallback search")
        return []

    query_vec = _run_in_thread(_vector_store_module._embed_query, resume_text)
    ranked = _run_in_thread(_rank_by_stored_embedding, query_vec, rows, top_k)

    jobs_by_id = {
        j.id: j
        for j in JobPosting.query.filter(JobPosting.id.in_([jid for jid, _ in ranked])).all()
    }
    candidates = [(jobs_by_id[jid], score) for jid, score in ranked if jid in jobs_by_id]
    return _analyze_candidates(app, resume_text, candidates)


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


def _analyze_candidates(app, resume_text: str, candidates: list[tuple[JobPosting, float]]) -> list[dict]:
    """Run _analyze_job over (job, similarity) pairs in parallel, keeping input order."""
    # Preserve ranking order even though futures complete out of order.
    ordered_results = [None] * len(candidates)

    with ThreadPoolExecutor(max_workers=_LLM_CONCURRENCY) as pool:
        future_to_idx = {
            pool.submit(_analyze_job, app, resume_text, job, score): idx
            for idx, (job, score) in enumerate(candidates)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                ordered_results[idx] = future.result()
            except Exception as e:
                job, score = candidates[idx]
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


def _dense_search_faiss_threaded(app, resume_text: str, candidate_k: int):
    """Run the SQLite dense-search branch inside a real OS thread, with app context.

    With VECTOR_BACKEND=redis this is Redis first, then the FAISS fallback.

    FAISS and the HuggingFace embedding model are CPU-bound C extensions gevent
    cannot patch, so they need a real thread (see _run_in_thread above). Real
    threads have no Flask app context, so a cold-start FAISS build (which
    queries JobPosting) needs one explicitly — same pattern as _analyze_job.
    """
    with app.app_context():
        return dense_search(resume_text, candidate_k)


@traceable(run_type="chain", name="job-matching")
def find_matching_jobs(resume_text, top_k=5, candidate_k=20):
    """
    Three-stage hybrid job matching: dense + sparse retrieval → RRF fusion → LLM analysis.

    Stage 1a (dense):  vector search (pgvector on Postgres, FAISS on SQLite) — catches
                        semantic similarity.
    Stage 1b (sparse): BM25 keyword search — catches exact tech-stack matches.
    Stage 1c (fusion): Reciprocal Rank Fusion merges both ranked lists.
    Stage 2 (rerank):  Parallel LLM analysis on the fused top_k candidates.
    """
    from flask import current_app
    app = current_app._get_current_object()

    try:
        # --- Stage 1a: dense search (pgvector on Postgres, FAISS on SQLite) ---
        # is_postgres() must be resolved here, on the request greenlet (which
        # has app context) — not inside _run_in_thread's real OS thread, which
        # has none and would make is_postgres() (and therefore dense_search)
        # silently and permanently take the FAISS branch on Postgres.
        #
        # The pgvector SELECT itself is a fast, indexed DB query — blocking
        # on it briefly is the same pre-existing tradeoff every other DB call
        # in this app already makes on the request greenlet; it is not
        # actually cooperative (psycopg-binary, pinned in requirements.txt,
        # uses a C waiter that gevent's monkey-patching does not touch), it's
        # just fast enough not to matter. The embedding itself is the real
        # CPU-bound cost (a sentence-transformers forward pass) and must not
        # run on the greenlet, so it's computed on a real OS thread via
        # _run_in_thread first, and only the resulting vector — plus the SQL
        # query — runs on the greenlet. Dispatches through the
        # jobs.vector_store module attribute (not a `from ... import
        # _embed_query`) so it's the same seam dense_search() itself uses
        # internally, and so test monkeypatches of
        # "jobs.vector_store._embed_query" still take effect here.
        # On SQLite with the Redis vector experiment on, dense_search() tries
        # Redis and then falls back to FAISS — so it stays on the thread below,
        # keeping that CPU-bound fallback off the greenlet.
        if is_postgres():
            query_vec = _run_in_thread(_vector_store_module._embed_query, resume_text)
            dense_results = dense_search(resume_text, candidate_k, query_vec=query_vec)
        else:
            dense_results = _run_in_thread(
                _dense_search_faiss_threaded, app, resume_text, candidate_k
            )

        if not dense_results:
            logger.warning("No dense results available, falling back to stored-embedding search")
            return find_matching_jobs_from_db(resume_text, top_k)

        dense_ranked = [job_id for job_id, _ in dense_results]
        # Similarity is already normalised 0..1 by the vector store.
        dense_score_map = dict(dense_results)

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
            (jobs_by_id[jid], dense_score_map.get(jid, 0.5))
            for jid in top_fused_ids
            if jid in jobs_by_id
        ]

        if not active_candidates:
            return []

        # --- Stage 2: parallel LLM analysis ---
        return _analyze_candidates(app, resume_text, active_candidates)

    except Exception as e:
        logger.error("Error in hybrid job matching: %s, falling back to stored-embedding search", e)
        return find_matching_jobs_from_db(resume_text, top_k)


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
