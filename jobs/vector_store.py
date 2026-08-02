"""
Job dense retrieval.

Single entry point for stage-1 dense search over job postings. On PostgreSQL
this is a pgvector index scan shared by every app instance; on SQLite it
delegates to the existing in-process FAISS index.

Both backends return (job_id, similarity) with similarity normalised to 0..1
(1.0 = identical), so callers need no backend-specific handling.
"""
import logging

from jobs.utils import get_embeddings, get_job_faiss_index
from services.pgvector_support import is_postgres, to_vector_literal

logger = logging.getLogger(__name__)


def _embed_query(query_text: str) -> list:
    """Embed the query. Separate function so tests can monkeypatch it."""
    return get_embeddings().embed_query(query_text)


def dense_search(query_text: str, k: int, query_vec: list | None = None) -> list[tuple[int, float]]:
    """Return the k nearest active jobs as (job_id, similarity), best first.

    query_vec: optional pre-computed embedding for query_text. Callers that
    are running on a gevent request greenlet should embed the query on a
    real OS thread first (embedding is a CPU-bound sentence-transformers
    forward pass) and pass the result here, so this function only issues
    the (I/O) SQL query on the greenlet. If omitted, the query is embedded
    internally via _embed_query — used by callers that already run on a
    real thread with app context (e.g. jobs/scout_agent.py) and by tests.
    """
    if k <= 0:
        return []
    if is_postgres():
        return _dense_search_pgvector(query_text, k, query_vec)
    return _dense_search_faiss(query_text, k)


def _dense_search_pgvector(query_text: str, k: int, query_vec: list | None = None) -> list[tuple[int, float]]:
    from sqlalchemy import text
    from models import db

    if query_vec is None:
        try:
            query_vec = _embed_query(query_text)
        except Exception as e:
            logger.error("Failed to embed job search query: %s", e)
            return []

    try:
        rows = db.session.execute(
            text(
                "SELECT id, 1 - (embedding_vec <=> CAST(:qvec AS vector)) AS similarity "
                "FROM job_postings "
                "WHERE is_active = true AND embedding_vec IS NOT NULL "
                "ORDER BY embedding_vec <=> CAST(:qvec AS vector) "
                "LIMIT :k"
            ),
            {"qvec": to_vector_literal(query_vec), "k": k},
        ).fetchall()
    except Exception as e:
        # On SQLite this genuinely falls back to a live FAISS index. On
        # Postgres there is no FAISS index to fall back to (get_job_faiss_index()
        # / build_job_faiss_index() return None there post-migration), so
        # _dense_search_faiss() just returns [] here — this is effectively a
        # fallback to an empty dense result, which then makes the caller
        # (find_matching_jobs) drop through to find_matching_jobs_old's
        # brute-force path, not a real FAISS search.
        logger.error("pgvector job search failed, falling back to FAISS: %s", e)
        db.session.rollback()
        return _dense_search_faiss(query_text, k)

    # Cosine similarity is -1..1; clamp so callers get a stable 0..1 score.
    return [(row[0], max(0.0, min(1.0, float(row[1])))) for row in rows]


def _dense_search_faiss(query_text: str, k: int) -> list[tuple[int, float]]:
    index = get_job_faiss_index()
    if index is None:
        logger.warning("No FAISS job index available")
        return []

    docs_with_scores = index.similarity_search_with_score(
        query_text, k=min(k, index.index.ntotal)
    )

    results = []
    for doc, distance in docs_with_scores:
        job_id = doc.metadata.get("job_id")
        if job_id is None:
            continue
        # FAISS returns squared L2 on normalised vectors; this is the same
        # conversion services/job_service.py used inline before.
        similarity = max(0.0, min(1.0, 1 - (distance ** 2 / 2)))
        results.append((job_id, similarity))
    return results
