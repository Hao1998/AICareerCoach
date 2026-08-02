"""
Job Utilities - Shared functions for job embedding and FAISS indexing

NOTE (future migration): the FAISS index here is in-process and saved to local
disk, so it cannot be shared across multiple app instances and must be rebuilt
per process. This is acceptable for the current single-/few-instance deployment.
When we scale horizontally, move embeddings into Postgres `pgvector` (already on
PostgreSQL in prod) so the vector store is shared and concurrent-write safe.
Tracked as a deliberate follow-up — keep FAISS for now.
"""

import logging
import os
import pickle
import re
import threading
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from models import db, JobPosting
from services.db_lock import safe_commit
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

logger = logging.getLogger(__name__)

_embeddings = None
_embeddings_lock = threading.Lock()


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                )
    return _embeddings

JOB_VECTOR_INDEX = 'job_vector_index'

_index_rebuild_lock = threading.RLock()

_faiss_cache = None
_faiss_cache_lock = threading.RLock()
_faiss_cache_mtime = 0

_bm25_cache: tuple | None = None
_bm25_lock = threading.RLock()


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25, preserving tech terms like C++, Node.js, .NET."""
    return re.findall(r'[a-zA-Z0-9][a-zA-Z0-9+#.\-]*', text.lower())

def compute_job_embedding(job):
    """Compute and store embedding for a single job"""
    job_text = job.get_job_text()
    vec = get_embeddings().embed_query(job_text)
    job.embedding = vec if isinstance(vec, list) else list(vec)
    job.embedding_updated_at = datetime.utcnow()
    return job.embedding

def compute_all_job_embeddings():
    """Pre-compute embeddings for all active jobs (run once or when jobs are updated)"""
    jobs = JobPosting.query.filter_by(is_active=True).all()

    updated_count = 0
    for job in jobs:
        if job.embedding is None:
            compute_job_embedding(job)
            updated_count += 1

    safe_commit()
    return updated_count


def sync_job_vectors() -> int:
    """Copy JSON embeddings into the native pgvector column.

    Set-based: one UPDATE, no Python loop. Rows are refreshed when the vector
    is missing or when embedding_updated_at has moved past the value recorded
    at the last sync, so re-fetched jobs pick up new embeddings.

    Returns the number of rows updated; 0 on non-PostgreSQL engines.
    """
    from services.pgvector_support import is_postgres
    if not is_postgres():
        return 0

    from sqlalchemy import text
    try:
        result = db.session.execute(text(
            "UPDATE job_postings "
            "SET embedding_vec = CAST(embedding::text AS vector), "
            "    embedding_vec_updated_at = embedding_updated_at "
            "WHERE embedding IS NOT NULL "
            "  AND (embedding_vec IS NULL "
            "       OR embedding_vec_updated_at IS DISTINCT FROM embedding_updated_at)"
        ))
        db.session.commit()
        if result.rowcount:
            logger.info("Synced %d job embeddings to pgvector", result.rowcount)
        return result.rowcount
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to sync job vectors: %s", e)
        return 0


def build_bm25_index():
    """Build in-memory BM25 index from all active jobs. Persist to Redis for fast recovery."""
    global _bm25_cache
    with _bm25_lock:
        jobs = JobPosting.query.filter_by(is_active=True).all()
        if not jobs:
            _bm25_cache = None
            return None
        corpus = [tokenize_for_bm25(job.get_job_text()) for job in jobs]
        job_ids = [job.id for job in jobs]
        bm25 = BM25Okapi(corpus)
        _bm25_cache = (bm25, job_ids)
        logger.info("Built BM25 index for %d jobs", len(jobs))

        try:
            from services.redis_client import get_redis_binary
            r = get_redis_binary()
            r.set('bm25:index', pickle.dumps(bm25), ex=86400)
            r.set('bm25:job_ids', pickle.dumps(job_ids), ex=86400)
        except Exception as e:
            logger.warning("Failed to persist BM25 to Redis: %s", e)

        return _bm25_cache


def get_bm25_index() -> tuple | None:
    """Return cached BM25 index. Tries in-memory → Redis → DB rebuild."""
    global _bm25_cache
    with _bm25_lock:
        if _bm25_cache is not None:
            return _bm25_cache

        try:
            from services.redis_client import get_redis_binary
            r = get_redis_binary()
            bm25_data = r.get('bm25:index')
            ids_data = r.get('bm25:job_ids')
            if bm25_data and ids_data:
                _bm25_cache = (pickle.loads(bm25_data), pickle.loads(ids_data))
                logger.info("Loaded BM25 index from Redis")
                return _bm25_cache
        except Exception:
            pass

        return build_bm25_index()


def invalidate_bm25_index():
    """Drop the BM25 cache (memory + Redis) so the next access rebuilds it."""
    global _bm25_cache
    with _bm25_lock:
        _bm25_cache = None
    try:
        from services.redis_client import get_redis_binary
        r = get_redis_binary()
        r.delete('bm25:index', 'bm25:job_ids')
    except Exception:
        pass


def build_job_faiss_index():
    """Build FAISS index from all active job embeddings for fast similarity search."""
    global _faiss_cache, _faiss_cache_mtime
    with _index_rebuild_lock:
        compute_all_job_embeddings()

        # On PostgreSQL the pgvector column is the real search index; keep it
        # current at the same points the FAISS index is rebuilt.
        sync_job_vectors()

        jobs = JobPosting.query.filter_by(is_active=True).filter(JobPosting.embedding.isnot(None)).all()

        if not jobs:
            logger.info("No jobs available to build index")
            return None

        job_texts = [job.get_job_text() for job in jobs]
        job_embeddings_list = [job.embedding for job in jobs]
        job_metadatas = [{"job_id": job.id} for job in jobs]

        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(job_texts, job_embeddings_list)),
            embedding=get_embeddings(),
            metadatas=job_metadatas,
            distance_metric="cosine"
        )

        vectorstore.save_local(JOB_VECTOR_INDEX)
        logger.info("Built FAISS index for %d jobs", len(jobs))

        with _faiss_cache_lock:
            _faiss_cache = vectorstore
            _faiss_cache_mtime = os.path.getmtime(
                os.path.join(JOB_VECTOR_INDEX, "index.faiss")
            )

        invalidate_bm25_index()

        return vectorstore


def get_job_faiss_index():
    """Load FAISS index from disk, caching in-process. Reloads if file changed."""
    global _faiss_cache, _faiss_cache_mtime

    index_path = os.path.join(JOB_VECTOR_INDEX, "index.faiss")

    if not os.path.exists(index_path):
        return build_job_faiss_index()

    current_mtime = os.path.getmtime(index_path)

    with _faiss_cache_lock:
        if _faiss_cache is not None and current_mtime == _faiss_cache_mtime:
            return _faiss_cache

        try:
            _faiss_cache = FAISS.load_local(
                JOB_VECTOR_INDEX,
                get_embeddings(),
                allow_dangerous_deserialization=True
            )
            _faiss_cache_mtime = current_mtime
            return _faiss_cache
        except Exception as e:
            logger.error("Failed to load FAISS index: %s", e)
            return build_job_faiss_index()


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors

    Args:
        vec1: First vector (numpy array)
        vec2: Second vector (numpy array)

    Returns:
        float: Cosine similarity score between -1 and 1
    """
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return sklearn_cosine_similarity(vec1, vec2)[0][0]


def compute_user_preference_embedding(user_id):
    """
    Learn user preferences from feedback history

    This creates a user preference vector by:
    1. Averaging embeddings of jobs marked 'interested' or 'applied'
    2. Averaging embeddings of jobs marked 'not_interested'
    3. Computing: preference = liked_jobs - 0.5 * rejected_jobs

    The resulting vector represents what the user likes in the embedding space.
    Job embeddings themselves are NOT changed - this is a separate user vector.

    Args:
        user_id: User ID to compute preferences for

    Returns:
        numpy.ndarray: User preference vector, or None if no feedback yet
    """
    from models import JobMatch

    # Get feedback history - jobs user was interested in
    interested = JobMatch.query.filter(
        JobMatch.user_id == user_id,
        JobMatch.user_feedback.in_(['interested', 'applied'])
    ).join(JobPosting).filter(
        JobPosting.embedding.isnot(None)
    ).all()

    # Get feedback history - jobs user was not interested in
    not_interested = JobMatch.query.filter_by(
        user_id=user_id,
        user_feedback='not_interested'
    ).join(JobPosting).filter(
        JobPosting.embedding.isnot(None)
    ).all()

    # Need at least some feedback to learn preferences
    if not interested and not not_interested:
        return None

    embedding_dim = 768
    if interested and interested[0].job.embedding is not None:
        embedding_dim = len(interested[0].job.embedding)
    elif not_interested and not_interested[0].job.embedding is not None:
        embedding_dim = len(not_interested[0].job.embedding)

    if interested:
        liked_embeddings = [np.array(match.job.embedding) for match in interested]
        liked_avg = np.mean(liked_embeddings, axis=0)
    else:
        liked_avg = np.zeros(embedding_dim)

    if not_interested:
        rejected_embeddings = [np.array(match.job.embedding) for match in not_interested]
        rejected_avg = np.mean(rejected_embeddings, axis=0)
    else:
        rejected_avg = np.zeros(embedding_dim)

    # User preference = what they like - 50% of what they don't like
    # This creates a vector pointing toward liked jobs and away from rejected ones
    preference_vector = liked_avg - 0.5 * rejected_avg

    # Normalize the vector
    norm = np.linalg.norm(preference_vector)
    if norm > 0:
        preference_vector = preference_vector / norm

    return preference_vector


def update_user_preferences(user_id):
    """
    Update user preference embedding in AgentConfig

    Call this after user provides feedback to update their learned preferences.

    Args:
        user_id: User ID to update preferences for

    Returns:
        bool: True if updated successfully, False if no preferences computed
    """
    from models import AgentConfig

    # Compute new preference vector
    preference_vector = compute_user_preference_embedding(user_id)

    if preference_vector is None:
        return False

    # Get or create agent config
    config = AgentConfig.query.filter_by(user_id=user_id).first()
    if not config:
        config = AgentConfig(user_id=user_id)
        db.session.add(config)

    config.preference_embedding = preference_vector.tolist()
    config.preference_updated_at = datetime.utcnow()

    safe_commit()

    print(f"Updated preference embedding for user {user_id}")
    return True


