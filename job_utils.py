"""
Job Utilities - Shared functions for job embedding and FAISS indexing
"""

import os
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from models import db, JobPosting
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


# Initialize embeddings (shared across the app)
embeddings = HuggingFaceEmbeddings()

JOB_VECTOR_INDEX = 'job_vector_index'

def compute_job_embedding(job):
    """Compute and store embedding for a single job"""
    job_text = job.get_job_text()
    job.embedding = embeddings.embed_query(job_text)
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

    db.session.commit()
    return updated_count


def build_job_faiss_index():
    """Build FAISS index from all active job embeddings for fast similarity search"""
    # First ensure all jobs have embeddings
    compute_all_job_embeddings()

    # Get all active jobs with embeddings
    jobs = JobPosting.query.filter_by(is_active=True).filter(JobPosting.embedding.isnot(None)).all()

    if not jobs:
        print("No jobs available to build index")
        return None

    # Extract embeddings and metadata
    job_texts = [job.get_job_text() for job in jobs]
    job_embeddings = [job.embedding for job in jobs]
    job_metadatas = [{"job_id": job.id} for job in jobs]

    # Create FAISS vector store
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(job_texts, job_embeddings)),
        embedding=embeddings,
        metadatas=job_metadatas,
        distance_metric="cosine"
    )

    # Save to disk
    vectorstore.save_local(JOB_VECTOR_INDEX)
    print(f"Built FAISS index for {len(jobs)} jobs")

    return vectorstore


def get_job_faiss_index():
    """Load or build FAISS index for job embeddings"""
    try:
        # Try to load existing index
        if os.path.exists(os.path.join(JOB_VECTOR_INDEX, "index.faiss")):
            vectorstore = FAISS.load_local(
                JOB_VECTOR_INDEX,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        else:
            # Build new index if doesn't exist
            return build_job_faiss_index()
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        # Rebuild if loading fails
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

    # Get embedding dimension (HuggingFace default is 768 for most models)
    embedding_dim = 768
    if interested and interested[0].job.embedding is not None:
        embedding_dim = len(interested[0].job.embedding)
    elif not_interested and not_interested[0].job.embedding is not None:
        embedding_dim = len(not_interested[0].job.embedding)

    # Average embeddings of liked jobs
    if interested:
        liked_embeddings = [match.job.embedding for match in interested]
        liked_avg = np.mean(liked_embeddings, axis=0)
    else:
        liked_avg = np.zeros(embedding_dim)

    # Average embeddings of rejected jobs
    if not_interested:
        rejected_embeddings = [match.job.embedding for match in not_interested]
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

    # Update preference embedding
    config.preference_embedding = preference_vector
    config.preference_updated_at = datetime.utcnow()

    db.session.commit()

    print(f"Updated preference embedding for user {user_id}")
    return True


