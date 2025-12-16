"""
Job Utilities - Shared functions for job embedding and FAISS indexing
"""
import os
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from models import db, JobPosting

# Initialize embeddings (shared across the app)
embeddings = HuggingFaceEmbeddings()

# Job vector index path
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
        metadatas=job_metadatas
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
