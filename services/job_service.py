"""
Job Service

Handles job matching using FAISS vector search and LLM analysis.
No Flask routes here — pure business logic.
"""

import json
import numpy as np

from job_utils import embeddings, get_job_faiss_index
from models import JobPosting
from services.llm_service import get_job_matching_chain


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

    resume_embedding = embeddings.embed_query(resume_text)

    matches = []
    for job in jobs:
        job_text = f"{job.title} {job.description} {job.requirements or ''}"
        job_embedding = embeddings.embed_query(job_text)
        similarity_score = calculate_embedding_similarity(resume_embedding, job_embedding)

        try:
            analysis_result = get_job_matching_chain().run(
                resume=resume_text[:3000],
                job_title=job.title,
                company=job.company,
                job_description=job.description[:1000],
                job_requirements=job.requirements[:1000] if job.requirements else "Not specified"
            )
            analysis = json.loads(analysis_result)
        except Exception:
            analysis = {
                "match_score": similarity_score * 100,
                "matched_skills": [],
                "skill_gaps": [],
                "recommendation": "Analysis not available"
            }

        matches.append({'job': job, 'similarity_score': similarity_score, 'analysis': analysis})

    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    return matches[:top_k]


def find_matching_jobs(resume_text, top_k=5, candidate_k=20):
    """
    [OPTIMIZED] Two-stage job matching using FAISS + LLM.

    Stage 1: Fast FAISS vector search retrieves top candidate_k jobs.
    Stage 2: LLM analysis only on the top_k candidates.
    ~60-100x faster than brute-force for large job databases.
    """
    try:
        job_index = get_job_faiss_index()

        if job_index is None:
            print("No job index available, falling back to old method")
            return find_matching_jobs_old(resume_text, top_k)

        docs_with_scores = job_index.similarity_search_with_score(
            resume_text,
            k=min(candidate_k, job_index.index.ntotal)
        )

        if not docs_with_scores:
            return []

        matches = []
        for doc, distance in docs_with_scores[:top_k]:
            job_id = doc.metadata.get("job_id")
            job = JobPosting.query.get(job_id)

            if not job or not job.is_active:
                continue

            # Convert L2 distance to cosine similarity approximation
            similarity_score = max(0, min(1, 1 - (distance ** 2 / 2)))

            try:
                analysis_result = get_job_matching_chain().run(
                    resume=resume_text[:3000],
                    job_title=job.title,
                    company=job.company,
                    job_description=job.description[:1000],
                    job_requirements=job.requirements[:1000] if job.requirements else "Not specified"
                )
                analysis = json.loads(analysis_result)
            except Exception as e:
                print(f"LLM analysis failed for job {job.id}: {e}")
                analysis = {
                    "match_score": similarity_score * 100,
                    "matched_skills": [],
                    "skill_gaps": [],
                    "recommendation": "Analysis not available"
                }

            matches.append({'job': job, 'similarity_score': similarity_score, 'analysis': analysis})

        return matches

    except Exception as e:
        print(f"Error in optimized job matching: {e}, falling back to old method")
        return find_matching_jobs_old(resume_text, top_k)
