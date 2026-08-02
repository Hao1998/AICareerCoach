"""
tests/test_find_matching_jobs_pgvector.py
==========================================
Regression test for C2: find_matching_jobs() must actually reach the pgvector
search path on PostgreSQL.

Before the fix, services/job_service.py dispatched dense_search() through
_run_in_thread(), which (when gevent is importable) hands the call to
gevent's real-OS-thread threadpool. Real threads have no Flask app context,
so is_postgres() inside dense_search() caught the resulting RuntimeError and
returned False -- meaning dense_search() ALWAYS took the FAISS branch on
Postgres, silently, with no error. This is exactly the scenario full unit
coverage of dense_search() in isolation (test_job_vector_store.py) could
never catch, because those tests call dense_search() directly from a
request-like app context, not via find_matching_jobs()'s real dispatch path.

These tests call find_matching_jobs() itself end-to-end against a real
Postgres-backed pg_app, and assert the pgvector code path was actually
exercised (and the FAISS fallback was not silently substituted).
"""
from datetime import datetime

import pytest

from models import db, JobPosting

DIM = 768


def _vec(val: float) -> list:
    return [val] * DIM


def _add_job(title="Backend Engineer", embedding=None):
    job = JobPosting(
        title=title,
        company="Acme",
        description="Build backend services in Python.",
        requirements="Python, SQL",
        source="test",
        is_active=True,
        embedding=embedding or _vec(0.1),
        embedding_updated_at=datetime(2026, 1, 1),
    )
    db.session.add(job)
    db.session.commit()
    return job


class _FakeMatchResult:
    match_score = 88
    matched_skills = ["Python"]
    skill_gaps = []
    recommendation = "Strong match"


@pytest.mark.postgres
def test_find_matching_jobs_reaches_pgvector_not_faiss(pg_app, monkeypatch):
    """find_matching_jobs() on Postgres must query pgvector, not silently
    fall back to FAISS (which has no local index in this test and would
    return no results, or worse, stale results from another test)."""
    with pg_app.app_context():
        job = _add_job()

        from jobs.utils import sync_job_vectors
        sync_job_vectors()

        monkeypatch.setattr("jobs.vector_store._embed_query", lambda text: _vec(0.1))
        monkeypatch.setattr("services.job_service.run_job_matching", lambda **kw: _FakeMatchResult())

        pgvector_calls = []
        faiss_calls = []

        from jobs import vector_store
        real_pgvector = vector_store._dense_search_pgvector
        real_faiss = vector_store._dense_search_faiss

        def _spy_pgvector(*args, **kwargs):
            pgvector_calls.append(1)
            return real_pgvector(*args, **kwargs)

        def _spy_faiss(*args, **kwargs):
            faiss_calls.append(1)
            return real_faiss(*args, **kwargs)

        monkeypatch.setattr("jobs.vector_store._dense_search_pgvector", _spy_pgvector)
        monkeypatch.setattr("jobs.vector_store._dense_search_faiss", _spy_faiss)

        from services.job_service import find_matching_jobs
        results = find_matching_jobs("Experienced Python backend engineer", top_k=1, candidate_k=5)

        assert pgvector_calls, "pgvector search path was never reached — bug regressed"
        assert not faiss_calls, "dense search silently fell back to FAISS instead of using pgvector"
        assert len(results) == 1
        assert results[0]["job"].id == job.id


@pytest.mark.postgres
def test_find_matching_jobs_dense_dispatch_does_not_use_thread_pool(pg_app, monkeypatch):
    """The dense-search SQL query itself must resolve is_postgres()/
    dense_search() on the request greenlet directly, not via _run_in_thread —
    dispatching the whole call through a real OS thread loses Flask app
    context and is exactly what caused the original bug.

    N2 regression: the query embedding IS CPU-bound (a sentence-transformers
    forward pass) and must go through _run_in_thread — only the embedding
    step, not the SQL query or dense_search() itself."""
    with pg_app.app_context():
        _add_job()

        from jobs.utils import sync_job_vectors
        sync_job_vectors()

        def _fake_embed_query(text):
            return _vec(0.1)

        monkeypatch.setattr("jobs.vector_store._embed_query", _fake_embed_query)
        monkeypatch.setattr("services.job_service.run_job_matching", lambda **kw: _FakeMatchResult())

        import services.job_service as job_service_module
        real_run_in_thread = job_service_module._run_in_thread
        thread_dispatched_fns = []

        def _tracking_run_in_thread(fn, *args, **kwargs):
            thread_dispatched_fns.append(getattr(fn, "__name__", repr(fn)))
            return real_run_in_thread(fn, *args, **kwargs)

        monkeypatch.setattr(job_service_module, "_run_in_thread", _tracking_run_in_thread)

        results = job_service_module.find_matching_jobs(
            "Experienced Python backend engineer", top_k=1, candidate_k=5
        )

        assert results, "expected at least one match"
        assert "dense_search" not in thread_dispatched_fns
        assert "_dense_search_faiss_threaded" not in thread_dispatched_fns
        assert "_fake_embed_query" in thread_dispatched_fns, (
            "the CPU-bound query embedding must be dispatched via "
            "_run_in_thread, not run directly on the gevent request greenlet"
        )
