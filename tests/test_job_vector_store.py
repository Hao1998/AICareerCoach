"""
tests/test_job_vector_store.py
==============================
dense_search returns (job_id, similarity) ordered best-first, with similarity
normalised to 0..1 on both backends.

The pgvector cases need a real Postgres; the FAISS fallback case runs on
SQLite with a stubbed index so no embedding model is loaded.
"""
from datetime import datetime

import pytest

from models import db, JobPosting

DIM = 768


def _vec(val: float) -> list:
    return [val] * DIM


def _add_job(title, embedding, is_active=True):
    job = JobPosting(
        title=title,
        company="Acme",
        description="Builds things",
        source="test",
        is_active=is_active,
        embedding=embedding,
        embedding_updated_at=datetime(2026, 1, 1),
    )
    db.session.add(job)
    db.session.commit()
    return job


@pytest.mark.postgres
def test_dense_search_orders_by_similarity(pg_app, monkeypatch):
    with pg_app.app_context():
        near = _add_job("near job", [0.1] * DIM)
        far = _add_job("far job", [0.1] * (DIM - 1) + [9.0])

        from jobs.utils import sync_job_vectors
        sync_job_vectors()

        monkeypatch.setattr(
            "jobs.vector_store._embed_query",
            lambda text: [0.1] * DIM,
        )

        from jobs.vector_store import dense_search
        results = dense_search("anything", k=5)

        ids = [job_id for job_id, _ in results]
        assert ids[0] == near.id
        assert far.id in ids


@pytest.mark.postgres
def test_dense_search_excludes_inactive_jobs(pg_app, monkeypatch):
    with pg_app.app_context():
        active = _add_job("active job", _vec(0.1))
        inactive = _add_job("inactive job", _vec(0.1), is_active=False)

        from jobs.utils import sync_job_vectors
        sync_job_vectors()

        monkeypatch.setattr("jobs.vector_store._embed_query", lambda text: _vec(0.1))

        from jobs.vector_store import dense_search
        ids = [job_id for job_id, _ in dense_search("anything", k=5)]

        assert active.id in ids
        assert inactive.id not in ids


@pytest.mark.postgres
def test_dense_search_similarity_within_unit_range(pg_app, monkeypatch):
    with pg_app.app_context():
        _add_job("a job", _vec(0.1))

        from jobs.utils import sync_job_vectors
        sync_job_vectors()

        monkeypatch.setattr("jobs.vector_store._embed_query", lambda text: _vec(0.1))

        from jobs.vector_store import dense_search
        for _, similarity in dense_search("anything", k=5):
            assert 0.0 <= similarity <= 1.0


def test_dense_search_uses_faiss_on_sqlite(app_sqlite, monkeypatch):
    """On SQLite dense_search delegates to the existing FAISS index."""
    class _FakeDoc:
        def __init__(self, job_id):
            self.metadata = {"job_id": job_id}

    class _FakeIndex:
        class index:
            ntotal = 2

        def similarity_search_with_score(self, query, k):
            return [(_FakeDoc(11), 0.0), (_FakeDoc(22), 1.0)]

    monkeypatch.setattr("jobs.vector_store.get_job_faiss_index", lambda: _FakeIndex())

    with app_sqlite.app_context():
        from jobs.vector_store import dense_search
        results = dense_search("anything", k=5)

    assert [job_id for job_id, _ in results] == [11, 22]
    # distance 0 -> similarity 1.0; distance 1 -> 1 - 1/2 = 0.5
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(0.5)


def test_dense_search_returns_empty_when_no_index(app_sqlite, monkeypatch):
    monkeypatch.setattr("jobs.vector_store.get_job_faiss_index", lambda: None)
    with app_sqlite.app_context():
        from jobs.vector_store import dense_search
        assert dense_search("anything", k=5) == []


@pytest.mark.postgres
def test_dense_search_pgvector_failure_rolls_back_and_falls_back_to_faiss(pg_app, monkeypatch):
    """C3 regression: a failed pgvector query must not leave the session's
    transaction aborted, or the next JobPosting query on the same session
    raises PendingRollbackError instead of succeeding.

    This is live today via jobs/scout_agent.py, which calls dense_search()
    directly on a thread that DOES have app context — a failed pgvector
    query there previously left the session aborted, so the very next
    JobPosting.query.get(job_id) call in _find_and_save_matches raised
    PendingRollbackError, failing the whole scout run.

    Uses a malformed vector literal (not a mocked exception) so Postgres
    itself genuinely aborts the transaction, mirroring
    tests/test_memory_pgvector_search.py::
    test_pgvector_search_failure_rolls_back_and_falls_back_to_cosine.
    """
    with pg_app.app_context():
        job = _add_job("fallback job", _vec(0.5))

        from jobs.utils import sync_job_vectors
        sync_job_vectors()

        monkeypatch.setattr("jobs.vector_store._embed_query", lambda text: _vec(0.5))
        # Force Postgres to genuinely abort the transaction on a malformed
        # vector literal, rather than mocking an exception in Python.
        monkeypatch.setattr("jobs.vector_store.to_vector_literal", lambda vec: "not-a-valid-vector-literal")

        class _FakeDoc:
            def __init__(self, job_id):
                self.metadata = {"job_id": job_id}

        class _FakeIndex:
            class index:
                ntotal = 1

            def similarity_search_with_score(self, query, k):
                return [(_FakeDoc(job.id), 0.0)]

        monkeypatch.setattr("jobs.vector_store.get_job_faiss_index", lambda: _FakeIndex())

        from jobs.vector_store import dense_search
        results = dense_search("anything", k=5)

        # Fallback still returns results without raising.
        assert [jid for jid, _ in results] == [job.id]

        # The session must not be left in an aborted transaction state --
        # this would raise sqlalchemy.exc.PendingRollbackError if C3's fix
        # (db.session.rollback() before the FAISS fallback) regressed.
        reloaded = JobPosting.query.get(job.id)
        assert reloaded is not None
        assert reloaded.title == "fallback job"
