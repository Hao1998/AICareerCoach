"""
tests/test_pgvector_sync.py
===========================
Verifies the JSON embedding column is copied into the native vector column,
and that a changed embedding refreshes a previously-synced vector.

Postgres-only: pgvector does not exist on SQLite.
"""
from datetime import datetime

import pytest
from sqlalchemy import text

from models import db, JobPosting, User, UserMemoryChunk

DIM = 768


def _vec(val: float) -> list:
    return [val] * DIM


def _add_job(embedding, updated_at):
    job = JobPosting(
        title="Engineer",
        company="Acme",
        description="Builds things",
        source="test",
        is_active=True,
        embedding=embedding,
        embedding_updated_at=updated_at,
    )
    db.session.add(job)
    db.session.commit()
    return job


@pytest.mark.postgres
def test_sync_job_vectors_populates_null_vectors(pg_app):
    with pg_app.app_context():
        job = _add_job(_vec(0.25), datetime(2026, 1, 1))

        from jobs.utils import sync_job_vectors
        updated = sync_job_vectors()

        assert updated == 1
        stored = db.session.execute(
            text("SELECT embedding_vec IS NOT NULL FROM job_postings WHERE id = :i"),
            {"i": job.id},
        ).scalar()
        assert stored is True


@pytest.mark.postgres
def test_sync_job_vectors_is_idempotent(pg_app):
    with pg_app.app_context():
        _add_job(_vec(0.25), datetime(2026, 1, 1))

        from jobs.utils import sync_job_vectors
        assert sync_job_vectors() == 1
        assert sync_job_vectors() == 0


@pytest.mark.postgres
def test_sync_job_vectors_refreshes_changed_embedding(pg_app):
    with pg_app.app_context():
        job = _add_job(_vec(0.25), datetime(2026, 1, 1))

        from jobs.utils import sync_job_vectors
        sync_job_vectors()

        job.embedding = _vec(0.75)
        job.embedding_updated_at = datetime(2026, 2, 1)
        db.session.commit()

        assert sync_job_vectors() == 1

        first_element = db.session.execute(
            text("SELECT (embedding_vec::text)::json->>0 FROM job_postings WHERE id = :i"),
            {"i": job.id},
        ).scalar()
        assert float(first_element) == pytest.approx(0.75)


@pytest.mark.postgres
def test_sync_chunks_to_pgvector_populates_null_vectors(pg_app):
    with pg_app.app_context():
        user = User(email="a@b.com", username="a", password_hash="x")
        db.session.add(user)
        db.session.commit()

        chunk = UserMemoryChunk(
            user_id=user.id,
            content="likes remote work",
            memory_type="fact",
            embedding=_vec(0.4),
            session_date=datetime(2026, 1, 1),
        )
        db.session.add(chunk)
        db.session.commit()

        from chatbot.memory import _sync_chunks_to_pgvector
        assert _sync_chunks_to_pgvector(pg_app) == 1
        assert _sync_chunks_to_pgvector(pg_app) == 0


def test_sync_job_vectors_noop_on_sqlite(app_sqlite):
    """On SQLite the sync must return 0 rather than raising."""
    with app_sqlite.app_context():
        from jobs.utils import sync_job_vectors
        assert sync_job_vectors() == 0
