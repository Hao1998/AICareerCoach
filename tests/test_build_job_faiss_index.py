"""
tests/test_build_job_faiss_index.py
====================================
N1 regression test: build_job_faiss_index() (and the /jobs/rebuild-index
endpoint built on top of it, controllers/job_controller.py:rebuild_job_index)
must distinguish "no jobs to index" from "indexed/synced fine, but this
backend has no FAISS vectorstore object" (pgvector on PostgreSQL).

Before the fix, build_job_faiss_index() returned a bare `None` for both
cases, and the rebuild-index endpoint used truthiness of that `None` to
decide success/failure -- meaning it always reported failure on Postgres
even when the sync fully succeeded.

The fix makes build_job_faiss_index() return a FaissBuildResult carrying
both `vectorstore` and `job_count`; it's truthy iff job_count > 0.
"""
from datetime import datetime

import pytest

from models import db, JobPosting

DIM = 768


def _vec(val: float) -> list:
    return [val] * DIM


def _add_job(title="Backend Engineer"):
    job = JobPosting(
        title=title,
        company="Acme",
        description="Build backend services in Python.",
        source="test",
        is_active=True,
        embedding=_vec(0.1),
        embedding_updated_at=datetime(2026, 1, 1),
    )
    db.session.add(job)
    db.session.commit()
    return job


@pytest.mark.postgres
def test_build_job_faiss_index_reports_success_on_postgres_with_jobs(pg_app):
    """On Postgres there's no FAISS vectorstore to return (pgvector is the
    real index there), but build_job_faiss_index() must still report success
    (via job_count / truthiness) when there were jobs to sync -- this is
    exactly what the /jobs/rebuild-index endpoint's `if result:` branch relies
    on."""
    with pg_app.app_context():
        _add_job()

        from jobs.utils import build_job_faiss_index
        result = build_job_faiss_index()

        assert result.vectorstore is None, "no FAISS vectorstore is expected on Postgres"
        assert result.job_count == 1
        assert bool(result) is True, (
            "build_job_faiss_index() must report success (truthy) on Postgres "
            "when jobs existed and were synced, even with no vectorstore object"
        )


@pytest.mark.postgres
def test_build_job_faiss_index_reports_failure_on_postgres_with_no_jobs(pg_app):
    """A genuinely empty jobs table is a real failure on both backends, not
    just a "wrong backend" case."""
    with pg_app.app_context():
        from jobs.utils import build_job_faiss_index
        result = build_job_faiss_index()

        assert result.vectorstore is None
        assert result.job_count == 0
        assert bool(result) is False


def test_build_job_faiss_index_reports_success_on_sqlite_with_jobs(app_sqlite, monkeypatch):
    """Existing SQLite behavior must be unchanged: a live vectorstore comes
    back and the result is truthy.

    Stubs the embedding model and FAISS.from_embeddings so this test stays
    hermetic/fast and, notably, sidesteps an unrelated pre-existing bug in
    this installed langchain-community version where FAISS.from_embeddings()
    doesn't accept the `distance_metric` kwarg jobs/utils.py passes it
    (langchain-community now expects `distance_strategy`) -- that bug is
    orthogonal to the N1 return-value-shape fix under test here and is
    flagged separately."""
    class _FakeEmbeddings:
        def embed_query(self, text):
            return _vec(0.1)

    class _FakeVectorstore:
        def save_local(self, path):
            import os
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "index.faiss"), "wb") as f:
                f.write(b"fake")

    with app_sqlite.app_context():
        import jobs.utils as jobs_utils

        _add_job()
        monkeypatch.setattr("jobs.utils.get_embeddings", lambda: _FakeEmbeddings())
        monkeypatch.setattr(
            "jobs.utils.FAISS.from_embeddings",
            staticmethod(lambda **kw: _FakeVectorstore()),
        )
        # Redirect the on-disk index to a throwaway path and reset the
        # in-process cache so this test can't pollute the real
        # job_vector_index/ directory or leak _faiss_cache into later tests.
        monkeypatch.setattr("jobs.utils.JOB_VECTOR_INDEX", "tests/_scratch_faiss_index")
        monkeypatch.setattr(jobs_utils, "_faiss_cache", None)
        monkeypatch.setattr(jobs_utils, "_faiss_cache_mtime", 0)

        from jobs.utils import build_job_faiss_index
        result = build_job_faiss_index()

        assert result.vectorstore is not None
        assert result.job_count == 1
        assert bool(result) is True

        import shutil
        shutil.rmtree("tests/_scratch_faiss_index", ignore_errors=True)
        monkeypatch.setattr(jobs_utils, "_faiss_cache", None)
        monkeypatch.setattr(jobs_utils, "_faiss_cache_mtime", 0)


def test_build_job_faiss_index_reports_failure_on_sqlite_with_no_jobs(app_sqlite):
    with app_sqlite.app_context():
        from jobs.utils import build_job_faiss_index
        result = build_job_faiss_index()

        assert result.vectorstore is None
        assert result.job_count == 0
        assert bool(result) is False


@pytest.mark.postgres
def test_rebuild_index_endpoint_branch_succeeds_on_postgres(pg_app):
    """Exercises the exact success/failure branch
    controllers/job_controller.py:rebuild_job_index() uses
    (`if result: ... success ... else: ... 400 failure ...`), against a real
    pgvector-backed app."""
    with pg_app.app_context():
        _add_job()

        from jobs.utils import build_job_faiss_index
        result = build_job_faiss_index()

        assert result, "endpoint would incorrectly report failure for a successful Postgres sync"


@pytest.mark.postgres
def test_rebuild_index_endpoint_branch_fails_on_empty_postgres(pg_app):
    with pg_app.app_context():
        from jobs.utils import build_job_faiss_index
        result = build_job_faiss_index()

        assert not result, "endpoint must still report failure for a genuinely empty jobs table"
