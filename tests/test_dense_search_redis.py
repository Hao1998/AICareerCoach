"""
tests/test_dense_search_redis.py
================================
dense_search() routing for the Redis vector search experiment.

With VECTOR_BACKEND=redis, Redis answers. Any Redis error, a missing index,
or an empty result falls back to the existing pgvector / FAISS path, so
job matching never breaks because of the experiment. With the switch off,
Redis is never touched.
"""
from datetime import datetime

import pytest
import redis

import jobs.vector_store as vs
from models import db, JobPosting

DIM = 768


@pytest.fixture
def redis_on(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "redis")
    # Postgres-backed tests run Alembic, whose fileConfig() disables every
    # existing logger — re-enable ours so caplog can see the fallback warnings.
    monkeypatch.setattr(vs.logger, "disabled", False)


class _FakeStore:
    def __init__(self, hits=None, raises=None, built=True):
        self.hits = hits or []
        self.raises = raises
        self.built = built
        self.calls = []

    def is_built(self):
        return self.built

    def search(self, query_vec, k):
        self.calls.append((query_vec, k))
        if self.raises:
            raise self.raises
        return self.hits


def _fake_store(monkeypatch, **kw):
    store = _FakeStore(**kw)
    monkeypatch.setattr(vs, "get_store", lambda: store)
    return store


def _faiss_fallback(monkeypatch, result=((1, 0.5),)):
    monkeypatch.setattr(vs, "is_postgres", lambda: False)
    monkeypatch.setattr(vs, "_dense_search_faiss", lambda q, k: list(result))


def test_redis_answers_when_enabled(monkeypatch, redis_on):
    store = _fake_store(monkeypatch, hits=[(7, 0.9)])
    monkeypatch.setattr(vs, "_dense_search_faiss", lambda *a: pytest.fail("should not fall back"))
    monkeypatch.setattr(vs, "_dense_search_pgvector", lambda *a: pytest.fail("should not fall back"))
    assert vs.dense_search("q", 5, query_vec=[0.0] * DIM) == [(7, 0.9)]
    assert store.calls == [([0.0] * DIM, 5)]


def test_redis_embeds_query_when_no_vector_given(monkeypatch, redis_on):
    store = _fake_store(monkeypatch, hits=[(7, 0.9)])
    monkeypatch.setattr(vs, "_embed_query", lambda text: [0.3] * DIM)
    vs.dense_search("q", 5)
    assert store.calls[0][0] == [0.3] * DIM


def test_falls_back_on_redis_error(monkeypatch, redis_on):
    _fake_store(monkeypatch, raises=redis.exceptions.ConnectionError("down"))
    _faiss_fallback(monkeypatch)
    assert vs.dense_search("q", 5, query_vec=[0.0] * DIM) == [(1, 0.5)]


def test_falls_back_when_index_missing(monkeypatch, redis_on, caplog):
    _fake_store(monkeypatch, raises=redis.exceptions.ResponseError("idx:jobs: no such index"))
    _faiss_fallback(monkeypatch)
    assert vs.dense_search("q", 5, query_vec=[0.0] * DIM) == [(1, 0.5)]
    assert "make redis-rebuild" in caplog.text


def test_falls_back_when_redis_returns_nothing(monkeypatch, redis_on, caplog):
    _fake_store(monkeypatch, hits=[])
    _faiss_fallback(monkeypatch)
    assert vs.dense_search("q", 5, query_vec=[0.0] * DIM) == [(1, 0.5)]
    assert "make redis-rebuild" in caplog.text


def test_falls_back_when_index_not_built(monkeypatch, redis_on, caplog):
    # e.g. after make redis-reset the relay recreates an EMPTY index and fills
    # it with a few outbox jobs — that partial index must not answer search.
    store = _fake_store(monkeypatch, hits=[(7, 0.9)], built=False)
    _faiss_fallback(monkeypatch)
    assert vs.dense_search("q", 5, query_vec=[0.0] * DIM) == [(1, 0.5)]
    assert store.calls == []
    assert "make redis-rebuild" in caplog.text


def test_off_never_touches_redis(monkeypatch):
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    monkeypatch.setattr(vs, "get_store", lambda: pytest.fail("redis used while off"))
    _faiss_fallback(monkeypatch)
    assert vs.dense_search("q", 5) == [(1, 0.5)]


class _FakeMatchResult:
    match_score = 88
    matched_skills = ["Python"]
    skill_gaps = []
    recommendation = "Strong match"


def test_find_matching_jobs_sqlite_redis_search_runs_off_greenlet(app_sqlite, monkeypatch, redis_on):
    job = JobPosting(
        title="Backend Engineer", company="Acme", description="Python services",
        source="test", embedding=[0.1] * DIM, embedding_updated_at=datetime(2026, 1, 1),
    )
    db.session.add(job)
    db.session.commit()

    store = _fake_store(monkeypatch, hits=[(job.id, 0.95)])
    monkeypatch.setattr(vs, "_dense_search_faiss", lambda *a: pytest.fail("should not fall back"))

    def _fake_embed_query(text):
        return [0.1] * DIM

    monkeypatch.setattr("jobs.vector_store._embed_query", _fake_embed_query)
    monkeypatch.setattr("services.job_service.run_job_matching", lambda **kw: _FakeMatchResult())
    monkeypatch.setattr("services.job_service.get_bm25_index", lambda: None)

    import services.job_service as job_service_module
    real_run_in_thread = job_service_module._run_in_thread
    dispatched = []

    def _tracking(fn, *args, **kwargs):
        dispatched.append(getattr(fn, "__name__", repr(fn)))
        return real_run_in_thread(fn, *args, **kwargs)

    monkeypatch.setattr(job_service_module, "_run_in_thread", _tracking)

    results = job_service_module.find_matching_jobs("Python backend engineer", top_k=1, candidate_k=5)

    # On SQLite the whole dense search (Redis, then any FAISS fallback) runs on
    # a real thread, so neither the embedding nor a fallback touches the greenlet.
    assert dispatched[0] == "_dense_search_faiss_threaded"
    assert store.calls and store.calls[0][0] == [0.1] * DIM
    assert [r["job"].id for r in results] == [job.id]


def test_sqlite_redis_failure_runs_faiss_fallback_off_greenlet(app_sqlite, monkeypatch, redis_on):
    job = JobPosting(
        title="Backend Engineer", company="Acme", description="Python services",
        source="test", embedding=[0.1] * DIM, embedding_updated_at=datetime(2026, 1, 1),
    )
    db.session.add(job)
    db.session.commit()

    _fake_store(monkeypatch, raises=redis.exceptions.ConnectionError("down"))
    monkeypatch.setattr("jobs.vector_store._embed_query", lambda text: [0.1] * DIM)
    monkeypatch.setattr("services.job_service.run_job_matching", lambda **kw: _FakeMatchResult())
    monkeypatch.setattr("services.job_service.get_bm25_index", lambda: None)

    import services.job_service as job_service_module
    real_run_in_thread = job_service_module._run_in_thread
    state = {"in_thread": False, "faiss_in_thread": None}

    def _tracking(fn, *args, **kwargs):
        state["in_thread"] = True
        try:
            return real_run_in_thread(fn, *args, **kwargs)
        finally:
            state["in_thread"] = False

    def _fake_faiss(query_text, k):
        state["faiss_in_thread"] = state["in_thread"]
        return [(job.id, 0.5)]

    monkeypatch.setattr(job_service_module, "_run_in_thread", _tracking)
    monkeypatch.setattr(vs, "_dense_search_faiss", _fake_faiss)

    results = job_service_module.find_matching_jobs("Python backend engineer", top_k=1, candidate_k=5)

    assert state["faiss_in_thread"] is True, "FAISS fallback ran on the request greenlet"
    assert [r["job"].id for r in results] == [job.id]
