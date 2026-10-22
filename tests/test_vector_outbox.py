"""
tests/test_vector_outbox.py
===========================
Transactional outbox for the Redis vector search experiment.

The after_flush listener must write a vector_outbox row in the same
transaction as every job change that affects search (new job, embedding
change, is_active change, delete), and nothing when VECTOR_BACKEND is off.
"""
import pytest

from models import db, JobPosting, VectorOutbox


@pytest.fixture
def redis_on(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "redis")


def _make_job(**overrides):
    fields = dict(
        title="Backend Engineer",
        company="Acme",
        description="Python services",
        embedding=[0.0] * 768,
    )
    fields.update(overrides)
    return JobPosting(**fields)


def _commit_job(**overrides):
    job = _make_job(**overrides)
    db.session.add(job)
    db.session.commit()
    return job


def _clear_outbox():
    VectorOutbox.query.delete()
    db.session.commit()


# ── listener ─────────────────────────────────────────────────────────────────

def test_new_job_writes_outbox_row(app_sqlite, redis_on):
    job = _commit_job()
    rows = VectorOutbox.query.all()
    assert [r.job_id for r in rows] == [job.id]
    assert rows[0].processed_at is None


def test_deactivate_writes_row(app_sqlite, redis_on):
    job = _commit_job()
    _clear_outbox()
    job.is_active = False
    db.session.commit()
    assert VectorOutbox.query.filter_by(job_id=job.id).count() == 1


def test_embedding_change_writes_row(app_sqlite, redis_on):
    job = _commit_job()
    _clear_outbox()
    job.embedding = [0.1] * 768
    db.session.commit()
    assert VectorOutbox.query.count() == 1


def test_unrelated_column_change_writes_nothing(app_sqlite, redis_on):
    job = _commit_job()
    _clear_outbox()
    job.title = "Renamed"
    db.session.commit()
    assert VectorOutbox.query.count() == 0


def test_hard_delete_writes_row(app_sqlite, redis_on):
    job = _commit_job()
    jid = job.id
    _clear_outbox()
    db.session.delete(job)
    db.session.commit()
    assert VectorOutbox.query.filter_by(job_id=jid).count() == 1


def test_off_writes_nothing(app_sqlite, monkeypatch):
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    db.session.add(_make_job())
    db.session.commit()
    assert VectorOutbox.query.count() == 0


def test_rollback_discards_outbox_row(app_sqlite, redis_on):
    db.session.add(_make_job())
    db.session.flush()
    db.session.rollback()
    assert VectorOutbox.query.count() == 0


def test_two_new_jobs_in_one_flush_write_two_rows(app_sqlite, redis_on):
    db.session.add_all([_make_job(), _make_job(title="Data Engineer")])
    db.session.commit()
    assert VectorOutbox.query.count() == 2


def test_missing_outbox_table_does_not_break_job_writes(app_sqlite, redis_on, caplog, monkeypatch):
    # Switch on before the migration ran: jobs must still save (Redis just
    # won't hear about them until the table exists and you rebuild).
    import services.vector_outbox as outbox_module
    monkeypatch.setattr(outbox_module, "_table_ready", False)
    # Postgres-backed tests run Alembic, whose fileConfig() disables existing loggers.
    monkeypatch.setattr(outbox_module.logger, "disabled", False)
    VectorOutbox.__table__.drop(db.engine)
    job = _commit_job()
    assert JobPosting.query.get(job.id) is not None
    assert "vector_outbox" in caplog.text
    VectorOutbox.__table__.create(db.engine)


# ── relay ────────────────────────────────────────────────────────────────────

import redis  # noqa: E402

from services.vector_outbox import relay_once  # noqa: E402


class FakeStore:
    """Records what the relay asks the Redis store to do."""

    def __init__(self, raise_on_write=None):
        self.raise_on_write = raise_on_write
        self.ensure_calls = 0
        self.upserts = []
        self.removes = []

    def ensure_index(self):
        self.ensure_calls += 1

    def upsert_job(self, job):
        if self.raise_on_write:
            raise self.raise_on_write
        self.upserts.append(job.id)

    def remove_job(self, job_id):
        if self.raise_on_write:
            raise self.raise_on_write
        self.removes.append(job_id)


def _unprocessed():
    return VectorOutbox.query.filter(VectorOutbox.processed_at.is_(None)).count()


def test_relay_upserts_active_job_and_marks_processed(app_sqlite, redis_on):
    job = _commit_job()
    store = FakeStore()
    assert relay_once(store) == [("upsert", job.id)]
    assert store.ensure_calls == 1
    assert store.upserts == [job.id]
    assert _unprocessed() == 0


def test_relay_deletes_inactive_job(app_sqlite, redis_on):
    job = _commit_job()
    relay_once(FakeStore())
    job.is_active = False
    db.session.commit()
    store = FakeStore()
    assert relay_once(store) == [("delete", job.id)]
    assert store.removes == [job.id]


def test_relay_deletes_hard_deleted_job(app_sqlite, redis_on):
    job = _commit_job()
    jid = job.id
    db.session.delete(job)
    db.session.commit()
    assert relay_once(FakeStore()) == [("delete", jid)]
    assert _unprocessed() == 0


def test_relay_deletes_job_without_embedding(app_sqlite, redis_on):
    job = _commit_job(embedding=None)
    store = FakeStore()
    assert relay_once(store) == [("delete", job.id)]
    assert store.upserts == []


def test_two_changes_one_write(app_sqlite, redis_on):
    job = _commit_job()
    job.embedding = [0.2] * 768
    db.session.commit()
    assert VectorOutbox.query.count() == 2
    store = FakeStore()
    assert relay_once(store) == [("upsert", job.id)]
    assert store.upserts == [job.id]
    assert _unprocessed() == 0


def test_redis_error_leaves_rows_unprocessed(app_sqlite, redis_on):
    _commit_job()
    with pytest.raises(redis.exceptions.ConnectionError):
        relay_once(FakeStore(raise_on_write=redis.exceptions.ConnectionError("down")))
    assert _unprocessed() == 1


def test_relay_writes_no_new_outbox_rows(app_sqlite, redis_on):
    _commit_job()
    relay_once(FakeStore())
    assert VectorOutbox.query.count() == 1


def test_relay_with_empty_outbox_does_nothing(app_sqlite, redis_on):
    assert relay_once(FakeStore()) == []


def test_relay_respects_batch_size(app_sqlite, redis_on):
    for i in range(3):
        _commit_job(title=f"Job {i}")
    assert len(relay_once(FakeStore(), batch_size=2)) == 2
    assert _unprocessed() == 1



# ── fast path (after_commit) ────────────────────────────────────────────────

@pytest.fixture
def fast_store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr("services.vector_outbox._fast_path_store", lambda: store)
    return store


def test_commit_applies_to_redis_immediately(app_sqlite, redis_on, fast_store):
    job = _commit_job()
    assert fast_store.upserts == [job.id]
    assert _unprocessed() == 0


def test_deactivate_is_applied_immediately(app_sqlite, redis_on, fast_store):
    job = _commit_job()
    job.is_active = False
    db.session.commit()
    assert fast_store.removes == [job.id]
    assert _unprocessed() == 0


def test_fast_path_failure_leaves_row_for_relay(app_sqlite, redis_on, monkeypatch, caplog):
    import services.vector_outbox as outbox_module
    monkeypatch.setattr(outbox_module.logger, "disabled", False)
    broken = FakeStore(raise_on_write=redis.exceptions.ConnectionError("down"))
    monkeypatch.setattr("services.vector_outbox._fast_path_store", lambda: broken)

    job = _commit_job()                       # the save itself must still succeed
    assert JobPosting.query.get(job.id) is not None
    assert _unprocessed() == 1
    assert "relay will retry" in caplog.text

    assert relay_once(FakeStore()) == [("upsert", job.id)]


def test_rolled_back_change_is_not_applied(app_sqlite, redis_on, fast_store):
    db.session.add(_make_job())
    db.session.flush()
    db.session.rollback()
    other = _commit_job(title="Committed later")
    assert fast_store.upserts == [other.id]   # only the committed job, not the rolled-back one


def test_fast_path_off_when_switch_off(app_sqlite, monkeypatch):
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    monkeypatch.setattr(
        "services.vector_outbox._fast_path_store",
        lambda: pytest.fail("fast path ran while VECTOR_BACKEND is off"),
    )
    _commit_job()


def test_commit_without_job_changes_skips_fast_path(app_sqlite, redis_on, monkeypatch):
    job = _commit_job()
    monkeypatch.setattr(
        "services.vector_outbox._fast_path_store",
        lambda: pytest.fail("fast path ran for a commit with no job changes"),
    )
    job.title = "Renamed"
    db.session.commit()


def test_fetched_jobs_are_in_redis_when_fetch_returns(app_sqlite, redis_on, fast_store, monkeypatch):
    """The Job Scout matches right after fetching — new jobs must already be searchable."""
    from jobs.fetchers.base import BaseJobFetcher

    class _OneJobFetcher(BaseJobFetcher):
        source_name = "test"

        def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs):
            return [{"id": 1, "title": "Backend Engineer", "company": "Acme", "description": "Python"}]

        def parse_job(self, raw):
            return {
                "title": raw["title"], "company": raw["company"], "description": raw["description"],
                "source": self.source_name, "source_id": str(raw["id"]),
            }

    def _fake_embed(job):
        job.embedding = [0.1] * 768

    monkeypatch.setattr("jobs.fetchers.base.compute_job_embedding", _fake_embed)
    monkeypatch.setattr("jobs.fetchers.base.build_job_faiss_index", lambda: None)

    stats = _OneJobFetcher().fetch_and_store_jobs()

    assert stats["stored"] == 1
    stored = JobPosting.query.filter_by(source="test").one()
    assert fast_store.upserts == [stored.id]
    assert _unprocessed() == 0


def test_fast_path_applies_more_than_one_batch(app_sqlite, redis_on, fast_store):
    db.session.add_all([_make_job(title=f"Job {i}") for i in range(150)])
    db.session.commit()
    assert len(fast_store.upserts) == 150
    assert _unprocessed() == 0
