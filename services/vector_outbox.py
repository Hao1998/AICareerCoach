"""
Transactional outbox for the Redis vector search experiment.

Writing (after_flush listener)
------------------------------
Whenever a flush inserts a JobPosting, deletes one, or changes its
`embedding` or `is_active`, a vector_outbox row is inserted through the
flush's own connection — so it commits or rolls back together with the job
change. This covers every ORM write path (fetchers, the add/delete routes,
the scout) without touching them. It does NOT see raw SQL writes; that gap is
what change data capture (reading the Postgres WAL) closes.

Only active when VECTOR_BACKEND=redis; otherwise the listener returns at once.

Fast path (after_commit)
------------------------
Right after a commit that wrote outbox rows, those rows are applied to Redis
straight away — so a new job is searchable within milliseconds, not at the
relay's next poll (the Job Scout matches immediately after fetching). It's
best-effort: if Redis is down or the process dies, the rows simply stay
unprocessed and the relay applies them. Rolled-back changes never get here.

Applying (relay_once)
---------------------
Reads unprocessed rows in id order, loads each job's CURRENT row, and either
upserts it into Redis (active, has an embedding) or deletes it (inactive,
no embedding, or gone). Because it applies current state rather than replaying
events, two quick changes to the same job collapse into one write, and
reprocessing a row after a crash is harmless (at-least-once delivery with
idempotent writes). If Redis fails, the transaction is rolled back and the
rows stay unprocessed for the next pass.
"""
import logging
from datetime import datetime

from sqlalchemy import event, insert, inspect
from sqlalchemy.orm import Session

from models import db, JobPosting, VectorOutbox
from services.db_lock import safe_commit
from services.vector_backend import is_redis_vectors

logger = logging.getLogger(__name__)

_SEARCH_FIELDS = ("embedding", "is_active")
_PENDING_KEY = "vector_outbox_pending_job_ids"
_table_ready = False


def _outbox_table_exists(connection) -> bool:
    """Checked until it's True once; a missing table only logs, never breaks writes."""
    global _table_ready
    if not _table_ready:
        _table_ready = inspect(connection).has_table(VectorOutbox.__tablename__)
        if not _table_ready:
            logger.error(
                "VECTOR_BACKEND=redis but the vector_outbox table is missing — run "
                "`flask --app wsgi db upgrade` (Postgres) then make redis-rebuild. "
                "Job changes are NOT being sent to Redis."
            )
    return _table_ready


def _changed_for_search(job) -> bool:
    attrs = inspect(job).attrs
    return any(getattr(attrs, name).history.has_changes() for name in _SEARCH_FIELDS)


def _after_flush(session, flush_context):
    if not is_redis_vectors():
        return

    job_ids = []
    for obj in session.new:
        if isinstance(obj, JobPosting):
            job_ids.append(obj.id)
    for obj in session.deleted:
        if isinstance(obj, JobPosting):
            job_ids.append(obj.id)
    for obj in session.dirty:
        if isinstance(obj, JobPosting) and _changed_for_search(obj):
            job_ids.append(obj.id)

    job_ids = list(dict.fromkeys(jid for jid in job_ids if jid is not None))
    if not job_ids:
        return

    connection = session.connection()
    if not _outbox_table_exists(connection):
        return
    connection.execute(
        insert(VectorOutbox.__table__),
        [{"job_id": jid} for jid in job_ids],
    )
    # Remember them for the fast path; forgotten again if the transaction rolls back.
    session.info.setdefault(_PENDING_KEY, set()).update(job_ids)


def _fast_path_store():
    """The store the fast path writes to. Tests replace this (see tests/conftest.py)."""
    from jobs.redis_vector_store import get_store
    return get_store()


def _after_commit(session):
    job_ids = session.info.pop(_PENDING_KEY, None)
    if not job_ids or not is_redis_vectors():
        return
    store = _fast_path_store()
    if store is None:
        return
    # The committing session can't run SQL inside after_commit, so use a fresh one.
    try:
        with Session(bind=session.get_bind()) as fast_session:
            while relay_once(store, session=fast_session, job_ids=job_ids):
                pass  # one batch per call; repeat until these jobs' rows are done
    except Exception as e:
        logger.warning("Redis fast path failed (%s); the relay will retry these jobs", e)


def _after_rollback(session):
    session.info.pop(_PENDING_KEY, None)


def register_outbox_listener(session) -> None:
    """Attach the listener to `session` (db.session). Safe to call repeatedly."""
    for name, fn in (
        ("after_flush", _after_flush),
        ("after_commit", _after_commit),
        ("after_rollback", _after_rollback),
    ):
        if not event.contains(session, name, fn):
            event.listen(session, name, fn)


def relay_once(store=None, batch_size: int = 100, session=None, job_ids=None) -> list[tuple[str, int]]:
    """Apply up to `batch_size` outbox rows to Redis. Returns [(action, job_id)].

    `session` defaults to db.session; `job_ids` limits it to those jobs' rows
    (the fast path passes the ids it just committed).
    """
    if store is None:
        from jobs.redis_vector_store import get_store
        store = get_store()
    if session is None:
        session = db.session

    query = session.query(VectorOutbox).filter(VectorOutbox.processed_at.is_(None))
    if job_ids is not None:
        query = query.filter(VectorOutbox.job_id.in_(list(job_ids)))
    query = query.order_by(VectorOutbox.id).limit(batch_size)
    if session.get_bind().dialect.name == "postgresql":
        # Lets two relays run side by side without applying the same rows.
        query = query.with_for_update(skip_locked=True)
    rows = query.all()
    if not rows:
        session.rollback()  # release the (empty) transaction / row locks
        return []

    rows_by_job: dict[int, list[VectorOutbox]] = {}
    for row in rows:
        rows_by_job.setdefault(row.job_id, []).append(row)

    actions = []
    try:
        store.ensure_index()
        now = datetime.utcnow()
        for job_id, job_rows in rows_by_job.items():
            job = session.get(JobPosting, job_id)
            if job is not None and job.is_active and job.embedding is not None:
                store.upsert_job(job)
                actions.append(("upsert", job_id))
            else:
                store.remove_job(job_id)
                actions.append(("delete", job_id))
            for row in job_rows:
                row.processed_at = now
    except Exception:
        session.rollback()
        raise

    if session is db.session:
        safe_commit()
    else:
        session.commit()
    return actions
