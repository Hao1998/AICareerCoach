# pgvector Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move job and memory similarity search from local-disk FAISS / sqlite-vec onto pgvector inside PostgreSQL, so search results are shared and consistent across multiple app instances.

**Architecture:** Embeddings already live in the database as dialect-neutral JSON columns (`JobPosting.embedding`, `UserMemoryChunk.embedding`) — those stay the source of truth and are not migrated. On PostgreSQL we add native `vector(768)` companion columns, kept in sync by set-based SQL, and search reads them via pgvector's `<=>` cosine-distance operator. On SQLite (local dev and the existing test suite) every search entry point falls back to the current FAISS / sqlite-vec code, unchanged.

**Tech Stack:** PostgreSQL 16 + pgvector, SQLAlchemy 2.x raw `text()` queries, Alembic (Flask-Migrate), pytest.

## Global Constraints

- Embedding dimension is **768** (`sentence-transformers/all-mpnet-base-v2`). Every vector column, cast, and fixture uses 768.
- The `vector` columns are **not** declared in `models.py`. They are created by raw Alembic DDL guarded on dialect and queried with raw SQL — the same pattern the existing `vec_user_memories` virtual table already uses. Declaring a pgvector type in `models.py` would break `db.create_all()` on SQLite in tests.
- No new Python dependency. Query parameters are passed as pgvector string literals and cast in SQL (`CAST(:qvec AS vector)`), so the `pgvector` Python package is not required.
- Architecture rule from CLAUDE.md: `services/` and `jobs/` must never import from `controllers/`. `models.py` imports nothing from the app.
- Eval gates from CLAUDE.md: `evals/job_match_eval.py` must pass after Task 3; `evals/memory_eval.py` must pass after Task 4.
- Existing tests must keep passing on SQLite with no external services: `python -m pytest` stays green throughout.
- pgvector-specific tests are skipped unless `TEST_DATABASE_URL` is set. They never run on SQLite.

---

## File Structure

| File | Responsibility |
|---|---|
| `services/pgvector_support.py` | **Create.** Dialect detection + vector literal formatting. Only this module knows how to spell a pgvector value. |
| `migrations/versions/<rev>_add_pgvector_columns.py` | **Create.** Extension, `vector(768)` columns, HNSW index. Postgres-only, no-op on SQLite. |
| `jobs/vector_store.py` | **Create.** Single entry point for job dense retrieval. Dispatches pgvector vs FAISS, returns a backend-independent result shape. |
| `tests/conftest.py` | **Create.** Shared `pg_app` fixture that skips when `TEST_DATABASE_URL` is unset. |
| `jobs/utils.py` | **Modify.** Add `sync_job_vectors()`; call it where the FAISS index is rebuilt. |
| `chatbot/memory.py` | **Modify.** Add `_search_memories_pgvector()` + Postgres branch in `search_memories()` and the sync helper. |
| `services/job_service.py:146-170` | **Modify.** Stage 1a calls `dense_search()` instead of touching FAISS directly. |
| `jobs/scout_agent.py:304-322` | **Modify.** Same substitution. |
| `docker-compose.yml` | **Modify.** Add a `postgres` service (pgvector image) for local Postgres-backed testing. |

`jobs/vector_store.py` exists so that `job_service.py` and `scout_agent.py` — the two dense-search callers — share one backend decision instead of each growing their own `if postgres:` branch.

---

### Task 1: pgvector plumbing — dialect helper, migration, local Postgres

**Files:**
- Create: `services/pgvector_support.py`
- Create: `migrations/versions/<generated>_add_pgvector_columns.py`
- Create: `tests/conftest.py`
- Modify: `docker-compose.yml`
- Test: `tests/test_pgvector_support.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `services.pgvector_support.EMBEDDING_DIM: int` — `768`
  - `services.pgvector_support.is_postgres() -> bool`
  - `services.pgvector_support.to_vector_literal(values: Sequence[float]) -> str`
  - DB columns `job_postings.embedding_vec vector(768)`, `job_postings.embedding_vec_updated_at timestamp`, `user_memory_chunks.embedding_vec vector(768)` (PostgreSQL only)

- [ ] **Step 1: Write the failing test**

Create `tests/test_pgvector_support.py`:

```python
"""
tests/test_pgvector_support.py
==============================
Dialect detection and vector literal formatting.

These run on SQLite (the default test DB), so is_postgres() must report False
and the literal formatter must work with no database at all.
"""
from factory import create_app
from services.pgvector_support import (
    EMBEDDING_DIM,
    is_postgres,
    to_vector_literal,
)


def test_embedding_dim_is_768():
    assert EMBEDDING_DIM == 768


def test_to_vector_literal_formats_pgvector_syntax():
    assert to_vector_literal([1.0, 2.5, -3.0]) == "[1.0,2.5,-3.0]"


def test_to_vector_literal_coerces_ints_and_numpy_scalars():
    import numpy as np
    assert to_vector_literal([1, 2]) == "[1.0,2.0]"
    assert to_vector_literal(np.array([0.5, 0.25])) == "[0.5,0.25]"


def test_is_postgres_false_on_sqlite():
    app = create_app('test', skip_api_check=True)
    with app.app_context():
        assert is_postgres() is False


def test_is_postgres_false_outside_app_context():
    # Celery beat imports these modules before an app context exists.
    assert is_postgres() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pgvector_support.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.pgvector_support'`

- [ ] **Step 3: Write the module**

Create `services/pgvector_support.py`:

```python
"""
pgvector support helpers.

Embeddings are stored in dialect-neutral JSON columns (JobPosting.embedding,
UserMemoryChunk.embedding) and those remain the source of truth. On PostgreSQL
we additionally maintain native `vector` columns so similarity search runs
inside the database and is shared by every app instance. On SQLite (local dev
and tests) is_postgres() reports False and callers fall back to the existing
FAISS / sqlite-vec paths.

Only this module knows how to spell a pgvector value.
"""
import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# sentence-transformers/all-mpnet-base-v2 output width.
EMBEDDING_DIM = 768


def is_postgres() -> bool:
    """True when the bound SQLAlchemy engine is PostgreSQL.

    Returns False outside an application context rather than raising, so
    module-level callers and background processes are safe.
    """
    from flask import current_app
    try:
        engine = current_app.extensions['sqlalchemy'].engine
    except (KeyError, RuntimeError, AttributeError):
        return False
    return engine.dialect.name == 'postgresql'


def to_vector_literal(values: Sequence[float]) -> str:
    """Format a float sequence as a pgvector literal: '[1.0,2.5,-3.0]'.

    Passed as a bind parameter and cast in SQL with CAST(:param AS vector),
    which avoids taking a dependency on the pgvector Python package.
    """
    return '[' + ','.join(repr(float(v)) for v in values) + ']'
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pgvector_support.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Generate the migration skeleton**

This is raw DDL, so use `revision`, **not** `migrate` — autogenerate cannot see columns that are absent from `models.py`.

```bash
flask --app wsgi db revision -m "add pgvector columns"
```

This creates a file under `migrations/versions/` with `down_revision = 'e9e97b70ce4c'` filled in automatically. Note the generated revision id.

- [ ] **Step 6: Write the migration body**

Replace the generated `upgrade()` / `downgrade()` with the following. Keep the auto-generated `revision` / `down_revision` lines exactly as they were.

```python
"""add pgvector columns

Adds native vector(768) companion columns on PostgreSQL for job and memory
similarity search. The JSON `embedding` columns remain the source of truth;
these are a derived, indexable representation.

No-op on SQLite — local development and the test suite keep using FAISS and
the sqlite-vec virtual table.
"""
from alembic import op

# revision identifiers — leave exactly as generated.
# revision = '<generated>'
# down_revision = 'e9e97b70ce4c'


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        return

    # Requires rds_superuser on RDS; the master user has it.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS embedding_vec vector(768)")
    op.execute(
        "ALTER TABLE job_postings "
        "ADD COLUMN IF NOT EXISTS embedding_vec_updated_at TIMESTAMP"
    )
    op.execute(
        "ALTER TABLE user_memory_chunks ADD COLUMN IF NOT EXISTS embedding_vec vector(768)"
    )

    # Job search is global across all active jobs, so an ANN index pays off.
    # The predicate matches the WHERE clause in jobs/vector_store.py so the
    # planner can actually use it.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_job_postings_embedding_vec "
        "ON job_postings USING hnsw (embedding_vec vector_cosine_ops) "
        "WHERE is_active = true AND embedding_vec IS NOT NULL"
    )

    # No ANN index on user_memory_chunks: every memory query filters by
    # user_id, and HNSW post-filtering would silently drop relevant rows.
    # The existing b-tree on user_id selects a small subset and Postgres
    # orders it exactly — correct results, no recall loss.


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        return

    op.execute("DROP INDEX IF EXISTS ix_job_postings_embedding_vec")
    op.execute("ALTER TABLE user_memory_chunks DROP COLUMN IF EXISTS embedding_vec")
    op.execute("ALTER TABLE job_postings DROP COLUMN IF EXISTS embedding_vec_updated_at")
    op.execute("ALTER TABLE job_postings DROP COLUMN IF EXISTS embedding_vec")
    # CREATE EXTENSION is intentionally not reversed — other objects may use it.
```

- [ ] **Step 7: Add Postgres to docker-compose**

Add this service to `docker-compose.yml` under `services:`, alongside `app`, `worker`, and `beat`:

```yaml
  # Local PostgreSQL with pgvector, for running the Postgres-backed tests.
  # Not used by `python app.py`, which stays on SQLite.
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: careercoach
      POSTGRES_PASSWORD: localdev
      POSTGRES_DB: careercoach
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U careercoach"]
      interval: 5s
      retries: 10
```

And add a top-level `volumes:` block at the end of the file (create it if the file has none):

```yaml
volumes:
  pgdata:
```

- [ ] **Step 8: Write the shared Postgres test fixture**

Create `tests/conftest.py`:

```python
"""
Shared pytest fixtures.

`pg_app` gives a real PostgreSQL-backed app with pgvector, used by the tests
that exercise the production search path. It skips when TEST_DATABASE_URL is
unset, so the default `python -m pytest` run stays SQLite-only and needs no
external services.

To run the Postgres tests locally:
    docker compose up -d postgres
    TEST_DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach \\
        python -m pytest -m postgres -v
"""
import os
import pytest

import config as config_module
from factory import create_app
from models import db

TEST_DATABASE_URL = os.environ.get('TEST_DATABASE_URL')


@pytest.fixture
def pg_app(monkeypatch):
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set — skipping Postgres-backed test")

    # Patch the config class BEFORE create_app. Flask-SQLAlchemy builds its
    # engine during init_app, so assigning SQLALCHEMY_DATABASE_URI to
    # app.config afterwards would leave the app pointing at SQLite.
    monkeypatch.setattr(
        config_module.TestConfig, 'SQLALCHEMY_DATABASE_URI', TEST_DATABASE_URL
    )
    monkeypatch.setattr(config_module.TestConfig, 'SQLALCHEMY_ENGINE_OPTIONS', {})

    application = create_app('test', skip_api_check=True)

    with application.app_context():
        from sqlalchemy import text
        assert db.engine.dialect.name == 'postgresql', (
            f"pg_app fixture is bound to {db.engine.dialect.name}, not postgresql"
        )
        db.create_all()
        with db.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(
                "ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS embedding_vec vector(768)"
            ))
            conn.execute(text(
                "ALTER TABLE job_postings "
                "ADD COLUMN IF NOT EXISTS embedding_vec_updated_at TIMESTAMP"
            ))
            conn.execute(text(
                "ALTER TABLE user_memory_chunks "
                "ADD COLUMN IF NOT EXISTS embedding_vec vector(768)"
            ))
            conn.commit()
        yield application
        db.session.remove()
        db.drop_all()
```

Register the marker so `-m postgres` works without warnings. Create `pytest.ini` at the project root:

```ini
[pytest]
testpaths = tests
markers =
    postgres: test requires a real PostgreSQL database with pgvector (set TEST_DATABASE_URL)
```

- [ ] **Step 9: Verify the migration is a no-op on SQLite**

Run:

```bash
flask --app wsgi db upgrade
```

Expected: completes without error against the local SQLite database, and `flask --app wsgi db current` reports the new revision. The `job_postings` table is unchanged — confirm with:

```bash
.venv/bin/python -c "import sqlite3;print([r[1] for r in sqlite3.connect('instance/career_coach.db').execute('PRAGMA table_info(job_postings)')])"
```

Expected: the output does **not** contain `embedding_vec`.

- [ ] **Step 10: Verify the migration works on Postgres**

```bash
docker compose up -d postgres
```

Then:

```bash
DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach flask --app wsgi db upgrade
```

Expected: completes without error. Confirm the column and index exist:

```bash
docker compose exec postgres psql -U careercoach -c "\d job_postings" -c "\di ix_job_postings_embedding_vec"
```

Expected: `embedding_vec | vector(768)` appears in the table, and the index is listed.

- [ ] **Step 11: Run the full suite to confirm nothing regressed**

Run: `python -m pytest -v`
Expected: all pre-existing tests PASS, plus the 5 new ones. No skips other than Postgres-marked tests (there are none yet).

- [ ] **Step 12: Commit**

```bash
git add services/pgvector_support.py migrations/versions tests/conftest.py tests/test_pgvector_support.py pytest.ini docker-compose.yml
git commit -m "feat: add pgvector columns and dialect-detection helpers"
```

---

### Task 2: Keep the vector columns in sync

**Files:**
- Modify: `jobs/utils.py` (add `sync_job_vectors`, call it from `build_job_faiss_index`)
- Modify: `chatbot/memory.py:283-313` (add `_sync_chunks_to_pgvector`, branch the existing dispatch)
- Test: `tests/test_pgvector_sync.py`

**Interfaces:**
- Consumes: `services.pgvector_support.is_postgres`
- Produces:
  - `jobs.utils.sync_job_vectors() -> int` — number of rows updated; returns `0` on non-Postgres
  - `chatbot.memory._sync_chunks_to_pgvector(app) -> int` — number of rows updated

Both syncs are set-based SQL. Jobs use `embedding_vec_updated_at` against the existing `embedding_updated_at` so re-fetched jobs get refreshed vectors. Memory chunks are append-only, so `embedding_vec IS NULL` is sufficient there — the same condition the existing sqlite-vec sync uses.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pgvector_sync.py`:

```python
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

from models import db, JobPosting, UserMemoryChunk

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
        chunk = UserMemoryChunk(
            user_id=1,
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
```

Add the SQLite fixture used by the last test to `tests/conftest.py`:

```python
@pytest.fixture
def app_sqlite():
    application = create_app('test', skip_api_check=True)
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pgvector_sync.py -v`
Expected: the four `postgres`-marked tests SKIP (no `TEST_DATABASE_URL`), and `test_sync_job_vectors_noop_on_sqlite` FAILs with `ImportError: cannot import name 'sync_job_vectors'`.

Then run against Postgres to see the real failures:

```bash
docker compose up -d postgres
```

```bash
TEST_DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach python -m pytest tests/test_pgvector_sync.py -v
```

Expected: all five FAIL on the missing `sync_job_vectors` / `_sync_chunks_to_pgvector` names.

- [ ] **Step 3: Implement the job sync**

Add to `jobs/utils.py`, after `compute_all_job_embeddings`:

```python
def sync_job_vectors() -> int:
    """Copy JSON embeddings into the native pgvector column.

    Set-based: one UPDATE, no Python loop. Rows are refreshed when the vector
    is missing or when embedding_updated_at has moved past the value recorded
    at the last sync, so re-fetched jobs pick up new embeddings.

    Returns the number of rows updated; 0 on non-PostgreSQL engines.
    """
    from services.pgvector_support import is_postgres
    if not is_postgres():
        return 0

    from sqlalchemy import text
    try:
        result = db.session.execute(text(
            "UPDATE job_postings "
            "SET embedding_vec = CAST(embedding::text AS vector), "
            "    embedding_vec_updated_at = embedding_updated_at "
            "WHERE embedding IS NOT NULL "
            "  AND (embedding_vec IS NULL "
            "       OR embedding_vec_updated_at IS DISTINCT FROM embedding_updated_at)"
        ))
        db.session.commit()
        if result.rowcount:
            logger.info("Synced %d job embeddings to pgvector", result.rowcount)
        return result.rowcount
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to sync job vectors: %s", e)
        return 0
```

- [ ] **Step 4: Call the job sync where the index is rebuilt**

In `jobs/utils.py`, inside `build_job_faiss_index()`, immediately after `compute_all_job_embeddings()` (currently line 143), insert:

```python
        # On PostgreSQL the pgvector column is the real search index; keep it
        # current at the same points the FAISS index is rebuilt.
        sync_job_vectors()
```

`build_job_faiss_index` is already called from `jobs/fetchers/base.py:154`, `jobs/scout_graph.py:213`, and `controllers/job_controller.py`, so every path that ingests jobs also syncs vectors. No other call sites need changing.

- [ ] **Step 5: Implement the memory sync**

In `chatbot/memory.py`, add after `_sync_chunks_to_vec` (which ends at line 313):

```python
def _sync_chunks_to_pgvector(app) -> int:
    """Copy JSON embeddings into user_memory_chunks.embedding_vec.

    Memory chunks are append-only, so a missing vector is the only condition
    that needs handling — the same semantics as _sync_chunks_to_vec.

    Returns the number of rows updated; 0 on non-PostgreSQL engines.
    """
    from sqlalchemy import text

    with app.app_context():
        from services.pgvector_support import is_postgres
        if not is_postgres():
            return 0

        engine = app.extensions['sqlalchemy'].engine
        try:
            with engine.connect() as conn:
                result = conn.execute(text(
                    "UPDATE user_memory_chunks "
                    "SET embedding_vec = CAST(embedding::text AS vector) "
                    "WHERE embedding IS NOT NULL AND embedding_vec IS NULL"
                ))
                conn.commit()
                if result.rowcount:
                    logger.info("Synced %d memory chunks to pgvector", result.rowcount)
                return result.rowcount
        except Exception as e:
            logger.error("Failed to sync memory chunks to pgvector: %s", e)
            return 0
```

- [ ] **Step 6: Branch the existing sync dispatch**

In `chatbot/memory.py`, replace the two lines at 283-284:

```python
        if _USE_VEC:
            _sync_chunks_to_vec(app)
```

with:

```python
        with app.app_context():
            from services.pgvector_support import is_postgres
            _is_pg = is_postgres()
        if _is_pg:
            _sync_chunks_to_pgvector(app)
        elif _USE_VEC:
            _sync_chunks_to_vec(app)
```

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/test_pgvector_sync.py -v`
Expected: 4 SKIPPED, 1 PASSED.

Run:

```bash
TEST_DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach python -m pytest tests/test_pgvector_sync.py -v
```

Expected: 5 PASSED.

- [ ] **Step 8: Confirm no regression**

Run: `python -m pytest -v`
Expected: all tests PASS or SKIP; zero failures.

- [ ] **Step 9: Commit**

```bash
git add jobs/utils.py chatbot/memory.py tests/conftest.py tests/test_pgvector_sync.py
git commit -m "feat: keep pgvector columns in sync with JSON embeddings"
```

---

### Task 3: Job dense search through pgvector

**Files:**
- Create: `jobs/vector_store.py`
- Modify: `jobs/__init__.py` (export `dense_search`)
- Modify: `services/job_service.py:146-170`
- Modify: `jobs/scout_agent.py:304-322`
- Test: `tests/test_job_vector_store.py`

**Interfaces:**
- Consumes: `services.pgvector_support.is_postgres`, `services.pgvector_support.to_vector_literal`, `jobs.utils.get_job_faiss_index`, `jobs.utils.get_embeddings`, `jobs.utils.sync_job_vectors` (Task 2), and the `pg_app` (Task 1) + `app_sqlite` (Task 2) pytest fixtures
- Produces: `jobs.vector_store.dense_search(query_text: str, k: int) -> list[tuple[int, float]]` — `(job_id, similarity)` pairs, best first, similarity clamped to `0.0..1.0` on both backends.

The normalised return shape is the point of this module: `job_service.py` currently converts raw FAISS L2 distance with `1 - (dist ** 2 / 2)` inline, and `scout_agent.py` handles scores separately. Both now receive the same already-normalised numbers.

- [ ] **Step 1: Write the failing test**

Create `tests/test_job_vector_store.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_job_vector_store.py -v`
Expected: 3 SKIPPED, 2 FAILED with `ModuleNotFoundError: No module named 'jobs.vector_store'`

- [ ] **Step 3: Write the module**

Create `jobs/vector_store.py`:

```python
"""
Job dense retrieval.

Single entry point for stage-1 dense search over job postings. On PostgreSQL
this is a pgvector index scan shared by every app instance; on SQLite it
delegates to the existing in-process FAISS index.

Both backends return (job_id, similarity) with similarity normalised to 0..1
(1.0 = identical), so callers need no backend-specific handling.
"""
import logging

from jobs.utils import get_embeddings, get_job_faiss_index
from services.pgvector_support import is_postgres, to_vector_literal

logger = logging.getLogger(__name__)


def _embed_query(query_text: str) -> list:
    """Embed the query. Separate function so tests can monkeypatch it."""
    return get_embeddings().embed_query(query_text)


def dense_search(query_text: str, k: int) -> list[tuple[int, float]]:
    """Return the k nearest active jobs as (job_id, similarity), best first."""
    if k <= 0:
        return []
    if is_postgres():
        return _dense_search_pgvector(query_text, k)
    return _dense_search_faiss(query_text, k)


def _dense_search_pgvector(query_text: str, k: int) -> list[tuple[int, float]]:
    from sqlalchemy import text
    from models import db

    try:
        query_vec = _embed_query(query_text)
    except Exception as e:
        logger.error("Failed to embed job search query: %s", e)
        return []

    try:
        rows = db.session.execute(
            text(
                "SELECT id, 1 - (embedding_vec <=> CAST(:qvec AS vector)) AS similarity "
                "FROM job_postings "
                "WHERE is_active = true AND embedding_vec IS NOT NULL "
                "ORDER BY embedding_vec <=> CAST(:qvec AS vector) "
                "LIMIT :k"
            ),
            {"qvec": to_vector_literal(query_vec), "k": k},
        ).fetchall()
    except Exception as e:
        logger.error("pgvector job search failed, falling back to FAISS: %s", e)
        return _dense_search_faiss(query_text, k)

    # Cosine similarity is -1..1; clamp so callers get a stable 0..1 score.
    return [(row[0], max(0.0, min(1.0, float(row[1])))) for row in rows]


def _dense_search_faiss(query_text: str, k: int) -> list[tuple[int, float]]:
    index = get_job_faiss_index()
    if index is None:
        logger.warning("No FAISS job index available")
        return []

    docs_with_scores = index.similarity_search_with_score(
        query_text, k=min(k, index.index.ntotal)
    )

    results = []
    for doc, distance in docs_with_scores:
        job_id = doc.metadata.get("job_id")
        if job_id is None:
            continue
        # FAISS returns squared L2 on normalised vectors; this is the same
        # conversion services/job_service.py used inline before.
        similarity = max(0.0, min(1.0, 1 - (distance ** 2 / 2)))
        results.append((job_id, similarity))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_job_vector_store.py -v`
Expected: 3 SKIPPED, 2 PASSED.

Run:

```bash
TEST_DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach python -m pytest tests/test_job_vector_store.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Export from the jobs package**

In `jobs/__init__.py`, add to the imports from `jobs.utils` block a new import line and extend `__all__`:

```python
from jobs.vector_store import dense_search
```

and add `"dense_search"` to the `__all__` list.

- [ ] **Step 6: Rewire job_service.py**

In `services/job_service.py`, change the import on line 17 from:

```python
from jobs.utils import get_embeddings, get_job_faiss_index, get_bm25_index, tokenize_for_bm25
```

to:

```python
from jobs.utils import get_embeddings, get_bm25_index, tokenize_for_bm25
from jobs.vector_store import dense_search
```

Then replace the block from `job_index = get_job_faiss_index()` (line 146) through the end of the `faiss_score_map` assignment (line 169) with:

```python
        # --- Stage 1a: dense search (pgvector on Postgres, FAISS on SQLite) ---
        dense_results = _run_in_thread(dense_search, resume_text, candidate_k)

        if not dense_results:
            logger.warning("No dense results available, falling back to brute-force")
            return find_matching_jobs_old(resume_text, top_k)

        dense_ranked = [job_id for job_id, _ in dense_results]
        # Similarity is already normalised 0..1 by the vector store.
        dense_score_map = dict(dense_results)
```

Then update the one downstream reference — replace `faiss_score_map.get(jid, 0.5)` with `dense_score_map.get(jid, 0.5)`.

Confirm no other references remain:

```bash
grep -n "faiss_score_map\|job_index" services/job_service.py
```

Expected: no output.

- [ ] **Step 7: Rewire scout_agent.py**

In `jobs/scout_agent.py`, change the import on line 18 from:

```python
from jobs.utils import get_job_faiss_index, build_job_faiss_index, cosine_similarity
```

to:

```python
from jobs.utils import build_job_faiss_index, cosine_similarity
from jobs.vector_store import dense_search
```

Replace lines 304-306:

```python
            job_index = get_job_faiss_index()
            if job_index is None:
                return {'analyzed': [], 'saved': []}
```

with nothing (delete them), and replace the search block at lines 320-326:

```python
            docs_with_scores = job_index.similarity_search_with_score(
                resume_text,
                k=min(20, job_index.index.ntotal)  # Get top 20 candidates
            )

            if not docs_with_scores:
                return {'analyzed': [], 'saved': []}
```

with:

```python
            # Stage 1: fast retrieval (pgvector on Postgres, FAISS on SQLite)
            docs_with_scores = dense_search(resume_text, k=20)

            if not docs_with_scores:
                return {'analyzed': [], 'saved': []}
```

`docs_with_scores` is now a list of `(job_id, similarity)` tuples rather than `(Document, distance)`. Inspect every downstream use in the same function and update it — the loop over `candidates` currently reads `doc.metadata['job_id']`, which must become the tuple's first element. Find them with:

```bash
grep -n "docs_with_scores\|doc.metadata\|for doc" jobs/scout_agent.py
```

Update each site so it unpacks `job_id, similarity` directly and loads the `JobPosting` by id.

- [ ] **Step 8: Verify both call sites still work end to end**

Run: `python -m pytest -v`
Expected: all PASS or SKIP, zero failures.

Run the eval gate required by CLAUDE.md for `services/job_service.py`:

```bash
python evals/job_match_eval.py
```

Expected: PASS. If the suite reports scores rather than pass/fail, record the numbers and compare against a run on `main` before these changes — they must not regress.

- [ ] **Step 9: Commit**

```bash
git add jobs/vector_store.py jobs/__init__.py services/job_service.py jobs/scout_agent.py tests/test_job_vector_store.py
git commit -m "feat: route job dense search through pgvector on Postgres"
```

---

### Task 4: Memory search through pgvector

**Files:**
- Modify: `chatbot/memory.py:316-331` (dispatch) and add `_search_memories_pgvector`
- Test: `tests/test_memory_pgvector_search.py`

**Interfaces:**
- Consumes: `services.pgvector_support.is_postgres`, `services.pgvector_support.to_vector_literal`
- Produces: `chatbot.memory._search_memories_pgvector(user_id: int, query_vec: list, top_k: int) -> str` — same newline-joined `"[date | type] content"` format the two existing search functions return.

This is where the cross-user crowding-out bug is fixed: `user_id` moves into the SQL `WHERE` clause, so it filters *before* the top-k cut rather than after.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_pgvector_search.py`:

```python
"""
tests/test_memory_pgvector_search.py
====================================
Postgres-backed memory search.

The isolation test is the important one: the sqlite-vec path fetched top_k*3
candidates globally and filtered by user afterwards, so a user's own memories
could be crowded out by other users'. The pgvector path filters in SQL, so a
user surrounded by many closer foreign chunks still gets their own results.
"""
from datetime import datetime

import pytest

from models import db, UserMemoryChunk

DIM = 768


def _vec(val: float) -> list:
    return [val] * DIM


def _fake_embeddings(val: float):
    class _FakeEmb:
        def embed_query(self, q):
            return _vec(val)
    return _FakeEmb()


def _insert_chunk(user_id, content, vec_val):
    chunk = UserMemoryChunk(
        user_id=user_id,
        content=content,
        memory_type="fact",
        embedding=_vec(vec_val),
        session_date=datetime(2026, 1, 1),
    )
    db.session.add(chunk)
    db.session.commit()
    return chunk


@pytest.mark.postgres
def test_pgvector_search_returns_nearest_first(pg_app, monkeypatch):
    with pg_app.app_context():
        _insert_chunk(1, "near chunk", 0.1)
        _insert_chunk(1, "far chunk", 0.9)

        from chatbot.memory import _sync_chunks_to_pgvector
        _sync_chunks_to_pgvector(pg_app)

        monkeypatch.setattr(
            "chatbot.memory._get_embeddings_for_memory",
            lambda: _fake_embeddings(0.1),
        )

        from chatbot.memory import search_memories
        result = search_memories(1, "test query", top_k=2)

        assert "near chunk" in result
        assert result.strip().split("\n")[0].endswith("near chunk")


@pytest.mark.postgres
def test_pgvector_search_isolates_by_user(pg_app, monkeypatch):
    with pg_app.app_context():
        _insert_chunk(1, "user1 chunk", 0.5)
        _insert_chunk(2, "user2 chunk", 0.5)

        from chatbot.memory import _sync_chunks_to_pgvector
        _sync_chunks_to_pgvector(pg_app)

        monkeypatch.setattr(
            "chatbot.memory._get_embeddings_for_memory",
            lambda: _fake_embeddings(0.5),
        )

        from chatbot.memory import search_memories
        result = search_memories(1, "anything", top_k=4)

        assert "user1 chunk" in result
        assert "user2 chunk" not in result


@pytest.mark.postgres
def test_pgvector_search_not_crowded_out_by_other_users(pg_app, monkeypatch):
    """20 closer foreign chunks must not displace the user's own match."""
    with pg_app.app_context():
        _insert_chunk(1, "my only chunk", 0.9)
        for i in range(20):
            _insert_chunk(2, f"foreign chunk {i}", 0.1)

        from chatbot.memory import _sync_chunks_to_pgvector
        _sync_chunks_to_pgvector(pg_app)

        monkeypatch.setattr(
            "chatbot.memory._get_embeddings_for_memory",
            lambda: _fake_embeddings(0.1),
        )

        from chatbot.memory import search_memories
        result = search_memories(1, "anything", top_k=4)

        assert "my only chunk" in result
        assert "foreign chunk" not in result


@pytest.mark.postgres
def test_pgvector_search_empty_for_unknown_user(pg_app, monkeypatch):
    with pg_app.app_context():
        monkeypatch.setattr(
            "chatbot.memory._get_embeddings_for_memory",
            lambda: _fake_embeddings(0.5),
        )
        from chatbot.memory import search_memories
        assert "No long-term memories" in search_memories(999, "anything", top_k=4)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
TEST_DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach python -m pytest tests/test_memory_pgvector_search.py -v
```

Expected: FAIL. `search_memories` currently has no Postgres branch, so it falls through to `_search_memories_cosine`; `test_pgvector_search_not_crowded_out_by_other_users` may pass by accident, but the others fail because `embedding_vec` is never consulted and ordering is wrong.

- [ ] **Step 3: Implement the pgvector search**

Add to `chatbot/memory.py`, immediately before `_search_memories_cosine`:

```python
def _search_memories_pgvector(user_id: int, query_vec, top_k: int) -> str:
    """Nearest-neighbour search via pgvector, filtered to one user in SQL.

    The user predicate is inside the WHERE clause, so it applies before the
    top-k cut. This is the fix for the sqlite-vec path's global candidate
    fetch, where another user's closer chunks could crowd out this user's.
    """
    from sqlalchemy import text
    from services.pgvector_support import to_vector_literal

    try:
        rows = db.session.execute(
            text(
                "SELECT content, memory_type, session_date "
                "FROM user_memory_chunks "
                "WHERE user_id = :uid AND embedding_vec IS NOT NULL "
                "ORDER BY embedding_vec <=> CAST(:qvec AS vector) "
                "LIMIT :k"
            ),
            {"uid": user_id, "qvec": to_vector_literal(query_vec), "k": top_k},
        ).fetchall()
    except Exception as e:
        logger.error("pgvector memory query failed, falling back to cosine: %s", e)
        return _search_memories_cosine(user_id, query_vec, top_k)

    if not rows:
        return "No long-term memories found for this user yet."

    lines = []
    for content, memory_type, session_date in rows:
        date_str = session_date.strftime("%Y-%m-%d") if session_date else "unknown date"
        lines.append(f"[{date_str} | {memory_type}] {content}")
    return "\n".join(lines)
```

- [ ] **Step 4: Add the dispatch branch**

In `chatbot/memory.py`, replace lines 328-330:

```python
    if _USE_VEC:
        return _search_memories_vec(user_id, query_vec, top_k)
    return _search_memories_cosine(user_id, query_vec, top_k)
```

with:

```python
    from services.pgvector_support import is_postgres
    if is_postgres():
        return _search_memories_pgvector(user_id, query_vec, top_k)
    if _USE_VEC:
        return _search_memories_vec(user_id, query_vec, top_k)
    return _search_memories_cosine(user_id, query_vec, top_k)
```

Also update the `search_memories` docstring at line 319 — it currently claims the Postgres path is an O(n) cosine scan, which this change makes false:

```python
    """Retrieve the most semantically relevant past memories for a given query.

    Uses pgvector on PostgreSQL, sqlite-vec ANN on SQLite when installed, and
    falls back to an O(n) cosine scan otherwise.
    """
```

- [ ] **Step 5: Run the tests**

```bash
TEST_DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach python -m pytest tests/test_memory_pgvector_search.py -v
```

Expected: 4 PASSED.

- [ ] **Step 6: Confirm the SQLite paths still work**

Run: `python -m pytest tests/test_memory_vec_search.py tests/test_sqlite_vec_engine.py -v`
Expected: all PASS — the four pre-existing sqlite-vec tests are untouched by this change.

Run: `python -m pytest -v`
Expected: zero failures.

- [ ] **Step 7: Run the eval gate**

Required by CLAUDE.md for changes to `chatbot/memory.py`:

```bash
python evals/memory_eval.py
```

Expected: PASS, with no regression against a `main` baseline run.

- [ ] **Step 8: Commit**

```bash
git add chatbot/memory.py tests/test_memory_pgvector_search.py
git commit -m "feat: route memory search through pgvector on Postgres"
```

---

### Task 5: Run the Postgres path in CI

**Files:**
- Create: `.github/workflows/test.yml`
- Modify: `CLAUDE.md` (test-count and architecture notes)
- Modify: `docs/superpowers/specs/2026-07-26-aws-deployment-design.md` (mark the pgvector step done)

**Interfaces:**
- Consumes: the `postgres` pytest marker and `TEST_DATABASE_URL` from Task 1.
- Produces: nothing consumed by later tasks. This is the gate that stops the production search path from silently rotting, since the default local run skips it.

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/test.yml`:

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  sqlite:
    name: pytest (SQLite, no services)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: pip
      - run: sudo apt-get update && sudo apt-get install -y libmagic1
      - run: pip install -r requirements.txt
      - run: python -m pytest -v

  postgres:
    name: pytest (PostgreSQL + pgvector)
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: careercoach
          POSTGRES_PASSWORD: localdev
          POSTGRES_DB: careercoach
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U careercoach"
          --health-interval 5s
          --health-timeout 5s
          --health-retries 10
    env:
      TEST_DATABASE_URL: postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: pip
      - run: sudo apt-get update && sudo apt-get install -y libmagic1
      - run: pip install -r requirements.txt
      - name: Verify the migration applies cleanly to PostgreSQL
        env:
          DATABASE_URL: postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach
        run: flask --app wsgi db upgrade
      - name: Run the full suite against PostgreSQL
        run: python -m pytest -v
      - name: Fail if the pgvector tests were skipped
        run: |
          python -m pytest -m postgres --collect-only -q | tail -1
          python -m pytest -m postgres -v 2>&1 | tee /tmp/pg.log
          if grep -q "skipped" /tmp/pg.log; then
            echo "pgvector tests were skipped — TEST_DATABASE_URL is not reaching pytest"
            exit 1
          fi
```

The last step exists because a skipped test is green. Without it, a broken `TEST_DATABASE_URL` would turn the entire Postgres job into a silent no-op that still reports success.

- [ ] **Step 2: Verify the workflow locally before pushing**

Reproduce what CI does, in order:

```bash
docker compose up -d postgres
```

```bash
DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach flask --app wsgi db upgrade
```

```bash
TEST_DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach python -m pytest -v
```

Expected: zero failures, and the `postgres`-marked tests report PASSED rather than SKIPPED.

- [ ] **Step 3: Update CLAUDE.md**

In the "Running the App" table, change the `Unit tests` row from:

```
| Unit tests | `python -m pytest` (32 tests, no API key needed) |
```

to:

```
| Unit tests | `python -m pytest` (no API key needed; pgvector tests skip without `TEST_DATABASE_URL`) |
| Unit tests (Postgres path) | `docker compose up -d postgres` then `TEST_DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach python -m pytest` |
```

Add a row to the "Key files at a glance" table:

```
| Job dense retrieval (pgvector / FAISS) | `jobs/vector_store.py` |
| pgvector dialect helpers | `services/pgvector_support.py` |
```

Add to "Critical Invariants — Never Break These":

```
- **Vector columns are not in `models.py`.** `embedding_vec` columns exist only
  on PostgreSQL, created by raw Alembic DDL and queried with raw SQL. Declaring
  a pgvector column type in `models.py` breaks `db.create_all()` on SQLite and
  therefore the entire test suite. The JSON `embedding` columns remain the
  source of truth; `embedding_vec` is derived.
```

- [ ] **Step 4: Update the deployment spec**

In `docs/superpowers/specs/2026-07-26-aws-deployment-design.md`, under "Rollout sequence", change step 1 to record completion:

```markdown
1. ~~**pgvector migration**~~ — **done.** Job embeddings and memory search now
   run on pgvector when the engine is PostgreSQL, verified by
   `evals/job_match_eval.py`, `evals/memory_eval.py`, and a Postgres CI job.
```

- [ ] **Step 5: Run the whole suite one final time**

Run: `python -m pytest -v`
Expected: zero failures.

```bash
python evals/run_all.py
```

Expected: all suites pass.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/test.yml CLAUDE.md docs/superpowers/specs/2026-07-26-aws-deployment-design.md
git commit -m "ci: run the test suite against PostgreSQL with pgvector"
```

---

## What this plan does not do

Stated so the next plan's author does not assume otherwise:

- **FAISS is not deleted.** It remains the SQLite backend and is still built by `build_job_faiss_index()`. Removing it is only possible once local development also runs on Postgres, which the "dialect dispatch" decision explicitly declined.
- **BM25 is untouched.** Sparse retrieval still uses the Redis-cached in-process index in `jobs/utils.py`. It has the same per-instance staleness characteristic as FAISS did, but it is rebuilt from the database rather than persisted to disk, so it self-heals on restart. Worth revisiting after deployment, not before.
- **The `job_vector_index/` directory is still written** on Postgres deployments, because `build_job_faiss_index()` runs unconditionally. It is harmless — just an unused file — but it means ECS tasks still need writable local storage.
- **`safe_commit()` is unchanged.** Whether the SQLite WAL-lock workaround can be simplified on Postgres is a separate question, listed in the deployment spec's code-changes table.
