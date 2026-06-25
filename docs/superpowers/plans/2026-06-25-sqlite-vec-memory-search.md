# sqlite-vec Memory Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the O(n) Python cosine-scan in `search_memories` with sqlite-vec ANN queries so memory retrieval stays fast as `UserMemoryChunk` rows accumulate per user.

**Architecture:** Load the sqlite-vec extension into the existing SQLAlchemy/SQLite connection via a `@event.listens_for(engine, "connect")` hook in `factory.py`. Maintain a `vec_user_memories` virtual table (rowid = `user_memory_chunks.id`, embedding float[768]) and query it with `WHERE embedding MATCH ?`. Fall back to the existing cosine scan on PostgreSQL deployments where the extension is unavailable.

**Tech Stack:** `sqlite-vec==0.1.9` (already pinned in requirements.txt), SQLAlchemy event hooks, Flask-Migrate for the virtual-table migration, raw `sqlite3` connection for the ANN query, existing HuggingFace `all-mpnet-base-v2` embeddings (768-dim).

## Global Constraints

- Python 3.12, Flask/SQLAlchemy stack as in `requirements.txt`.
- Never import from `controllers/` in `services/` or `agents/`. `models.py` imports nothing from the app.
- All DB operations in background threads must use `with app.app_context():`.
- Use `services/db_lock.safe_commit()` inside job fetchers/background jobs; `db.session.commit()` elsewhere.
- Embedding model is fixed: `sentence-transformers/all-mpnet-base-v2` → 768 float32 dimensions.
- Run migrations via `flask --app wsgi db migrate` / `flask --app wsgi db upgrade` — never hand-edit migration files.
- The sqlite-vec changes must not break the 32-test offline suite: `python -m pytest`.
- Do **not** touch the job/FAISS path (`jobs/utils.py`, `services/job_service.py`) — scope is memory only.

---

### Task 1: Install sqlite-vec and verify it loads in this environment

**Why:** `sqlite-vec` requires a native C extension that is loaded at runtime via `enable_load_extension`. On some machines (especially macOS with system Python), this call is blocked. Verifying before writing any integration code means we fail fast on an environmental blocker rather than discovering it mid-implementation.

**Benefit:** Zero wasted implementation effort if the extension can't load. One snippet run here saves potentially reverting Tasks 2–5.

**Files:**
- No source files changed.
- Test: run a quick one-off Python snippet (shown below).

**Interfaces:**
- Produces: confidence that `sqlite_vec.load(conn)` and `conn.execute("SELECT vec_version()")` work on this machine.

- [ ] **Step 1: Install the package**

```bash
pip install sqlite-vec==0.1.9
```

Expected output contains: `Successfully installed sqlite-vec-0.1.9`  
(The package is already pinned in `requirements.txt`, so this just installs it into the active venv.)

- [ ] **Step 2: Confirm the extension loads and can create a virtual table**

```bash
python3 - <<'EOF'
import sqlite3, sqlite_vec, struct

conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)

print("vec_version:", conn.execute("SELECT vec_version()").fetchone()[0])

conn.execute("CREATE VIRTUAL TABLE t USING vec0(embedding float[4])")

def ser(floats):
    return struct.pack(f"{len(floats)}f", *floats)

conn.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [ser([0.1, 0.2, 0.3, 0.4])])
conn.execute("INSERT INTO t(rowid, embedding) VALUES (2, ?)", [ser([0.9, 0.8, 0.7, 0.6])])

rows = conn.execute(
    "SELECT rowid, distance FROM t WHERE embedding MATCH ? ORDER BY distance LIMIT 2",
    [ser([0.1, 0.2, 0.3, 0.4])]
).fetchall()
print("ANN results (rowid, distance):", rows)  # expect rowid 1 first (distance ≈ 0)
assert rows[0][0] == 1, "nearest row should be rowid 1"
print("OK")
EOF
```

Expected output:
```
vec_version: v0.1.9
ANN results (rowid, distance): [(1, 0.0), (2, ...)]
OK
```

- [ ] **Step 3: Commit (no source changes — just confirms env readiness)**

```bash
git add requirements.txt  # already pinned; no change expected
git status  # confirm nothing untracked needs committing
```

If `requirements.txt` already has `sqlite-vec==0.1.9` and is unchanged, skip the commit.

---

### Task 2: Wire sqlite-vec into the SQLAlchemy engine

**Why:** SQLAlchemy manages a connection pool, creating new DBAPI connections transparently. If we only load the extension once, connections created later (new requests, background threads) will be missing it, causing `no such function: vec_version` errors at runtime. The `@event.listens_for(engine, "connect")` hook fires for every new connection, ensuring the extension is always present.

**Benefit:** All future tasks and the rest of the codebase get vec0 support for free — no manual setup per connection or per thread. Also adds the `_is_sqlite()` guard that makes the entire feature safely skip on PostgreSQL deployments.

**Files:**
- Modify: `factory.py` (add extension-load hook + `_is_sqlite()` helper)

**Interfaces:**
- Produces: `_is_sqlite(app) -> bool` — importable from `factory` by the migration and tests.
- Produces: Every SQLAlchemy connection to a SQLite DB automatically has sqlite-vec loaded before any SQL is run.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sqlite_vec_engine.py`:

```python
"""
tests/test_sqlite_vec_engine.py
================================
Verifies that the SQLAlchemy engine created by create_app() has sqlite-vec
loaded and can execute vec0 virtual-table queries.

Uses TestConfig (in-memory SQLite) so no disk state is touched.
"""
import pytest
from factory import create_app
from config import TestConfig


@pytest.fixture
def app():
    application = create_app(TestConfig)
    with application.app_context():
        yield application


def test_sqlite_vec_extension_loaded(app):
    """vec_version() is callable — extension loaded on every connection."""
    from sqlalchemy import text
    with app.extensions['sqlalchemy'].engine.connect() as conn:
        version = conn.execute(text("SELECT vec_version()")).scalar()
    assert version.startswith("v"), f"unexpected version: {version}"


def test_vec0_virtual_table_query(app):
    """Can create a temporary vec0 table and do an ANN query end-to-end."""
    import struct
    from sqlalchemy import text

    def ser(floats):
        return struct.pack(f"{len(floats)}f", *floats)

    with app.extensions['sqlalchemy'].engine.connect() as conn:
        conn.execute(text("CREATE VIRTUAL TABLE tmp_vec USING vec0(embedding float[4])"))
        conn.execute(
            text("INSERT INTO tmp_vec(rowid, embedding) VALUES (42, :v)"),
            {"v": ser([0.1, 0.2, 0.3, 0.4])}
        )
        row = conn.execute(
            text("SELECT rowid, distance FROM tmp_vec WHERE embedding MATCH :v ORDER BY distance LIMIT 1"),
            {"v": ser([0.1, 0.2, 0.3, 0.4])}
        ).fetchone()
    assert row[0] == 42
    assert row[1] < 0.001
```

- [ ] **Step 2: Run the test — verify it fails**

```bash
python -m pytest tests/test_sqlite_vec_engine.py -v
```

Expected: `FAILED` — `OperationalError: no such function: vec_version`

- [ ] **Step 3: Add the extension-load hook to `factory.py`**

Open `factory.py`. Find the `create_app()` function. After `db.init_app(app)` (or wherever the SQLAlchemy extension is initialised), add:

```python
# ── sqlite-vec extension (SQLite only) ──────────────────────────────────────
def _is_sqlite(application) -> bool:
    uri = application.config.get("SQLALCHEMY_DATABASE_URI", "")
    return uri.startswith("sqlite")


def _register_sqlite_vec(application):
    """Load the sqlite-vec extension on every new DBAPI connection."""
    if not _is_sqlite(application):
        return
    try:
        import sqlite_vec
        from sqlalchemy import event
        engine = application.extensions['sqlalchemy'].engine

        @event.listens_for(engine, "connect")
        def _load_vec(dbapi_conn, _record):
            dbapi_conn.enable_load_extension(True)
            sqlite_vec.load(dbapi_conn)
            dbapi_conn.enable_load_extension(False)

    except ImportError:
        application.logger.warning("sqlite-vec not installed; memory ANN search unavailable")


# call it right after db.init_app(app):
_register_sqlite_vec(app)
```

Make `_is_sqlite` importable at module level (outside `create_app`) so the migration can use it.

> **Exact placement in `factory.py`:** right after the line `db.init_app(app)`. Add `_register_sqlite_vec(app)` on the very next line inside `create_app()`. The two helper functions (`_is_sqlite`, `_register_sqlite_vec`) go at module level, just before `create_app`.

- [ ] **Step 4: Run the tests — they should pass**

```bash
python -m pytest tests/test_sqlite_vec_engine.py -v
```

Expected: `PASSED` for both tests.

- [ ] **Step 5: Confirm full suite still passes**

```bash
python -m pytest -v
```

Expected: all 32 existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add factory.py tests/test_sqlite_vec_engine.py
git commit -m "feat: load sqlite-vec extension into SQLAlchemy engine on connect"
```

---

### Task 3: Create the `vec_user_memories` virtual table via migration

**Why:** The vec0 virtual table must exist in the DB before any code can query it. Using Flask-Migrate keeps schema changes in the same versioned migration chain as all other DB changes — so `flask db upgrade` is the single deployment step, just like every other schema change in this project. Hand-creating the table outside the migration system would leave it out of version control and break the upgrade path for any future deployment.

**Benefit:** The table is created automatically on `flask db upgrade`, making deployment to staging/production a one-liner. The `IF NOT EXISTS` guard and the `dialect.name == "sqlite"` check make the migration re-runnable and Postgres-safe with no extra configuration.

**Files:**
- New migration file (generated by Flask-Migrate, then hand-reviewed): `migrations/versions/<rev>_add_vec_user_memories.py`

**Interfaces:**
- Produces: `vec_user_memories` virtual table in the SQLite DB: `rowid INTEGER PRIMARY KEY`, `embedding float[768]`. `rowid` is the FK to `user_memory_chunks.id`.
- Consumes: `_is_sqlite` from `factory` (to skip on Postgres).

- [ ] **Step 1: Generate the migration scaffold**

```bash
flask --app wsgi db migrate -m "add vec_user_memories virtual table"
```

This creates a new file under `migrations/versions/`. Open it.

- [ ] **Step 2: Replace the auto-generated body with the correct virtual-table DDL**

The generated `upgrade()` and `downgrade()` will be empty (Alembic can't introspect virtual tables). Replace them:

```python
from alembic import op
import sqlalchemy as sa


def upgrade():
    # Virtual tables are SQLite-only; skip on Postgres.
    bind = op.get_bind()
    if not bind.dialect.name == "sqlite":
        return
    op.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_user_memories "
        "USING vec0(embedding float[768])"
    )


def downgrade():
    bind = op.get_bind()
    if not bind.dialect.name == "sqlite":
        return
    op.execute("DROP TABLE IF EXISTS vec_user_memories")
```

- [ ] **Step 3: Apply the migration**

```bash
flask --app wsgi db upgrade
```

Expected output ends with: `Running upgrade ... -> <rev>, add vec_user_memories virtual table`

- [ ] **Step 4: Verify the table exists**

```bash
python3 - <<'EOF'
from factory import create_app
from config import DevelopmentConfig
app = create_app(DevelopmentConfig)
with app.app_context():
    from sqlalchemy import text
    with app.extensions['sqlalchemy'].engine.connect() as conn:
        row = conn.execute(text("SELECT name FROM sqlite_master WHERE name='vec_user_memories'")).fetchone()
    print("table exists:", row)
    assert row is not None, "vec_user_memories missing"
    print("OK")
EOF
```

Expected: `table exists: ('vec_user_memories',)` / `OK`

- [ ] **Step 5: Commit the migration**

```bash
git add migrations/versions/
git commit -m "feat: add vec_user_memories virtual table migration"
```

---

### Task 4: Backfill existing `UserMemoryChunk` embeddings into the vec table

**Why:** The `user_memory_chunks` table already contains embeddings stored as JSON for every conversation session indexed so far. The new `vec_user_memories` table starts empty — if we skip this step, the ANN search path will return zero results for all existing users until they accumulate enough new sessions to re-populate it. That would be a silent regression: users would lose access to their entire memory history the moment we deploy.

**Benefit:** All existing users' memories are immediately searchable via sqlite-vec on first deploy, with no data loss or "warm-up" period. The script is idempotent (delete-then-insert), so it can be safely re-run if the migration is repeated on a test environment.

**Files:**
- New script: `scripts/backfill_vec_memories.py`

**Interfaces:**
- Consumes: `UserMemoryChunk.id`, `UserMemoryChunk.embedding` (JSON list of 768 floats).
- Produces: rows inserted into `vec_user_memories` where `rowid = UserMemoryChunk.id`.

- [ ] **Step 1: Write the backfill script**

Create `scripts/backfill_vec_memories.py`:

```python
"""
One-time backfill: copy embeddings from user_memory_chunks.embedding (JSON)
into the vec_user_memories virtual table.

Run once after the migration:
    python scripts/backfill_vec_memories.py
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from factory import create_app
from config import DevelopmentConfig
from models import UserMemoryChunk, db

EMBEDDING_DIM = 768


def ser(floats: list) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


def backfill():
    app = create_app(DevelopmentConfig)
    with app.app_context():
        chunks = (UserMemoryChunk.query
                  .filter(UserMemoryChunk.embedding.isnot(None))
                  .all())
        print(f"Found {len(chunks)} chunks with embeddings to backfill.")

        from sqlalchemy import text
        engine = app.extensions['sqlalchemy'].engine

        inserted = 0
        skipped = 0
        with engine.connect() as conn:
            for chunk in chunks:
                emb = chunk.embedding
                if not isinstance(emb, list) or len(emb) != EMBEDDING_DIM:
                    skipped += 1
                    continue
                # Upsert: delete then insert so re-runs are idempotent.
                conn.execute(
                    text("DELETE FROM vec_user_memories WHERE rowid = :rid"),
                    {"rid": chunk.id}
                )
                conn.execute(
                    text("INSERT INTO vec_user_memories(rowid, embedding) VALUES (:rid, :emb)"),
                    {"rid": chunk.id, "emb": ser(emb)}
                )
                inserted += 1
            conn.commit()

        print(f"Backfilled {inserted} rows. Skipped {skipped} (bad shape).")


if __name__ == "__main__":
    backfill()
```

- [ ] **Step 2: Run the backfill**

```bash
python scripts/backfill_vec_memories.py
```

Expected output (numbers will vary):
```
Found N chunks with embeddings to backfill.
Backfilled N rows. Skipped 0 (bad shape).
```

- [ ] **Step 3: Spot-check the vec table row count**

```bash
python3 - <<'EOF'
from factory import create_app
from config import DevelopmentConfig
app = create_app(DevelopmentConfig)
with app.app_context():
    from sqlalchemy import text
    with app.extensions['sqlalchemy'].engine.connect() as conn:
        n_vec = conn.execute(text("SELECT count(*) FROM vec_user_memories")).scalar()
        n_chunks = conn.execute(text(
            "SELECT count(*) FROM user_memory_chunks WHERE embedding IS NOT NULL"
        )).scalar()
    print(f"vec rows={n_vec}, source rows={n_chunks}")
    assert n_vec == n_chunks, "mismatch!"
    print("OK")
EOF
```

- [ ] **Step 4: Commit**

```bash
git add scripts/backfill_vec_memories.py
git commit -m "feat: backfill existing memory embeddings into vec_user_memories"
```

---

### Task 5: Replace `search_memories` + keep vec table in sync on writes

**Why:** This is the core of the migration. The existing `search_memories` loads every `UserMemoryChunk` row for a user into Python memory and scores them one-by-one with numpy cosine arithmetic — O(n) per query with no early exit. As a user accumulates dozens of sessions over months, this becomes noticeable latency in every chat turn that triggers the `search_memory` tool. The vec0 ANN query pushes the similarity computation into C inside the DB engine and returns only the `top_k` rows, making the query O(log n) regardless of history length. The `_sync_chunks_to_vec` call inside `index_session_memories` ensures the vec table is kept current for all future writes, so the backfill from Task 4 never needs to be re-run.

**Benefit:** Memory search latency stays flat as user history grows — the slow path only applied as long as the legacy cosine code was in place. The `_USE_VEC` flag and preserved `_search_memories_cosine` path guarantee the feature degrades gracefully on Postgres without requiring any configuration change. Public signature of `search_memories` is unchanged so `chatbot/tools.py` needs no edits.

**Files:**
- Modify: `chatbot/memory.py` — replace `search_memories`, update `index_session_memories`.

**Interfaces:**
- Consumes: `vec_user_memories` virtual table (rowid = `user_memory_chunks.id`, embedding float[768]).
- Produces: `search_memories(user_id, query, top_k=4) -> str` — same signature, same return format. Falls back to existing O(n) cosine on Postgres.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_vec_search.py`:

```python
"""
tests/test_memory_vec_search.py
================================
Offline tests for the sqlite-vec-backed search_memories function.

Uses a real in-memory SQLite DB (TestConfig) with the sqlite-vec extension
loaded by the factory. Embeddings are tiny 4-float vectors so no model is
needed — we monkeypatch get_embeddings().embed_query to return controlled
vectors.

What each test covers
---------------------
test_search_memories_returns_nearest_chunk
    Given two chunks with known embeddings, the one closest to the query
    embedding is returned first.

test_search_memories_isolates_by_user
    A chunk belonging to user_id=2 must never appear in user_id=1 results.

test_search_memories_empty
    Returns the "no memories found" string when the user has no chunks.

test_search_memories_falls_back_on_non_sqlite
    When the DB is not SQLite, search_memories must still return a result
    (via the legacy cosine fallback). This test monkeypatches _is_sqlite to
    return False to simulate Postgres without needing a Postgres DB.
"""
import struct
import pytest
from factory import create_app
from config import TestConfig
from models import db, UserMemoryChunk
from datetime import datetime

FAKE_DIM = 768  # must match real dim so struct packing works


def _ser(floats):
    return struct.pack(f"{len(floats)}f", *floats)


def _vec(val: float) -> list:
    """Return a 768-dim vector where every element is `val`."""
    return [val] * FAKE_DIM


@pytest.fixture
def app(monkeypatch):
    application = create_app(TestConfig)
    with application.app_context():
        db.create_all()
        # Create vec table
        from sqlalchemy import text
        with application.extensions['sqlalchemy'].engine.connect() as conn:
            conn.execute(text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_user_memories "
                "USING vec0(embedding float[768])"
            ))
            conn.commit()
        yield application
        db.drop_all()


def _insert_chunk(app, user_id, content, vec_val, memory_type="fact"):
    """Insert a UserMemoryChunk + corresponding vec row."""
    from sqlalchemy import text
    emb = _vec(vec_val)
    chunk = UserMemoryChunk(
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        embedding=emb,
        session_date=datetime(2026, 1, 1),
    )
    db.session.add(chunk)
    db.session.flush()  # get chunk.id
    with app.extensions['sqlalchemy'].engine.connect() as conn:
        conn.execute(
            text("INSERT INTO vec_user_memories(rowid, embedding) VALUES (:rid, :emb)"),
            {"rid": chunk.id, "emb": _ser(emb)}
        )
        conn.commit()
    db.session.commit()
    return chunk


def test_search_memories_returns_nearest_chunk(app, monkeypatch):
    chunk_near = _insert_chunk(app, user_id=1, content="near chunk", vec_val=0.1)
    _insert_chunk(app, user_id=1, content="far chunk", vec_val=0.9)

    # Query embedding close to chunk_near
    monkeypatch.setattr(
        "chatbot.memory._get_embeddings_for_memory",
        lambda: type("E", (), {"embed_query": lambda self, q: _vec(0.1)})()
    )

    from chatbot.memory import search_memories
    result = search_memories(1, "test query", top_k=2)
    assert "near chunk" in result
    lines = result.strip().split("\n")
    assert lines[0].endswith("near chunk"), f"unexpected order: {result}"


def test_search_memories_isolates_by_user(app, monkeypatch):
    _insert_chunk(app, user_id=1, content="user1 chunk", vec_val=0.5)
    _insert_chunk(app, user_id=2, content="user2 chunk", vec_val=0.5)

    monkeypatch.setattr(
        "chatbot.memory._get_embeddings_for_memory",
        lambda: type("E", (), {"embed_query": lambda self, q: _vec(0.5)})()
    )

    from chatbot.memory import search_memories
    result = search_memories(1, "anything", top_k=4)
    assert "user1 chunk" in result
    assert "user2 chunk" not in result


def test_search_memories_empty(app, monkeypatch):
    monkeypatch.setattr(
        "chatbot.memory._get_embeddings_for_memory",
        lambda: type("E", (), {"embed_query": lambda self, q: _vec(0.5)})()
    )
    from chatbot.memory import search_memories
    result = search_memories(999, "anything", top_k=4)
    assert "No long-term memories" in result


def test_search_memories_falls_back_on_non_sqlite(app, monkeypatch):
    _insert_chunk(app, user_id=1, content="fallback chunk", vec_val=0.5)

    monkeypatch.setattr(
        "chatbot.memory._get_embeddings_for_memory",
        lambda: type("E", (), {"embed_query": lambda self, q: _vec(0.5)})()
    )
    # Pretend we are not on SQLite
    import chatbot.memory as mem_mod
    monkeypatch.setattr(mem_mod, "_USE_VEC", False)

    from chatbot.memory import search_memories
    result = search_memories(1, "anything", top_k=4)
    assert "fallback chunk" in result
```

- [ ] **Step 2: Run the tests — verify they fail**

```bash
python -m pytest tests/test_memory_vec_search.py -v
```

Expected: all 4 tests fail — `_get_embeddings_for_memory` and `_USE_VEC` don't exist yet.

- [ ] **Step 3: Rewrite `chatbot/memory.py` — replace `search_memories` and update `index_session_memories`**

At the top of `chatbot/memory.py`, add these imports after the existing ones:

```python
import struct

# Set to False at module load if sqlite-vec is unavailable or DB is Postgres.
# Checked at search time so tests can monkeypatch it.
_USE_VEC = False

try:
    import sqlite_vec as _sqlite_vec  # noqa: F401
    _USE_VEC = True
except ImportError:
    pass


def _get_embeddings_for_memory():
    """Thin wrapper around get_embeddings() — exists so tests can monkeypatch it."""
    from jobs.utils import get_embeddings
    return get_embeddings()


def _serialize_f32(floats: list) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)
```

Replace the existing `search_memories` function (lines 264–305 in the current file) with:

```python
def search_memories(user_id: int, query: str, top_k: int = 4) -> str:
    """Retrieve the most semantically relevant past memories for a given query.

    Uses sqlite-vec ANN when available; falls back to O(n) cosine on Postgres.
    """
    try:
        query_vec = _get_embeddings_for_memory().embed_query(query)
    except Exception as e:
        logger.error("Failed to embed memory search query: %s", e)
        return "Memory search temporarily unavailable."

    if _USE_VEC:
        return _search_memories_vec(user_id, query_vec, top_k)
    return _search_memories_cosine(user_id, query_vec, top_k)


def _search_memories_vec(user_id: int, query_vec: list, top_k: int) -> str:
    """ANN search via sqlite-vec virtual table."""
    from sqlalchemy import text
    from flask import current_app

    engine = current_app.extensions['sqlalchemy'].engine
    query_bytes = _serialize_f32(query_vec)

    try:
        with engine.connect() as conn:
            # Get top_k*3 candidates from the vec index (no user filter there),
            # then filter to this user in Python — avoids assuming aux-column
            # support in vec0 0.1.x.
            rows = conn.execute(
                text(
                    "SELECT rowid, distance "
                    "FROM vec_user_memories "
                    "WHERE embedding MATCH :q "
                    "ORDER BY distance "
                    "LIMIT :lim"
                ),
                {"q": query_bytes, "lim": top_k * 3}
            ).fetchall()
    except Exception as e:
        logger.error("sqlite-vec query failed, falling back to cosine: %s", e)
        return _search_memories_cosine(user_id, list(query_vec), top_k)

    if not rows:
        return "No long-term memories found for this user yet."

    candidate_ids = [r[0] for r in rows]
    distance_by_id = {r[0]: r[1] for r in rows}

    # Filter to this user and preserve ANN distance order.
    chunks = (UserMemoryChunk.query
              .filter(UserMemoryChunk.id.in_(candidate_ids))
              .filter_by(user_id=user_id)
              .all())

    if not chunks:
        return "No long-term memories found for this user yet."

    chunks.sort(key=lambda c: distance_by_id.get(c.id, float("inf")))
    chunks = chunks[:top_k]

    lines = []
    for chunk in chunks:
        date_str = chunk.session_date.strftime("%Y-%m-%d") if chunk.session_date else "unknown date"
        lines.append(f"[{date_str} | {chunk.memory_type}] {chunk.content}")
    return "\n".join(lines)


def _search_memories_cosine(user_id: int, query_vec, top_k: int) -> str:
    """Legacy O(n) cosine scan — fallback for Postgres or when sqlite-vec unavailable."""
    import numpy as np

    chunks = (UserMemoryChunk.query
              .filter_by(user_id=user_id)
              .filter(UserMemoryChunk.embedding.isnot(None))
              .order_by(UserMemoryChunk.session_date.desc())
              .all())

    if not chunks:
        return "No long-term memories found for this user yet."

    query_np = np.array(query_vec)
    scored = []
    for chunk in chunks:
        try:
            chunk_vec = np.array(chunk.embedding)
            sim = float(np.dot(query_np, chunk_vec) / (
                np.linalg.norm(query_np) * np.linalg.norm(chunk_vec) + 1e-9
            ))
            scored.append((sim, chunk))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top:
        return "No relevant memories found."

    lines = []
    for _score, chunk in top:
        date_str = chunk.session_date.strftime("%Y-%m-%d") if chunk.session_date else "unknown date"
        lines.append(f"[{date_str} | {chunk.memory_type}] {chunk.content}")
    return "\n".join(lines)
```

Now update `index_session_memories` to also insert into the vec table after each `UserMemoryChunk` is committed. Find the `db.session.add(chunk)` loop (around line 248) and add the vec insert after `safe_commit()`:

```python
        try:
            safe_commit()
            logger.info("Indexed %d memory chunks for user %s", len(chunks_to_index), user_id)
        except Exception as e:
            logger.error("Failed to commit memory chunks: %s", e)
            return  # nothing to vec-index if commit failed

        # Sync to vec table (SQLite only — no-op if _USE_VEC is False).
        if _USE_VEC:
            _sync_chunks_to_vec(app)


def _sync_chunks_to_vec(app):
    """Insert any UserMemoryChunk rows missing from vec_user_memories."""
    from sqlalchemy import text

    engine = app.extensions['sqlalchemy'].engine
    try:
        with engine.connect() as conn:
            # Find chunks that have an embedding but no vec row yet.
            missing = conn.execute(text(
                "SELECT c.id, c.embedding "
                "FROM user_memory_chunks c "
                "LEFT JOIN vec_user_memories v ON v.rowid = c.id "
                "WHERE c.embedding IS NOT NULL AND v.rowid IS NULL"
            )).fetchall()

            for row_id, embedding_json in missing:
                if not isinstance(embedding_json, list):
                    continue
                conn.execute(
                    text("INSERT INTO vec_user_memories(rowid, embedding) VALUES (:rid, :emb)"),
                    {"rid": row_id, "emb": _serialize_f32(embedding_json)}
                )
            conn.commit()
    except Exception as e:
        logger.error("Failed to sync memory chunks to vec table: %s", e)
```

> Note: `_sync_chunks_to_vec` is called after `safe_commit()` inside `index_session_memories`, so `chunk.id` is guaranteed to be populated.

- [ ] **Step 4: Run the new tests**

```bash
python -m pytest tests/test_memory_vec_search.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest -v
```

Expected: all tests pass (32 existing + 4 new + 2 engine tests = 38).

- [ ] **Step 6: Commit**

```bash
git add chatbot/memory.py tests/test_memory_vec_search.py
git commit -m "feat: replace O(n) cosine memory search with sqlite-vec ANN"
```

---

### Task 6: Manual smoke test — end-to-end in the running app

**Why:** Unit tests in Tasks 2 and 5 use in-memory SQLite with a fixture-created vec table and monkeypatched embeddings. They cannot catch integration problems that only appear with real data: embedding shape mismatches from the actual HuggingFace model, the connection-pool hook firing correctly in a live Flask request context, or the `search_memory` LangChain tool actually invoking `search_memories` end-to-end through the agent. These are exactly the class of bugs that unit tests miss.

**Benefit:** Gives us a human-in-the-loop confirmation that the new path is correct before marking the feature done. Catches any remaining environment-specific issue (e.g. macOS `enable_load_extension` restriction in a specific Python build) before it hits a real user session.

**Files:**
- No code changes. This task is verification only.

**Interfaces:**
- Consumes: the running app (`python app.py`) + a test user account with at least one prior chat session.

- [ ] **Step 1: Start the app**

```bash
python app.py
```

Navigate to `http://localhost:5001`. Log in with a test account (or register one).

- [ ] **Step 2: Trigger memory indexing**

Send a few messages in the chat that mention career facts:
```
I have 7 years of experience in data engineering. I want remote-only roles.
```
Wait > 5 minutes OR restart the app to trigger `close_and_summarize_session` (which calls `index_session_memories`).

- [ ] **Step 3: Verify the vec table has rows**

```bash
python3 - <<'EOF'
from factory import create_app
from config import DevelopmentConfig
app = create_app(DevelopmentConfig)
with app.app_context():
    from sqlalchemy import text
    with app.extensions['sqlalchemy'].engine.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM vec_user_memories")).scalar()
    print(f"vec_user_memories row count: {n}")
EOF
```

- [ ] **Step 4: Invoke `search_memory` tool through the chat**

In the chat UI, ask something that should trigger the `search_memory` tool:
```
What do you know about my career goals from previous sessions?
```

Verify the AI responds with facts from prior sessions (not "no memories found").

- [ ] **Step 5: Check app logs for the vec path being used**

In the terminal running `python app.py`, there should be no `sqlite-vec query failed` error lines. The memory search should silently succeed.

- [ ] **Step 6: Final commit note**

No code changes in this task. If smoke test reveals issues, fix them in `chatbot/memory.py` and create a new commit before marking this task done.

---

## Self-Review

**Spec coverage:**
- ✅ sqlite-vec installed and verified (Task 1)
- ✅ Extension loaded into SQLAlchemy engine (Task 2)
- ✅ Virtual table created via proper migration (Task 3)
- ✅ Existing embeddings backfilled (Task 4)
- ✅ `search_memories` replaced with ANN path (Task 5)
- ✅ `index_session_memories` keeps vec table in sync (Task 5)
- ✅ Postgres fallback via `_USE_VEC` flag (Task 5)
- ✅ All 32 existing tests must still pass (checked in Tasks 2 and 5)
- ✅ Smoke test in live app (Task 6)
- ✅ FAISS job path untouched (scope constraint honoured)

**Placeholder scan:** None found — all steps have actual code.

**Type consistency:**
- `_serialize_f32(list) -> bytes` — used consistently in Task 4 (backfill), Task 5 (index + search).
- `_get_embeddings_for_memory()` — introduced in Task 5 step 3, patched in test step 1. Name matches.
- `_USE_VEC: bool` — introduced in Task 5 step 3, patched in test `test_search_memories_falls_back_on_non_sqlite`. Name matches.
- `vec_user_memories` table name — consistent across migration (Task 3), backfill (Task 4), sync (Task 5).
