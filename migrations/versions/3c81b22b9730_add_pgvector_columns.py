"""add pgvector columns

Adds native vector(768) companion columns on PostgreSQL for job and memory
similarity search. The JSON `embedding` columns remain the source of truth;
these are a derived, indexable representation.

No-op on SQLite — local development and the test suite keep using FAISS and
the sqlite-vec virtual table.

Revision ID: 3c81b22b9730
Revises: e9e97b70ce4c
Create Date: 2026-08-02 14:44:57.805477

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '3c81b22b9730'
down_revision = 'e9e97b70ce4c'
branch_labels = None
depends_on = None


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
