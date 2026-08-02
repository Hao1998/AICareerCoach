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
