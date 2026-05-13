"""
Database Write Helpers

Provides safe_commit() and safe_flush() with automatic rollback on failure.
No locking needed — PostgreSQL handles concurrent writes natively with MVCC.
"""

import logging

from models import db

logger = logging.getLogger(__name__)


def safe_commit():
    """db.session.commit() with automatic rollback on failure."""
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise


def safe_flush():
    """db.session.flush() with automatic rollback on failure."""
    try:
        db.session.flush()
    except Exception:
        db.session.rollback()
        raise
