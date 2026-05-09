"""
Global Database Write Lock

Serializes all DB write operations (commit/flush) across threads to prevent
concurrent write conflicts. Works with any database backend but is especially
critical for SQLite which only allows one writer at a time.

Usage:
    from services.db_lock import safe_commit, safe_flush

    # Instead of: db.session.commit()
    safe_commit()

    # Instead of: db.session.flush()
    safe_flush()

    # For batch operations (multiple adds + one commit):
    from services.db_lock import db_write_lock
    with db_write_lock:
        db.session.add(obj1)
        db.session.add(obj2)
        db.session.commit()
"""

import logging
import threading

from models import db

logger = logging.getLogger(__name__)

db_write_lock = threading.RLock()


def safe_commit():
    """Thread-safe db.session.commit() with automatic rollback on failure."""
    with db_write_lock:
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            raise


def safe_flush():
    """Thread-safe db.session.flush()."""
    with db_write_lock:
        try:
            db.session.flush()
        except Exception:
            db.session.rollback()
            raise
