"""
tests/test_sqlite_vec_engine.py
================================
Verifies that the SQLAlchemy engine created by create_app() has sqlite-vec
loaded and can execute vec0 virtual-table queries.

Uses TestConfig (in-memory SQLite) so no disk state is touched.
"""
import pytest
from factory import create_app


@pytest.fixture
def app():
    application = create_app('test', skip_api_check=True)
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
