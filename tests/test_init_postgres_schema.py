"""
tests/test_init_postgres_schema.py
==================================
`make db-init` (scripts/init_postgres_schema.py) must build a working schema
from an EMPTY Postgres database: create_all() for the models, the raw-DDL
pgvector migration on top, and alembic_version at head.

Regression: once a migration after the pgvector one creates a models.py table
(vector_outbox), create_all() has already built it, so upgrading straight to
head failed with DuplicateTable.
"""
import pytest
from sqlalchemy import inspect, text

import config as config_module
from models import db
from tests.conftest import TEST_DATABASE_URL


def _wipe(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))


@pytest.mark.postgres
def test_db_init_builds_schema_from_empty_database(monkeypatch):
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set — skipping Postgres-backed test")

    # Point the script's 'development' app at the test DB, never at .env's DATABASE_URL.
    monkeypatch.setattr(config_module.DevelopmentConfig, "SQLALCHEMY_DATABASE_URI", TEST_DATABASE_URL)
    monkeypatch.setattr(config_module.DevelopmentConfig, "SQLALCHEMY_ENGINE_OPTIONS", {})
    monkeypatch.delenv("FLASK_ENV", raising=False)

    import scripts.init_postgres_schema as init_script
    from factory import create_app

    app = create_app("development", skip_api_check=True)
    with app.app_context():
        assert db.engine.url.render_as_string(hide_password=False) == TEST_DATABASE_URL
        _wipe(db.engine)

    try:
        init_script.main()

        with app.app_context():
            insp = inspect(db.engine)
            assert insp.has_table("vector_outbox")
            job_cols = {c["name"] for c in insp.get_columns("job_postings")}
            assert {"embedding_vec", "embedding_vec_updated_at"} <= job_cols
            head = db.session.execute(text("SELECT version_num FROM alembic_version")).scalar()
            assert head == "ef2562d3d929"

        init_script.main()  # re-run on an existing schema must be a no-op, not an error
    finally:
        with app.app_context():
            _wipe(db.engine)
