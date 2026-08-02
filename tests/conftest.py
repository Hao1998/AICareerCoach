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


@pytest.fixture
def app_sqlite():
    application = create_app('test', skip_api_check=True)
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()
