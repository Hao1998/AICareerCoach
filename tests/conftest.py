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
        # db.create_all() builds every table/column tracked by models.py, which
        # mirrors the schema as of migration e9e97b70ce4c (the pgvector columns
        # are deliberately NOT SQLAlchemy model columns — see
        # services/pgvector_support.py — so create_all() never touches them).
        db.create_all()

        # Run the real Alembic migration for the pgvector columns/index instead
        # of hand-rolling the DDL here, so schema drift between this fixture
        # and migrations/versions/3c81b22b9730_add_pgvector_columns.py (extension
        # creation, columns, HNSW index) would be caught by the tests rather
        # than silently diverging. Stamp to the revision create_all() already
        # matches, then let Alembic apply only the pgvector migration on top —
        # running the full chain from empty would fail, since earlier
        # migrations assume their target tables/columns don't exist yet.
        from flask_migrate import stamp, upgrade
        stamp(revision='e9e97b70ce4c')
        upgrade(revision='head')

        yield application
        db.session.remove()
        db.drop_all()
        with db.engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
            conn.commit()


@pytest.fixture
def app_sqlite():
    application = create_app('test', skip_api_check=True)
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()
