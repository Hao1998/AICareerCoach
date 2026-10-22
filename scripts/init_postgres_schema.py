"""
Create or upgrade the schema on a PostgreSQL database (e.g. the one from
docker-compose.db.yml).

    python scripts/init_postgres_schema.py

Why not just `flask db upgrade`? The migration chain can't build a schema from
an empty database — its earliest revisions ALTER tables that no revision
creates. So on an empty database this does the same bootstrap as
tests/conftest.py `pg_app`: db.create_all() (matches revision e9e97b70ce4c),
stamp that revision, run the raw-DDL pgvector migration, then stamp head
(later migrations only add models.py tables create_all() already built).

Safe to re-run: once alembic_version exists it only runs `upgrade`.
Uses DATABASE_URL from the environment / .env.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from flask_migrate import stamp, upgrade
from sqlalchemy import inspect

from factory import create_app
from models import db

# Last revision whose schema db.create_all() reproduces exactly (see conftest).
CREATE_ALL_REVISION = 'e9e97b70ce4c'
# The raw-DDL pgvector migration (columns/index absent from models.py), which
# create_all() can't build and must therefore really run. Later migrations only
# add models.py tables that create_all() has already built, so they're stamped.
RAW_DDL_REVISION = '3c81b22b9730'


def main():
    app = create_app(os.getenv('FLASK_ENV', 'development'), skip_api_check=True)
    with app.app_context():
        if db.engine.dialect.name != 'postgresql':
            sys.exit(f"DATABASE_URL points at {db.engine.dialect.name}, not PostgreSQL — "
                     "nothing to do (SQLite builds its schema on startup).")

        if inspect(db.engine).has_table('alembic_version'):
            print("Existing schema found — applying pending migrations.")
        else:
            print("Empty database — creating tables and stamping "
                  f"{CREATE_ALL_REVISION}.")
            db.create_all()
            stamp(revision=CREATE_ALL_REVISION)
            upgrade(revision=RAW_DDL_REVISION)
            stamp(revision='head')

        upgrade(revision='head')
        print("Schema is up to date.")


if __name__ == '__main__':
    main()
