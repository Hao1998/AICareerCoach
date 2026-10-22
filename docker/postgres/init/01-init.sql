-- Runs once, only when .data/postgres is empty (first `up`).
-- 1. pgvector in the dev database (the pgvector migration also does this; doing
--    it here means it never depends on the app role's privileges).
-- 2. A separate careercoach_test database: the Postgres test fixture
--    (tests/conftest.py pg_app) calls db.drop_all() on teardown, so pointing
--    TEST_DATABASE_URL at the dev database would wipe your local data.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE DATABASE careercoach_test OWNER careercoach;
\connect careercoach_test
CREATE EXTENSION IF NOT EXISTS vector;
