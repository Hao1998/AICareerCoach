.DEFAULT_GOAL := help

VENV        := .venv
PYTHON      := $(VENV)/bin/python
FLASK       := $(VENV)/bin/flask
PYTEST      := $(VENV)/bin/pytest
CELERY      := $(VENV)/bin/celery

# Every target depends on this so a stray system `python`/`flask` on PATH
# can never silently run instead of the project's venv.
$(PYTHON):
	@echo "No .venv found (or it's missing python). Create it first:"; \
	echo "  python3 -m venv .venv && $(VENV)/bin/pip install -r requirements.txt"; \
	exit 1

.PHONY: help
help:
	@echo "AiCareerCoach — make targets (all run inside .venv, no activation needed)"
	@echo ""
	@echo "  make run              Start dev server (python app.py, port 5001)"
	@echo "  make prod             Start production server (gunicorn wsgi:app)"
	@echo "  make migrate MSG=... Generate a new migration"
	@echo "  make upgrade          Apply migrations (flask db upgrade)"
	@echo "  make evals            Run all evals"
	@echo "  make mcp              Start MCP server"
	@echo "  make worker           Start Celery worker (scout queue)"
	@echo "  make beat             Start Celery beat scheduler"
	@echo "  make test             Run unit tests (SQLite-backed)"
	@echo "  make test-pg          Run unit tests against Postgres (careercoach_test DB)"
	@echo "  make db-up            Start local Postgres (docker-compose.db.yml, data in .data/)"
	@echo "  make db-init          Create/upgrade the schema on DATABASE_URL (Postgres)"
	@echo "  make db-down          Stop local Postgres (data is kept)"
	@echo "  make db-reset         Delete local Postgres data and start fresh"
	@echo "  make db-psql          Open psql in the local Postgres container"
	@echo "  make redis-up         Start the vector-search Redis (docker-compose.redis.yml, port 6380)"
	@echo "  make redis-down       Stop the vector-search Redis (data is kept)"
	@echo "  make redis-reset      Delete vector-search Redis data and start fresh"
	@echo "  make redis-cli        Open redis-cli in the vector-search Redis container"
	@echo "  make redis-rebuild    Reload every active job from the DB into the Redis index"
	@echo "  make redis-relay      Apply vector_outbox rows to Redis, printing each change"
	@echo "  make shell            Open a Python shell inside the venv"

.PHONY: run
run: $(PYTHON)
	$(PYTHON) app.py

.PHONY: prod
prod: $(PYTHON)
	$(VENV)/bin/gunicorn wsgi:app

.PHONY: migrate
migrate: $(PYTHON)
	@if [ -z "$(MSG)" ]; then echo "Usage: make migrate MSG=\"description\""; exit 1; fi
	$(FLASK) --app wsgi db migrate -m "$(MSG)"

.PHONY: upgrade
upgrade: $(PYTHON)
	$(FLASK) --app wsgi db upgrade

.PHONY: evals
evals: $(PYTHON)
	$(PYTHON) evals/run_all.py

.PHONY: mcp
mcp: $(PYTHON)
	$(PYTHON) mcp_server.py

.PHONY: worker
worker: $(PYTHON)
	$(CELERY) -A celery_worker.celery worker --loglevel=info -Q scout

.PHONY: beat
beat: $(PYTHON)
	$(CELERY) -A celery_worker.celery beat --loglevel=info

.PHONY: test
test: $(PYTHON)
	$(PYTEST)

.PHONY: test-pg
test-pg: $(PYTHON) db-up
	TEST_DATABASE_URL=$(TEST_DATABASE_URL) $(PYTEST)

# ── Local Postgres (docker-compose.db.yml) ────────────────────────────────────
DB_COMPOSE        := docker compose -f docker-compose.db.yml
# Separate DB: the pg_app fixture drops all tables on teardown.
TEST_DATABASE_URL := postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach_test

.PHONY: db-up
db-up:
	@mkdir -p .data/postgres
	$(DB_COMPOSE) up -d --wait

.PHONY: db-init
db-init: $(PYTHON) db-up
	$(PYTHON) scripts/init_postgres_schema.py

.PHONY: db-down
db-down:
	$(DB_COMPOSE) down

# Wipe all local Postgres data; the next db-up re-initialises from scratch.
# Deletes only pgdata/ INSIDE the bind mount, never the mounted dir itself:
# Docker Desktop for macOS keeps a stale handle to a deleted-then-recreated
# mount source, and the container then crash-loops on "mkdir: No such file".
.PHONY: db-reset
db-reset:
	$(DB_COMPOSE) down
	rm -rf .data/postgres/pgdata
	$(MAKE) db-up

.PHONY: db-psql
db-psql:
	$(DB_COMPOSE) exec postgres psql -U careercoach -d careercoach

# ── Vector-search Redis (docker-compose.redis.yml) ───────────────────────────
REDIS_COMPOSE     := docker compose -f docker-compose.redis.yml

.PHONY: redis-up
redis-up:
	@mkdir -p .data/redis
	$(REDIS_COMPOSE) up -d --wait

.PHONY: redis-down
redis-down:
	$(REDIS_COMPOSE) down

# Same rule as db-reset: delete the files INSIDE the bind mount, never the
# mounted dir itself.
.PHONY: redis-reset
redis-reset:
	$(REDIS_COMPOSE) down
	rm -rf .data/redis/appendonlydir .data/redis/dump.rdb
	$(MAKE) redis-up

.PHONY: redis-cli
redis-cli:
	$(REDIS_COMPOSE) exec redis redis-cli

.PHONY: redis-rebuild
redis-rebuild: $(PYTHON)
	VECTOR_BACKEND=redis $(FLASK) --app wsgi redis-vectors rebuild

.PHONY: redis-relay
redis-relay: $(PYTHON)
	VECTOR_BACKEND=redis $(FLASK) --app wsgi redis-vectors relay --watch

.PHONY: shell
shell: $(PYTHON)
	$(PYTHON)
