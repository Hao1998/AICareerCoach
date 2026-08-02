# AiCareerCoach — CLAUDE.md

AI-powered career coaching platform. Users upload resumes, get LLM-driven analysis, receive job matches from multiple external sources, and interact with a structured multi-agent AI via a chat interface. The Job Scout Agent runs on a schedule to surface new matches automatically.

---

## Running the App

| Task | Command |
|---|---|
| Dev server | `python app.py` (port 5001) |
| Production | `gunicorn wsgi:app` |
| DB migrate | `flask --app wsgi db migrate -m "description"` |
| DB upgrade | `flask --app wsgi db upgrade` |
| Evals | `python evals/run_all.py` |
| MCP server | `python mcp_server.py` |
| Celery worker | `celery -A celery_worker.celery worker --loglevel=info -Q scout` |
| Celery beat | `celery -A celery_worker.celery beat --loglevel=info` |
| Unit tests | `python -m pytest` (no API key needed; pgvector tests skip without `TEST_DATABASE_URL`) |
| Unit tests (Postgres path) | `docker compose up -d postgres` then `TEST_DATABASE_URL=postgresql+psycopg://careercoach:localdev@localhost:5432/careercoach python -m pytest` |

**Required env vars before starting:**

| Variable | Required | Purpose |
|---|---|---|
| `XAI_API_KEY` | Yes | xAI / Grok LLM |
| `SECRET_KEY` | Yes (prod) | Flask session signing |
| `ADZUNA_APP_ID` | Yes | Adzuna job fetcher |
| `ADZUNA_APP_KEY` | Yes | Adzuna job fetcher |
| `DATABASE_URL` | No | Defaults to SQLite at `instance/career_coach.db` |
| `REDIS_URL` | No | Defaults to `redis://localhost:6379/0` |
| `CHECKPOINT_DB_PATH` | No | LangGraph checkpoint DB; defaults to SQLite |
| `USE_CELERY` | No | `true` to enable Celery+Beat scheduling; defaults to APScheduler |
| `CELERY_BROKER_URL` | No | Broker URL for Celery; defaults to `REDIS_URL` |
| `CELERY_RESULT_BACKEND` | No | Result backend for Celery; defaults to `REDIS_URL` |
| `OTEL_EXPORTER_ENDPOINT` | No | OpenTelemetry tracing (opt-in) |
| `SENTRY_DSN` | No | Sentry error tracking (opt-in) |

---

## Architecture

Strict one-way dependency chain — **never import upward**.

```
controllers/      ← HTTP layer; thin, no business logic
    ↓
services/         ← business logic; no Flask request objects
    ↓
models.py         ← SQLAlchemy models only; no logic

agents/           ← specialist single-task LLM workers (JobAnalyst,
    ↓               ResumeTailoring, KeywordResearch, SecurityGuard …)
services/         ← agents are called FROM services/, not the other way
    ↑
chatbot/          ← conversational orchestration layer only
  agent.py        ← CareerCoachChatbot: LangGraph ReAct agent, system prompt, streaming
  tools.py        ← LangChain @tool wrappers around services/ calls
  planner.py      ← long-horizon Plan→Execute→Replan loop
  memory.py       ← session summarisation, conversation history

jobs/fetchers/    ← external API adapters; inherit BaseJobFetcher
    ↓
jobs/utils.py     ← embedding + FAISS index helpers
jobs/schedule_selector.py  ← pure function: due_user_ids(configs, now_utc)

tasks/            ← Celery task definitions (only used when USE_CELERY=true)
  celery_app.py   ← make_celery(), task registration, Beat schedule
  scout_tasks.py  ← run_scout_for_user_logic, dispatch_due_scouts_logic
celery_worker.py  ← Celery entry point (`celery -A celery_worker.celery worker`)
```

**`chatbot/` vs `agents/` — the key distinction:**

| Layer | Role | Calls into |
|---|---|---|
| `chatbot/` | Conversational orchestration. Runs the LangGraph `create_react_agent`, manages the user-facing chat loop, tools, memory, and planning. | `services/` only (never directly into `agents/`) |
| `agents/` | Specialist LLM workers. Each does exactly one task (score a job, tailor a resume, extract keywords). Structured Pydantic output. | `services/` (for helpers like embeddings) |

`chatbot/tools.py` surfaces agent capabilities to the chat interface, but always via the `services/` abstraction — never by importing an agent directly.

### Legacy files — do not edit

These root-level files are superseded by the structured packages above. They remain for reference only and will be removed. **Always use the canonical locations instead.**

| Legacy file | Canonical location |
|---|---|
| `job_scout_agent.py` | `jobs/scout_agent.py` |
| `job_fetcher.py` | `jobs/fetcher.py` + `jobs/fetchers/` |
| `job_utils.py` | `jobs/utils.py` |
| `agent_scheduler.py` | `jobs/scheduler.py` (scheduling logic) |
| `chatbot_legacy.py` | `chatbot/` package |

---

### App-level singletons (stored on `app.extensions`)

| Key | Type | Initialized in |
|---|---|---|
| `app.extensions['llm']` | `ChatXAI` | `services/llm_service.py` `get_llm()` |
| `app.extensions['scheduler']` | `APScheduler` | `jobs/scheduler.py` `init_scheduler()` — only when `USE_CELERY` is false |
| `app.extensions['celery']` | `Celery` | `tasks/celery_app.py` `make_celery()` — only when `USE_CELERY=true` |

Access from request context: `current_app.extensions['scheduler']`  
Background threads still need `with app.app_context():` — the extensions dict is not thread-local, but DB sessions are.  
Celery worker tasks also need `with app.app_context():` — they run in a separate process with no active request.

### Key files at a glance

| What you want to change | File |
|---|---|
| App wiring, blueprints, extensions | `factory.py` |
| Config / env vars | `config.py` |
| DB models | `models.py` |
| Job matching (FAISS + LLM) | `services/job_service.py` |
| Resume parsing + Q&A | `services/resume_service.py` |
| LLM chains (non-agent) | `services/llm_service.py` |
| Conversational agent (chat loop) | `chatbot/agent.py` |
| Chat tools (what the AI can invoke) | `chatbot/tools.py` |
| Long-horizon career planner | `chatbot/planner.py` |
| Chat memory / summarisation | `chatbot/memory.py` |
| Specialist agent registry + cache | `agents/coordinator.py` |
| Job fetcher registry | `jobs/fetchers/registry.py` |
| APScheduler scheduling (default mode) | `jobs/scheduler.py` |
| Celery task definitions + Beat schedule | `tasks/celery_app.py` |
| Scout task logic (testable, broker-free) | `tasks/scout_tasks.py` |
| Per-user timezone schedule selection | `jobs/schedule_selector.py` |
| Semantic cache bypass rules | `services/semantic_cache.py` `DEFAULT_BYPASS_PREFIXES` |
| Request validation | `schemas/request_schemas.py` |
| Structured LLM output schemas | `schemas/output_schemas.py` |
| Prompt injection guard | `services/input_guard.py` |
| Job dense retrieval (pgvector / FAISS) | `jobs/vector_store.py` |
| pgvector dialect helpers | `services/pgvector_support.py` |

---

## Adding New Features

### A. New Route / Blueprint

1. Create `controllers/foo_controller.py`:
   ```python
   from flask import Blueprint
   foo_bp = Blueprint('foo', __name__)

   @foo_bp.route('/foo')
   def index(): ...
   ```
2. Register in `factory.py` `create_app()`:
   ```python
   from controllers.foo_controller import foo_bp
   app.register_blueprint(foo_bp)
   ```
3. Always reference routes as `url_for('foo.index')` — blueprint prefix required.
4. Validate incoming JSON at the controller boundary using `schemas/request_schemas.py` before passing to services.

### B. New Specialist Agent

1. Create `agents/foo_agent.py`, subclass `BaseAgent`:
   ```python
   from agents.base_agent import BaseAgent

   class FooAgent(BaseAgent):
       MODEL = "grok-3-mini"       # use grok-3 for complex reasoning
       SYSTEM_PROMPT = "You are..."

       def run(self, input_text: str) -> MyOutputSchema:
           return self._invoke_structured(MyOutputSchema, input_text)
   ```
2. Declare a Pydantic output schema in `schemas/output_schemas.py` and pass it to `_invoke_structured()`.
3. Cache the agent instance on `app.extensions` via `agents/coordinator.py` so it's created once per process.
4. Register any new chat-facing capability in `chatbot/tools.py`.

**Model selection guidance:**
- `grok-3-mini` — classification, keyword extraction, simple scoring
- `grok-3` — resume analysis, tailoring, roadmaps, complex multi-step reasoning

### C. New Job Source Fetcher

1. Create `jobs/fetchers/bar.py`, subclass `BaseJobFetcher`:
   ```python
   from jobs.fetchers.base import BaseJobFetcher

   class BarFetcher(BaseJobFetcher):
       source_name = "bar"

       def fetch_jobs(self, keywords=None, location=None, max_jobs=50, **kwargs) -> list[dict]:
           ...  # call external API, return raw response list

       def parse_job(self, raw: dict) -> dict:
           return {
               'title': raw['title'],
               'company': raw['company'],
               'location': raw.get('location'),
               'job_type': raw.get('type'),
               'description': raw['description'],
               'requirements': raw.get('requirements'),
               'salary_range': raw.get('salary'),
               'source': self.source_name,
               'source_id': str(raw['id']),
               'source_url': raw.get('url'),
           }
   ```
2. Register in `jobs/fetchers/registry.py`:
   ```python
   from jobs.fetchers.bar import BarFetcher
   FETCHER_REGISTRY['bar'] = BarFetcher
   USER_VISIBLE_SOURCES.append('bar')  # omit if not user-selectable
   ```
3. The base class handles dedup, embedding, FAISS rebuild, and DB commit automatically.

### D. New DB Column or Table

1. Edit `models.py` — add columns or new `db.Model` subclasses.
2. Run:
   ```
   flask --app wsgi db migrate -m "add foo to bar"
   flask --app wsgi db upgrade
   ```
3. Never hand-edit the generated migration file unless fixing a known Alembic limitation.
4. Never create one-off migration scripts at the project root — use Flask-Migrate exclusively.

### E. New Structured LLM Output

1. Add a Pydantic class in `schemas/output_schemas.py`.
2. Use `agent._invoke_structured(MySchema, input_text)` or `llm.with_structured_output(MySchema)` — never parse free-text LLM output with string manipulation.

---

## Security Rules

These apply to **all contributors and AI agents** working on this codebase.

### Prompt Injection

Every piece of untrusted external data — resume text, job descriptions, user chat input — is a potential injection vector. Always handle it as follows:

- **In agents:** pass untrusted content in the `human` turn wrapped in `<untrusted_data>` tags, never interpolated into `SYSTEM_PROMPT`.
  ```python
  input_text = f"<untrusted_data>{resume_text}</untrusted_data>\nTask: analyse this resume."
  self._invoke_structured(ResumeAnalysis, input_text)
  ```
- **In services / chains:** run user input through `services/input_guard.py` before building any LLM prompt.
- **New prompts added anywhere in the codebase must follow the same pattern.** When adding a new prompt, ask: "could a user craft input that changes what this prompt does?" If yes, add `<untrusted_data>` wrapping and pipe through `input_guard.py`.
- `agents/security_guard_agent.py` classifies suspicious inputs — invoke it for any user-provided free-text that will reach an agent.

### Other Security Rules

- Never interpolate `request` data directly into SQL — always use SQLAlchemy ORM or parameterised queries.
- Never store secrets in code — all credentials go through `config.py` from environment variables.
- Rate limiting is applied globally in `factory.py` via `flask-limiter`. Do not bypass it; add per-route overrides using `@limiter.limit()` decorator only when justified.
- File uploads are restricted to `config.UPLOAD_FOLDER`. Never construct file paths from user input without sanitisation.
- All `/api/` routes that modify state require `@login_required`.

---

## Eval / Quality Gates

Run the relevant eval suite after touching any of the listed components. All evals live in `evals/` and share fixtures from `evals/datasets.py`.

| Eval file | Run when you touch |
|---|---|
| `evals/chat_eval.py` | `chatbot/`, `controllers/chat_controller.py`, `chatbot/tools.py` |
| `evals/job_match_eval.py` | `services/job_service.py`, `agents/job_analyst_agent.py`, FAISS logic |
| `evals/tailoring_eval.py` | `agents/resume_tailoring_agent.py`, resume tailoring prompts |
| `evals/memory_eval.py` | `chatbot/memory.py`, `UserMemoryChunk` model, memory summarisation |

```bash
# Run all evals
python evals/run_all.py

# Run a single suite
python evals/chat_eval.py
```

Evals must pass before merging any change that touches the logic above.

---

## Critical Invariants — Never Break These

- **No reverse imports.** `services/` and `agents/` must never import from `controllers/`. `models.py` imports nothing from the app.
- **App context in background threads.** Any DB operation outside a request must be wrapped: `with app.app_context(): ...`
- **`safe_commit()` for SQLite.** Use `services/db_lock.safe_commit()` inside job fetchers and background jobs (prevents WAL lock contention). Regular `db.session.commit()` is fine everywhere else.
- **Structured output only.** LLM responses that feed application logic must go through Pydantic schemas via `with_structured_output()`. Free-text parsing is banned.
- **Semantic cache bypass list.** Job-specific, resume-specific, and conversational chat prompts must stay in `services/semantic_cache.py`'s `DEFAULT_BYPASS_PREFIXES` list (single source of truth, imported by `factory.py`). Chat prompts in particular must bypass because identical system prompts cause different user messages to collide in the cache, producing wrong answers and silent "(no response)" in the streaming UI.
- **`url_for` with blueprint prefix.** Always `url_for('blueprint_name.function_name')` — bare function names will raise `BuildError` at runtime.
- **Vector columns are not in `models.py`.** `embedding_vec` columns exist only
  on PostgreSQL, created by raw Alembic DDL and queried with raw SQL. Declaring
  a pgvector column type in `models.py` breaks `db.create_all()` on SQLite and
  therefore the entire test suite. The JSON `embedding` columns remain the
  source of truth; `embedding_vec` is derived.
