# AI Career Coach

> A production-grade agentic AI system that autonomously scouts jobs, matches candidates using semantic search, and coaches users through a multi-tool conversational agent — not a wrapper around an API call.

![Python](https://img.shields.io/badge/Python-3.12+-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-purple) ![LangChain](https://img.shields.io/badge/LangChain-RAG_Pipeline-green) ![MCP](https://img.shields.io/badge/MCP-Tool_Server-orange) ![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

<p align="center">
  <img src="docs/job-matching.png" width="700" alt="AI-powered job matching with skill gap analysis and match scoring" />
</p>
<p align="center"><em>Semantic job matching — AI scores each role, identifies skill gaps, and gives personalized recommendations</em></p>

<p align="center">
  <img src="docs/resume-tailoring.png" width="700" alt="AI-tailored resume rewriting per job posting" />
</p>
<p align="center"><em>Resume tailoring — AI rewrites your CV to match specific job requirements, highlighting relevant experience</em></p>

<p align="center">
  <img src="docs/prep-roadmap.png" width="700" alt="AI-generated interview preparation roadmap with phased learning plan" />
</p>
<p align="center"><em>Prep roadmap — generates a phased study plan with curated resources tailored to the specific role and your skill gaps</em></p>

<p align="center">
  <img src="docs/chat-ui.png" width="700" alt="Conversational career coach with tool-calling agent" />
</p>
<p align="center"><em>Agentic chatbot — uses tools to search jobs, analyze resumes, and surface matches mid-conversation</em></p>

<p align="center">
  <img src="docs/agent-dashboard.png" width="700" alt="Autonomous Job Scout Agent dashboard with scheduling and run history" />
</p>
<p align="center"><em>Job Scout Agent — runs autonomously on a schedule, configurable threshold, full run history</em></p>

## What This Demonstrates

This is a solo-built, end-to-end AI application — not a tutorial follow-along. It showcases:

| Skill | Implementation |
|-------|---------------|
| **Multi-agent orchestration** | Coordinator routes tasks to 5 specialist agents with model-specific cost/quality tradeoffs (Grok-3 vs Grok-3-mini) |
| **LangGraph state machines** | Job Scout runs as a graph with conditional edges, parallel execution, and checkpoint persistence |
| **RAG with evaluation** | FAISS vector store + contextual retrieval, measured by RAGAS eval suite |
| **Production engineering** | Rate limiting, semantic caching, DB indexing, prompt injection guards, streaming SSE, OpenTelemetry, Sentry |
| **Tool-use agents** | Chatbot with tool-calling (resume lookup, job search, skill gaps) + session-aware memory + task planner |
| **MCP integration** | Exposes 6 capabilities as MCP tools with per-user auth scoping and async audit logging |
| **Multi-source job fetching** | Pluggable fetcher registry aggregates 8 job board APIs in parallel |

---

## Highlights

- **Multi-agent architecture** — Coordinator delegates to 5 specialist agents (security guard, keyword research, job search planner, job analyst, resume tailoring) on different models for cost/quality optimization
- **LangGraph state machine** — Job Scout Agent uses a graph-based workflow with conditional routing, parallel node execution, and checkpoint-based persistence
- **RAG pipeline** — Resume Q&A powered by FAISS vector store with HuggingFace `all-mpnet-base-v2` embeddings and contextual retrieval
- **Semantic job matching** — Cosine similarity on embeddings + LLM-scored analysis produces ranked matches with skill gap identification
- **Multi-source job fetching** — Pluggable fetcher registry (Adzuna, Remotive, Jobicy, RemoteOK, Himalayas, The Muse, Arbeitnow, Greenhouse) runs in parallel via a thread pool
- **MCP server** — Exposes 6 tools via Model Context Protocol with per-user API key scoping (zero cross-user access by design) and fire-and-forget audit logging
- **Prompt injection guard** — Regex + heuristic input validation layer protects both chat and PDF-extracted resume content
- **Streaming chat** — Server-Sent Events and SocketIO for real-time AI responses with session-aware conversation memory
- **Task planner** — Chatbot includes a planner module for breaking complex requests into structured steps
- **Observability** — LangSmith tracing, OpenTelemetry (OTLP), Sentry error monitoring, structured logging with request IDs
- **Eval suite** — Automated evaluation harness using RAGAS for chat, job matching, memory, and resume tailoring quality
- **Production-ready** — Docker + Gunicorn + gevent workers, Flask-Limiter (Redis-backed), database indexing, semantic caching, Flask-Migrate for schema versioning

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Client (Tailwind CSS + Framer Motion + SocketIO)               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Flask Blueprints (Controllers)                                  │
│  auth · resume · jobs · agent · chat                             │
└───────────────┬──────────────────────────────┬──────────────────┘
                │                              │
┌───────────────▼───────────┐  ┌───────────────▼──────────────────┐
│  Services Layer            │  │  Agent System                     │
│  llm · resume · job        │  │  coordinator → 5 specialist agents│
│  input_guard · streaming   │  │  (LangGraph state machine)        │
│  semantic_cache · redis     │  │  chatbot: agent + memory + planner│
│  telemetry · logging        │  │                                   │
└───────────────┬───────────┘  └───────────────┬──────────────────┘
                │                              │
┌───────────────▼───────────┐  ┌───────────────▼──────────────────┐
│  Data Layer                │  │  Job Fetcher Registry             │
│  SQLAlchemy ORM            │  │  Adzuna · Remotive · Jobicy       │
│  FAISS vector indices      │  │  RemoteOK · Himalayas · TheMuse   │
│  LangGraph checkpoints     │  │  Arbeitnow · Greenhouse           │
└───────────────────────────┘  └──────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask, Gunicorn + gevent, Flask-SocketIO |
| AI/LLM | LangChain, LangGraph, xAI Grok-3 / Grok-3-mini |
| Embeddings | HuggingFace `all-mpnet-base-v2`, FAISS |
| Database | SQLAlchemy, Flask-Migrate (Alembic), SQLite / PostgreSQL |
| Caching | Redis (rate limiting + session), semantic LLM cache |
| Security | Flask-Login, Flask-Limiter, input guard (prompt injection detection) |
| Observability | LangSmith tracing, OpenTelemetry (OTLP), Sentry, structured logging |
| Evaluation | RAGAS, custom eval harness |
| External APIs | Adzuna, Remotive, Jobicy, RemoteOK, Himalayas, The Muse, Arbeitnow, Greenhouse |
| MCP | FastMCP (6 tools + 2 resources, per-user auth, async audit log) |
| Frontend | Jinja2, Tailwind CSS, Framer Motion, SSE + SocketIO streaming |
| Deployment | Docker (multi-stage), docker-compose, Nginx |

## Key Features

### Multi-Agent Job Scout
An autonomous agent that runs on a configurable schedule (APScheduler), searching for jobs matching your resume profile. Built as a LangGraph state machine with:
- Security guard agent (Grok-3-mini) — validates all inputs before processing
- Keyword research agent (Grok-3-mini) — extracts search terms from resume
- Job search planner agent — structures the search strategy
- Job analyst agent (Grok-3) — scores and ranks results
- Resume tailoring agent (Grok-3) — suggests resume modifications per job

### Multi-Source Job Fetching
A pluggable fetcher registry aggregates listings from 8 job board APIs in parallel:
- **Adzuna** — general job board with location filtering and salary data
- **Remotive, Jobicy, RemoteOK** — remote-first job boards
- **Himalayas, The Muse** — tech-focused and culture-forward listings
- **Arbeitnow** — EU / international roles
- **Greenhouse** — direct company ATS listings

### Semantic Resume Matching
1. PDF text extraction (PyPDF2 + pdfplumber)
2. Chunk and embed with HuggingFace `all-mpnet-base-v2` sentence transformers
3. Store in FAISS index for fast cosine-similarity search
4. LLM-scored match analysis with skill gap identification
5. Personalized recommendations for each job match

### Conversational Career Coach
- Tool-calling agent with access to resume data, job matches, and skill gaps
- Task planner for breaking complex multi-step requests into structured plans
- Session-aware memory with automatic summarization at session boundaries
- Intent detection for routing (job search, resume feedback, general advice)
- Streaming responses via Server-Sent Events and SocketIO

### MCP Server Integration
Exposes career coaching capabilities as MCP tools for use in Claude Code or any MCP client:
- `find_matching_jobs` — semantic job search against your resume
- `analyze_resume` — full resume analysis
- `get_skill_gaps` — targeted gap analysis for a specific job
- `ask_resume_question` — natural language Q&A over your resume
- `get_recent_job_matches` — retrieve your latest match history
- `trigger_job_scout` — on-demand agent run

Two MCP resources are also exposed: `resume://me/active` and `resume://me/analysis`.

### Security & Reliability
- Input validation guard against prompt injection (chat + PDF vectors)
- Per-user API key scoping on MCP server (cross-user access architecturally impossible)
- Async audit logging on every MCP tool call (fire-and-forget, zero latency impact)
- Rate limiting on all endpoints (Redis-backed, user-aware key)
- Database indexes for query performance
- Semantic caching to reduce redundant LLM calls (cosine threshold 0.90, bypass rules for job-specific prompts)

## Getting Started

### Prerequisites
- Python 3.12+
- xAI API key ([platform.x.ai](https://platform.x.ai))
- Adzuna API credentials ([developer.adzuna.com](https://developer.adzuna.com)) — optional, other free sources work without a key
- Redis — optional (falls back to in-memory for dev)

### Installation

```bash
git clone https://github.com/Hao1998/AiCareerCoach.git
cd AiCareerCoach

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
# Development
python app.py  # → http://localhost:5001

# Production
gunicorn wsgi:app -c gunicorn.conf.py

# Docker
docker compose up --build

# MCP Server
python mcp_server.py  # → http://localhost:8001
```

### Database Migrations

```bash
flask --app wsgi db migrate -m "describe change"
flask --app wsgi db upgrade

# SQLite → PostgreSQL migration
python scripts/migrate_sqlite_to_postgres.py
```

## Project Structure

```
├── app.py                  # Entry point
├── wsgi.py                 # Gunicorn / migration entrypoint
├── factory.py              # App factory (create_app)
├── config.py               # Environment configs
├── models.py               # SQLAlchemy models (User, Resume, JobPosting,
│                           #   JobMatch, AgentConfig, ChatMessage,
│                           #   UserMemoryChunk, TaskPlan, PlanStep, AgentRunHistory)
├── controllers/            # Flask blueprints (route handlers)
│   ├── auth_controller.py
│   ├── resume_controller.py
│   ├── job_controller.py
│   ├── agent_controller.py
│   └── chat_controller.py
├── services/               # Business logic layer
│   ├── llm_service.py
│   ├── resume_service.py
│   ├── job_service.py
│   ├── input_guard.py
│   ├── streaming.py
│   ├── semantic_cache.py
│   ├── redis_client.py
│   ├── db_lock.py
│   ├── logging_config.py
│   └── telemetry.py
├── agents/                 # Multi-agent system
│   ├── base_agent.py       # Shared agent base class
│   ├── coordinator.py      # Model routing & agent lifecycle
│   ├── security_guard_agent.py
│   ├── keyword_research_agent.py
│   ├── job_search_planner_agent.py
│   ├── job_analyst_agent.py
│   └── resume_tailoring_agent.py
├── chatbot/                # Conversational agent
│   ├── agent.py            # AgentExecutor + intent detection
│   ├── memory.py           # Session-aware conversation memory
│   ├── planner.py          # Task planning for complex requests
│   └── tools.py            # Tool definitions
├── jobs/                   # Job fetching & scout orchestration
│   ├── fetcher.py          # AdzunaJobFetcher (primary)
│   ├── fetchers/           # Pluggable multi-source fetcher registry
│   │   ├── registry.py     # Parallel fetch coordinator
│   │   ├── adzuna.py
│   │   ├── remotive.py
│   │   ├── jobicy.py
│   │   ├── remoteok.py
│   │   ├── himalayas.py
│   │   ├── themuse.py
│   │   ├── arbeitnow.py
│   │   └── greenhouse.py
│   ├── scheduler.py        # APScheduler integration
│   ├── scout_agent.py      # JobScoutAgent orchestrator
│   ├── scout_graph.py      # LangGraph state machine definition
│   └── utils.py            # Embedding helpers, FAISS index management
├── schemas/                # Request/response validation
│   ├── request_schemas.py
│   ├── output_schemas.py
│   └── validate.py
├── mcp_server.py           # MCP tool server (FastMCP)
├── evals/                  # Evaluation suite (RAGAS)
│   ├── chat_eval.py
│   ├── job_match_eval.py
│   ├── memory_eval.py
│   ├── tailoring_eval.py
│   └── run_all.py
├── scripts/
│   └── migrate_sqlite_to_postgres.py
├── migrations/             # Alembic version scripts
├── templates/              # Jinja2 frontend
├── Dockerfile              # Multi-stage Docker build (python:3.12-slim)
├── docker-compose.yml
├── nginx.conf
└── gunicorn.conf.py
```

## Evaluation

The project includes an automated eval suite for measuring AI quality:

```bash
python evals/run_all.py
```

Evaluates: chat response quality, job match accuracy, memory retrieval, and resume tailoring fidelity using RAGAS metrics.

## License

MIT
