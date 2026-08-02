# AWS Deployment Design — AiCareerCoach

**Date:** 2026-07-26
**Status:** Approved design, pending implementation plan

---

## Goal

Deploy AiCareerCoach to AWS as a production-grade service: multiple app
instances behind a load balancer, managed Postgres and Redis, durable file
storage, secrets outside the codebase, and automatic deploys on merge to `main`.

Target scale is "growing production" — real users, meaningful traffic, and a
setup that scales out without a rewrite.

---

## Architecture

```
                    Route 53 (DNS)
                          │
                    ACM cert (HTTPS)
                          │
              Application Load Balancer
              (sticky sessions, 400s idle)
                          │
        ┌─────────────────┴─────────────────┐
        │        ECS Fargate cluster        │
        │                                   │
        │  web service      (N tasks)       │
        │  worker service   (M tasks)       │
        │  beat service     (exactly 1)     │
        └─────────────────┬─────────────────┘
                          │
      ┌───────────┬───────┴───────┬────────────┐
      │           │               │            │
  RDS Postgres  ElastiCache      S3       Secrets Manager
  (+ pgvector)    Redis      (uploads)     (API keys)
      │           │               │            │
      └───────────┴───────┬───────┴────────────┘
                          │
                  CloudWatch Logs
```

All three ECS services run **the same Docker image** with different startup
commands. One build, one image tag, three deployments — this keeps the web tier
and the background workers permanently in sync.

| Service | Command | Tasks | Notes |
|---|---|---|---|
| web | `gunicorn -c gunicorn.conf.py wsgi:app` | 2+ | Behind the ALB |
| worker | `celery -A celery_worker.celery worker -Q scout` | 1+ | Scales with job volume |
| beat | `celery -A celery_worker.celery beat` | **exactly 1** | Two would double-fire scheduled scouts |

`USE_CELERY=true` in all three. APScheduler is a development-only path; in AWS,
scheduling belongs to Beat so the web tier can scale freely.

---

## AWS Components

### ECS on Fargate
Runs containers without managing EC2 instances. Chosen over Elastic Beanstalk
because the app is three cooperating processes from one codebase, which
Beanstalk's single-application model fits poorly. Chosen over plain EC2 because
"growing production" makes manual patching and restart handling a recurring
cost.

Task sizing must account for the sentence-transformers model baked into the
image (`Dockerfile:25`) plus the FAISS index held in process. Start at
**2 vCPU / 4 GB** for web and worker; measure and adjust. This is a starting
point to be validated under load, not a derived figure.

### ECR
Private registry for the Docker image. Enable image scanning on push.

### RDS for PostgreSQL
Replaces the SQLite default in `config.py`. SQLite cannot back multiple
containers — the file lives on ephemeral container storage and is lost on every
deploy.

**The `pgvector` extension must be enabled.** See "Vector storage" below; this
is not optional given the current code.

Multi-AZ for automatic failover. Automated backups with a 7-day retention
window.

### ElastiCache for Redis
Serves three distinct roles, all already present in the code:
1. Celery broker and result backend (`config.py` `CELERY_BROKER_URL`)
2. Flask-Limiter storage (rate limiting shared across instances)
3. **Socket.IO `message_queue`** (`factory.py:112`) — without this, a WebSocket
   event emitted by one container never reaches a client connected to another.

### S3
Resume uploads move off local disk. `config.UPLOAD_FOLDER` currently points at a
project-relative directory, which on Fargate is wiped on every restart and is
not shared between tasks. Uploaded resumes would silently disappear.

Bucket configuration: versioning on, public access blocked, server-side
encryption enabled, access via a task IAM role (never access keys in env vars).

### Secrets Manager
`XAI_API_KEY`, `SECRET_KEY`, `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`, the RDS
password, and `SENTRY_DSN` are injected into containers as environment
variables by the ECS task definition at startup. Nothing sensitive is baked
into the image or committed.

Note that `config.py` currently defaults `SECRET_KEY` to
`'dev-secret-key-change-in-production'`. Production must fail loudly rather than
start with this default — a weak signing key means forgeable sessions.

### CloudWatch
Container stdout/stderr ship to CloudWatch Logs via the `awslogs` driver.
Alarms on: ALB 5xx rate, ECS service unhealthy task count, RDS CPU and free
storage, Celery queue depth.

### Route 53 + ACM
DNS and a free managed TLS certificate terminated at the ALB. No domain is
registered yet — see "Decisions" below for what that means for HTTPS and when
it must be resolved.

---

## Vector storage — the substantive change

The app maintains two vector stores, and both need work before it can run on
more than one instance.

### 1. Job matching (FAISS)

`jobs/utils.py` builds a FAISS index and saves it to a local directory. The
module docstring already identifies this as a scaling blocker and names the fix:

> "the FAISS index here is in-process and saved to local disk, so it cannot be
> shared across multiple app instances… move embeddings into Postgres
> `pgvector`"

With N Fargate tasks, each container holds its own index. A job fetched by the
worker is invisible to web containers until they independently rebuild, and
every deploy discards all indexes. Job match results become inconsistent
between requests depending on which container serves them.

**Decision:** migrate job embeddings to `pgvector` on RDS as part of this
deployment. The code already anticipates this, and the alternative — an EFS
volume shared between tasks — adds a filesystem dependency and concurrent-write
hazards to solve a problem Postgres solves natively.

### 2. Conversation memory (sqlite-vec)

`chatbot/memory.py` uses the `sqlite-vec` extension for ANN search over user
memory chunks, with an explicit fallback:

> "Uses sqlite-vec ANN when available; falls back to O(n) cosine on Postgres"

Moving to RDS therefore does not break memory search — results stay correct —
but it silently degrades every lookup to a linear scan. The fallback filters by
`user_id` before scanning (`chatbot/memory.py:384`), so the cost grows with a
*single user's* memory count rather than the size of the whole table. A user
with tens of memories will not notice; a long-tenured heavy user with thousands
will, as gradually increasing chat latency with no error to trace it to. On
SQLite this path was never the hot one; on Postgres it becomes the only one.

`pgvector` also fixes a correctness weakness in the current SQLite fast path:
`_search_memories_vec` retrieves the `top_k * 3` nearest chunks globally and
only then filters to the requesting user (`chatbot/memory.py:346`), because
vec0 v0.1.x cannot filter inside the index. As the user base grows, a user's
own relevant memories can be crowded out of the candidate set by other users'
chunks and never surface. pgvector supports filtered index scans, so the
user predicate applies before the top-k cut.

**Decision:** port memory chunk search to `pgvector` in the same pass. The
`vec_user_memories` virtual table and its migration
(`e9e97b70ce4c`) remain SQLite-only and stay in place for local development —
`_register_sqlite_vec` in `factory.py` is already conditional on the dialect.

This is the largest code change in the deployment and should be sequenced
before the ECS cutover, with `evals/memory_eval.py` and
`evals/job_match_eval.py` as the acceptance gates (per CLAUDE.md's eval
requirements).

---

## WebSocket handling

`controllers/ws_chat_controller.py` streams chat responses over Socket.IO. Three
requirements follow:

1. **ALB sticky sessions must be enabled.** Socket.IO's handshake and subsequent
   frames must land on the same container.
2. **ALB idle timeout must exceed the gunicorn timeout.** `gunicorn.conf.py` sets
   `timeout = 300` for long LLM completions; the ALB default of 60s would sever
   streaming responses mid-answer. Set the ALB to ~400s.
3. **Redis `message_queue` is mandatory in production.** Already wired at
   `factory.py:112` and correctly skipped in development.

The existing gevent-websocket worker class in `gunicorn.conf.py` is already the
right choice and needs no change.

---

## Code changes required

| Change | File(s) | Why |
|---|---|---|
| Job embeddings → pgvector | `jobs/utils.py`, `services/job_service.py` | FAISS on local disk cannot be shared across tasks |
| Memory search → pgvector | `chatbot/memory.py` | Avoids the O(n) per-user fallback scan on Postgres; also allows filtering by user inside the index |
| Uploads → S3 | `config.py`, `services/resume_service.py` | Container disk is ephemeral and unshared |
| `SECRET_KEY` fails closed in prod | `config.py` | Dev default must never reach production |
| `CHECKPOINT_DB_PATH` → Postgres DSN | env config only | LangGraph checkpoints must survive restarts; already supported |
| Add deploy workflow | `.github/workflows/deploy.yml` | CI/CD |
| Review `safe_commit()` | `services/db_lock.py` callers | The SQLite WAL-lock workaround is unnecessary on Postgres, but removal must be verified, not assumed |

The existing `Dockerfile` needs no structural change — it is already multi-stage,
runs as a non-root user, and defines a healthcheck. `docker-compose.yml` remains
the local development entry point and stays as-is.

---

## CI/CD

GitHub Actions on push to `main`:

1. Run `python -m pytest` — fail the build on any failure
2. Run the eval suites touched by the diff (per the CLAUDE.md eval gate table)
3. Build the Docker image, tag it with the commit SHA
4. Authenticate to AWS via OIDC — a federated role, **no long-lived AWS keys in
   GitHub secrets**
5. Push the image to ECR
6. Run `flask --app wsgi db upgrade` as a one-off ECS task and wait for success
7. Update the three ECS services to the new image tag
8. Wait for the deployment to stabilize; roll back on failure

Migrations run before the service update so new code never meets an old schema.
Deployments use ECS rolling updates with the ALB draining connections, so there
is no downtime for the web tier.

---

## Environments

Two: `staging` and `production`, in separate AWS accounts if practical,
otherwise separate VPCs. Staging gets single-AZ RDS and one task per service to
keep costs down. The pipeline deploys to staging on every merge and to
production on a tagged release.

---

## Network layout

A VPC with public and private subnets across two availability zones.

- ALB in the public subnets — the only internet-facing component
- ECS tasks in private subnets, reached only by the ALB
- RDS and ElastiCache in private subnets, reachable only from the ECS security
  group
- A NAT gateway provides outbound access for the xAI and Adzuna API calls

Security groups are chained by reference (ALB → ECS → RDS/Redis) rather than by
CIDR block, so the rules stay correct if subnets change.

---

## Cost

A ballpark for the smallest viable production footprint — two web tasks, one
worker, one beat, `db.t4g.small` Multi-AZ RDS, `cache.t4g.micro` Redis, one ALB,
one NAT gateway:

**roughly $150–250/month** before traffic-driven scaling.

This is an order-of-magnitude estimate for planning, not a quote. Multi-AZ RDS
and the NAT gateway are the two largest line items and both have cheaper
single-AZ / VPC-endpoint variants if the budget is tighter than the availability
requirement.

---

## Rollout sequence

1. ~~**pgvector migration**~~ — **done.** Job embeddings and memory search now
   run on pgvector when the engine is PostgreSQL, verified by
   `evals/job_match_eval.py`, `evals/memory_eval.py`, and a Postgres CI job.
2. **S3 uploads** — no back-fill needed; this is a fresh start.
3. **Terraform backend bootstrap** — S3 state bucket and DynamoDB lock table.
4. **Infrastructure** — VPC, RDS, ElastiCache, S3, Secrets Manager, ECR.
5. **Staging ECS deployment** — validate the full stack end to end, ALB locked
   to known source IPs while it is HTTP-only.
6. **CI/CD pipeline** — automate what step 5 did by hand.
7. **Domain + HTTPS** — register in Route 53, issue the ACM certificate, and
   move the ALB to an HTTPS listener with an HTTP redirect.
8. **Production** — `flask db upgrade` against an empty RDS instance, then DNS.

Steps 1 and 2 are ordinary application work and carry the most risk; the AWS
steps are largely mechanical once they land. Step 7 gates any public launch.

---

## Decisions

**Fresh start — no data migration.** There is no production SQLite database to
carry over. The production cutover is `flask db upgrade` against an empty RDS
instance, and step 2's "migration path for existing files in `uploads/`" is
dropped. This removes the riskiest part of the rollout.

**No domain yet.** The ALB's own DNS name
(`<name>-<id>.<region>.elb.amazonaws.com`) is the initial endpoint. This has a
consequence worth stating up front: **an ALB hostname cannot have an ACM
certificate attached, so there is no HTTPS until a domain exists.** Running the
chat interface — including login and resume uploads — over plain HTTP is not
acceptable beyond a private test.

Registering a domain is therefore a prerequisite for anything user-facing, not
an optional polish step. It can be registered through Route 53 in minutes and
costs roughly $12–15/year. The plan will provision Route 53 and ACM as normal
and leave the hostname as the single value to fill in; until then, the staging
ALB should be restricted to known source IPs via its security group.

**Terraform.** Infrastructure lives in `infra/` as Terraform, with a `modules/`
directory for the shared building blocks and `envs/staging` + `envs/production`
composing them with different sizing. State goes in an S3 backend with DynamoDB
locking — created once by hand, since the backend cannot bootstrap itself.
