# Chatbot Capability Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Constrain what the chatbot's LLM layers can invoke on their own — bounding cost, requiring user consent for destructive actions, and denying the autonomous planner access to gated capabilities — while consolidating 12 tools to 9 for selection accuracy.

**Architecture:** Three new service modules (`rate_limit`, `pending_actions`, `gated_actions`) sit under `chatbot/tools.py`. Gated capabilities keep their `@tool` wrapper so the model can still *propose* them, but the wrapper only mints a single-use Redis nonce; the real implementation lives in `gated_actions.py` and runs only from an authenticated `POST /api/chat/confirm`. `build_tools` gains a required `surface` keyword so the planner physically cannot receive gated tools.

**Tech Stack:** Flask, Flask-SocketIO, SQLAlchemy, LangChain/LangGraph, Redis (`redis==7.4.0`), pytest.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-13-chatbot-capability-hardening-design.md`. Read it before starting.
- **No reverse imports.** `services/` and `chatbot/` must never import from `controllers/`. (CLAUDE.md)
- **`safe_commit()`** from `services/db_lock` for any commit in a background thread or job fetcher.
- **Structured output only.** No free-text parsing of LLM responses.
- **`url_for` always takes the blueprint prefix**: `url_for('chat.confirm_action')`.
- **Untrusted external data** (resume text, job descriptions, memory) must be wrapped in `<untrusted_data>` tags in any new prompt.
- **No new Python dependencies.** Redis fakes in tests are hand-rolled — `fakeredis` is not in `requirements.txt` and must not be added.
- **Never `from services.redis_client import get_redis`.** That binds the name at import time and defeats the `fake_redis` fixture's monkeypatch, so tests silently hit the developer's real Redis. Always `from services import redis_client` and call `redis_client.get_redis()` at use time. (A real Redis is running on localhost in this environment, so the failure is silent rather than loud.)
- **Python interpreter: `.venv/bin/python`.** Bare `python` is NOT on PATH in this environment — every command in this plan that says `python` must be run as `.venv/bin/python`. Run from the repo root.
- **Test command:** `.venv/bin/python -m pytest` (SQLite; pgvector tests skip without `TEST_DATABASE_URL`). Baseline at branch point: **53 passed, 19 skipped**.
- **Eval command:** `.venv/bin/python evals/chat_eval.py` — required after Tasks 9–12 per CLAUDE.md. It calls `load_dotenv()` itself and `.env` already holds `XAI_API_KEY`, so no key needs to be supplied.
- **Rate-limit budgets, exact values:** `trigger_job_scout_agent` = 3/hour/user; `create_career_plan` = 5/day/user; WebSocket chat = 20/min and 200/day per user.
- **Nonce TTL, exact value:** 300 seconds.

---

## Sequencing Note

Tasks 1–8 use the **current** tool names. Tasks 10–11 rename them. This ordering is deliberate: it keeps the security work reviewable against the code as it exists today, and confines rename churn to two late tasks with a full call-site checklist. Do not rename early.

---

### Task 1: Test infrastructure — fake Redis + eval baseline

**Files:**
- Modify: `tests/conftest.py`
- Create: `docs/superpowers/plans/chat-eval-baseline.txt`

**Interfaces:**
- Consumes: nothing
- Produces: `FakeRedis` class and `fake_redis` pytest fixture in `tests/conftest.py`. Every later task's tests import `FakeRedis` via `from tests.conftest import FakeRedis` or use the `fake_redis` fixture, which monkeypatches `services.redis_client.get_redis` to return the fake.

- [ ] **Step 1: Record the chat_eval baseline**

The spec (§5) requires a baseline recorded *before* any tool consolidation, so a regression in Task 10–12 is observed rather than inferred.

```bash
python evals/chat_eval.py 2>&1 | tee docs/superpowers/plans/chat-eval-baseline.txt
```

If this fails because `XAI_API_KEY` is unset, stop and ask the user for the key — do not proceed to Task 10 without a baseline.

- [ ] **Step 2: Add FakeRedis and the fixture to conftest.py**

Append to `tests/conftest.py`:

```python
class FakePipeline:
    """Minimal pipeline supporting the incr/ttl chain used by services/rate_limit.py."""

    def __init__(self, client):
        self._client = client
        self._ops = []

    def incr(self, key):
        self._ops.append(('incr', key))
        return self

    def ttl(self, key):
        self._ops.append(('ttl', key))
        return self

    def execute(self):
        results = [getattr(self._client, op)(key) for op, key in self._ops]
        self._ops = []
        return results


class FakeRedis:
    """In-memory stand-in for redis.Redis(decode_responses=True).

    Implements only the operations this codebase uses. TTLs are stored but
    never expire on their own — call expire_now(key) to simulate expiry.
    """

    def __init__(self):
        self.store = {}
        self.ttls = {}

    def incr(self, key):
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    def ttl(self, key):
        if key not in self.store:
            return -2
        return self.ttls.get(key, -1)

    def expire(self, key, seconds):
        if key not in self.store:
            return False
        self.ttls[key] = seconds
        return True

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        removed = 0
        for key in keys:
            if key in self.store:
                del self.store[key]
                self.ttls.pop(key, None)
                removed += 1
        return removed

    def pipeline(self):
        return FakePipeline(self)

    def expire_now(self, key):
        """Test helper — simulate TTL expiry."""
        self.store.pop(key, None)
        self.ttls.pop(key, None)


@pytest.fixture
def fake_redis(monkeypatch):
    """Patch every get_redis() call site to return one shared FakeRedis."""
    client = FakeRedis()
    monkeypatch.setattr('services.redis_client.get_redis', lambda: client)
    return client
```

- [ ] **Step 3: Write a test proving the fake behaves like Redis**

Create `tests/test_fake_redis.py`:

```python
"""
tests/test_fake_redis.py
========================
Guards the FakeRedis test double itself. Every rate-limit and pending-action
test depends on this behaving like redis.Redis(decode_responses=True), so a
silent divergence here would make those suites pass against fiction.
"""

from tests.conftest import FakeRedis


def test_incr_starts_at_one_and_accumulates():
    r = FakeRedis()
    assert r.incr("k") == 1
    assert r.incr("k") == 2


def test_ttl_is_minus_two_for_missing_key():
    r = FakeRedis()
    assert r.ttl("nope") == -2


def test_ttl_is_minus_one_for_key_without_expiry():
    r = FakeRedis()
    r.incr("k")
    assert r.ttl("k") == -1


def test_expire_sets_ttl_and_returns_true():
    r = FakeRedis()
    r.incr("k")
    assert r.expire("k", 60) is True
    assert r.ttl("k") == 60


def test_set_nx_does_not_overwrite():
    r = FakeRedis()
    assert r.set("k", "first", nx=True) is True
    assert r.set("k", "second", nx=True) is None
    assert r.get("k") == "first"


def test_delete_returns_count_removed():
    r = FakeRedis()
    r.set("a", "1")
    assert r.delete("a", "missing") == 1
    assert r.get("a") is None


def test_pipeline_executes_ops_in_order():
    r = FakeRedis()
    pipe = r.pipeline()
    pipe.incr("k")
    pipe.ttl("k")
    count, ttl = pipe.execute()
    assert count == 1
    assert ttl == -1
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_fake_redis.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_fake_redis.py docs/superpowers/plans/chat-eval-baseline.txt
git commit -m "test: add FakeRedis double and record chat_eval baseline"
```

---

### Task 2: Rate-limit service

**Files:**
- Create: `services/rate_limit.py`
- Test: `tests/test_rate_limit.py`

**Interfaces:**
- Consumes: `services.redis_client.get_redis`, `FakeRedis` from Task 1
- Produces:
  - `allow(scope: str, user_id: int, budgets: Sequence[tuple[int, int]]) -> bool` — increments counters; returns `False` if any budget is exhausted.
  - `would_allow(scope: str, user_id: int, budgets: Sequence[tuple[int, int]]) -> bool` — read-only check, no increment.
  - `SCOUT_BUDGETS: list[tuple[int, int]]` = `[(3, 3600)]`
  - `PLAN_BUDGETS: list[tuple[int, int]]` = `[(5, 86400)]`
  - `CHAT_BUDGETS: list[tuple[int, int]]` = `[(20, 60), (200, 86400)]`
  - A budget is `(limit, window_seconds)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rate_limit.py`:

```python
"""
tests/test_rate_limit.py
========================
Feature tested: per-user, per-capability rate limiting (spec §2.2, C1/C2).

Why this exists
---------------
flask-limiter only sees HTTP routes. The WebSocket chat path and the
background planner both invoke capabilities without passing through a route,
so the budget has to live in a service both can call. These tests pin the
fixed-window semantics that service provides.

What each test covers
---------------------
test_allow_permits_calls_up_to_the_limit
    A budget of (3, 3600) lets exactly 3 calls through.
test_allow_denies_the_call_past_the_limit
    The 4th call returns False.
test_allow_sets_a_ttl_on_first_increment
    The window is bounded — the counter key gets the budget's TTL.
test_allow_repairs_a_missing_ttl
    A key that somehow lost its TTL gets one re-applied, so a counter can
    never become permanent and lock a user out forever.
test_allow_isolates_users
    User 1 exhausting the budget does not affect user 2.
test_allow_isolates_scopes
    The scout budget and the plan budget are independent counters.
test_allow_requires_every_budget_to_pass
    With [(2, 60), (100, 86400)], the tighter budget is what denies.
test_would_allow_does_not_increment
    The read-only check can be called repeatedly without consuming budget.
test_would_allow_reflects_exhaustion
    After the budget is spent, would_allow returns False.
"""

from services.rate_limit import allow, would_allow


def test_allow_permits_calls_up_to_the_limit(fake_redis):
    assert [allow("scout", 1, [(3, 3600)]) for _ in range(3)] == [True, True, True]


def test_allow_denies_the_call_past_the_limit(fake_redis):
    for _ in range(3):
        allow("scout", 1, [(3, 3600)])
    assert allow("scout", 1, [(3, 3600)]) is False


def test_allow_sets_a_ttl_on_first_increment(fake_redis):
    allow("scout", 1, [(3, 3600)])
    key = "ratelimit:scout:1:3600"
    assert fake_redis.ttl(key) == 3600


def test_allow_repairs_a_missing_ttl(fake_redis):
    key = "ratelimit:scout:1:3600"
    fake_redis.store[key] = 1          # counter with no TTL
    allow("scout", 1, [(3, 3600)])
    assert fake_redis.ttl(key) == 3600


def test_allow_isolates_users(fake_redis):
    for _ in range(3):
        allow("scout", 1, [(3, 3600)])
    assert allow("scout", 2, [(3, 3600)]) is True


def test_allow_isolates_scopes(fake_redis):
    for _ in range(3):
        allow("scout", 1, [(3, 3600)])
    assert allow("plan", 1, [(3, 3600)]) is True


def test_allow_requires_every_budget_to_pass(fake_redis):
    budgets = [(2, 60), (100, 86400)]
    assert allow("chat", 1, budgets) is True
    assert allow("chat", 1, budgets) is True
    assert allow("chat", 1, budgets) is False


def test_would_allow_does_not_increment(fake_redis):
    for _ in range(10):
        assert would_allow("scout", 1, [(3, 3600)]) is True
    assert allow("scout", 1, [(3, 3600)]) is True


def test_would_allow_reflects_exhaustion(fake_redis):
    for _ in range(3):
        allow("scout", 1, [(3, 3600)])
    assert would_allow("scout", 1, [(3, 3600)]) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rate_limit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.rate_limit'`

- [ ] **Step 3: Write the implementation**

Create `services/rate_limit.py`:

```python
"""
Per-User Rate Limiting

flask-limiter only observes HTTP routes. The WebSocket chat path and the
background planner both invoke capabilities without passing through a route,
so budgets that must hold everywhere live here and are called from inside the
capability rather than from a decorator.

A budget is a (limit, window_seconds) pair. Fixed-window counters in Redis,
keyed by scope + user + window.
"""

from typing import Sequence

from services.redis_client import get_redis

# Exact budgets from the design spec §2.2.
SCOUT_BUDGETS: list[tuple[int, int]] = [(3, 3600)]        # 3 per hour
PLAN_BUDGETS: list[tuple[int, int]] = [(5, 86400)]        # 5 per day
CHAT_BUDGETS: list[tuple[int, int]] = [(20, 60), (200, 86400)]


def _key(scope: str, user_id: int, window_seconds: int) -> str:
    return f"ratelimit:{scope}:{user_id}:{window_seconds}"


def allow(scope: str, user_id: int, budgets: Sequence[tuple[int, int]]) -> bool:
    """Consume one unit against every budget. False if any is exhausted.

    Note: when an earlier budget passes and a later one denies, the earlier
    counter has still been incremented. That over-counts rather than
    under-counts, which is the safe direction for a spend limit.
    """
    redis = get_redis()
    for limit, window_seconds in budgets:
        key = _key(scope, user_id, window_seconds)

        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        count, ttl = pipe.execute()

        # ttl < 0 means the key has no expiry (-1) or vanished between the two
        # ops (-2). Re-apply it so a counter can never become permanent and
        # lock the user out for good.
        if ttl is None or ttl < 0:
            redis.expire(key, window_seconds)

        if int(count) > limit:
            return False
    return True


def would_allow(scope: str, user_id: int, budgets: Sequence[tuple[int, int]]) -> bool:
    """Read-only check. Does not consume budget.

    Used to avoid offering the user a confirmation button for an action that
    would be refused the moment they click it.
    """
    redis = get_redis()
    for limit, window_seconds in budgets:
        raw = redis.get(_key(scope, user_id, window_seconds))
        if raw is not None and int(raw) >= limit:
            return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rate_limit.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add services/rate_limit.py tests/test_rate_limit.py
git commit -m "feat: add per-user rate limit service"
```

---

### Task 3: Rate-limit the WebSocket chat path (C1)

**Files:**
- Modify: `controllers/ws_chat_controller.py:45-76`
- Test: `tests/test_ws_chat_rate_limit.py`

**Interfaces:**
- Consumes: `allow`, `CHAT_BUDGETS` from Task 2
- Produces: nothing consumed by later tasks

**Context:** `/api/chat` carries `@limiter.limit("20 per minute; 200 per day")`. The WebSocket path — which is what the widget actually uses — has none. This closes that gap.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ws_chat_rate_limit.py`:

```python
"""
tests/test_ws_chat_rate_limit.py
================================
Feature tested: per-user rate limiting on the WebSocket chat entrypoint
(spec §2.2, C1).

Background
----------
flask-limiter decorates /api/chat but cannot see socket.io events, so the
primary chat path shipped with no frequency limit at all. acquire_stream_slot
bounds concurrency to one in-flight message but not rate: send, await done,
send again, forever.

These tests exercise the guard function directly rather than standing up a
socket.io client, so they stay fast and have no server dependency. The handler
calls this same function before acquiring the stream slot.

What each test covers
---------------------
test_chat_rate_guard_allows_within_budget
    The first 20 messages in a minute pass.
test_chat_rate_guard_denies_past_budget
    The 21st is refused.
test_chat_rate_guard_is_checked_before_the_stream_slot
    A refused message must not consume the single concurrency slot, or a
    rate-limited user would be locked out for the slot's 300s TTL.
"""

from controllers.ws_chat_controller import chat_rate_guard
from services.streaming import acquire_stream_slot


def test_chat_rate_guard_allows_within_budget(fake_redis):
    assert all(chat_rate_guard(7) for _ in range(20))


def test_chat_rate_guard_denies_past_budget(fake_redis):
    for _ in range(20):
        chat_rate_guard(7)
    assert chat_rate_guard(7) is False


def test_chat_rate_guard_is_checked_before_the_stream_slot(fake_redis):
    for _ in range(20):
        chat_rate_guard(7)
    assert chat_rate_guard(7) is False
    # The slot must still be free — the guard runs first and returns early.
    assert acquire_stream_slot(7) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ws_chat_rate_limit.py -v`
Expected: FAIL — `ImportError: cannot import name 'chat_rate_guard'`

- [ ] **Step 3: Implement the guard and wire it in**

In `controllers/ws_chat_controller.py`, add to the imports:

```python
from services.rate_limit import allow, CHAT_BUDGETS
```

Add this function above `handle_chat_message`:

```python
def chat_rate_guard(user_id: int) -> bool:
    """Consume one chat message against the user's budget.

    Mirrors the /api/chat route's flask-limiter config, which socket.io
    events bypass entirely.
    """
    return allow("chat_ws", user_id, CHAT_BUDGETS)
```

In `handle_chat_message`, insert the check between the `scan_message` guard and
`acquire_stream_slot` — the ordering matters, see the test:

```python
    user_id = current_user.id

    try:
        within_budget = chat_rate_guard(user_id)
    except Exception:
        logger.exception("Redis unavailable while rate limiting user %s", user_id)
        emit('error', {'error': 'Chat is temporarily unavailable. Please try again shortly.'})
        return

    if not within_budget:
        emit('error', {'error': 'Too many messages. Please slow down.'})
        return

    try:
        slot_acquired = acquire_stream_slot(user_id)
```

Note the existing `user_id = current_user.id` line moves above this block; do not duplicate it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ws_chat_rate_limit.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add controllers/ws_chat_controller.py tests/test_ws_chat_rate_limit.py
git commit -m "fix: rate limit the WebSocket chat path to match /api/chat"
```

---

### Task 4: Bound plan creation and execution (C2 plan budget, C3)

**Files:**
- Modify: `chatbot/planner.py` (`execute_plan`)
- Modify: `chatbot/tools.py` (`create_career_plan`)
- Test: `tests/test_planner_bounds.py`

**Interfaces:**
- Consumes: `allow`, `PLAN_BUDGETS` from Task 2
- Produces: `MAX_PLAN_ITERATIONS: int = 12` module constant in `chatbot/planner.py`

**Context:** `create_career_plan` (`chatbot/tools.py:454`) spawns a daemon thread running `execute_plan`'s plan→execute→replan loop. Two things are unbounded: how often a user may create plans, and how many times the loop may iterate. `create_career_plan` is not a gated capability (spec §3.1 — costly but constructive), so its budget is enforced in the tool itself rather than at a confirmation step.

- [ ] **Step 1: Read the loop before changing it**

Run: `sed -n '390,470p' chatbot/planner.py`

Identify the `while` loop and how it terminates today. The cap goes on that loop's iteration count. Confirm whether an iteration counter already exists before adding one.

- [ ] **Step 2: Write the failing test**

Create `tests/test_planner_bounds.py`:

```python
"""
tests/test_planner_bounds.py
============================
Feature tested: the hard iteration cap on the autonomous plan execution loop
(spec §2.2, C3).

Background
----------
create_career_plan spawns a daemon thread running execute_plan's
plan -> execute -> replan loop. Without a cap, a replanner that keeps
producing new steps consumes LLM budget until the process dies. Nobody is
watching that thread.

What each test covers
---------------------
test_max_plan_iterations_is_defined
    The cap exists as a module constant, so it is one reviewable line.
test_execute_plan_stops_at_the_iteration_cap
    A plan whose replanner always yields another step terminates at the cap
    rather than looping, and records a terminal status.
test_create_career_plan_enforces_the_daily_budget
    5 plans per day per user. Unlike the gated capabilities, this budget is
    consumed in the tool itself — create_career_plan needs no confirmation,
    so there is no later step at which to charge it.
"""

import json

import chatbot.planner as planner_module
from chatbot.tools import build_tools
from models import TaskPlan, PlanStep, db


def test_max_plan_iterations_is_defined():
    assert isinstance(planner_module.MAX_PLAN_ITERATIONS, int)
    assert planner_module.MAX_PLAN_ITERATIONS > 0


def test_execute_plan_stops_at_the_iteration_cap(app_sqlite, monkeypatch):
    app = app_sqlite
    monkeypatch.setattr(planner_module, "MAX_PLAN_ITERATIONS", 3)

    with app.app_context():
        plan = TaskPlan(user_id=1, goal="loop forever", status="running")
        db.session.add(plan)
        db.session.commit()
        plan_id = plan.id

        # A replanner that never runs out of work.
        call_count = {"n": 0}

        def _always_another_step(*args, **kwargs):
            call_count["n"] += 1
            step = PlanStep(
                plan_id=plan_id,
                step_order=call_count["n"],
                description=f"step {call_count['n']}",
                tool_name=None,
                status="pending",
            )
            db.session.add(step)
            db.session.commit()
            return step

        monkeypatch.setattr(planner_module, "get_llm", lambda: None)
        monkeypatch.setattr(planner_module, "_next_step", _always_another_step, raising=False)

        planner_module.execute_plan(app, 1, plan_id)

        refreshed = TaskPlan.query.get(plan_id)
        assert refreshed.status in ("done", "failed", "abandoned")
        assert call_count["n"] <= 3


def test_create_career_plan_enforces_the_daily_budget(app_sqlite, fake_redis, monkeypatch):
    app = app_sqlite

    created = {"n": 0}

    def _fake_generate_plan(app_arg, uid, goal, llm):
        created["n"] += 1
        plan = TaskPlan(user_id=uid, goal=goal, status="running")
        db.session.add(plan)
        db.session.commit()
        return plan

    # create_career_plan does `from chatbot.planner import generate_plan,
    # execute_plan` INSIDE the function body, so patching the planner module's
    # attributes works — the lookup happens at call time.
    monkeypatch.setattr(planner_module, "generate_plan", _fake_generate_plan)
    monkeypatch.setattr(planner_module, "execute_plan", lambda *a, **k: None)
    # The tool imports get_llm from services.llm_service, not from planner.
    monkeypatch.setattr("services.llm_service.get_llm", lambda: None)

    with app.app_context():
        tools = {t.name: t for t in build_tools(app, 1)}
        create = tools["create_career_plan"]

        outcomes = []
        for _ in range(6):
            outcomes.append(json.loads(create.invoke("become an ML engineer")))
            # Clear the active plan between calls. Without this, calls 2-6
            # short-circuit on "you already have an active plan" and never
            # reach the budget check at all.
            TaskPlan.query.filter_by(user_id=1).update({"status": "abandoned"})
            db.session.commit()

        assert created["n"] == 5, "5 plans should be created before the budget refuses"
        assert outcomes[5]["success"] is False
        assert "limit" in outcomes[5]["error"].lower()
```

**Placement matters:** the budget check goes *after* the active-plan early return (Step 5), so a call refused for "you already have a plan" does not consume budget — only real creations are charged. The test abandons the plan between iterations precisely because of that ordering.

**Note:** this test calls `build_tools(app, 1)` with the current signature. Task 7 makes `surface` a required keyword — its Step 5 call-site sweep updates this line to `build_tools(app, 1, surface="chat")`.

**Note for the implementer:** the monkeypatch target `_next_step` is a placeholder for whatever the loop actually calls to obtain the next step — Step 1 tells you its real name. Adjust the patch target to match, keeping the assertion identical.

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_planner_bounds.py -v`
Expected: FAIL — `AttributeError: module 'chatbot.planner' has no attribute 'MAX_PLAN_ITERATIONS'`

- [ ] **Step 4: Implement the cap**

In `chatbot/planner.py`, near `SYNTHESIS_MARKER`:

```python
# Hard ceiling on the plan -> execute -> replan loop. execute_plan runs in an
# unsupervised daemon thread; without this, a replanner that keeps producing
# steps burns LLM budget until the process dies.
MAX_PLAN_ITERATIONS = 12
```

In `execute_plan`, add an iteration counter to the loop and terminate on the cap:

```python
        iterations = 0
        while True:
            iterations += 1
            if iterations > MAX_PLAN_ITERATIONS:
                logger.warning(
                    "[execute_plan] Plan %d hit MAX_PLAN_ITERATIONS (%d) — terminating",
                    plan_id, MAX_PLAN_ITERATIONS,
                )
                plan.status = 'failed'
                safe_commit()
                break
```

Place this at the top of the existing loop body, before any LLM call.

- [ ] **Step 5: Enforce the plan-creation budget**

In `chatbot/tools.py`, add to the imports:

```python
from services.rate_limit import allow, PLAN_BUDGETS
```

In the `create_career_plan` tool, insert the budget check immediately after the
`existing = get_active_plan(user_id)` block and before the `try:`:

```python
            if not allow("plan_create", user_id, PLAN_BUDGETS):
                return json.dumps({
                    "success": False,
                    "error": "You've reached the limit of 5 new career plans per day. "
                             "Your existing plan's roadmap is still available.",
                })
```

The budget is consumed here rather than at a confirmation step because
`create_career_plan` is not a gated capability (spec §3.1) — there is no later
moment at which to charge it.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_planner_bounds.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add chatbot/planner.py chatbot/tools.py tests/test_planner_bounds.py
git commit -m "fix: cap plan execution iterations and plan creation rate"
```

---

### Task 5: Pending-action store

**Files:**
- Create: `services/pending_actions.py`
- Test: `tests/test_pending_actions.py`

**Interfaces:**
- Consumes: `services.redis_client.get_redis`, `fake_redis` fixture from Task 1
- Produces:
  - `propose(user_id: int, capability: str, args: dict, label: str) -> str` — returns the nonce
  - `claim(user_id: int, nonce: str) -> dict | None` — returns `{"user_id", "capability", "args", "label"}` or `None`
  - `TTL_SECONDS: int = 300`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pending_actions.py`:

```python
"""
tests/test_pending_actions.py
=============================
Feature tested: the single-use nonce store backing the confirmation gate
(spec §3.2, §3.3).

The property under test
-----------------------
The client transmits ONLY a nonce. It cannot supply the capability name or
the arguments — those are fixed server-side at proposal time. Nothing
downstream (the model, injected text, a tampered client) can alter what
actually runs. These tests pin that property.

What each test covers
---------------------
test_propose_returns_a_nonce_and_stores_the_payload
    The capability and args are recoverable by nonce.
test_propose_sets_the_ttl
    Unconfirmed actions expire rather than lingering.
test_propose_generates_distinct_nonces
    Two proposals never collide.
test_claim_returns_the_stored_payload
    The round trip preserves capability and args exactly.
test_claim_is_single_use
    A second claim of the same nonce returns None — defeats replay.
test_claim_rejects_another_users_nonce
    User B cannot execute an action proposed for user A.
test_claim_does_not_consume_another_users_nonce
    Critically: a rejected cross-user claim must NOT delete the nonce, or
    user B could grief user A by burning every pending action.
test_claim_returns_none_for_expired_nonce
test_claim_returns_none_for_unknown_nonce
"""

from services.pending_actions import propose, claim, TTL_SECONDS


def test_propose_returns_a_nonce_and_stores_the_payload(fake_redis):
    nonce = propose(1, "abandon_career_plan", {"reason": "restarting"}, "Abandon plan?")
    assert isinstance(nonce, str) and nonce
    assert fake_redis.get(f"pending_action:{nonce}") is not None


def test_propose_sets_the_ttl(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    assert fake_redis.ttl(f"pending_action:{nonce}") == TTL_SECONDS


def test_propose_generates_distinct_nonces(fake_redis):
    a = propose(1, "abandon_career_plan", {}, "x")
    b = propose(1, "abandon_career_plan", {}, "x")
    assert a != b


def test_claim_returns_the_stored_payload(fake_redis):
    nonce = propose(1, "trigger_job_scout_agent", {"reason": "user asked"}, "Run scout?")
    claimed = claim(1, nonce)
    assert claimed["capability"] == "trigger_job_scout_agent"
    assert claimed["args"] == {"reason": "user asked"}
    assert claimed["label"] == "Run scout?"


def test_claim_is_single_use(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    assert claim(1, nonce) is not None
    assert claim(1, nonce) is None


def test_claim_rejects_another_users_nonce(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    assert claim(2, nonce) is None


def test_claim_does_not_consume_another_users_nonce(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    claim(2, nonce)
    assert claim(1, nonce) is not None


def test_claim_returns_none_for_expired_nonce(fake_redis):
    nonce = propose(1, "abandon_career_plan", {}, "Abandon plan?")
    fake_redis.expire_now(f"pending_action:{nonce}")
    assert claim(1, nonce) is None


def test_claim_returns_none_for_unknown_nonce(fake_redis):
    assert claim(1, "never-issued") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pending_actions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.pending_actions'`

- [ ] **Step 3: Write the implementation**

Create `services/pending_actions.py`:

```python
"""
Pending Action Store

Backs the chat confirmation gate. A gated capability does not execute when the
model calls it — it proposes, storing the capability name and its
server-resolved arguments under a single-use nonce. The client is handed only
the nonce, so nothing downstream can alter what eventually runs.
"""

import json
import secrets

# Import the MODULE, not the function. `from services.redis_client import
# get_redis` binds the name at import time, which defeats the test fixture's
# monkeypatch of services.redis_client.get_redis — tests would then write to
# the developer's real Redis instead of the in-memory double, and silently
# pass or fail on real state. Resolve the attribute at call time instead.
from services import redis_client

TTL_SECONDS = 300

_PREFIX = "pending_action:"


def propose(user_id: int, capability: str, args: dict, label: str) -> str:
    """Store a proposed action and return its single-use nonce."""
    nonce = secrets.token_urlsafe(24)
    payload = json.dumps({
        "user_id": user_id,
        "capability": capability,
        "args": args,
        "label": label,
    })
    redis_client.get_redis().set(f"{_PREFIX}{nonce}", payload, ex=TTL_SECONDS)
    return nonce


def claim(user_id: int, nonce: str) -> dict | None:
    """Consume a pending action. Returns None if unknown, expired, or not this user's.

    A nonce belonging to another user is left in place deliberately. Deleting
    it on a failed ownership check would let any authenticated user burn every
    pending action issued to anyone else.
    """
    redis = redis_client.get_redis()
    key = f"{_PREFIX}{nonce}"

    raw = redis.get(key)
    if raw is None:
        return None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        redis.delete(key)
        return None

    if data.get("user_id") != user_id:
        return None

    redis.delete(key)
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pending_actions.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add services/pending_actions.py tests/test_pending_actions.py
git commit -m "feat: add single-use pending action store for confirmation gate"
```

---

### Task 6: Gated action implementations

**Files:**
- Create: `chatbot/gated_actions.py`
- Modify: `chatbot/tools.py` (extract the two implementations)
- Test: `tests/test_gated_actions.py`

**Interfaces:**
- Consumes: `allow`, `SCOUT_BUDGETS` from Task 2
- Produces:
  - `GATED_TOOL_NAMES: frozenset[str]` = `{"trigger_job_scout_agent", "abandon_career_plan"}`
  - `run_job_scout(app, user_id: int, reason: str) -> dict`
  - `abandon_plan(app, user_id: int, reason: str) -> dict`
  - `CONFIRMED_EXECUTORS: dict[str, Callable]` mapping capability name → executor
  - `execute_confirmed(app, user_id: int, pending: dict) -> dict` — dispatches; returns `{"success": bool, "message": str}`

**Context:** These are the real implementations lifted out of the `@tool` wrappers in `chatbot/tools.py:221-237` (`trigger_job_scout_agent`) and `chatbot/tools.py:566-581` (`abandon_career_plan`). In Task 7 the wrappers become propose-only; these functions become the only path that actually does the work.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gated_actions.py`:

```python
"""
tests/test_gated_actions.py
===========================
Feature tested: the real implementations behind the two gated capabilities,
and the dispatcher the confirm endpoint calls (spec §3).

Background
----------
After the confirmation gate lands, the @tool wrappers for these capabilities
only propose. These functions are the sole path that actually runs the work,
reached exclusively from POST /api/chat/confirm.

What each test covers
---------------------
test_gated_tool_names_lists_both_capabilities
test_abandon_plan_marks_the_active_plan_abandoned
test_abandon_plan_reports_failure_when_no_active_plan
test_run_job_scout_enforces_the_hourly_budget
    The budget is consumed at execution, not at proposal, so proposing
    repeatedly without confirming cannot exhaust it.
test_execute_confirmed_dispatches_to_the_named_capability
test_execute_confirmed_rejects_an_unknown_capability
    A capability name that is not in the registry must never be called, even
    though the name can only come from server-side storage.
"""

import pytest

from chatbot import gated_actions
from models import TaskPlan, db


def test_gated_tool_names_lists_both_capabilities():
    assert gated_actions.GATED_TOOL_NAMES == frozenset({
        "trigger_job_scout_agent",
        "abandon_career_plan",
    })


def test_abandon_plan_marks_the_active_plan_abandoned(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        db.session.add(TaskPlan(user_id=1, goal="become an ML engineer", status="running"))
        db.session.commit()

        result = gated_actions.abandon_plan(app, 1, "changed my mind")

        assert result["success"] is True
        assert TaskPlan.query.filter_by(user_id=1).first().status == "abandoned"


def test_abandon_plan_reports_failure_when_no_active_plan(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        result = gated_actions.abandon_plan(app, 1, "")
        assert result["success"] is False


def test_run_job_scout_enforces_the_hourly_budget(app_sqlite, fake_redis, monkeypatch):
    app = app_sqlite

    class _FakeScheduler:
        def trigger_manual_run(self, user_id):
            return {"status": "success", "matches_found": 2, "jobs_analyzed": 5, "jobs_fetched": 9}

    app.extensions['scheduler'] = _FakeScheduler()

    with app.app_context():
        results = [gated_actions.run_job_scout(app, 1, "test") for _ in range(4)]

    assert [r["success"] for r in results[:3]] == [True, True, True]
    assert results[3]["success"] is False
    assert "limit" in results[3]["message"].lower()


def test_execute_confirmed_dispatches_to_the_named_capability(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        db.session.add(TaskPlan(user_id=1, goal="g", status="running"))
        db.session.commit()

        result = gated_actions.execute_confirmed(app, 1, {
            "capability": "abandon_career_plan",
            "args": {"reason": "done"},
        })

        assert result["success"] is True


def test_execute_confirmed_rejects_an_unknown_capability(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        result = gated_actions.execute_confirmed(app, 1, {
            "capability": "delete_everything",
            "args": {},
        })
        assert result["success"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gated_actions.py -v`
Expected: FAIL — `ImportError: cannot import name 'gated_actions' from 'chatbot'`

- [ ] **Step 3: Write the implementation**

Create `chatbot/gated_actions.py`:

```python
"""
Gated Capability Implementations

The two capabilities that require explicit user confirmation. Their @tool
wrappers in chatbot/tools.py only propose an action; these functions are the
only code path that actually performs it, and they are reached exclusively
from POST /api/chat/confirm after the user clicks.

Rate limits are consumed HERE, at execution — not at proposal — so a model
proposing repeatedly without the user confirming cannot exhaust the budget.
"""

import logging

from services.db_lock import safe_commit
from services.rate_limit import allow, SCOUT_BUDGETS

logger = logging.getLogger(__name__)

GATED_TOOL_NAMES = frozenset({
    "trigger_job_scout_agent",
    "abandon_career_plan",
})


def run_job_scout(app, user_id: int, reason: str) -> dict:
    """Fetch from all enabled sources and analyse. Costs real money per run."""
    if not allow("scout_manual", user_id, SCOUT_BUDGETS):
        return {
            "success": False,
            "message": "You've reached the limit of 3 manual scout runs per hour. "
                       "The scheduled runs will keep finding matches in the meantime.",
        }

    with app.app_context():
        try:
            result = app.extensions['scheduler'].trigger_manual_run(user_id)
            matches = result.get('matches_found', 0)
            return {
                "success": result['status'] == 'success',
                "message": (
                    f"Job Scout finished — analysed {result.get('jobs_analyzed', 0)} jobs "
                    f"from {result.get('jobs_fetched', 0)} fetched and saved {matches} new "
                    f"{'match' if matches == 1 else 'matches'}."
                ),
            }
        except Exception as exc:
            logger.exception("run_job_scout failed for user %s", user_id)
            return {"success": False, "message": f"The Job Scout run failed: {exc}"}


def abandon_plan(app, user_id: int, reason: str) -> dict:
    """Mark the user's active career plan abandoned. Destroys work, no undo."""
    with app.app_context():
        from chatbot.planner import get_active_plan

        plan = get_active_plan(user_id)
        if not plan:
            return {"success": False, "message": "You have no active plan to abandon."}

        goal = plan.goal
        plan.status = 'abandoned'
        safe_commit()
        return {
            "success": True,
            "message": f"Plan '{goal}' has been abandoned. You can create a new one anytime.",
        }


CONFIRMED_EXECUTORS = {
    "trigger_job_scout_agent": lambda app, user_id, args: run_job_scout(
        app, user_id, args.get("reason", "")
    ),
    "abandon_career_plan": lambda app, user_id, args: abandon_plan(
        app, user_id, args.get("reason", "")
    ),
}


def execute_confirmed(app, user_id: int, pending: dict) -> dict:
    """Dispatch a claimed pending action to its executor.

    The capability name can only have come from server-side storage, but the
    registry lookup is still authoritative — an unrecognised name is refused
    rather than reflected into a call.
    """
    capability = pending.get("capability")
    executor = CONFIRMED_EXECUTORS.get(capability)
    if executor is None:
        logger.error("execute_confirmed got unknown capability %r for user %s", capability, user_id)
        return {"success": False, "message": "That action is no longer available."}

    return executor(app, user_id, pending.get("args") or {})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gated_actions.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add chatbot/gated_actions.py tests/test_gated_actions.py
git commit -m "feat: extract gated capability implementations"
```

---

### Task 7: Surface-scoped build_tools + propose-only gated wrappers (G1, G3)

**Files:**
- Modify: `chatbot/tools.py` (`build_tools` signature; the two gated wrappers)
- Modify: `chatbot/agent.py` (its `build_tools` call)
- Modify: `controllers/ws_chat_controller.py` (its `build_tools` call)
- Modify: `chatbot/planner.py:377` (its `build_tools` call)
- Test: `tests/test_tool_surfaces.py`

**Interfaces:**
- Consumes: `GATED_TOOL_NAMES` from Task 6; `propose` from Task 5; `would_allow`, `SCOUT_BUDGETS` from Task 2
- Produces: `build_tools(app, user_id, *, surface: str, progress_cb=None) -> list` where `surface` is `"chat"` or `"planner"`. **`surface` is keyword-only and has no default** — a caller who forgets it gets a `TypeError` at the call site rather than silently receiving gated tools.

**Context — why this task exists:** `chatbot/planner.py:377` calls `build_tools(app, user_id)` and receives the full tool list including gated capabilities, then dispatches by LLM-chosen name (`planner.py:423`) inside a background daemon thread where no user can click anything. Without this task, a plan step invoking a gated capability would mint a nonce, emit to nobody, and hand the replan loop a condition it can never satisfy.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tool_surfaces.py`:

```python
"""
tests/test_tool_surfaces.py
===========================
Feature tested: surface-scoped tool construction (spec §3.5, G1 and G3).

The hole this closes
--------------------
chatbot/planner.py builds the full tool list and dispatches by LLM-chosen
name inside an unsupervised daemon thread. A confirmation gate is meaningless
there — nobody is present to click. So the planner surface must not receive
gated capabilities at all.

What each test covers
---------------------
test_chat_surface_includes_gated_tools
    The model must still be able to PROPOSE them, so they stay in the chat list.
test_planner_surface_excludes_gated_tools
    The planner physically cannot invoke what it never receives.
test_surface_is_required
    Omitting it raises TypeError rather than defaulting to the permissive
    value. This is the G3 defence: a future caller that forgets fails loudly.
test_gated_tool_proposes_without_executing
    Calling the chat-surface wrapper stores a pending action and returns
    confirm_required — it must NOT perform the work.
test_gated_tool_does_not_consume_rate_budget_on_propose
    Budget is spent at confirm time (Task 6), not here.
"""

import json

import pytest

from chatbot.tools import build_tools
from chatbot.gated_actions import GATED_TOOL_NAMES
from services.rate_limit import would_allow, SCOUT_BUDGETS


def _names(tools):
    return {t.name for t in tools}


def test_chat_surface_includes_gated_tools(app_sqlite):
    tools = build_tools(app_sqlite, 1, surface="chat")
    assert GATED_TOOL_NAMES <= _names(tools)


def test_planner_surface_excludes_gated_tools(app_sqlite):
    tools = build_tools(app_sqlite, 1, surface="planner")
    assert _names(tools).isdisjoint(GATED_TOOL_NAMES)


def test_surface_is_required(app_sqlite):
    with pytest.raises(TypeError):
        build_tools(app_sqlite, 1)


def test_gated_tool_proposes_without_executing(app_sqlite, fake_redis):
    app = app_sqlite
    from models import TaskPlan, db

    with app.app_context():
        db.session.add(TaskPlan(user_id=1, goal="become an ML engineer", status="running"))
        db.session.commit()

        tools = {t.name: t for t in build_tools(app, 1, surface="chat")}
        raw = tools["abandon_career_plan"].invoke("changed my mind")
        payload = json.loads(raw)

        assert payload["action"] == "confirm_required"
        assert payload["nonce"]
        # The plan must be untouched — proposing is not doing.
        assert TaskPlan.query.filter_by(user_id=1).first().status == "running"


def test_gated_tool_does_not_consume_rate_budget_on_propose(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        tools = {t.name: t for t in build_tools(app, 1, surface="chat")}
        for _ in range(5):
            tools["trigger_job_scout_agent"].invoke("please run it")
        assert would_allow("scout_manual", 1, SCOUT_BUDGETS) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tool_surfaces.py -v`
Expected: FAIL — `TypeError: build_tools() got an unexpected keyword argument 'surface'`

- [ ] **Step 3: Change the build_tools signature and filter**

In `chatbot/tools.py`, add imports at the top:

```python
from chatbot.gated_actions import GATED_TOOL_NAMES
from services.pending_actions import propose
from services.rate_limit import would_allow, SCOUT_BUDGETS
```

Change the signature at `chatbot/tools.py:24`:

```python
def build_tools(app, user_id, *, surface: str, progress_cb=None):
    """Return the list of LangChain tools bound to app + user_id.

    surface: 'chat' or 'planner'. Keyword-only with no default on purpose —
        the planner runs in a background thread with no user present, so it
        must never receive a capability that requires confirmation. A caller
        who forgets this argument gets a TypeError instead of silently
        receiving the permissive set.

    progress_cb: optional callable(label: str) pushed mid-tool to update the UI.
    """
```

Replace the return statement at `chatbot/tools.py:583-585`:

```python
    all_tools = [find_top_jobs, get_resume_info, trigger_job_scout_agent, get_recent_matches,
                 explain_feature, search_job_by_title, tailor_resume_to_job, get_user_preferences,
                 search_memory, create_career_plan, get_career_plan_status, abandon_career_plan]

    if surface == "chat":
        return all_tools
    return [t for t in all_tools if t.name not in GATED_TOOL_NAMES]
```

- [ ] **Step 4: Rewrite the two gated wrappers as propose-only**

Replace the body of `trigger_job_scout_agent` (`chatbot/tools.py:221-237`):

```python
    @tool
    def trigger_job_scout_agent(reason: str) -> str:
        """Ask the user to confirm running the Job Scout Agent, which searches all enabled job sources for new matches. Use this when the user asks to run the agent, scan for new jobs, or do an automatic job search. This does NOT run the scout — it shows the user a confirmation button."""
        if surface != "chat":
            raise RuntimeError(
                f"trigger_job_scout_agent requires a confirmable surface; got {surface!r}"
            )

        if not would_allow("scout_manual", user_id, SCOUT_BUDGETS):
            return json.dumps({
                "success": False,
                "error": "You've reached the limit of 3 manual scout runs per hour. "
                         "The scheduled runs will keep finding matches in the meantime.",
            })

        label = "Run the Job Scout now?"
        nonce = propose(user_id, "trigger_job_scout_agent", {"reason": reason}, label)
        return json.dumps({
            "success": True,
            "action": "confirm_required",
            "nonce": nonce,
            "label": label,
            "note": "Confirmation required. The user has been shown a button. "
                    "Tell them to click it. Do not call this tool again.",
        })
```

Replace the body of `abandon_career_plan` (`chatbot/tools.py:566-581`):

```python
    @tool
    def abandon_career_plan(reason: str = "") -> str:
        """Ask the user to confirm abandoning their current active career plan. Use this when the user wants to cancel, restart, or change their career plan. This does NOT abandon the plan — it shows the user a confirmation button."""
        if surface != "chat":
            raise RuntimeError(
                f"abandon_career_plan requires a confirmable surface; got {surface!r}"
            )

        with app.app_context():
            from chatbot.planner import get_active_plan

            plan = get_active_plan(user_id)
            if not plan:
                return json.dumps({"success": False, "error": "No active plan to abandon."})

            # Resolve the arguments server-side and freeze them into the pending
            # action. The client only ever echoes the nonce back.
            label = f"Abandon your plan '{plan.goal}'?"
            nonce = propose(user_id, "abandon_career_plan", {"reason": reason}, label)
            return json.dumps({
                "success": True,
                "action": "confirm_required",
                "nonce": nonce,
                "label": label,
                "note": "Confirmation required. The user has been shown a button. "
                        "Tell them to click it. Do not call this tool again.",
            })
```

- [ ] **Step 5: Update all three call sites**

In `chatbot/agent.py`, find the `build_tools(` call and add `surface="chat"`:

```python
                tools = build_tools(app, user_id, surface="chat", progress_cb=handler.push_progress)
```

In `controllers/ws_chat_controller.py:119`:

```python
                tools = build_tools(app, user_id, surface="chat", progress_cb=handler.push_progress)
```

In `chatbot/planner.py:377`:

```python
            tools = build_tools(app, user_id, surface="planner")
```

Also update `tests/test_planner_bounds.py`, which calls `build_tools(app, 1)`:

```python
        tools = {t.name: t for t in build_tools(app, 1, surface="chat")}
```

Verify no call site was missed:

```bash
grep -rn "build_tools(" --include="*.py" . | grep -v "def build_tools"
```

Every result — including those under `tests/` — must pass `surface=`.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/test_tool_surfaces.py -v && python -m pytest`
Expected: 5 passed in the new file; no regressions elsewhere.

- [ ] **Step 7: Commit**

```bash
git add chatbot/tools.py chatbot/agent.py controllers/ws_chat_controller.py chatbot/planner.py tests/test_tool_surfaces.py
git commit -m "feat: surface-scoped tools; gated capabilities propose instead of execute"
```

---

### Task 8: Confirm endpoint + intent wiring

**Files:**
- Modify: `controllers/chat_controller.py` (add `POST /api/chat/confirm`)
- Modify: `chatbot/agent.py:182-213` (`_extract_intent`)
- Test: `tests/test_confirm_endpoint.py`, `tests/test_intent_extraction.py`

**Interfaces:**
- Consumes: `claim` from Task 5; `execute_confirmed` from Task 6; the `confirm_required` tool payload from Task 7
- Produces: route `chat.confirm_action` at `POST /api/chat/confirm`; `_extract_intent` now also returns `("confirm_required", '{"nonce": ..., "label": ...}')`

- [ ] **Step 1: Write the failing intent test**

Append to `tests/test_intent_extraction.py`:

```python
def test_extract_intent_confirm_required():
    """A gated tool's propose payload surfaces as a confirm_required intent
    carrying the nonce and the human-readable label for the button."""
    steps = [("abandon_career_plan", json.dumps({
        "success": True,
        "action": "confirm_required",
        "nonce": "abc123",
        "label": "Abandon your plan 'Become an ML engineer'?",
    }))]
    intent, action_data = _extract_intent(steps)
    assert intent == "confirm_required"
    parsed = json.loads(action_data)
    assert parsed["nonce"] == "abc123"
    assert parsed["label"] == "Abandon your plan 'Become an ML engineer'?"


def test_extract_intent_ignores_confirm_payload_without_nonce():
    """A malformed propose payload must not produce a button with no nonce."""
    steps = [("abandon_career_plan", json.dumps({
        "success": True,
        "action": "confirm_required",
        "label": "Abandon?",
    }))]
    intent, _ = _extract_intent(steps)
    assert intent is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_intent_extraction.py -v`
Expected: FAIL — the two new tests fail; existing tests still pass.

- [ ] **Step 3: Extend _extract_intent**

In `chatbot/agent.py`, inside the `for tool_name, tool_output in tool_steps:` loop, add a branch that runs for **any** tool name (gated names are checked from the registry, not hardcoded), placed before the existing `if tool_name == "find_top_jobs":`:

```python
    from chatbot.gated_actions import GATED_TOOL_NAMES

    intent = None
    action_data = None
    for tool_name, tool_output in tool_steps:
        if tool_name in GATED_TOOL_NAMES:
            try:
                parsed = json.loads(tool_output) if isinstance(tool_output, str) else tool_output
                if parsed.get("action") == "confirm_required" and parsed.get("nonce"):
                    intent = "confirm_required"
                    action_data = json.dumps({
                        "nonce": parsed["nonce"],
                        "label": parsed.get("label", "Confirm"),
                    })
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        elif tool_name == "find_top_jobs":
```

Note the existing `if tool_name == "find_top_jobs":` becomes `elif`.

- [ ] **Step 4: Run to verify the intent tests pass**

Run: `python -m pytest tests/test_intent_extraction.py -v`
Expected: all pass, including the two new ones.

- [ ] **Step 5: Write the failing endpoint test**

Create `tests/test_confirm_endpoint.py`:

```python
"""
tests/test_confirm_endpoint.py
==============================
Feature tested: POST /api/chat/confirm — the only path that executes a gated
capability (spec §3.2, §3.3).

The property under test
-----------------------
The client sends ONLY a nonce. It cannot name the capability or supply
arguments. These tests assert that a client attempting to do so is ignored,
and that ownership, single-use, and expiry all hold at the HTTP boundary.

What each test covers
---------------------
test_confirm_requires_authentication
test_confirm_rejects_a_missing_nonce
test_confirm_rejects_an_unknown_nonce
test_confirm_executes_the_stored_capability
test_confirm_ignores_client_supplied_capability_and_args
    The decisive test: a client posting its own capability/args alongside a
    valid nonce gets the STORED action, not the one it asked for.
test_confirm_is_single_use
"""

import json

import pytest

from models import User, TaskPlan, db
from services.pending_actions import propose


@pytest.fixture
def client_with_user(app_sqlite, fake_redis):
    app = app_sqlite
    with app.app_context():
        user = User(username="tester", email="t@example.com")
        user.set_password("pw-not-used-in-test")
        db.session.add(user)
        db.session.commit()
        user_id = user.id

    client = app.test_client()
    with client.session_transaction() as session:
        session['_user_id'] = str(user_id)
        session['_fresh'] = True
    return app, client, user_id


def test_confirm_requires_authentication(app_sqlite, fake_redis):
    client = app_sqlite.test_client()
    response = client.post('/api/chat/confirm', json={"nonce": "x"})
    assert response.status_code in (302, 401)


def test_confirm_rejects_a_missing_nonce(client_with_user):
    _, client, _ = client_with_user
    response = client.post('/api/chat/confirm', json={})
    assert response.status_code == 400


def test_confirm_rejects_an_unknown_nonce(client_with_user):
    _, client, _ = client_with_user
    response = client.post('/api/chat/confirm', json={"nonce": "never-issued"})
    assert response.status_code == 400


def test_confirm_executes_the_stored_capability(client_with_user):
    app, client, user_id = client_with_user
    with app.app_context():
        db.session.add(TaskPlan(user_id=user_id, goal="g", status="running"))
        db.session.commit()

    nonce = propose(user_id, "abandon_career_plan", {"reason": "done"}, "Abandon 'g'?")
    response = client.post('/api/chat/confirm', json={"nonce": nonce})

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    with app.app_context():
        assert TaskPlan.query.filter_by(user_id=user_id).first().status == "abandoned"


def test_confirm_ignores_client_supplied_capability_and_args(client_with_user):
    app, client, user_id = client_with_user
    with app.app_context():
        db.session.add(TaskPlan(user_id=user_id, goal="g", status="running"))
        db.session.commit()

    nonce = propose(user_id, "abandon_career_plan", {"reason": "done"}, "Abandon 'g'?")
    response = client.post('/api/chat/confirm', json={
        "nonce": nonce,
        "capability": "trigger_job_scout_agent",
        "args": {"reason": "attacker supplied"},
    })

    assert response.status_code == 200
    # The STORED capability ran, not the one the client named.
    with app.app_context():
        assert TaskPlan.query.filter_by(user_id=user_id).first().status == "abandoned"


def test_confirm_is_single_use(client_with_user):
    app, client, user_id = client_with_user
    with app.app_context():
        db.session.add(TaskPlan(user_id=user_id, goal="g", status="running"))
        db.session.commit()

    nonce = propose(user_id, "abandon_career_plan", {"reason": ""}, "Abandon 'g'?")
    assert client.post('/api/chat/confirm', json={"nonce": nonce}).status_code == 200
    assert client.post('/api/chat/confirm', json={"nonce": nonce}).status_code == 400
```

(`User.set_password` is defined at `models.py:33` and hashes with pbkdf2:sha256 — the fixture above is correct as written.)

- [ ] **Step 6: Run to verify failure**

Run: `python -m pytest tests/test_confirm_endpoint.py -v`
Expected: FAIL — 404 on the route.

- [ ] **Step 7: Add the endpoint**

In `controllers/chat_controller.py`, add to the imports:

```python
from services.pending_actions import claim
from chatbot.gated_actions import execute_confirmed
```

Add the route after `chat_api`:

```python
@chat_bp.route('/api/chat/confirm', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def confirm_action():
    """Execute a previously proposed gated capability.

    The client sends ONLY a nonce. The capability name and its arguments were
    resolved server-side when the action was proposed and are read from the
    store — anything else in the request body is ignored on purpose, so
    neither the model nor a tampered client can change what runs.
    """
    body = request.get_json(silent=True) or {}
    nonce = body.get('nonce')
    if not isinstance(nonce, str) or not nonce:
        return jsonify({"success": False, "error": "Missing confirmation token."}), 400

    pending = claim(current_user.id, nonce)
    if pending is None:
        return jsonify({
            "success": False,
            "error": "This action has expired or was already used. Ask me again if you still want it.",
        }), 400

    try:
        result = execute_confirmed(current_app._get_current_object(), current_user.id, pending)
    except Exception:
        logger.exception("Confirmed action failed for user %s", current_user.id)
        return jsonify({"success": False, "error": "That action failed. Please try again."}), 500

    return jsonify({
        "success": result.get("success", False),
        "message": result.get("message", ""),
    })
```

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest tests/test_confirm_endpoint.py tests/test_intent_extraction.py -v && python -m pytest`
Expected: all pass, no regressions.

- [ ] **Step 9: Commit**

```bash
git add controllers/chat_controller.py chatbot/agent.py tests/test_confirm_endpoint.py tests/test_intent_extraction.py
git commit -m "feat: add POST /api/chat/confirm and confirm_required intent"
```

---

### Task 9: Client confirmation button — and fix the stored XSS found alongside it

**Files:**
- Modify: `templates/chat_widget.html:308-327` (`attachActionButtons`)

**Interfaces:**
- Consumes: the `confirm_required` intent and `{nonce, label}` action_data from Task 8
- Produces: nothing consumed by later tasks

**SECURITY — read before starting.** `attachActionButtons` currently builds the tailor button with `insertAdjacentHTML` and interpolates `job.title` and `job.company` directly (lines 315–326). Those values come from `JobPosting` rows populated by the external job-board fetchers. **A job posting whose title contains markup is stored XSS in the chat widget.** This is a live vulnerability today, independent of anything else in this plan, and it sits in the exact function this task modifies. Fix it here.

This was not in the design spec — it was found while mapping this task. Flag it to the reviewer in the commit message.

- [ ] **Step 1: Rewrite attachActionButtons using DOM construction**

Replace the whole function (`templates/chat_widget.html:308-327`):

```javascript
    function chatActionButton(label) {
        const btn = document.createElement('button');
        btn.className = 'chat-action-btn';
        btn.type = 'button';
        btn.textContent = label;          // textContent, never innerHTML
        return btn;
    }

    function chatActionLink(href, label) {
        const a = document.createElement('a');
        a.className = 'chat-action-btn';
        a.href = href;
        a.textContent = label;            // textContent, never innerHTML
        return a;
    }

    function attachActionButtons(div, intent, actionData) {
        if (!actionData) return;

        if (intent === 'redirect_to_jobs' && actionData.job_ids) {
            const ids = actionData.job_ids.join(',');
            const label = 'View ' + actionData.job_ids.length + ' Matching Jobs';
            div.appendChild(document.createElement('br'));
            div.appendChild(chatActionLink('/jobs?ids=' + encodeURIComponent(ids), label));
        }

        if (intent === 'open_tailor_modal' && actionData.job_id) {
            // job.title and job.company originate from external job boards.
            // They MUST go through textContent — building this with
            // insertAdjacentHTML was a stored XSS.
            const job = actionData.job || {};
            const name = job.title
                ? job.title + (job.company ? ' at ' + job.company : '')
                : 'Job';
            const before = actionData.ats_before;
            const after = actionData.ats_after;
            const score = (before && before !== '?' && after && after !== '?')
                ? ' (' + before + '% → ' + after + '%)'
                : '';
            const id = encodeURIComponent(actionData.job_id);
            const link = chatActionLink(
                '/jobs?ids=' + id + '&tailor=' + id,
                'View ATS Results for ' + name + score
            );
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            div.appendChild(document.createElement('br'));
            div.appendChild(link);
        }

        if (intent === 'confirm_required' && actionData.nonce) {
            // The label embeds the user's plan goal — untrusted text, so
            // textContent only (chatActionButton handles that).
            const btn = chatActionButton(actionData.label || 'Confirm');
            btn.addEventListener('click', function () {
                btn.disabled = true;
                btn.textContent = 'Working…';
                fetch('/api/chat/confirm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ nonce: actionData.nonce })
                })
                    .then(function (r) { return r.json(); })
                    .catch(function () { return { success: false, error: 'Network error.' }; })
                    .then(function (data) {
                        btn.remove();
                        // The result is rendered directly, NOT sent back through
                        // the LLM — the outcome of a destructive action must not
                        // be reportable by a hallucination.
                        const out = document.createElement('div');
                        out.className = 'chat-msg assistant';
                        out.innerHTML = formatContent(
                            data.message || data.error || 'Done.'
                        );
                        messagesEl.appendChild(out);
                        scrollToBottom();
                    });
            });
            div.appendChild(document.createElement('br'));
            div.appendChild(btn);
        }
    }
```

- [ ] **Step 2: Verify the XSS is closed**

Start the app and confirm manually — there is no automated JS test harness in this repo.

```bash
python app.py
```

Insert a job posting with a markup-bearing title via the Add Job page (`/add-job`), title:

```
Engineer <img src=x onerror="document.title='XSS'">
```

Ask the chatbot to tailor your resume to that job. The button must render the tag as **literal text**, and `document.title` must be unchanged. Before this task, the same input executes.

- [ ] **Step 3: Verify the confirm button works end to end**

With the app running and a career plan active, ask the chatbot to cancel your plan. Confirm:
1. A button appears reading `Abandon your plan '<goal>'?`
2. The plan is **still active** before you click (check `/agent-dashboard`)
3. Clicking it produces the result message and the plan becomes abandoned
4. Clicking the same button again is impossible — it was removed

- [ ] **Step 4: Run the suite for regressions**

Run: `python -m pytest`
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add templates/chat_widget.html
git commit -m "fix: render chat action buttons via DOM, closing stored XSS

attachActionButtons interpolated job.title and job.company into
insertAdjacentHTML. Those values come from external job boards, so a
posting with a markup-bearing title executed in the chat widget. All
action buttons now build through textContent.

Found while adding the confirm_required button; not part of the
original design spec. Worth a reviewer's attention."
```

---

### Task 10: Merge the two history tools and drop the two non-tools

**Files:**
- Modify: `chatbot/tools.py` (merge `get_recent_matches` + `get_user_preferences`; delete `explain_feature`, `get_career_plan_status`)
- Modify: `chatbot/agent.py` (`build_system_prompt` — pre-load plan status, add feature blurb)
- Modify: `services/streaming.py:52` (`_TOOL_LABELS`)
- Test: `tests/test_tool_registry.py`

**Interfaces:**
- Consumes: `build_tools(..., surface=...)` from Task 7
- Produces: tool `get_job_history(limit: int = 5) -> str`; helper `plan_status_summary(user_id: int) -> str` in `chatbot/agent.py`

**Context:** Spec §4.2 (merge pair 2) and §4.3 (both removals). `explain_feature` is a static dict with no DB or LLM call — a constant, not a tool. `get_career_plan_status` is one cheap query useful on every turn, which is the signal it belongs in the prompt.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tool_registry.py`:

```python
"""
tests/test_tool_registry.py
===========================
Feature tested: the consolidated tool set (spec §4).

Why a registry test
-------------------
Tool-selection accuracy degrades with confusable, overlapping tools. This
test pins the exact set so a future addition is a deliberate, reviewed change
rather than drift.

What each test covers
---------------------
test_chat_surface_exposes_exactly_the_expected_tools
test_removed_tools_are_gone
test_get_job_history_returns_matches_and_preferences_together
    The merge must not lose either half of what the two old tools returned.
"""

import json

from chatbot.tools import build_tools

EXPECTED_CHAT_TOOLS = {
    "find_top_jobs",
    "search_job_by_title",
    "get_resume_info",
    "search_memory",
    "get_job_history",
    "tailor_resume_to_job",
    "create_career_plan",
    "trigger_job_scout_agent",
    "abandon_career_plan",
}


def test_chat_surface_exposes_exactly_the_expected_tools(app_sqlite):
    tools = {t.name for t in build_tools(app_sqlite, 1, surface="chat")}
    assert tools == EXPECTED_CHAT_TOOLS


def test_removed_tools_are_gone(app_sqlite):
    tools = {t.name for t in build_tools(app_sqlite, 1, surface="chat")}
    assert "explain_feature" not in tools
    assert "get_career_plan_status" not in tools
    assert "get_recent_matches" not in tools
    assert "get_user_preferences" not in tools


def test_get_job_history_returns_matches_and_preferences_together(app_sqlite):
    app = app_sqlite
    with app.app_context():
        tools = {t.name: t for t in build_tools(app, 1, surface="chat")}
        payload = json.loads(tools["get_job_history"].invoke({"limit": 5}))

        assert payload["success"] is True
        # Both halves of the two merged tools are present.
        assert "matches" in payload
        assert "personalization_active" in payload
        assert "liked_jobs" in payload
        assert "disliked_jobs" in payload
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_tool_registry.py -v`
Expected: FAIL — the expected set does not match.

- [ ] **Step 3: Add get_job_history, replacing the two old tools**

In `chatbot/tools.py`, delete the `get_recent_matches` tool (lines 239-269) and the `get_user_preferences` tool (lines 394-432), and add:

```python
    @tool
    def get_job_history(limit: int = 5) -> str:
        """Get the user's recent job matches AND the job preferences the AI has learned from their feedback. Use this when the user asks about their matches, previous results, match history, what you've learned about them, their preferences, or how personalization works for them."""
        with app.app_context():
            try:
                matches = (JobMatch.query
                           .options(joinedload(JobMatch.job))  # type: ignore[arg-type]
                           .filter_by(user_id=user_id)
                           .order_by(JobMatch.created_at.desc())
                           .limit(limit).all())

                config = AgentConfig.query.filter_by(user_id=user_id).first()
                liked = JobMatch.query.filter(
                    JobMatch.user_id == user_id,
                    JobMatch.user_feedback.in_(['interested', 'applied'])
                ).order_by(JobMatch.feedback_at.desc()).limit(10).all()
                disliked = JobMatch.query.filter_by(
                    user_id=user_id, user_feedback='not_interested'
                ).order_by(JobMatch.feedback_at.desc()).limit(5).all()

                has_preferences = config is not None and config.preference_embedding is not None

                return json.dumps({
                    "success": True,
                    "matches": [
                        {
                            "id": m.id,
                            "job_title": m.job.title if m.job else "Unknown",
                            "company": m.job.company if m.job else "Unknown",
                            "match_score": m.match_score,
                            "feedback": m.user_feedback,
                            "created_at": m.created_at.isoformat() if m.created_at else None,
                        }
                        for m in matches
                    ],
                    "personalization_active": has_preferences,
                    "liked_count": len(liked),
                    "disliked_count": len(disliked),
                    "liked_jobs": [
                        {"title": m.job.title, "company": m.job.company, "feedback": m.user_feedback}
                        for m in liked if m.job
                    ],
                    "disliked_jobs": [
                        {"title": m.job.title, "company": m.job.company}
                        for m in disliked if m.job
                    ],
                    "message": (
                        f"Personalization is active — learned from {len(liked)} liked "
                        f"and {len(disliked)} disliked jobs."
                        if has_preferences else
                        "No preferences learned yet. Rate matches as 'interested' or "
                        "'not interested' to enable personalized recommendations."
                    ),
                })
            except Exception as e:
                logger.error("get_job_history error: %s", e)
                return json.dumps({"success": False, "error": str(e)})
```

Delete the `explain_feature` tool (lines 271-287) and the `get_career_plan_status` tool (lines 516-564) entirely.

Update the return block:

```python
    all_tools = [find_top_jobs, get_resume_info, trigger_job_scout_agent, get_job_history,
                 search_job_by_title, tailor_resume_to_job, search_memory,
                 create_career_plan, abandon_career_plan]

    if surface == "chat":
        return all_tools
    return [t for t in all_tools if t.name not in GATED_TOOL_NAMES]
```

- [ ] **Step 4: Add the plan-status pre-load to the system prompt**

In `chatbot/agent.py`, add above `build_system_prompt`:

```python
def plan_status_summary(user_id: int) -> str:
    """One-line summary of the user's active plan, pre-loaded into the prompt.

    This replaces the get_career_plan_status tool. It is one cheap query that
    is useful context on every turn, which is the signal it belongs in the
    prompt rather than behind a tool call.
    """
    from chatbot.planner import get_active_plan, SYNTHESIS_MARKER
    from models import PlanStep

    plan = get_active_plan(user_id)
    if not plan:
        return "Active career plan: none."

    steps = PlanStep.query.filter_by(plan_id=plan.id).all()
    done = sum(1 for s in steps if s.status == 'done')
    synthesis = (PlanStep.query
                 .filter_by(plan_id=plan.id, status='done')
                 .filter(PlanStep.description.startswith(SYNTHESIS_MARKER))
                 .first())

    roadmap_note = (
        " The roadmap is ready — offer to walk them through it."
        if synthesis else
        " Still running."
    )
    return (
        f"Active career plan: '{plan.goal}' — {done}/{len(steps)} steps complete."
        f"{roadmap_note}"
    )
```

Change the signature and body of `build_system_prompt` to accept and embed it:

```python
def build_system_prompt(user, resume, agent_config, liked_count=0, disliked_count=0,
                        plan_status="Active career plan: none.") -> str:
```

and include `{plan_status}` in the returned f-string, inside `<trusted_instructions>` (the plan goal is user-authored, but it is the user's own text and already appears in their own chat history).

Update both callers to pass it. In `controllers/ws_chat_controller.py:114`:

```python
                from chatbot.agent import plan_status_summary
                system_prompt = build_system_prompt(
                    user, resume, config, liked_count, disliked_count,
                    plan_status=plan_status_summary(user_id),
                )
```

Apply the same change to the `build_system_prompt(` call in `chatbot/agent.py`. Find every caller:

```bash
grep -rn "build_system_prompt(" --include="*.py" .
```

- [ ] **Step 5: Add the feature blurb to the prompt**

Replace the deleted `explain_feature` dict with this condensed block inside the system prompt's `<trusted_instructions>`:

```
App features you can explain directly (no tool needed):
- Resume upload: PDF upload gives AI analysis, a vector index for Q&A, and a skills/experience summary.
- Job matching: FAISS vector search narrows candidates, then the LLM scores each against the resume for match score, matched skills, gaps, and recommendations.
- Job Scout Agent: runs on a schedule or on request; fetches from Adzuna, Remotive, Jobicy, RemoteOK, Himalayas, The Muse, and Arbeitnow, analyses against the resume, and saves high-quality matches. Configured from the Agent Dashboard.
- Resume Q&A: ask anything about the resume; answers come from its vector index.
- Interview roadmap: a phased prep plan with skills, resources, projects, milestones, and progressive questions.
- Job feedback: rating matches interested / not interested / applied teaches the system the user's preferences.
- Resume tailoring: ATS-optimizes the resume for a target job — keyword gaps, rewritten summary, reordered skills, reframed bullets.
- Agent config: schedule time, timezone, match threshold, max results, and Adzuna search preferences.
```

- [ ] **Step 6: Update the tool label map**

In `services/streaming.py:52`, replace `_TOOL_LABELS`:

```python
_TOOL_LABELS: dict[str, str] = {
    "find_top_jobs":           "Searching jobs…",
    "get_resume_info":         "Reading your resume…",
    "tailor_resume_to_job":    "Tailoring your resume…",
    "get_job_history":         "Reviewing your match history…",
    "search_job_by_title":     "Looking up job…",
    "trigger_job_scout_agent": "Preparing job scout run…",
    "search_memory":           "Searching memory…",
    "create_career_plan":      "Creating your career plan…",
    "abandon_career_plan":     "Checking your active plan…",
}
```

- [ ] **Step 7: Fix the stale reference in create_career_plan**

`create_career_plan`'s success message tells the model to "Use get_career_plan_status to check progress" — that tool no longer exists. In `chatbot/tools.py`, change that message to:

```python
                    "message": (
                        "Your plan has been created and is now executing autonomously in the "
                        "background. Progress appears in your context at the start of each "
                        "turn — just ask and I'll tell you where it's up to."
                    ),
```

- [ ] **Step 8: Run the suite**

Run: `python -m pytest tests/test_tool_registry.py -v && python -m pytest`
Expected: 3 passed in the new file; no regressions.

- [ ] **Step 9: Commit**

```bash
git add chatbot/tools.py chatbot/agent.py controllers/ws_chat_controller.py services/streaming.py tests/test_tool_registry.py
git commit -m "refactor: merge history tools, move explain_feature and plan status to prompt"
```

---

### Task 11: Rename the confusable job-search pair and regenerate the planner vocabulary (G2, G4)

**Files:**
- Modify: `chatbot/tools.py` (two renames)
- Modify: `chatbot/agent.py:182` (`_extract_intent` literal), `build_system_prompt`
- Modify: `services/streaming.py:52` (`_TOOL_LABELS`)
- Modify: `chatbot/planner.py:32` (`TOOL_DESCRIPTIONS`), `:56` (`PLANNER_PROMPT`)
- Test: `tests/test_tool_registry.py`, `tests/test_planner_vocabulary.py`

**Interfaces:**
- Consumes: the consolidated tool set from Task 10
- Produces: `find_jobs_matching_resume` (was `find_top_jobs`), `lookup_job_by_title` (was `search_job_by_title`); `chatbot.planner.tool_descriptions(app, user_id) -> str`

**Context:** Spec §4.2 pair 1 (rename, don't merge) and §3.5 G2/G4. `TOOL_DESCRIPTIONS` is a hand-maintained string listing 8 of the 12 tools — already drifted from reality — and `PLANNER_PROMPT` instructs the LLM to schedule `trigger_job_scout_agent`, which the planner surface no longer receives.

- [ ] **Step 1: Update the expected registry and add the planner vocabulary test**

In `tests/test_tool_registry.py`, change `EXPECTED_CHAT_TOOLS`:

```python
EXPECTED_CHAT_TOOLS = {
    "find_jobs_matching_resume",
    "lookup_job_by_title",
    "get_resume_info",
    "search_memory",
    "get_job_history",
    "tailor_resume_to_job",
    "create_career_plan",
    "trigger_job_scout_agent",
    "abandon_career_plan",
}
```

Create `tests/test_planner_vocabulary.py`:

```python
"""
tests/test_planner_vocabulary.py
================================
Feature tested: the planner's tool vocabulary (spec §3.5, G2 and G4).

Background
----------
TOOL_DESCRIPTIONS was a hand-maintained string listing 8 of the 12 tools —
already drifted — and PLANNER_PROMPT explicitly told the LLM to schedule
trigger_job_scout_agent as Phase 1 Step 4. The planner surface no longer
receives gated capabilities, so a plan naming one would just be skipped.
Generating the vocabulary from the registry makes drift impossible.

What each test covers
---------------------
test_tool_descriptions_lists_every_planner_tool
test_tool_descriptions_names_no_gated_capability
    The prompt must not invite the model to schedule something it cannot run.
test_planner_prompt_does_not_mention_gated_capabilities
test_unknown_tool_in_a_plan_step_degrades_gracefully
    A stale plan referencing a removed tool must skip, not crash.
"""

from chatbot.planner import tool_descriptions, PLANNER_PROMPT
from chatbot.tools import build_tools
from chatbot.gated_actions import GATED_TOOL_NAMES


def test_tool_descriptions_lists_every_planner_tool(app_sqlite):
    text = tool_descriptions(app_sqlite, 1)
    for tool in build_tools(app_sqlite, 1, surface="planner"):
        assert tool.name in text


def test_tool_descriptions_names_no_gated_capability(app_sqlite):
    text = tool_descriptions(app_sqlite, 1)
    for name in GATED_TOOL_NAMES:
        assert name not in text


def test_planner_prompt_does_not_mention_gated_capabilities():
    rendered = str(PLANNER_PROMPT)
    for name in GATED_TOOL_NAMES:
        assert name not in rendered


def test_unknown_tool_in_a_plan_step_degrades_gracefully(app_sqlite):
    tools_by_name = {t.name: t for t in build_tools(app_sqlite, 1, surface="planner")}
    assert "trigger_job_scout_agent" not in tools_by_name
    # This mirrors the guard at chatbot/planner.py:429 — an unrecognised
    # tool_name falls to the "Unknown tool ... Skipping." branch.
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_tool_registry.py tests/test_planner_vocabulary.py -v`
Expected: FAIL — `ImportError: cannot import name 'tool_descriptions'` and registry mismatch.

- [ ] **Step 3: Perform the renames**

In `chatbot/tools.py`:
- `def find_top_jobs(query: str)` → `def find_jobs_matching_resume(query: str)`
- `def search_job_by_title(title: str)` → `def lookup_job_by_title(title: str)`
- Update the `all_tools` list to the new names.
- Update the three `logger.error("find_top_jobs error: ...")` style strings to match.

Sharpen the two docstrings so they no longer overlap:

```python
    @tool
    def find_jobs_matching_resume(query: str) -> str:
        """Find jobs that SEMANTICALLY match the user's resume, ranked by fit. Use this when the user wants recommendations, matches, or 'jobs for me' — anything where the resume is the basis for the search. NOT for looking up one specific known job title; use lookup_job_by_title for that. The query parameter is exactly what the user said — include location, seniority, job type, and count if they mentioned them."""
```

```python
    @tool
    def lookup_job_by_title(title: str) -> str:
        """Look up specific job postings by LITERAL title or company name, returning their IDs. Use this when the user names a concrete role or employer — e.g. before tailoring a resume to it. NOT for resume-based recommendations; use find_jobs_matching_resume for that. Accepts 'AI Developer', 'AI Developer at Intellivon', or a company name alone."""
```

Also sharpen the pair the spec flags in §4.2 row 3:

```python
    @tool
    def get_resume_info(question: str) -> str:
        """Answer questions from the text of the user's uploaded RESUME DOCUMENT — their skills, experience, employment history, qualifications. NOT for things the user said in past chats; use search_memory for that."""
```

```python
    @tool
    def search_memory(query: str) -> str:
        """Recall what the user SAID IN PAST CONVERSATIONS — stated career goals, preferences, decisions, context they volunteered. NOT for resume content; use get_resume_info for that. Formulate the query as a short description of what to recall, e.g. 'user remote work preference'."""
```

- [ ] **Step 4: Update every hardcoded call site**

`chatbot/agent.py` `_extract_intent`:

```python
        elif tool_name == "find_jobs_matching_resume":
```

`services/streaming.py` `_TOOL_LABELS` — rename the two keys:

```python
    "find_jobs_matching_resume": "Searching jobs…",
    "lookup_job_by_title":       "Looking up job…",
```

Verify nothing was missed:

```bash
grep -rn "find_top_jobs\|search_job_by_title\|get_recent_matches\|get_user_preferences\|explain_feature\|get_career_plan_status" --include="*.py" --include="*.html" . | grep -v docs/
```

Expected: no results outside `docs/`. Anything remaining is a missed call site.

- [ ] **Step 5: Generate TOOL_DESCRIPTIONS from the registry**

In `chatbot/planner.py`, delete the `TOOL_DESCRIPTIONS` constant (lines 31-39) and add:

```python
def tool_descriptions(app, user_id: int) -> str:
    """Render the planner's tool vocabulary from the actual planner-surface registry.

    Generated rather than hand-maintained: the previous hardcoded string had
    already drifted to listing 8 of 12 tools, and it named a gated capability
    the planner surface cannot receive.
    """
    from chatbot.tools import build_tools

    lines = ["Available tools:"]
    for i, tool in enumerate(build_tools(app, user_id, surface="planner"), start=1):
        summary = (tool.description or "").split(". ")[0].strip()
        lines.append(f"{i}. {tool.name} — {summary}")
    return "\n".join(lines)
```

Update `generate_plan` to pass `tool_descriptions(app, user_id)` where it currently passes `TOOL_DESCRIPTIONS` into the `{tool_descriptions}` template variable.

- [ ] **Step 6: Remove the scout from PLANNER_PROMPT (G2)**

In `PLANNER_PROMPT`, replace the Phase 1 block:

```
PHASE 1 — Context Gathering (1 to 3 tool steps, in this order as needed):
  Step 1: get_resume_info — ask a specific question to extract the user's current skills and experience relevant to the target role
  Step 2: find_jobs_matching_resume — search for jobs matching the target role to understand what the market requires
  Step 3: get_job_history — check the user's existing matches and learned preferences for this role
```

and update the Rules block:

```
Rules:
- Total steps = Phase 1 steps (1-3) + 1 synthesis step. Do not add any other step types.
- tool_input for get_resume_info must be a specific question, e.g. "What are the user's skills in LangChain, Python, and cloud platforms relevant to an AI Engineer role?"
- tool_input for find_jobs_matching_resume must be a descriptive query, e.g. "Senior AI Agentic Engineer remote"
- tool_input for get_job_history must be a plain integer as a string, e.g. "10"
- Be specific in every description — name the actual role and technologies involved
```

- [ ] **Step 7: Run the suite**

Run: `python -m pytest -v`
Expected: all pass. Pay attention to `tests/test_agent_graph.py` and `tests/test_intent_extraction.py`, which may carry old tool names — update them if so.

- [ ] **Step 8: Commit**

```bash
git add chatbot/tools.py chatbot/agent.py chatbot/planner.py services/streaming.py tests/
git commit -m "refactor: rename confusable job tools, generate planner vocabulary from registry"
```

---

### Task 12: Prompt reduction and eval verification

**Files:**
- Modify: `chatbot/agent.py` `build_system_prompt`
- Test: `evals/chat_eval.py` (run, do not modify)

**Interfaces:**
- Consumes: everything from Tasks 10–11
- Produces: nothing

**Context:** Spec §4.5. The prompt enumerates all 12 tools in a numbered list *and* carries 16 numbered guidelines, 6 of which are pure "when the user asks X, use tool Y" routing that the tool schemas already declare.

- [ ] **Step 1: Delete the numbered tool list**

In `build_system_prompt`, remove the entire block from `You have access to the following tools to help the user:` through item `12. abandon_career_plan - ...`. The tool schemas are authoritative; the list is a second, drift-prone copy.

- [ ] **Step 2: Collapse the guidelines**

Replace the 16 numbered guidelines with these, dropping every pure routing restatement (old items 2, 5, 6, 10, 12, 13):

```
Guidelines:
1. Be friendly, professional, and encouraging. Keep responses concise but informative.
2. If the user has not uploaded a resume yet, guide them to do so before anything else.
3. After finding jobs, tell the user they can click the "View Matching Jobs" button.
4. If personalization is active, mention that results reflect their feedback.
5. If a tool returns an error, explain it helpfully and suggest a next step.
6. Use the user's career context from previous sessions to personalise advice. When the current conversation is sparse and you are giving personalised advice, check long-term memory first.
7. When tailoring a resume to a job: if the job appeared earlier in this conversation, reuse its job_id directly. Otherwise look the job up first, pick the best match, then tailor. Present the ATS score improvement, missing keywords, the rewritten summary, and the top rewritten bullets. Never ask the user to paste a job description — always search the database.
8. Use career plans only for goals that genuinely need multiple steps (role transitions, interview prep, roadmaps), never for single-step questions.
9. Some actions need the user's explicit confirmation. When a tool tells you a confirmation button has been shown, say so plainly and ask them to click it. Do not claim the action has happened, and do not call that tool again.
```

Guideline 9 is load-bearing for Task 7's propose-only wrappers — without it the model narrates completed actions that have not run.

- [ ] **Step 3: Verify the prompt still contains the required blocks**

Run:

```bash
python -c "
from factory import create_app
from chatbot.agent import build_system_prompt
app = create_app('test', skip_api_check=True)
with app.app_context():
    from models import User
    u = User(username='demo', email='d@e.com')
    p = build_system_prompt(u, None, None, 0, 0, plan_status='Active career plan: none.')
    assert '<trusted_instructions>' in p
    assert 'Active career plan' in p
    assert 'App features you can explain' in p
    assert 'find_jobs_matching_resume' not in p, 'tool list should be gone'
    print('OK — prompt length:', len(p))
"
```

Expected: `OK — prompt length: <n>` and no assertion error.

- [ ] **Step 4: Run the eval and compare against the baseline**

```bash
python evals/chat_eval.py 2>&1 | tee /tmp/chat-eval-after.txt
diff docs/superpowers/plans/chat-eval-baseline.txt /tmp/chat-eval-after.txt
```

Expected: pass rate equal to or better than the Task 1 baseline. **If it regressed, stop and report the diff — do not proceed.** Consolidation was justified on selection accuracy, so a drop invalidates the change.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add chatbot/agent.py
git commit -m "refactor: cut duplicated tool list and routing guidelines from system prompt"
```

---

### Task 13: Threat model document

**Files:**
- Create: `docs/THREAT_MODEL.md`

**Interfaces:**
- Consumes: the completed implementation
- Produces: nothing

**Context:** Spec §6. This is a deliverable in its own right, and the spec is explicit that recording *why a flow controller was not built* is the point — stating it, grounded in a data-flow trace, is a stronger artifact than building one the threat model does not justify.

- [ ] **Step 1: Write the document**

Create `docs/THREAT_MODEL.md` covering, in this order:

1. **Scope** — the chatbot and its tool surface; what is out of scope.
2. **Data-flow trace** — reproduce spec §1.3's table, *re-verified against the code as it now stands*. Do not copy blindly; confirm each claim.
3. **Existing structural mitigations** — closure-bound `user_id`; `<untrusted_data>` wrapping in `agents/`; structured Pydantic output; the escape-then-allow-bold-only sanitizer in `formatContent`; `services/input_guard.scan_message` on both entrypoints.
4. **Controls added by this work** — capability-level rate limits, the confirmation gate and its nonce properties, surface-scoped tool construction, the planner iteration cap.
5. **The XSS found and fixed in Task 9** — `attachActionButtons` interpolating external job titles into `insertAdjacentHTML`. Include it; a threat model that omits the one real vulnerability found during the work is not credible.
6. **Residual risks, ranked** — spec §1.4, updated to reflect what Tasks 2–12 closed. Ranking manipulation via spam job postings stays open; say so, and say why no published pattern addresses it.
7. **Patterns considered and rejected** — spec §1.5 verbatim, including the Flow Controller and why.
8. **References** — the three papers cited in the spec.

- [ ] **Step 2: Verify every code claim in the document**

For each factual claim about the code, run the check and confirm:

```bash
grep -rn "user_id" chatbot/tools.py | grep -c "filter_by"
grep -rn "untrusted_data" agents/
grep -n "escapeHtml" templates/chat_widget.html
grep -rn "build_tools(" --include="*.py" . | grep -v "def build_tools"
```

A threat model with a stale claim is worse than none. Fix any that no longer hold.

- [ ] **Step 3: Commit**

```bash
git add docs/THREAT_MODEL.md
git commit -m "docs: add chatbot threat model"
```

---

## Final Verification

- [ ] `python -m pytest` — full suite passes
- [ ] `python evals/chat_eval.py` — at or above the Task 1 baseline
- [ ] `grep -rn "find_top_jobs\|search_job_by_title\|get_recent_matches\|get_user_preferences\|explain_feature\|get_career_plan_status" --include="*.py" --include="*.html" . | grep -v docs/` — no results
- [ ] `grep -rn "build_tools(" --include="*.py" . | grep -v "def build_tools"` — every call passes `surface=`
- [ ] Manual: XSS probe from Task 9 Step 2 renders as literal text
- [ ] Manual: confirm button flow from Task 9 Step 3 works end to end
- [ ] Manual: the planner cannot schedule a gated capability — create a career plan and confirm no plan step names one

## Spec Coverage Map

Every requirement in the design spec, and the task that implements it. Use this to check nothing was dropped.

| Spec section | Requirement | Task |
|---|---|---|
| §2.2 C1 | WebSocket chat rate limit (20/min, 200/day) | 3 |
| §2.2 C2 | `trigger_job_scout_agent` 3/hour | 6 (consumed at confirm) |
| §2.2 C2 | `create_career_plan` 5/day | 4 |
| §2.2 C3 | Planner loop iteration cap | 4 |
| §3.1 | Gate scope: scout + abandon only | 6, 7 |
| §3.2 | Nonce store, propose/claim flow | 5, 7, 8 |
| §3.3 | Client sends only the nonce | 5, 8, 9 |
| §3.4 | Result bypasses the LLM; model told "shown", not "done" | 7 (tool `note`), 9, 12 (guideline 9) |
| §3.5 G1 | Surface-scoped `build_tools` | 7 |
| §3.5 G2 | Scout removed from planner vocabulary | 11 |
| §3.5 G3 | Fail loudly outside a confirmable surface | 7 |
| §3.5 G4 | `TOOL_DESCRIPTIONS` generated from registry | 11 |
| §4.2 | Three confusable pairs: rename, merge, sharpen | 10, 11 |
| §4.3 | Remove `explain_feature`, `get_career_plan_status` | 10 |
| §4.4 | Resulting 9-tool set | 10, 11 (registry test) |
| §4.5 | Prompt reduction | 12 |
| §4.6 | Hardcoded tool-name call sites | 10, 11 |
| §5 | Eval baseline before consolidation | 1, 12 |
| §6 | Threat model document | 13 |
| — | Stored XSS in `attachActionButtons` (found during planning, not in spec) | 9 |
