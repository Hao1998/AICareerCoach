# Chatbot Capability Hardening — Design

**Date:** 2026-08-13
**Status:** Approved, ready for implementation planning
**Scope:** `chatbot/`, `controllers/ws_chat_controller.py`, `controllers/chat_controller.py`, `services/streaming.py`, `templates/chat_widget.html`

---

## 1. Motivation and Threat Model

### 1.1 The original concern

The chatbot exposes 12 LangChain tools to a `create_react_agent` loop. The stated worry
was that a prompt injection could drive the agent to "run amok" — in particular to read or
modify **other users'** data.

### 1.2 What the code actually shows

That specific threat does not exist, and saying so plainly is more useful than building a
defence against it.

`build_tools(app, user_id, progress_cb)` closes over `user_id`. **No tool accepts a user
identifier as a parameter.** Every query is `filter_by(user_id=user_id)`. There is no
argument an injection could poison to cross the user boundary. The capability boundary for
tenancy already exists, enforced by closure rather than by prompt.

### 1.3 Where untrusted data actually flows

Tracing every ingress that reaches an LLM:

| Vector | Adversary-controlled? | Reaches the **chat** model? | Notes |
|---|---|---|---|
| Job descriptions (7 external boards) | **Yes** — the only true one | **No** | `find_top_jobs` returns `{id, title, company, location, match_score}`; `search_job_by_title` returns `{id, title, company, location}`; `get_recent_matches` returns titles + scores. Raw description text is never projected into chat context. |
| Resume text | No | Yes, via `get_resume_info` | User's own upload — self-injection only |
| Long-term memory | No | Yes, wrapped in `<untrusted_data>` | Derived from the user's own past messages |
| Chat input | No | Yes | The user, and `services/input_guard.scan_message` runs on both entrypoints |

Raw job text *does* reach an LLM — but only `JobAnalystAgent` and `ResumeTailoringAgent`.
Both already wrap it in `<untrusted_data>`, run under a narrow single-purpose system
prompt, hold **no tools**, and return Pydantic structured output. Untrusted content is
processed in isolation and only a schema-constrained result escapes — structurally the
LLM Map-Reduce pattern from Beurer-Kellner et al., already in place, at the layer where
the data actually enters.

### 1.4 The residual risk, ranked

1. **Unsupervised autonomous invocation.** `chatbot/planner.py` builds the full tool list
   and dispatches by LLM-chosen name inside an unbounded background daemon thread, with
   no user present. This — not the chat loop — is the genuine "run amok" surface. See §3.5.
2. **Cost / resource exhaustion.** Real today, with zero attackers. See §2.
3. **Unconfirmed destructive action.** `abandon_career_plan` destroys work with no undo
   and no confirmation. A confused model suffices; no attacker needed.
4. **Tool-selection error.** 12 tools with overlapping descriptions degrade routing
   accuracy. A reliability problem that presents as a correctness problem.
5. **Ranking manipulation.** A spam job posting stuffed with keywords to over-rank itself
   into a user's matches. Genuinely unmitigated — and *not* fixable by any of the six
   published design patterns, because producing a ranking requires reading the text.
   Accepted as out of scope; recorded here so it is not mistaken for an oversight.

### 1.5 Patterns considered and rejected

| Pattern | Why rejected |
|---|---|
| **Action-Selector** | Defined as agents that trigger tools but cannot act on responses — no feedback loop. The chatbot's entire value *is* the feedback loop (read tool output, discuss it). Mutually exclusive with the product. The paper also warns selector reliability degrades as action count grows, which cuts against this app specifically. |
| **Flow Controller / phase FSM** (per `AITicketRouting/CHATBOT_FLOW_ARCHITECTURE.md`) | Ticket deflection is a funnel with an end state, so its phase space is finite. Career coaching has no funnel — a literal transplant yields either one `AWAITING_ANYTHING` phase or a phase explosion. |
| **Dual LLM / CaMeL** | Requires the privileged model to never see untrusted content. Already effectively true here (§1.3), so the machinery buys nothing. |
| **Plan-Then-Execute** | Strong control-flow integrity, but heavy for one-shot turns like "what are my skills?", which are most of the traffic. |

**Conclusion:** the architectural boundary is already correctly placed. This design does
not relocate it. It closes the three concrete gaps in §1.4 (1–3) and documents the whole
picture, including what was deliberately *not* built.

---

## 2. Cost Containment

### 2.1 The defect

`/api/chat` carries `@limiter.limit("20 per minute; 200 per day")`. The **WebSocket path
— which is what the widget actually uses — has no rate limit at all**; `flask-limiter`
does not observe socket.io events. `acquire_stream_slot` (`services/streaming.py:34`)
bounds *concurrency* to one in-flight message per user via a Redis `SET NX EX 300`, but
not *frequency*: send, await `done`, send again, indefinitely.

Each such turn may invoke `trigger_job_scout_agent` (multi-source fetch + `grok-3` per
job) or `create_career_plan` (spawns a daemon thread running plan→execute→replan).

The primary entrypoint therefore has strictly weaker limits than the secondary one.

### 2.2 Changes

**C1 — Bring the WS path to parity.** Add a Redis-backed per-user counter in
`handle_chat_message`, checked *before* `acquire_stream_slot`, with the same
`20/min; 200/day` budget as the HTTP route. `services/redis_client.get_redis()` is already
a hard dependency on this path, so no new infrastructure. On exceed, `emit('error', ...)`
with the same message shape the client already renders.

**C2 — Per-capability budgets, enforced inside the capability.**

| Capability | Budget |
|---|---|
| `trigger_job_scout_agent` | 3 / hour / user |
| `create_career_plan` | 5 / day / user |

Enforced *inside the capability function*, not at the route. A limit on the route is
bypassed by every other caller; a limit at the capability is not. This matters concretely:
`chatbot/planner.py:377` builds and invokes tools directly (see §3.5), so a route-level
limit would not cover plan execution at all. Capability-level enforcement covers the HTTP
path, the WS path, and the planner uniformly.

On exceed, return the existing `{"success": false, "error": "..."}` shape so the model
explains the limit conversationally rather than surfacing an exception.

**C3 — Bound the planner loop.** `execute_plan` runs in an unsupervised daemon thread
(`chatbot/tools.py:481-495`). Add a hard integer cap on total steps and replan iterations,
persisted on `TaskPlan`, so a runaway plan terminates deterministically instead of
consuming LLM budget until the process dies.

---

## 3. Confirmation Gate

### 3.1 Scope

| Capability | Gate | Rationale |
|---|---|---|
| `abandon_career_plan` | **Confirm** + rate limit | Destroys work, no undo |
| `trigger_job_scout_agent` | **Confirm** + rate limit | Real money per run |
| `create_career_plan` | Rate limit only | Costly but constructive; already refuses when a plan is active |
| `tailor_resume_to_job` | Rate limit only | Overwrites `JobMatch.tailoring_result`, but that is derived and regenerable |

### 3.2 Mechanism

Gated tools **remain in the model's tool list** — the model must still know they exist and
be able to propose them. What changes is that invoking one no longer executes it.

```
model calls abandon_career_plan(reason="…")
  │
  ├─ tool does NOT execute. It:
  │    1. resolves arguments server-side (looks up the active plan and its goal)
  │    2. stores {user_id, capability, resolved_args} in Redis under a nonce, TTL 5 min
  │    3. returns to the model:
  │       "Confirmation required. The user has been shown a button.
  │        Do not call this tool again."
  │
  ├─ controller emits intent='confirm_required',
  │    action_data={nonce, label: "Abandon plan 'Transition to ML Engineer'?"}
  │
  ├─ attachActionButtons() renders the button
  │
  └─ user clicks → POST /api/chat/confirm {nonce}
       endpoint (@login_required):
         load nonce from Redis
         assert stored user_id == current_user.id
         delete nonce (single-use)
         execute capability with the STORED args
```

### 3.3 The property that matters

**The client transmits only the nonce.** It cannot supply the capability name or the
arguments — those were fixed server-side at proposal time. Nothing downstream (the model,
injected text, or a tampered client) can alter what actually runs. Single-use plus TTL
defeats replay. This is the same reasoning that puts rate limits at the capability rather
than the route: the authority lives with the code that holds it, not with any caller.

### 3.4 Two deliberate choices

- **The confirmed result does not re-enter the LLM.** The endpoint executes and returns a
  rendered result string, which the client appends as an assistant message directly. One
  fewer LLM call, fully deterministic, and the outcome of a destructive action cannot be
  misreported by a hallucination.
- **The model is told the button was shown, not that the action succeeded.** Otherwise it
  narrates "I've abandoned your plan" before the user has clicked.

### 3.5 The planner is a second invocation path — and it breaks the gate

`chatbot/planner.py` invokes tools independently of the chat agent:

- `execute_plan` calls `build_tools(app, user_id)` (`planner.py:377`) and receives the
  **full** tool list, gated capabilities included.
- `PLANNER_PROMPT` explicitly instructs the LLM to include `trigger_job_scout_agent` as
  Phase 1 Step 4 (`planner.py:56`).
- It runs in a background daemon thread. **No user is present to click a button.**
- It dispatches by LLM-chosen `tool_name` string (`planner.py:423`).

Under §3.2 as written, a plan step invoking a gated capability would mint a Redis nonce,
emit to no socket, and return "confirmation required" to a replan loop that can never
satisfy it — retrying indefinitely on LLM budget. This is the single clearest instance of
the "run amok" concern that motivated this work, and it lives in the *autonomous* path,
not the chat path.

**Fixes:**

**G1 — Surface-scoped tool construction.** `build_tools(app, user_id, surface=...)` where
`surface` is `'chat'` or `'planner'`. The `'planner'` surface omits gated capabilities
entirely, so `tools_by_name` cannot contain them. Capability metadata declares which
surfaces may invoke it; the surface filter is the enforcement point.

**G2 — Remove the scout from the planner's vocabulary.** Drop
`trigger_job_scout_agent` from `TOOL_DESCRIPTIONS` (`planner.py:32`) and from
`PLANNER_PROMPT`'s Phase 1 Step 4. Plans rely on existing matches via `get_job_history`.
The existing `elif next_step.tool_name:` branch (`planner.py:429`) already degrades an
unrecognised tool to `"Unknown tool … Skipping."` rather than crashing, so a stale plan
referencing the removed tool fails safe.

**G3 — Fail loudly outside a confirmable surface.** Defence in depth: if a gated
capability is somehow invoked where no confirmation channel exists, it raises rather than
creating an orphan nonce. G1 should make this unreachable; it exists so that a future
caller which forgets `surface` fails visibly instead of silently no-opping.

**G4 — Derive `TOOL_DESCRIPTIONS` from the registry.** It is currently a hand-maintained
string listing 8 of the 12 tools — already drifted. Generate it from the tool objects so
it cannot drift again after §4's renames.

### 3.6 Honest characterisation

This is a **consent and reliability** control, not primarily a security one. Given §1.3,
the failure mode it prevents is mostly "the model got confused," not "an attacker got in."
That is still worth preventing, and it should be documented as such.

---

## 4. Tool Consolidation

### 4.1 Principle

Tool-selection errors arise from **confusable pairs**, not from raw count. Merge where
descriptions overlap; rename where concepts are distinct but names collide; move to the
prompt anything that is not really a tool. Consolidating past that point trades a real
accuracy gain for a smaller number.

Result: **12 → 9 tools**, plus a materially smaller system prompt.

### 4.2 The three confusable pairs

| Pair | Failure mode | Fix |
|---|---|---|
| `find_top_jobs` / `search_job_by_title` | Both read as "search for jobs" | **Rename, don't merge** — semantic resume-match vs. literal title lookup are genuinely different operations. → `find_jobs_matching_resume` / `lookup_job_by_title` |
| `get_recent_matches` / `get_user_preferences` | Both query `JobMatch` feedback history with overlapping payloads | **Merge** → `get_job_history()` returns recent matches *and* learned preferences in one payload |
| `get_resume_info` / `search_memory` | Both read as "recall something about the user" | **Sharpen descriptions** — one reads the resume document, the other reads past conversations. Make the boundary explicit in each docstring. |

### 4.3 Two removals

- **`explain_feature`** — a static dict lookup with no DB or LLM call. It is a constant,
  not a tool. Condense to ~150 words in the system prompt.
- **`get_career_plan_status`** — one cheap deterministic query that is useful context on
  *every* turn. Pre-load active-plan status into the system prompt instead. Needing it
  every turn is precisely the signal that it belongs in the prompt rather than behind a
  call.

### 4.4 Resulting tool set

```
reads    find_jobs_matching_resume · lookup_job_by_title
         get_resume_info · search_memory · get_job_history
writes   tailor_resume_to_job · create_career_plan          (rate-limited)
gated    trigger_job_scout_agent · abandon_career_plan      (confirm + rate-limited)
```

### 4.5 Prompt reduction

`build_system_prompt` (`chatbot/agent.py:42`) currently enumerates all 12 tools in a
numbered list **and** carries 16 numbered guidelines — much of it restating what the tool
schemas already declare, with 6 of the 16 guidelines being pure "when the user asks X, use
tool Y" routing. The model is told the same thing twice in two formats.

- Delete the numbered tool list entirely; the tool schemas are authoritative.
- Collapse the guidelines to those encoding real policy, dropping the routing restatements.
- Add the pre-loaded plan status (§4.3) and the condensed feature blurb (§4.3).

### 4.6 Call sites that hardcode tool names

Renames require updating:

| Location | What |
|---|---|
| `chatbot/agent.py:182` `_extract_intent` | Matches `"find_top_jobs"` and `"tailor_resume_to_job"` as string literals |
| `services/streaming.py:52` `_TOOL_LABELS` | Maps 9 of the 12 tool names to progress labels; unmapped names fall back to `f"Running {name}…"` |
| `chatbot/agent.py:42` `build_system_prompt` | The numbered tool list being deleted (§4.5) |
| `chatbot/planner.py:32` `TOOL_DESCRIPTIONS` | Hand-maintained list of 8 of the 12 tools — replaced by generation from the registry per §3.5 G4 |
| `chatbot/planner.py:56` `PLANNER_PROMPT` | Names `get_resume_info`, `find_top_jobs`, `get_recent_matches`, `trigger_job_scout_agent` inline in the Phase 1 step template |

---

## 5. Testing and Verification

| Area | Verification |
|---|---|
| Rate limits (C1) | Unit test with a faked Redis against the WS handler: N messages succeed, N+1 emits the limit error. The HTTP route's existing `flask-limiter` config is unchanged and already covered. |
| Rate limits (C2) | Unit tests per capability with a faked Redis: N calls succeed, N+1 returns `{"success": false, …}`. Assert the limit holds when the capability is invoked from the planner path, not just from chat. |
| Planner bound (C3) | Test that a plan exceeding the step cap terminates and records terminal status. |
| Confirmation gate | Unit tests: gated tool returns pending without executing; nonce is single-use; a nonce belonging to user A is rejected for user B; expired nonce is rejected; client-supplied capability/args are ignored. |
| Surface scoping (G1–G3) | Assert `build_tools(surface='planner')` omits every gated capability; assert a gated capability invoked without a confirmation channel raises rather than minting a nonce. |
| Tool consolidation | `evals/chat_eval.py` per CLAUDE.md — covers `chatbot/` and `chatbot/tools.py`, both touched throughout §4. |
| Regression | `python -m pytest` |

**Baseline requirement.** Record a `chat_eval.py` run *before* any §4 change. Without a
baseline, a consolidation-caused regression can only be inferred, not observed.

---

## 6. Deliverable: Threat Model Document

Separate from the code changes, produce a threat-model document capturing §1: every
ingress, what it reaches, what is already mitigated and by which mechanism, and the
residual risks ranked — **including §1.5, the patterns considered and rejected, with
reasons**.

Stating why a flow controller was *not* built, grounded in a data-flow trace, is a stronger
engineering artifact than building one that the threat model does not justify. It also
pre-empts the obvious review question rather than inviting it.

---

## 7. Explicitly Out of Scope

- **Flow Controller / Router→Controller→Renderer rewrite.** Rejected in §1.5. The
  capability boundary it would establish already exists.
- **Ranking manipulation via spam job postings** (§1.4 item 5). Real and unmitigated;
  not addressable by these patterns. Recorded, accepted, not fixed here.
- **Cross-user access controls.** Already enforced by closure-bound `user_id` (§1.2).
- **Changes to `agents/`.** The `<untrusted_data>` wrapping and structured-output
  discipline there are already correct (§1.3).

---

## References

- Beurer-Kellner et al., [*Design Patterns for Securing LLM Agents against Prompt Injections*](https://arxiv.org/abs/2506.08837) (arXiv:2506.08837)
- Willison, [*Design Patterns for Securing LLM Agents against Prompt Injections*](https://simonwillison.net/2025/Jun/13/prompt-injection-design-patterns/)
- Debenedetti et al., [*Defeating Prompt Injections by Design* (CaMeL)](https://arxiv.org/abs/2503.18813) (arXiv:2503.18813)
- `AITicketRouting/CHATBOT_FLOW_ARCHITECTURE.md` — the Flow Controller pattern evaluated in §1.5
