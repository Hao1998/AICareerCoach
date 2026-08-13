"""
Long-Horizon Task Planner

Three-phase loop: Plan → Execute → Replan
  - Planner:   LLM generates a multi-step plan from a user goal + context
  - Executor:  Walks through steps, calling existing tools or LLM reasoning
  - Replanner: After each step, LLM reviews results and adjusts remaining steps

The plan is persisted in TaskPlan / PlanStep so it survives across sessions.
"""

import json
import logging
import traceback
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from models import PlanStep, TaskPlan, db
from schemas.output_schemas import CareerRoadmap, PlanResult, ReplanResult
from services.db_lock import safe_commit, safe_flush

# Marker used to identify the final synthesis step
SYNTHESIS_MARKER = "SYNTHESIZE:"

# Hard ceiling on the plan -> execute -> replan loop. execute_plan runs in an
# unsupervised daemon thread, so this is the only thing bounding its LLM
# spend. Kept at the value the loop has always used.
MAX_PLAN_ITERATIONS = 15

logger = logging.getLogger(__name__)

# ── Available tools the planner can reference ─────────────────────────────────

TOOL_DESCRIPTIONS = """Available tools:
1. get_resume_info(question: str) — Answer questions about the user's resume, skills, experience
2. find_top_jobs(query: str) — Find top 5 matching jobs based on resume
3. get_recent_matches(limit: int) — Get recent job match history
4. trigger_job_scout_agent(reason: str) — Run the job scout to fetch and match new jobs
5. search_job_by_title(title: str) — Search the job database by title
6. tailor_resume_to_job(job_id: int) — ATS-optimize resume for a specific job
7. search_memory(query: str) — Search long-term memory of past conversations
8. get_user_preferences(dummy: str) — Show learned job preferences from feedback"""


# ── Planner ───────────────────────────────────────────────────────────────────

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a career coaching task planner. Generate a two-phase plan to help a user transition into a target role.

{tool_descriptions}

Your plan MUST follow this exact structure — no exceptions:

PHASE 1 — Context Gathering (1 to 4 tool steps, in this order as needed):
  Step 1: get_resume_info — ask a specific question to extract the user's current skills and experience relevant to the target role
  Step 2: find_top_jobs — search for jobs matching the target role to understand what the market requires
  Step 3: get_recent_matches — check if the user already has good job matches for this role
  Step 4: trigger_job_scout_agent — ONLY include if fresh job data is needed (i.e. step 3 might return stale or insufficient matches)

PHASE 2 — Synthesis (ALWAYS exactly one final step):
  Last step: tool_name must be null, description must start with "SYNTHESIZE: " followed by a one-line summary of what to synthesize.
  This step will use all Phase 1 results to produce a complete career roadmap.

Rules:
- Total steps = Phase 1 steps (1-4) + 1 synthesis step. Do not add any other step types.
- tool_input for get_resume_info must be a specific question, e.g. "What are Hao's skills in LangChain, Python, and cloud platforms relevant to an AI Engineer role?"
- tool_input for find_top_jobs must be a descriptive query, e.g. "Senior AI Agentic Engineer remote"
- tool_input for get_recent_matches must be a plain integer as a string, e.g. "10"
- tool_input for trigger_job_scout_agent must be a short reason string
- Be specific in every description — name the actual role and technologies involved
"""),
    ("human", """User goal: {goal}

User context:
{user_context}

Previous memories (if any):
{memories}

Generate the Phase 1 + Phase 2 plan."""),
])


@traceable(run_type="chain", name="task-planner-generate")
def generate_plan(app, user_id: int, goal: str, llm) -> TaskPlan:
    """Generate a new TaskPlan from a user goal. Persists to DB."""
    from chatbot.memory import search_memories
    from chatbot.agent import _load_context

    with app.app_context():
        user, resume, config, liked_count, disliked_count = _load_context(app, user_id)

        user_context_parts = [f"Name: {user.full_name or user.username}"]
        if resume and resume.analysis:
            user_context_parts.append(f"Resume summary: {resume.analysis[:400]}")
        if config and config.conversation_summary:
            user_context_parts.append(f"Session history: {config.conversation_summary[:300]}")
        if config and config.explicit_preferences:
            user_context_parts.append(f"Preferences: {json.dumps(config.explicit_preferences)}")
        user_context = "\n".join(user_context_parts)

        try:
            memories = search_memories(user_id, goal, top_k=3)
        except Exception:
            memories = "No memories available."

        chain = PLANNER_PROMPT | llm.with_structured_output(PlanResult)
        plan_result: PlanResult = chain.invoke({
            "goal": goal,
            "user_context": user_context,
            "memories": memories,
            "tool_descriptions": TOOL_DESCRIPTIONS,
        })

        task_plan = TaskPlan(user_id=user_id, goal=goal, status='active')
        db.session.add(task_plan)
        safe_flush()

        for step in plan_result.steps:
            db.session.add(PlanStep(
                plan_id=task_plan.id,
                step_order=step.step_order,
                description=step.description,
                tool_name=step.tool_name,
                tool_input=step.tool_input,
                status='pending',
            ))

        safe_commit()
        logger.info("Created plan %d with %d steps for user %d: %s",
                     task_plan.id, len(plan_result.steps), user_id, goal)
        return task_plan


# ── Executor ──────────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="task-planner-execute-step")
def execute_step(app, user_id: int, step: PlanStep, tools_by_name: dict, llm) -> str:
    """Execute a single PlanStep. Returns a result summary string."""
    with app.app_context():
        step.status = 'running'
        safe_commit()

        if step.tool_name and step.tool_name in tools_by_name:
            tool_fn = tools_by_name[step.tool_name]
            tool_input = step.tool_input or ""
            try:
                result = tool_fn.invoke(tool_input)
            except Exception as e:
                result = f"Tool error: {e}"
                logger.error("Step %d tool '%s' failed: %s", step.id, step.tool_name, e)
        elif step.tool_name and step.tool_name not in tools_by_name:
            result = f"Unknown tool '{step.tool_name}'. Skipping."
            logger.warning("Plan step %d references unknown tool: %s", step.id, step.tool_name)
        else:
            reasoning_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a career coach. Analyze the information and provide insights."),
                ("human", "{task}"),
            ])
            chain = reasoning_prompt | llm
            response = chain.invoke({"task": step.description})
            result = response.content if hasattr(response, 'content') else str(response)

        if len(result) > 2000:
            result = result[:2000] + "... (truncated)"

        step.status = 'done'
        step.result_summary = result
        step.completed_at = datetime.utcnow()
        safe_commit()

        logger.info("Completed step %d (tool=%s) for plan step_order=%d",
                     step.id, step.tool_name or 'reasoning', step.step_order)
        return result


# ── Roadmap Synthesis ─────────────────────────────────────────────────────────

ROADMAP_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior career coach producing a personalised career transition roadmap.
You have already gathered all context needed (resume analysis, job market data, existing matches).
Your job is to synthesise everything into one clear, actionable roadmap the user can follow immediately.

Output a CareerRoadmap with these sections:
- current_state: Where the user is today — their relevant skills, years of experience, and key strengths
- target_state: What the target role actually requires — skills, experience level, domain knowledge
- skill_gaps: Bullet list of specific missing skills or experience areas
- strengths: Bullet list of existing skills that transfer directly to the target role
- learning_path: 2-4 time-boxed phases, each with a focus topic, specific resources (named courses, certifications, or projects), and a measurable milestone
- application_strategy: Concrete advice on when to start applying, what roles to target first as stepping stones, and how to position themselves
- target_companies: Specific company names or types drawn from the job search results
- resume_tips: 3-5 specific ATS and content tips tailored to this exact role
- timeline_summary: A single paragraph summarising the full journey from today to landing the target role

Be specific — name actual courses (e.g. "DeepLearning.AI LangChain course"), real frameworks, real companies.
Do not be generic. Every recommendation must be grounded in the gathered context below.
"""),
    ("human", """Career goal: {goal}

User context:
{user_context}

Gathered context from research steps:
{gathered_context}

Produce the career roadmap now."""),
])


def _format_roadmap(roadmap: CareerRoadmap) -> str:
    """Convert a CareerRoadmap schema into a readable markdown string for storage."""
    lines = []

    lines.append("## Where You Are Now")
    lines.append(roadmap.current_state)

    lines.append("\n## Where You Need to Be")
    lines.append(roadmap.target_state)

    lines.append("\n## Your Strengths (Already Have)")
    for s in roadmap.strengths:
        lines.append(f"- {s}")

    lines.append("\n## Skill Gaps (Need to Learn)")
    for g in roadmap.skill_gaps:
        lines.append(f"- {g}")

    lines.append("\n## Learning Path")
    for phase in roadmap.learning_path:
        lines.append(f"\n**{phase.timeframe} — {phase.focus}**")
        for r in phase.resources:
            lines.append(f"  - {r}")
        lines.append(f"  *Milestone: {phase.milestone}*")

    lines.append("\n## Job Application Strategy")
    lines.append(roadmap.application_strategy)

    if roadmap.target_companies:
        lines.append("\n## Companies to Target")
        for c in roadmap.target_companies:
            lines.append(f"- {c}")

    lines.append("\n## Resume Tips for This Role")
    for tip in roadmap.resume_tips:
        lines.append(f"- {tip}")

    lines.append("\n## Timeline Summary")
    lines.append(roadmap.timeline_summary)

    return "\n".join(lines)


# ── Replanner ─────────────────────────────────────────────────────────────────

REPLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a career coaching task replanner. Review the progress so far and decide whether the remaining steps need adjustment.

{tool_descriptions}

Rules:
- If the plan is on track, return the remaining steps unchanged.
- If new information changes what's needed, add/remove/modify steps.
- If the goal is already achieved, set is_complete = true and return an empty steps list.
- Renumber updated_steps starting from 1.
- Keep the plan focused — don't add unnecessary steps.
"""),
    ("human", """Original goal: {goal}

Completed steps and results:
{completed_summary}

Remaining steps (before adjustment):
{remaining_summary}

Latest step result:
{latest_result}

Should the remaining plan be adjusted? Return the updated remaining steps."""),
])


@traceable(run_type="chain", name="task-planner-replan")
def replan(app, plan: TaskPlan, latest_result: str, llm) -> bool:
    """Review and optionally adjust remaining steps. Returns True if plan is complete."""
    with app.app_context():
        all_steps = PlanStep.query.filter_by(plan_id=plan.id).order_by(PlanStep.step_order).all()
        completed = [s for s in all_steps if s.status == 'done']
        remaining = [s for s in all_steps if s.status == 'pending']

        if not remaining:
            plan.status = 'completed'
            safe_commit()
            return True

        completed_summary = "\n".join(
            f"Step {s.step_order}: {s.description}\n  Result: {(s.result_summary or 'N/A')[:300]}"
            for s in completed
        ) or "None yet."

        remaining_summary = "\n".join(
            f"Step {s.step_order}: {s.description} (tool: {s.tool_name or 'reasoning'})"
            for s in remaining
        )

        chain = REPLAN_PROMPT | llm.with_structured_output(ReplanResult)
        replan_result: ReplanResult = chain.invoke({
            "goal": plan.goal,
            "completed_summary": completed_summary,
            "remaining_summary": remaining_summary,
            "latest_result": latest_result[:500],
            "tool_descriptions": TOOL_DESCRIPTIONS,
        })

        if replan_result.is_complete:
            for s in remaining:
                s.status = 'skipped'
            plan.status = 'completed'
            safe_commit()
            logger.info("Plan %d marked complete by replanner", plan.id)
            return True

        max_order = max((s.step_order for s in completed), default=0)
        new_steps = replan_result.updated_steps

        for i, old_step in enumerate(remaining):
            if i < len(new_steps):
                old_step.step_order = max_order + new_steps[i].step_order
                old_step.description = new_steps[i].description
                old_step.tool_name = new_steps[i].tool_name
                old_step.tool_input = new_steps[i].tool_input
            else:
                old_step.status = 'skipped'

        for i in range(len(remaining), len(new_steps)):
            db.session.add(PlanStep(
                plan_id=plan.id,
                step_order=max_order + new_steps[i].step_order,
                description=new_steps[i].description,
                tool_name=new_steps[i].tool_name,
                tool_input=new_steps[i].tool_input,
                status='pending',
            ))

        safe_commit()
        logger.info("Replanned plan %d: %d remaining steps", plan.id, len(new_steps))
        return False


# ── Main loop: Plan → Execute → Replan ────────────────────────────────────────

def execute_plan(app, user_id: int, plan_id: int, progress_cb=None) -> dict:
    """Execute an already-generated plan step by step.

    Runs inside a SINGLE app context for the entire loop so that
    Flask-SQLAlchemy never removes the shared session mid-execution.
    Designed to be called from a background daemon thread.
    """
    from services.llm_service import get_llm
    from chatbot.tools import build_tools

    def _progress(msg):
        if progress_cb:
            progress_cb(msg)

    logger.info("[execute_plan] ▶ START plan_id=%d user_id=%d", plan_id, user_id)

    try:
        with app.app_context():
            # ── bootstrap ────────────────────────────────────────────────────
            plan = TaskPlan.query.get(plan_id)
            if not plan:
                logger.error("[execute_plan] ✗ Plan %d not found in DB", plan_id)
                return {}

            logger.info("[execute_plan] Plan found — goal='%s' status=%s", plan.goal, plan.status)

            llm = get_llm()
            logger.info("[execute_plan] LLM ready")

            tools = build_tools(app, user_id)
            tools_by_name = {t.name: t for t in tools}
            logger.info("[execute_plan] Tools built: %s", list(tools_by_name.keys()))

            # Build user context string once — reused by the synthesis step
            from chatbot.agent import _load_context
            user, resume, config, _, _ = _load_context(app, user_id)
            user_context_parts = [f"Name: {user.full_name or user.username}"]
            if resume and resume.analysis:
                user_context_parts.append(f"Resume summary: {resume.analysis[:500]}")
            if config and config.explicit_preferences:
                user_context_parts.append(f"Preferences: {json.dumps(config.explicit_preferences)}")
            user_context_str = "\n".join(user_context_parts)

            step_results = []

            for iteration in range(MAX_PLAN_ITERATIONS):
                # ── pick next pending step ───────────────────────────────────
                next_step = (PlanStep.query
                             .filter_by(plan_id=plan_id, status='pending')
                             .order_by(PlanStep.step_order)
                             .first())

                if not next_step:
                    plan.status = 'completed'
                    safe_commit()
                    logger.info("[execute_plan] ✓ No more pending steps — plan %d completed", plan_id)
                    break

                logger.info(
                    "[execute_plan] Iteration %d — step %d: '%s' (tool=%s)",
                    iteration, next_step.step_order, next_step.description, next_step.tool_name,
                )
                _progress(f"Step {next_step.step_order}: {next_step.description}")

                # ── mark running ─────────────────────────────────────────────
                next_step.status = 'running'
                safe_commit()

                # ── execute ──────────────────────────────────────────────────
                result = ""
                if next_step.tool_name and next_step.tool_name in tools_by_name:
                    tool_fn = tools_by_name[next_step.tool_name]
                    tool_input = next_step.tool_input or ""
                    try:
                        result = tool_fn.invoke(tool_input)
                        logger.info("[execute_plan] Step %d tool '%s' ✓", next_step.step_order, next_step.tool_name)
                    except Exception as tool_exc:
                        result = f"Tool error: {tool_exc}"
                        logger.error("[execute_plan] Step %d tool '%s' ✗: %s",
                                     next_step.step_order, next_step.tool_name, tool_exc)
                elif next_step.tool_name:
                    result = f"Unknown tool '{next_step.tool_name}'. Skipping."
                    logger.warning("[execute_plan] Step %d unknown tool '%s'",
                                   next_step.step_order, next_step.tool_name)
                elif next_step.description.startswith(SYNTHESIS_MARKER):
                    # ── roadmap synthesis step ────────────────────────────────
                    logger.info("[execute_plan] Step %d is synthesis — building roadmap", next_step.step_order)
                    _progress("Building your career roadmap...")
                    try:
                        # Collect all prior done-step results as context
                        done_steps = (PlanStep.query
                                      .filter(PlanStep.plan_id == plan_id,
                                              PlanStep.status == 'done')
                                      .order_by(PlanStep.step_order)
                                      .all())
                        gathered_context = "\n\n".join(
                            f"[Step {s.step_order} — {s.description}]\n{s.result_summary or 'No result'}"
                            for s in done_steps
                        ) or "No context gathered."

                        chain = ROADMAP_SYNTHESIS_PROMPT | llm.with_structured_output(CareerRoadmap)
                        roadmap: CareerRoadmap = chain.invoke({
                            "goal": plan.goal,
                            "user_context": user_context_str,
                            "gathered_context": gathered_context,
                        })
                        result = _format_roadmap(roadmap)
                        logger.info("[execute_plan] Step %d synthesis ✓ (%d chars)", next_step.step_order, len(result))
                    except Exception as synth_exc:
                        result = f"Reasoning error: {synth_exc}"
                        logger.error("[execute_plan] Step %d synthesis ✗: %s",
                                     next_step.step_order, synth_exc)
                else:
                    # reasoning-only step (non-synthesis)
                    reasoning_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a career coach. Analyze the information and provide insights."),
                        ("human", "{task}"),
                    ])
                    try:
                        resp = (reasoning_prompt | llm).invoke({"task": next_step.description})
                        result = resp.content if hasattr(resp, 'content') else str(resp)
                        logger.info("[execute_plan] Step %d reasoning ✓", next_step.step_order)
                    except Exception as reason_exc:
                        result = f"Reasoning error: {reason_exc}"
                        logger.error("[execute_plan] Step %d reasoning ✗: %s",
                                     next_step.step_order, reason_exc)

                # Truncate tool/reasoning results but not the synthesis roadmap
                is_synthesis = next_step.description.startswith(SYNTHESIS_MARKER)
                if not is_synthesis and len(result) > 2000:
                    result = result[:2000] + "... (truncated)"

                # ── mark done ────────────────────────────────────────────────
                next_step.status = 'done'
                next_step.result_summary = result
                next_step.completed_at = datetime.utcnow()
                safe_commit()
                logger.info("[execute_plan] Step %d saved as done", next_step.step_order)

                step_results.append({
                    "step": next_step.step_order,
                    "description": next_step.description,
                    "tool": next_step.tool_name,
                    "result_preview": result[:300],
                })

                # ── replan (only on failure) ──────────────────────────────────
                # If the step succeeded, trust the original plan and move to the
                # next step. Calling the replanner after every success causes the
                # LLM to regenerate the same step in a loop (observed in logs).
                step_failed = result.startswith((
                    "Tool error:", "Unknown tool", "Reasoning error:"
                ))

                if not step_failed:
                    remaining_count = (PlanStep.query
                                       .filter_by(plan_id=plan_id, status='pending')
                                       .count())
                    if remaining_count == 0:
                        plan.status = 'completed'
                        safe_commit()
                        logger.info("[execute_plan] ✓ No remaining steps — plan %d completed", plan_id)
                        break
                    logger.info("[execute_plan] Step succeeded — skipping replan, %d steps left", remaining_count)
                    continue  # straight to next iteration

                # Step failed → ask LLM to adjust the remaining steps
                logger.info("[execute_plan] Step failed — calling replanner")
                _progress("Adjusting plan after error...")
                try:
                    all_steps_now = (PlanStep.query
                                     .filter_by(plan_id=plan_id)
                                     .order_by(PlanStep.step_order)
                                     .all())
                    completed = [s for s in all_steps_now if s.status == 'done']
                    remaining = [s for s in all_steps_now if s.status == 'pending']

                    if not remaining:
                        plan.status = 'completed'
                        safe_commit()
                        logger.info("[execute_plan] ✓ No remaining steps after failure — plan %d done", plan_id)
                        break

                    completed_summary = "\n".join(
                        f"Step {s.step_order}: {s.description}\n  Result: {(s.result_summary or 'N/A')[:200]}"
                        for s in completed
                    ) or "None yet."
                    remaining_summary = "\n".join(
                        f"Step {s.step_order}: {s.description} (tool: {s.tool_name or 'reasoning'})"
                        for s in remaining
                    )

                    chain = REPLAN_PROMPT | llm.with_structured_output(ReplanResult)
                    replan_result: ReplanResult = chain.invoke({
                        "goal": plan.goal,
                        "completed_summary": completed_summary,
                        "remaining_summary": remaining_summary,
                        "latest_result": result[:500],
                        "tool_descriptions": TOOL_DESCRIPTIONS,
                    })

                    if replan_result.is_complete:
                        for s in remaining:
                            s.status = 'skipped'
                        plan.status = 'completed'
                        safe_commit()
                        logger.info("[execute_plan] ✓ Replanner marked plan %d complete", plan_id)
                        break

                    max_order = max((s.step_order for s in completed), default=0)
                    new_steps = replan_result.updated_steps
                    for i, old_step in enumerate(remaining):
                        if i < len(new_steps):
                            old_step.step_order = max_order + new_steps[i].step_order
                            old_step.description = new_steps[i].description
                            old_step.tool_name = new_steps[i].tool_name
                            old_step.tool_input = new_steps[i].tool_input
                        else:
                            old_step.status = 'skipped'
                    for i in range(len(remaining), len(new_steps)):
                        db.session.add(PlanStep(
                            plan_id=plan_id,
                            step_order=max_order + new_steps[i].step_order,
                            description=new_steps[i].description,
                            tool_name=new_steps[i].tool_name,
                            tool_input=new_steps[i].tool_input,
                            status='pending',
                        ))
                    safe_commit()
                    logger.info("[execute_plan] Replanned after failure — %d remaining steps", len(new_steps))

                except Exception as replan_exc:
                    logger.error("[execute_plan] Replan failed (continuing anyway): %s", replan_exc)

            else:
                # Every terminal path inside the loop breaks. Reaching here means
                # the iteration budget was exhausted with steps still pending.
                # Without this, plan.status stays 'active' forever — and because
                # get_active_plan() filters on 'active' and create_career_plan
                # refuses while one exists, the user would be permanently locked
                # out of creating new plans.
                logger.warning(
                    "[execute_plan] Plan %d exhausted MAX_PLAN_ITERATIONS (%d) "
                    "with steps still pending — marking failed",
                    plan_id, MAX_PLAN_ITERATIONS,
                )
                plan.status = 'failed'
                safe_commit()

            # ── final summary ────────────────────────────────────────────────
            db.session.refresh(plan)
            all_steps = PlanStep.query.filter_by(plan_id=plan_id).order_by(PlanStep.step_order).all()
            done_count = len([s for s in all_steps if s.status == 'done'])
            logger.info("[execute_plan] ■ FINISHED plan %d — %d/%d steps done, status=%s",
                        plan_id, done_count, len(all_steps), plan.status)

            return {
                "plan_id": plan_id,
                "goal": plan.goal,
                "status": plan.status,
                "steps_completed": done_count,
                "steps_total": len(all_steps),
                "step_results": step_results,
            }

    except Exception as exc:
        logger.error("[execute_plan] ✗ UNHANDLED ERROR for plan %d:\n%s",
                     plan_id, traceback.format_exc())
        # Leaving status='active' here would permanently lock the user out of
        # creating new plans (get_active_plan filters on 'active', and
        # create_career_plan refuses while one exists). Best-effort terminal
        # status; never let bookkeeping mask the original error.
        try:
            with app.app_context():
                db.session.rollback()
                failed_plan = db.session.get(TaskPlan, plan_id)
                if failed_plan is not None and failed_plan.status == 'active':
                    failed_plan.status = 'failed'
                    safe_commit()
        except Exception:
            logger.exception("[execute_plan] Could not mark plan %d failed", plan_id)
        return {}


def run_plan(app, user_id: int, goal: str, progress_cb=None) -> dict:
    """Generate and execute a plan synchronously (kept for direct callers)."""
    from services.llm_service import get_llm

    def _progress(msg):
        if progress_cb:
            progress_cb(msg)

    with app.app_context():
        llm = get_llm()
        _progress("Creating your personalized plan...")
        plan = generate_plan(app, user_id, goal, llm)

    return execute_plan(app, user_id, plan.id, progress_cb=progress_cb)


def get_active_plan(user_id: int) -> TaskPlan | None:
    """Get the user's current active plan, if any."""
    return (TaskPlan.query
            .filter_by(user_id=user_id, status='active')
            .order_by(TaskPlan.created_at.desc())
            .first())


def format_plan_status(plan: TaskPlan) -> str:
    """Format a plan's current status for display in chat."""
    steps = PlanStep.query.filter_by(plan_id=plan.id).order_by(PlanStep.step_order).all()

    # If there's a completed synthesis step, surface the roadmap first
    synthesis_step = next(
        (s for s in steps if s.description.startswith(SYNTHESIS_MARKER) and s.status == 'done'),
        None
    )
    if synthesis_step and synthesis_step.result_summary:
        return (
            f"**Goal:** {plan.goal}\n"
            f"**Status:** {plan.status}\n\n"
            f"{synthesis_step.result_summary}"
        )

    # No synthesis yet — show step-by-step progress
    lines = [f"**Goal:** {plan.goal}", f"**Status:** {plan.status}", ""]
    for s in steps:
        icon = {"done": "✓", "running": "⟳", "skipped": "—", "pending": "○"}.get(s.status, "?")
        label = s.description.replace(SYNTHESIS_MARKER, "Roadmap synthesis:").strip()
        lines.append(f"  [{icon}] Step {s.step_order}: {label}")
        if s.result_summary and s.status == 'done' and not s.description.startswith(SYNTHESIS_MARKER):
            lines.append(f"      → {s.result_summary[:400]}")
    return "\n".join(lines)
