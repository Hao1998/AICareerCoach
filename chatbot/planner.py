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
import time
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from models import PlanStep, TaskPlan, db
from schemas.output_schemas import PlanResult, ReplanResult


def _commit_with_retry(max_retries=3, delay=0.5):
    """Commit with retry for SQLite 'database is locked' errors."""
    for attempt in range(max_retries):
        try:
            db.session.commit()
            return
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                db.session.rollback()
                time.sleep(delay * (attempt + 1))
                logger.warning("DB locked, retrying commit (attempt %d)", attempt + 1)
            else:
                db.session.rollback()
                raise

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
    ("system", """You are a career coaching task planner. Given a user's career goal and their current context, generate a concrete, actionable plan.

{tool_descriptions}

Rules:
- Each step must map to exactly one tool call OR be a reasoning-only step (tool_name = null).
- Keep plans between 3-7 steps. Don't over-plan.
- Order steps logically — later steps can depend on earlier results.
- tool_input should be the actual argument string you'd pass to the tool.
- For reasoning-only steps, tool_name and tool_input should be null.
- Be specific — "Analyze resume for Python skills" is better than "Look at resume".
"""),
    ("human", """User goal: {goal}

User context:
{user_context}

Previous memories (if any):
{memories}

Generate a plan to achieve this goal."""),
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
        db.session.flush()

        for step in plan_result.steps:
            db.session.add(PlanStep(
                plan_id=task_plan.id,
                step_order=step.step_order,
                description=step.description,
                tool_name=step.tool_name,
                tool_input=step.tool_input,
                status='pending',
            ))

        _commit_with_retry()
        logger.info("Created plan %d with %d steps for user %d: %s",
                     task_plan.id, len(plan_result.steps), user_id, goal)
        return task_plan


# ── Executor ──────────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="task-planner-execute-step")
def execute_step(app, user_id: int, step: PlanStep, tools_by_name: dict, llm) -> str:
    """Execute a single PlanStep. Returns a result summary string."""
    with app.app_context():
        step.status = 'running'
        _commit_with_retry()

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
        _commit_with_retry()

        logger.info("Completed step %d (tool=%s) for plan step_order=%d",
                     step.id, step.tool_name or 'reasoning', step.step_order)
        return result


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
            _commit_with_retry()
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
            _commit_with_retry()
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

        _commit_with_retry()
        logger.info("Replanned plan %d: %d remaining steps", plan.id, len(new_steps))
        return False


# ── Main loop: Plan → Execute → Replan ────────────────────────────────────────

@traceable(run_type="chain", name="task-planner-run")
def run_plan(app, user_id: int, goal: str, progress_cb=None) -> dict:
    """Full plan-execute-replan loop. Returns a summary of the entire run.

    progress_cb: optional callable(message: str) for streaming updates.
    """
    from services.llm_service import get_llm
    from chatbot.tools import build_tools

    def _progress(msg):
        if progress_cb:
            progress_cb(msg)

    with app.app_context():
        llm = get_llm()

        _progress("Creating your personalized plan...")
        plan = generate_plan(app, user_id, goal, llm)

        tools = build_tools(app, user_id, progress_cb=_progress)
        tools_by_name = {t.name: t for t in tools}

        step_results = []
        max_iterations = 15

        for iteration in range(max_iterations):
            next_step = (PlanStep.query
                         .filter_by(plan_id=plan.id, status='pending')
                         .order_by(PlanStep.step_order)
                         .first())

            if not next_step:
                plan.status = 'completed'
                _commit_with_retry()
                break

            _progress(f"Step {next_step.step_order}: {next_step.description}")
            result = execute_step(app, user_id, next_step, tools_by_name, llm)
            step_results.append({
                "step": next_step.step_order,
                "description": next_step.description,
                "tool": next_step.tool_name,
                "result_preview": result[:300],
            })

            _progress("Reviewing progress...")
            is_complete = replan(app, plan, result, llm)
            if is_complete:
                break

        db.session.refresh(plan)
        all_steps = PlanStep.query.filter_by(plan_id=plan.id).order_by(PlanStep.step_order).all()

        return {
            "plan_id": plan.id,
            "goal": plan.goal,
            "status": plan.status,
            "steps_completed": len([s for s in all_steps if s.status == 'done']),
            "steps_total": len(all_steps),
            "step_results": step_results,
        }


def get_active_plan(user_id: int) -> TaskPlan | None:
    """Get the user's current active plan, if any."""
    return (TaskPlan.query
            .filter_by(user_id=user_id, status='active')
            .order_by(TaskPlan.created_at.desc())
            .first())


def format_plan_status(plan: TaskPlan) -> str:
    """Format a plan's current status for display in chat."""
    steps = PlanStep.query.filter_by(plan_id=plan.id).order_by(PlanStep.step_order).all()
    lines = [f"**Goal:** {plan.goal}", f"**Status:** {plan.status}", ""]
    for s in steps:
        icon = {"done": "done", "running": "running", "skipped": "skipped", "pending": "pending"}.get(s.status, "?")
        lines.append(f"  [{icon}] Step {s.step_order}: {s.description}")
        if s.result_summary and s.status == 'done':
            preview = s.result_summary[:150]
            lines.append(f"         Result: {preview}")
    return "\n".join(lines)
