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
