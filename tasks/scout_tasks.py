"""
Job Scout task logic — decoupled from the Flask web process.

The Job Scout used to run inside APScheduler on the Flask process, competing with
request workers. It now runs as Celery tasks (see tasks/celery_app.py): a Beat
periodic task (dispatch) fans out per-user run tasks to Celery workers.

The real work lives in plain `*_logic` functions that take their dependencies as
arguments, so they are unit-testable without a broker. The Celery task wrappers
in register_scout_tasks() are thin glue around them.
"""

import logging
from datetime import datetime

import pytz

from jobs.schedule_selector import due_user_ids

logger = logging.getLogger(__name__)


def run_scout_for_user_logic(flask_app, user_id, run_type="scheduled", run_id=None):
    """Run the Job Scout for one user inside the Flask app context."""
    from jobs.scout_agent import JobScoutAgent
    with flask_app.app_context():
        agent = JobScoutAgent(flask_app)
        return agent.run_for_user(user_id, run_type, run_id)


def _default_fetch_configs():
    from models import AgentConfig
    return AgentConfig.query.filter_by(is_enabled=True).all()


def dispatch_due_scouts_logic(flask_app, enqueue, fetch_configs=None, now_utc=None):
    """Enqueue a scout run for every enabled user whose local time is due now.

    enqueue:        callable(user_id) — in prod, queues a Celery task.
    fetch_configs:  callable() -> list of AgentConfig (defaults to a DB query).
    now_utc:        override for testing; defaults to the current UTC time.
    Returns the list of dispatched user_ids.
    """
    now_utc = now_utc or datetime.now(pytz.UTC)
    with flask_app.app_context():
        configs = (fetch_configs or _default_fetch_configs)()
        due = due_user_ids(configs, now_utc)
        for user_id in due:
            enqueue(user_id)
        logger.info("dispatch_due_scouts: %d user(s) due, enqueued", len(due))
        return due
