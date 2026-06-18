"""
Celery application factory for AICareerCoach.

Decouples the Job Scout from the Flask web process: scout runs execute on Celery
workers, and Celery Beat fires a once-per-minute dispatch task that fans out
per-user runs for whoever is due in their own timezone. This replaces the
in-process APScheduler, so scheduled work no longer competes with request workers
and the app can scale to multiple instances.

Run the workers alongside the web app:

    celery -A celery_worker.celery worker --loglevel=info
    celery -A celery_worker.celery beat   --loglevel=info

(see celery_worker.py for the module Celery looks up).
"""

import logging

from celery import Celery

logger = logging.getLogger(__name__)


def make_celery(flask_app):
    """Create a Celery app bound to the Flask app's config and context."""
    broker = flask_app.config.get("CELERY_BROKER_URL") or flask_app.config["REDIS_URL"]
    backend = flask_app.config.get("CELERY_RESULT_BACKEND") or broker

    celery = Celery(flask_app.import_name, broker=broker, backend=backend)
    celery.conf.update(flask_app.config.get("CELERY_CONFIG", {}))

    # Every task runs inside a Flask app context so DB sessions / extensions work.
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask

    _register_scout_tasks(celery, flask_app)

    if not hasattr(flask_app, "extensions"):
        flask_app.extensions = {}
    flask_app.extensions["celery"] = celery
    return celery


def _register_scout_tasks(celery, flask_app):
    """Register the Job Scout tasks and the Beat dispatch schedule."""
    from tasks.scout_tasks import run_scout_for_user_logic, dispatch_due_scouts_logic

    @celery.task(name="scout.run_for_user")
    def run_scout_for_user(user_id, run_type="scheduled", run_id=None):
        return run_scout_for_user_logic(flask_app, user_id, run_type, run_id)

    @celery.task(name="scout.dispatch_due")
    def dispatch_due_scouts():
        return dispatch_due_scouts_logic(
            flask_app,
            enqueue=lambda uid: run_scout_for_user.delay(uid, "scheduled"),
        )

    # Beat checks every minute; due_user_ids() does the per-timezone matching.
    celery.conf.beat_schedule = {
        "dispatch-due-scouts-every-minute": {
            "task": "scout.dispatch_due",
            "schedule": 60.0,
        }
    }
    celery.conf.timezone = "UTC"

    # Expose handles so callers (e.g. manual triggers) can enqueue directly.
    celery.run_scout_for_user = run_scout_for_user
    celery.dispatch_due_scouts = dispatch_due_scouts
    return celery
