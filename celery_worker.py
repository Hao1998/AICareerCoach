"""
Celery worker entry point.

Start the worker and the Beat scheduler as separate processes alongside the web app
(set USE_CELERY=true in the environment):

    celery -A celery_worker.celery worker --loglevel=info -Q scout
    celery -A celery_worker.celery beat   --loglevel=info

The Beat process fires scout.dispatch_due every minute; the worker executes the
fanned-out scout.run_for_user tasks. This keeps scheduled Job Scout runs off the
Flask web workers and lets the web tier scale to multiple instances.
"""

from factory import create_app
from tasks.celery_app import make_celery

flask_app = create_app()
celery = flask_app.extensions.get("celery") or make_celery(flask_app)
