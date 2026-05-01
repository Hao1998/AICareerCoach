"""
WSGI Entry Point

Use for production deployment:  gunicorn wsgi:app
Use for Flask-Migrate commands: flask --app wsgi db migrate

The factory pattern allows migrations to run without requiring API keys.
"""

# gevent monkey patching must happen before any other imports so that
# stdlib socket/threading/ssl are replaced with gevent-aware versions.
# This turns blocking I/O (Grok API, Adzuna, DB) into cooperative yields,
# allowing thousands of SSE streams to run on a small number of OS threads.
try:
    from gevent import monkey
    monkey.patch_all()
except ImportError:
    pass  # gevent not installed (e.g. during local dev with flask run)

import os
from factory import create_app

env = os.getenv('FLASK_ENV', 'development')

# skip_api_check=True so `flask db migrate` works without XAI_API_KEY set
app = create_app(config_name=env, skip_api_check=True)
