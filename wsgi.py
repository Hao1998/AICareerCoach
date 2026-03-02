"""
WSGI Entry Point

Use for production deployment:  gunicorn wsgi:app
Use for Flask-Migrate commands: flask --app wsgi db migrate

The factory pattern allows migrations to run without requiring API keys.
"""

import os
from factory import create_app

env = os.getenv('FLASK_ENV', 'development')

# skip_api_check=True so `flask db migrate` works without XAI_API_KEY set
app = create_app(config_name=env, skip_api_check=True)
