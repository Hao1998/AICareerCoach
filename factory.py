"""
Application Factory

Creates and configures the Flask application.
- Registers all blueprints (controllers)
- Initialises extensions (db, login, migrate)
- Registers template filters
- Starts the scheduler
"""

import json
import os

from flask import Flask
from flask_login import LoginManager
from flask_migrate import Migrate

from models import db, User
from config import config

login_manager = LoginManager()
migrate = Migrate()


def create_app(config_name='default', skip_api_check=False):
    """
    Create and configure the Flask application.

    Args:
        config_name: 'development', 'production', or 'test'
        skip_api_check: Skip API key validation (useful for migrations)
    """
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # ── Extensions ────────────────────────────────────────────────────────────
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # ── Directories ───────────────────────────────────────────────────────────
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['JOB_VECTOR_INDEX'], exist_ok=True)

    # ── Blueprints ────────────────────────────────────────────────────────────
    from controllers.auth_controller import auth_bp
    from controllers.resume_controller import resume_bp
    from controllers.job_controller import job_bp
    from controllers.agent_controller import agent_bp
    from controllers.chat_controller import chat_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(resume_bp)
    app.register_blueprint(job_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(chat_bp)

    # ── Template Filters ──────────────────────────────────────────────────────
    @app.template_filter('from_json')
    def from_json_filter(value):
        if value:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    @app.template_filter('local_time')
    def local_time_filter(dt, timezone='UTC', fmt='%B %d, %Y at %I:%M %p'):
        if dt is None:
            return 'N/A'
        from zoneinfo import ZoneInfo
        utc_dt = dt.replace(tzinfo=ZoneInfo('UTC'))
        local_dt = utc_dt.astimezone(ZoneInfo(timezone))
        return local_dt.strftime(fmt)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    if not skip_api_check:
        from agent_scheduler import init_scheduler
        from job_scout_agent import JobScoutAgent
        scheduler = init_scheduler(app, JobScoutAgent)
        app.extensions['scheduler'] = scheduler

        _validate_api_keys(app)

    return app


def _validate_api_keys(app):
    """Raise if required API keys are missing"""
    if not app.config.get('XAI_API_KEY'):
        raise RuntimeError(
            "XAI_API_KEY environment variable is not set. "
            "Please set it in your environment or .env file."
        )
