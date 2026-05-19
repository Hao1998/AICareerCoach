"""
Application Factory

Creates and configures the Flask application.
- Registers all blueprints (controllers)
- Initialises extensions (db, login, migrate)
- Registers template filters
- Starts the scheduler
"""

import hashlib
import json
import os

from flask import Flask, jsonify, render_template, request, url_for
from flask_login import LoginManager, current_user
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from langchain_core.globals import set_llm_cache
from flask_socketio import SocketIO

from models import db, User
from config import config
from services.semantic_cache import SemanticCache

login_manager = LoginManager()
migrate = Migrate()
socketio = SocketIO()


def _rate_limit_key():
    """Use authenticated user ID as the rate-limit key, fall back to IP."""
    if current_user.is_authenticated:
        return f"user:{current_user.id}"
    return get_remote_address()


limiter = Limiter(
    key_func=_rate_limit_key,
    default_limits=[],
    storage_uri=os.environ.get('REDIS_URL', 'memory://'),
)


def create_app(config_name='default', skip_api_check=False):
    """
    Create and configure the Flask application.

    Args:
        config_name: 'development', 'production', or 'test'
        skip_api_check: Skip API key validation (useful for migrations)
    """
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # ── Semantic LLM Cache ────────────────────────────────────────────────────
    # whose meaning is close enough (cosine similarity >= 0.90).
    # This means rephrased queries like "analyze my resume" and "review my CV"
    # both hit the same cached response — unlike exact-match caching.
    # Uses the same HuggingFace embedding model already used for job matching.
    set_llm_cache(SemanticCache(
        embedding_model=None,
        score_threshold=0.90,
        ttl_seconds=3600,
        # Job-specific prompts embed similarly across different jobs because the
        # resume text dominates the vector, causing false cache hits. These are
        # cached at the DB level (JobMatch) instead, so bypass semantic caching.
        bypass_prefixes=[
            "You are the Job Analyst Agent",       # JobAnalystAgent system prompt
            "You are the Resume Tailoring Agent",  # ResumeTailoringAgent system prompt
            "Role: You are an AI Career Coach creating personalized interview",  # roadmap
            "You are a senior career coach",
        ],
    ))

    # ── Observability ─────────────────────────────────────────────────────────
    from services.logging_config import configure_logging
    from services.telemetry import init_request_id, init_telemetry, init_sentry

    configure_logging(app)
    init_request_id(app)
    init_telemetry(app)
    init_sentry(app)

    # ── Extensions ────────────────────────────────────────────────────────────
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    limiter.init_app(app)
    # message_queue is only needed for multi-worker gunicorn (routes SocketIO
    # events through Redis so all workers see them). For single-process dev
    # (python app.py) it causes a startup deadlock — skip it in development.
    _mq = None if config_name == 'development' else app.config.get('REDIS_URL')
    socketio.init_app(app,
        cors_allowed_origins="*",
        async_mode='gevent',
        message_queue=_mq,
    )

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

    import controllers.ws_chat_controller  # noqa: F401 — registers SocketIO events

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

    # ── Static asset cache-busting ─────────────────────────────────────────
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000

    @app.context_processor
    def asset_hash():
        _cache = {}
        def hashed_url(filename):
            if filename in _cache:
                return _cache[filename]
            filepath = os.path.join(app.static_folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    h = hashlib.md5(f.read()).hexdigest()[:8]
                result = url_for('static', filename=filename) + '?v=' + h
            except FileNotFoundError:
                result = url_for('static', filename=filename)
            _cache[filename] = result
            return result
        return {'hashed_url': hashed_url}

    # ── Health checks ─────────────────────────────────────────────────────────
    @app.route('/health')
    def health():
        from sqlalchemy import text
        from services.redis_client import get_redis
        status = {"status": "healthy", "db": False, "redis": False}
        code = 200
        try:
            db.session.execute(text("SELECT 1"))
            status["db"] = True
        except Exception:
            code = 503


        try:
            get_redis().ping()
            status["redis"] = True
        except Exception:
            code = 503
        status["status"] = "healthy" if code == 200 else "degraded"
        return jsonify(status), code

    @app.route('/ready')
    def ready():
        from sqlalchemy import text
        from services.redis_client import get_redis
        checks = {"db": False, "redis": False, "scheduler": False}
        try:
            db.session.execute(text("SELECT 1"))
            checks["db"] = True
        except Exception:
            pass
        try:
            get_redis().ping()
            checks["redis"] = True
        except Exception:
            pass
        scheduler = app.extensions.get('scheduler')
        checks["scheduler"] = scheduler.is_running() if scheduler else False
        all_ok = all(checks.values())
        return jsonify({"status": "ready" if all_ok else "not_ready", **checks}), 200 if all_ok else 503

    # ── Global error handlers ────────────────────────────────────────────────
    def _wants_json():
        return request.is_json or request.path.startswith('/api/')

    @app.errorhandler(404)
    def not_found(e):
        if _wants_json():
            return jsonify({"success": False, "error": "Not found"}), 404
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        if _wants_json():
            return jsonify({"success": False, "error": "Internal server error"}), 500
        return render_template('errors/500.html'), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        # HTTPException subclasses (404, 405, etc.) should keep their own status code
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            return e
        app.logger.exception("Unhandled exception: %s", e)
        if _wants_json():
            return jsonify({"success": False, "error": "An unexpected error occurred"}), 500
        return render_template('errors/500.html'), 500

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
