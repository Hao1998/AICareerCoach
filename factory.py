"""
Application Factory

This module provides the create_app() function that initializes the Flask application.
Using the factory pattern allows:
- Running migrations without requiring API keys
- Testing with different configurations
- Multiple app instances
"""

import os
from flask import Flask
from flask_login import LoginManager
from flask_migrate import Migrate
from models import db, User
from config import config

# Global references (initialized in create_app)
login_manager = LoginManager()
migrate = Migrate()

def create_app(config_name='default', skip_api_check=False):
    """
    Create and configure the Flask application

    Args:
        config_name: Configuration to use ('development', 'production', 'test')
        skip_api_check: If True, don't validate API keys (useful for migrations)

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    # Configure Flask-Login
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['JOB_VECTOR_INDEX'], exist_ok=True)

    # Validate API keys (skip during migrations)
    if not skip_api_check:
        with app.app_context():
            validate_api_keys(app)

    return app


def validate_api_keys(app):
    """
    Validate required API keys are set

    Raises:
        RuntimeError: If required API keys are missing
    """
    if not app.config.get('XAI_API_KEY'):
        raise RuntimeError(
            "XAI_API_KEY environment variable is not set. "
            "Please set it in your environment or .env file."
        )

    # Add other API key validations here
    # if not app.config.get('ADZUNA_APP_ID'):
    #     raise RuntimeError("ADZUNA_APP_ID environment variable is not set")


def get_llm(app=None):
    """
    Get or create LLM instance (lazy initialization)

    This defers LLM creation until actually needed, avoiding
    API key checks during imports or migrations.

    Args:
        app: Flask app instance (uses current_app if not provided)

    Returns:
        ChatXAI instance
    """
    from flask import current_app
    from langchain_xai import ChatXAI

    if app is None:
        app = current_app

    # Store LLM in app extensions
    if not hasattr(app, 'extensions'):
        app.extensions = {}

    if 'llm' not in app.extensions:
        api_key = app.config.get('XAI_API_KEY')
        if not api_key:
            raise RuntimeError("XAI_API_KEY not configured")

        app.extensions['llm'] = ChatXAI(
            model="grok-3",
            temperature=0,
            api_key=api_key
        )

    return app.extensions['llm']