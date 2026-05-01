"""
Application Configuration
"""
import os

# Absolute path to the project root — works regardless of the working directory
# the process is launched from (e.g. Claude Desktop spawning the MCP server).
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(_PROJECT_ROOT, 'instance', 'career_coach.db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # With gevent, many greenlets share one worker process. Increase pool size
    # so they don't queue up waiting for a DB connection.
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 20,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 1800,
    }

    # API Keys (optional for migrations)
    XAI_API_KEY = os.environ.get('XAI_API_KEY')
    ADZUNA_APP_ID = os.environ.get('ADZUNA_APP_ID')
    ADZUNA_APP_KEY = os.environ.get('ADZUNA_APP_KEY')

    # Upload configuration — absolute so the server works from any working directory
    UPLOAD_FOLDER = os.path.join(_PROJECT_ROOT, 'uploads')

    # Job vector index — same reason
    JOB_VECTOR_INDEX = os.path.join(_PROJECT_ROOT, 'job_vector_index')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

class TestConfig(Config):
    """Test configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'test': TestConfig,
    'default': DevelopmentConfig
}