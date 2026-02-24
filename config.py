"""
Application Configuration
"""
import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///career_coach.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # API Keys (optional for migrations)
    XAI_API_KEY = os.environ.get('XAI_API_KEY')
    ADZUNA_APP_ID = os.environ.get('ADZUNA_APP_ID')
    ADZUNA_APP_KEY = os.environ.get('ADZUNA_APP_KEY')

    # Upload configuration
    UPLOAD_FOLDER = 'uploads'

    # Job vector index
    JOB_VECTOR_INDEX = 'job_vector_index'

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