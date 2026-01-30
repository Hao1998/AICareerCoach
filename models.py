from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

import json
import json

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """Model for user authentication and profile"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)



    # Relationships
    resumes = db.relationship('Resume', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    job_matches = db.relationship('JobMatch', backref='user', lazy='dynamic', cascade='all, delete-orphan')


    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')


    def check_password(self, password):
        """Verify password against hash"""
        return check_password_hash(self.password_hash, password)



    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

    def __repr__(self):
        return f'<User {self.username}>'



class Resume(db.Model):
    """Model for storing user resume information"""
    __tablename__ = 'resumes'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    analysis = db.Column(db.Text)  # JSON string of resume analysis
    is_active = db.Column(db.Boolean, default=True)
    # Relationships
    job_matches = db.relationship('JobMatch', backref='resume', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'analysis': json.loads(self.analysis) if self.analysis else None,
            'is_active': self.is_active
        }

    def __repr__(self):
        return f'<Resume {self.filename}>'


class JobPosting(db.Model):
    """Model for storing job postings"""
    __tablename__ = 'job_postings'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200))
    job_type = db.Column(db.String(50))  # Full-time, Part-time, Contract, etc.
    description = db.Column(db.Text, nullable=False)
    requirements = db.Column(db.Text)
    salary_range = db.Column(db.String(100))
    posted_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    # Store embeddings as JSON string for simplicity
    embedding_ids = db.Column(db.Text)  # Store FAISS indices (deprecated, use embedding instead)

    # Pre-computed embedding vector (stored as pickled numpy array)
    embedding = db.Column(db.PickleType)  # Store job embedding for fast retrieval
    embedding_updated_at = db.Column(db.DateTime)  # Track when embedding was last updated

    def get_job_text(self):
        """Get combined text representation of the job for embedding"""
        return f"{self.title} {self.description} {self.requirements or ''}"

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'company': self.company,
            'location': self.location,
            'job_type': self.job_type,
            'description': self.description,
            'requirements': self.requirements,
            'salary_range': self.salary_range,
            'posted_date': self.posted_date.isoformat() if self.posted_date else None,
            'is_active': self.is_active
        }


class JobMatch(db.Model):
    """Model for storing CV-Job match results"""
    __tablename__ = 'job_matches'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=False, index=True)
    resume_filename = db.Column(db.String(200), nullable=False)  # Keep for backward compatibility
    job_id = db.Column(db.Integer, db.ForeignKey('job_postings.id'), nullable=False)
    match_score = db.Column(db.Float, nullable=False)
    matched_skills = db.Column(db.Text)  # JSON string of matched skills
    gaps = db.Column(db.Text)  # JSON string of skill gaps
    recommendation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    agent_generated = db.Column(db.Boolean, default=False)  # Whether found by agent
    agent_run_id = db.Column(db.Integer, db.ForeignKey('agent_run_history.id'), nullable=True)
    user_feedback = db.Column(db.String(50))  # 'interested', 'not_interested', 'applied', etc.
    feedback_at = db.Column(db.DateTime)

    job = db.relationship('JobPosting', backref='matches')

    def to_dict(self):
        return {
            'id': self.id,
            'resume_filename': self.resume_filename,
            'job_id': self.job_id,
            'match_score': self.match_score,
            'matched_skills': json.loads(self.matched_skills) if self.matched_skills else [],
            'gaps': json.loads(self.gaps) if self.gaps else [],
            'recommendation': self.recommendation,
            'agent_generated': self.agent_generated,
            'user_feedback': self.user_feedback,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'job': self.job.to_dict() if self.job else None
        }


class AgentConfig(db.Model):
    """Model for storing user's Job Scout Agent configuration"""
    __tablename__ = 'agent_configs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True, index=True)
    is_enabled = db.Column(db.Boolean, default=True)
    schedule_time = db.Column(db.String(5), default='09:00')  # HH:MM format
    timezone = db.Column(db.String(50), default='UTC')  # User's timezone (e.g., 'America/New_York')
    schedule_frequency = db.Column(db.String(20), default='daily')  # 'daily', 'weekly', etc.
    match_threshold = db.Column(db.Float, default=75.0)  # Minimum match score to notify
    max_results_per_run = db.Column(db.Integer, default=10)  # Max matches per run
    last_run_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = db.relationship('User', backref=db.backref('agent_config', uselist=False))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'is_enabled': self.is_enabled,
            'schedule_time': self.schedule_time,
            'timezone': self.timezone,
            'schedule_frequency': self.schedule_frequency,
            'match_threshold': self.match_threshold,
            'max_results_per_run': self.max_results_per_run,
            'last_run_at': self.last_run_at.isoformat() if self.last_run_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def __repr__(self):
        return f'<AgentConfig user_id={self.user_id} enabled={self.is_enabled}>'


class AgentRunHistory(db.Model):
    """Model for storing Job Scout Agent execution history"""
    __tablename__ = 'agent_run_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    run_type = db.Column(db.String(20), nullable=False)  # 'scheduled' or 'manual'
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='running')  # 'running', 'completed', 'failed'
    jobs_fetched = db.Column(db.Integer, default=0)
    jobs_analyzed = db.Column(db.Integer, default=0)
    matches_found = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text)
    results_summary = db.Column(db.Text)  # JSON string with detailed results

    # Relationships
    user = db.relationship('User', backref='agent_runs')
    matches = db.relationship('JobMatch', backref='agent_run', foreign_keys='JobMatch.agent_run_id')

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'run_type': self.run_type,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'jobs_fetched': self.jobs_fetched,
            'jobs_analyzed': self.jobs_analyzed,
            'matches_found': self.matches_found,
            'error_message': self.error_message,
            'results_summary': json.loads(self.results_summary) if self.results_summary else None
        }

    def __repr__(self):
        return f'<AgentRunHistory id={self.id} user_id={self.user_id} status={self.status}>'
