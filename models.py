from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

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
    embedding_ids = db.Column(db.Text)  # Store FAISS indices

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
    resume_filename = db.Column(db.String(200), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job_postings.id'), nullable=False)
    match_score = db.Column(db.Float, nullable=False)
    matched_skills = db.Column(db.Text)  # JSON string of matched skills
    gaps = db.Column(db.Text)  # JSON string of skill gaps
    recommendation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'job': self.job.to_dict() if self.job else None
        }
