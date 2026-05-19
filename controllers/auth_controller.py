"""
Auth Controller

Handles user authentication and the main index route.
Blueprint: 'auth'
"""

import os
from datetime import datetime

from flask import Blueprint, request, render_template, redirect, url_for, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user
from flask_limiter.errors import RateLimitExceeded

from sqlalchemy.orm import joinedload

from factory import limiter
from models import db, User, Resume, JobMatch
from services.db_lock import safe_commit
from form import LoginForm, RegistrationForm

auth_bp = Blueprint('auth', __name__)


@auth_bp.errorhandler(RateLimitExceeded)
def handle_rate_limit(e):
    flash('Too many attempts. Please try again later.', 'error')
    return redirect(url_for('auth.login')), 429


@auth_bp.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)


@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute; 100 per hour", methods=['POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('auth.index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid email or password', 'error')
            return redirect(url_for('auth.login'))

        user.last_login = datetime.utcnow()
        safe_commit()
        login_user(user, remember=form.remember_me.data)
        flash(f'Welcome back, {user.username}!', 'success')

        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('auth.index')
        return redirect(next_page)

    return render_template('login.html', form=form)


@auth_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute; 20 per hour", methods=['POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('auth.index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            full_name=form.full_name.data
        )
        user.set_password(form.password.data)
        db.session.add(user)
        safe_commit()

        user_upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(user.id))
        user_vector_dir = os.path.join('vector_index', str(user.id))
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_vector_dir, exist_ok=True)

        flash('Congratulations, you are now registered! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/profile')
@login_required
def profile():
    resumes = current_user.resumes.filter_by(is_active=True).order_by(Resume.uploaded_at.desc()).all()
    recent_matches = (current_user.job_matches
                      .options(joinedload(JobMatch.job))
                      .order_by(JobMatch.created_at.desc())
                      .limit(10).all())
    return render_template('profile.html', user=current_user, resumes=resumes, recent_matches=recent_matches)
