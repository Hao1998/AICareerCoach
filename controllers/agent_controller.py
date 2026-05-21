"""
Agent Controller

Handles Job Scout Agent configuration, triggering, history, and feedback.
Blueprint: 'agent'
"""

import json
import time
from datetime import datetime

from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify, current_app, Response
from flask_login import login_required, current_user

from sqlalchemy.orm import joinedload

from pydantic import ValidationError

from models import db, AgentConfig, AgentRunHistory, JobMatch
from jobs.fetchers.registry import USER_VISIBLE_SOURCES
from services.db_lock import safe_commit
from job_utils import update_user_preferences
from schemas.request_schemas import AgentConfigUpdateRequest, FeedbackRequest
from schemas.validate import validate_json

agent_bp = Blueprint('agent', __name__)


def _get_scheduler():
    """Helper to retrieve the scheduler from app extensions"""
    return current_app.extensions.get('scheduler')


@agent_bp.route('/agent/trigger', methods=['POST'])
@login_required
def trigger_agent():
    try:
        latest_resume = current_user.resumes.filter_by(is_active=True).first()
        if not latest_resume:
            return jsonify({
                'success': False,
                'error': 'Please upload a resume first before running the Job Scout Agent'
            }), 400

        result = _get_scheduler().trigger_manual_run_async(current_user.id)
        return jsonify({'success': True, 'run_id': result['run_id']})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@agent_bp.route('/agent/stream/<int:run_id>')
@login_required
def stream_agent_run(run_id):
    """SSE endpoint — pushes progress events to the browser as they are emitted."""
    from job_scout_agent import get_run_events, cleanup_run_progress

    # Verify this run belongs to the current user
    run = AgentRunHistory.query.filter_by(id=run_id, user_id=current_user.id).first()
    if not run:
        return jsonify({'error': 'Run not found'}), 404

    def event_stream():
        index = 0
        try:
            while True:
                events, done = get_run_events(run_id, since=index)
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n"
                    index += 1
                if done:
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    break
                time.sleep(0.4)
        finally:
            cleanup_run_progress(run_id)

    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@agent_bp.route('/agent/config', methods=['GET', 'POST'])
@login_required
def agent_config():
    config = AgentConfig.query.filter_by(user_id=current_user.id).first()
    if not config:
        config = AgentConfig(user_id=current_user.id)
        db.session.add(config)
        safe_commit()

    if request.method == 'POST':
        try:
            if request.is_json:
                raw = request.get_json(silent=True) or {}
                try:
                    parsed = AgentConfigUpdateRequest(**raw)
                except ValidationError as e:
                    details = [
                        f"{'.'.join(str(loc) for loc in err['loc']) or '<root>'}: {err['msg']}"
                        for err in e.errors()
                    ]
                    return jsonify({
                        "success": False,
                        "error": "Validation failed",
                        "details": details,
                    }), 400
                data = parsed.model_dump(exclude_unset=True)
            else:
                data = request.form

            schedule_changed = False
            enabled_changed = False

            new_schedule_time = data.get('schedule_time', config.schedule_time)
            if new_schedule_time != config.schedule_time:
                schedule_changed = True
                config.schedule_time = new_schedule_time

            new_timezone = data.get('timezone', config.timezone)
            if new_timezone != config.timezone:
                schedule_changed = True
                config.timezone = new_timezone

            raw_enabled = data.get('is_enabled', True)
            if isinstance(raw_enabled, bool):
                new_is_enabled = raw_enabled
            else:
                new_is_enabled = str(raw_enabled).lower() in ['true', '1', 'on']
            if new_is_enabled != config.is_enabled:
                enabled_changed = True
                config.is_enabled = new_is_enabled

            config.match_threshold = float(data.get('match_threshold', config.match_threshold))
            config.max_results_per_run = int(data.get('max_results_per_run', config.max_results_per_run))

            if 'adzuna_location' in data:
                location_value = data.get('adzuna_location', '').strip()
                config.adzuna_location = location_value if location_value else None

            if 'adzuna_max_jobs' in data:
                try:
                    max_jobs_value = int(data.get('adzuna_max_jobs', 20))
                    if 1 <= max_jobs_value <= 200:
                        config.adzuna_max_jobs = max_jobs_value
                except (ValueError, TypeError):
                    pass

            if 'adzuna_max_days_old' in data:
                try:
                    max_days_old_value = int(data.get('adzuna_max_days_old', 30))
                    if 1 <= max_days_old_value <= 365:
                        config.adzuna_max_days_old = max_days_old_value
                except (ValueError, TypeError):
                    pass

            # Collect enabled sources from checkboxes (form sends one value per checked box)
            if request.is_json:
                selected_sources = data.get('enabled_sources', [])
                if isinstance(selected_sources, str):
                    selected_sources = [selected_sources]
            else:
                selected_sources = request.form.getlist('enabled_sources')
            selected_sources = [s for s in selected_sources if s in USER_VISIBLE_SOURCES]
            config.enabled_sources = selected_sources if selected_sources else ['adzuna']

            safe_commit()

            if schedule_changed or enabled_changed:
                _get_scheduler().rebuild_schedule()

            if request.is_json:
                return jsonify({'success': True, 'message': 'Configuration updated successfully',
                                'config': config.to_dict()})
            else:
                flash('Agent configuration updated successfully', 'success')
                return redirect(url_for('agent.agent_dashboard'))

        except Exception as e:
            if request.is_json:
                return jsonify({'success': False, 'error': str(e)}), 500
            else:
                flash(f'Error updating configuration: {str(e)}', 'error')
                return redirect(url_for('agent.agent_dashboard'))

    return render_template('agent_config.html', config=config, user=current_user,
                           available_sources=USER_VISIBLE_SOURCES)


@agent_bp.route('/agent/history')
@login_required
def agent_history():
    page = request.args.get('page', 1, type=int)
    runs = AgentRunHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(AgentRunHistory.started_at.desc()).paginate(page=page, per_page=20, error_out=False)

    config = AgentConfig.query.filter_by(user_id=current_user.id).first()
    return render_template('agent_history.html', runs=runs, config=config, user=current_user)


@agent_bp.route('/agent/dashboard')
@login_required
def agent_dashboard():
    config = AgentConfig.query.filter_by(user_id=current_user.id).first()
    if not config:
        config = AgentConfig(user_id=current_user.id)
        db.session.add(config)
        safe_commit()

    recent_runs = AgentRunHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(AgentRunHistory.started_at.desc()).limit(10).all()

    recent_matches = JobMatch.query.options(joinedload(JobMatch.job)).filter_by(
        user_id=current_user.id, agent_generated=True
    ).order_by(JobMatch.created_at.desc()).limit(10).all()

    next_run = _get_scheduler().get_next_run_time()

    return render_template('agent_dashboard.html', config=config, recent_runs=recent_runs,
                           recent_matches=recent_matches, next_run=next_run, user=current_user)


@agent_bp.route('/agent/matches/<int:run_id>')
@login_required
def agent_run_matches(run_id):
    run = AgentRunHistory.query.filter_by(id=run_id, user_id=current_user.id).first_or_404()
    matches = JobMatch.query.options(joinedload(JobMatch.job)).filter_by(
        agent_run_id=run_id, user_id=current_user.id
    ).order_by(JobMatch.match_score.desc()).all()
    return render_template('agent_run_matches.html', run=run, matches=matches, user=current_user)


@agent_bp.route('/agent/feedback/<int:match_id>', methods=['POST'])
@login_required
def agent_match_feedback(match_id):
    try:
        match = JobMatch.query.filter_by(id=match_id, user_id=current_user.id).first_or_404()

        if request.is_json:
            try:
                parsed = FeedbackRequest(**(request.get_json(silent=True) or {}))
            except ValidationError as e:
                details = [
                    f"{'.'.join(str(loc) for loc in err['loc']) or '<root>'}: {err['msg']}"
                    for err in e.errors()
                ]
                return jsonify({
                    'success': False,
                    'error': 'Validation failed',
                    'details': details,
                }), 400
            feedback = parsed.feedback
        else:
            feedback = request.form.get('feedback')
            if feedback not in ['interested', 'not_interested', 'applied']:
                return jsonify({'success': False, 'error': 'Invalid feedback value'}), 400

        match.user_feedback = feedback
        match.feedback_at = datetime.utcnow()
        safe_commit()

        try:
            update_user_preferences(current_user.id)
        except Exception as pref_error:
            print(f"Warning: Failed to update user preferences: {pref_error}")

        if request.is_json:
            return jsonify({'success': True, 'message': 'Feedback recorded successfully'})
        else:
            flash('Thank you for your feedback!', 'success')
            return redirect(url_for('agent.agent_dashboard'))

    except Exception as e:
        if request.is_json:
            return jsonify({'success': False, 'error': str(e)}), 500
        else:
            flash(f'Error recording feedback: {str(e)}', 'error')
            return redirect(url_for('agent.agent_dashboard'))


@agent_bp.route('/agent/preferences')
@login_required
def agent_preferences():
    """Return what preferences have been learned from the user's feedback history."""
    config = AgentConfig.query.filter_by(user_id=current_user.id).first()

    liked = JobMatch.query.options(joinedload(JobMatch.job)).filter(
        JobMatch.user_id == current_user.id,
        JobMatch.user_feedback.in_(['interested', 'applied'])
    ).order_by(JobMatch.feedback_at.desc()).all()

    disliked = JobMatch.query.options(joinedload(JobMatch.job)).filter_by(
        user_id=current_user.id,
        user_feedback='not_interested'
    ).order_by(JobMatch.feedback_at.desc()).all()

    return jsonify({
        "personalization_active": config is not None and config.preference_embedding is not None,
        "liked_count": len(liked),
        "disliked_count": len(disliked),
        "liked_jobs": [
            {
                "title": m.job.title,
                "company": m.job.company,
                "feedback": m.user_feedback,
                "feedback_at": m.feedback_at.isoformat() if m.feedback_at else None
            }
            for m in liked if m.job
        ],
        "disliked_jobs": [
            {
                "title": m.job.title,
                "company": m.job.company,
                "feedback_at": m.feedback_at.isoformat() if m.feedback_at else None
            }
            for m in disliked if m.job
        ],
        "preference_updated_at": (
            config.preference_updated_at.isoformat()
            if config and config.preference_updated_at else None
        )
    })


@agent_bp.route('/agent/status')
@login_required
def agent_status():
    config = AgentConfig.query.filter_by(user_id=current_user.id).first()
    if not config:
        config = AgentConfig(user_id=current_user.id)
        db.session.add(config)
        safe_commit()

    latest_run = AgentRunHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(AgentRunHistory.started_at.desc()).first()

    scheduler = _get_scheduler()
    next_run = scheduler.get_next_run_time()

    return jsonify({
        'is_enabled': config.is_enabled,
        'last_run': latest_run.to_dict() if latest_run else None,
        'next_run': next_run.isoformat() if next_run else None,
        'scheduler_running': scheduler.is_running()
    })
