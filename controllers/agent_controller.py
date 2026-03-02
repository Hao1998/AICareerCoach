"""
Agent Controller

Handles Job Scout Agent configuration, triggering, history, and feedback.
Blueprint: 'agent'
"""

from datetime import datetime

from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user

from models import db, AgentConfig, AgentRunHistory, JobMatch
from job_utils import update_user_preferences

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

        result = _get_scheduler().trigger_manual_run(current_user.id)

        if result['status'] == 'success':
            return jsonify({
                'success': True,
                'message': f"Agent run completed! Found {result['matches_found']} new matches.",
                'jobs_fetched': result['jobs_fetched'],
                'jobs_analyzed': result['jobs_analyzed'],
                'matches_found': result['matches_found']
            })
        else:
            return jsonify({'success': False, 'error': result.get('error', 'Agent run failed')}), 500

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@agent_bp.route('/agent/config', methods=['GET', 'POST'])
@login_required
def agent_config():
    config = AgentConfig.query.filter_by(user_id=current_user.id).first()
    if not config:
        config = AgentConfig(user_id=current_user.id)
        db.session.add(config)
        db.session.commit()

    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form

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

            new_is_enabled = data.get('is_enabled', 'true').lower() in ['true', '1', 'on']
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

            db.session.commit()

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

    return render_template('agent_config.html', config=config, user=current_user)


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
        db.session.commit()

    recent_runs = AgentRunHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(AgentRunHistory.started_at.desc()).limit(10).all()

    recent_matches = JobMatch.query.filter_by(
        user_id=current_user.id, agent_generated=True
    ).order_by(JobMatch.created_at.desc()).limit(10).all()

    next_run = _get_scheduler().get_next_run_time()

    return render_template('agent_dashboard.html', config=config, recent_runs=recent_runs,
                           recent_matches=recent_matches, next_run=next_run, user=current_user)


@agent_bp.route('/agent/matches/<int:run_id>')
@login_required
def agent_run_matches(run_id):
    run = AgentRunHistory.query.filter_by(id=run_id, user_id=current_user.id).first_or_404()
    matches = JobMatch.query.filter_by(
        agent_run_id=run_id, user_id=current_user.id
    ).order_by(JobMatch.match_score.desc()).all()
    return render_template('agent_run_matches.html', run=run, matches=matches, user=current_user)


@agent_bp.route('/agent/feedback/<int:match_id>', methods=['POST'])
@login_required
def agent_match_feedback(match_id):
    try:
        match = JobMatch.query.filter_by(id=match_id, user_id=current_user.id).first_or_404()
        data = request.get_json() if request.is_json else request.form
        feedback = data.get('feedback')

        if feedback not in ['interested', 'not_interested', 'applied']:
            return jsonify({'success': False, 'error': 'Invalid feedback value'}), 400

        match.user_feedback = feedback
        match.feedback_at = datetime.utcnow()
        db.session.commit()

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


@agent_bp.route('/agent/status')
@login_required
def agent_status():
    config = AgentConfig.query.filter_by(user_id=current_user.id).first()
    if not config:
        config = AgentConfig(user_id=current_user.id)
        db.session.add(config)
        db.session.commit()

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
