"""
Job Controller

Handles all job listing, fetching, matching, and CRUD operations.
Blueprint: 'jobs'
"""

import json

from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user

from models import db, JobPosting, JobMatch, Resume, AgentConfig
from job_fetcher import fetch_jobs_from_adzuna
from job_utils import compute_job_embedding, compute_all_job_embeddings, build_job_faiss_index
from services.resume_service import extract_text_from_pdf
from services.llm_service import get_resume_analysis_chain, get_job_matching_chain
from services.job_service import find_matching_jobs

job_bp = Blueprint('jobs', __name__)


@job_bp.route('/jobs')
@login_required
def list_jobs():
    ids_param = request.args.get('ids', '')
    is_chat_filtered = False

    if ids_param:
        try:
            id_list = [int(x.strip()) for x in ids_param.split(',') if x.strip()]
            if id_list:
                jobs_dict = {j.id: j for j in JobPosting.query.filter(JobPosting.id.in_(id_list)).all()}
                jobs = [jobs_dict[jid] for jid in id_list if jid in jobs_dict]
                is_chat_filtered = True
                return render_template('jobs.html', jobs=jobs, user=current_user,
                                       is_chat_filtered=is_chat_filtered)
        except (ValueError, TypeError):
            pass

    jobs = JobPosting.query.filter_by(is_active=True).order_by(JobPosting.posted_date.desc()).all()
    return render_template('jobs.html', jobs=jobs, user=current_user, is_chat_filtered=is_chat_filtered)


@job_bp.route('/jobs/fetch', methods=['GET', 'POST'])
@login_required
def fetch_jobs():
    if request.method == 'POST':
        try:
            keywords = request.form.get('keywords', '').strip() or None
            location = request.form.get('location', '').strip() or None
            max_jobs = int(request.form.get('max_jobs', 50))
            max_days_old = int(request.form.get('max_days_old', 30))

            if max_jobs < 1 or max_jobs > 200:
                return render_template('fetch_jobs.html',
                                       error="Please enter a number between 1 and 200 for max jobs")

            config = AgentConfig.query.filter_by(user_id=current_user.id).first()
            if not config:
                config = AgentConfig(user_id=current_user.id)
                db.session.add(config)

            config.adzuna_location = location
            config.adzuna_max_jobs = max_jobs
            config.adzuna_max_days_old = max_days_old
            db.session.commit()

            stats = fetch_jobs_from_adzuna(keywords=keywords, location=location,
                                           max_jobs=max_jobs, max_days_old=max_days_old)

            if stats['errors'] > 0:
                return render_template('fetch_jobs.html',
                                       error='; '.join(stats['error_messages']), stats=stats)

            return render_template('fetch_jobs.html', success=True, stats=stats)

        except ValueError as e:
            return render_template('fetch_jobs.html', error=str(e))
        except Exception as e:
            return render_template('fetch_jobs.html', error=f"Unexpected error: {str(e)}")

    return render_template('fetch_jobs.html')


@job_bp.route('/api/jobs/fetch', methods=['POST'])
@login_required
def fetch_jobs_api():
    try:
        data = request.get_json() or {}
        keywords = data.get('keywords')
        location = data.get('location')
        max_jobs = int(data.get('max_jobs', 50))
        max_days_old = int(data.get('max_days_old', 30))

        if max_jobs < 1 or max_jobs > 200:
            return jsonify({'success': False, 'error': 'max_jobs must be between 1 and 200'}), 400

        config = AgentConfig.query.filter_by(user_id=current_user.id).first()
        if not config:
            config = AgentConfig(user_id=current_user.id)
            db.session.add(config)

        if location is not None:
            config.adzuna_location = location if location.strip() else None
        if max_jobs:
            config.adzuna_max_jobs = max_jobs
        if max_days_old:
            config.adzuna_max_days_old = max_days_old
        db.session.commit()

        stats = fetch_jobs_from_adzuna(keywords=keywords, location=location,
                                       max_jobs=max_jobs, max_days_old=max_days_old)

        if stats['errors'] > 0:
            return jsonify({'success': False, 'stats': stats,
                            'error': '; '.join(stats['error_messages'])}), 500

        return jsonify({'success': True, 'stats': stats})

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f"Unexpected error: {str(e)}"}), 500


@job_bp.route('/jobs/add', methods=['GET', 'POST'])
@login_required
def add_job():
    if request.method == 'POST':
        job = JobPosting(
            title=request.form['title'],
            company=request.form['company'],
            location=request.form.get('location', ''),
            job_type=request.form.get('job_type', ''),
            description=request.form['description'],
            requirements=request.form.get('requirements', ''),
            salary_range=request.form.get('salary_range', '')
        )
        compute_job_embedding(job)
        db.session.add(job)
        db.session.commit()

        try:
            build_job_faiss_index()
        except Exception as e:
            print(f"Warning: Failed to rebuild job index: {e}")

        flash('Job posted successfully!', 'success')
        return redirect(url_for('jobs.list_jobs'))

    return render_template('add_job.html', user=current_user)


@job_bp.route('/jobs/<int:job_id>')
@login_required
def view_job(job_id):
    job = JobPosting.query.get_or_404(job_id)
    return render_template('view_job.html', job=job, user=current_user)


@job_bp.route('/jobs/rebuild-index', methods=['POST'])
@login_required
def rebuild_job_index():
    try:
        updated_count = compute_all_job_embeddings()
        vectorstore = build_job_faiss_index()

        if vectorstore:
            total_jobs = JobPosting.query.filter_by(is_active=True).count()
            return jsonify({"success": True,
                            "message": f"Successfully rebuilt job index with {total_jobs} jobs",
                            "updated_embeddings": updated_count})
        else:
            return jsonify({"success": False, "message": "No jobs available to build index"}), 400

    except Exception as e:
        return jsonify({"success": False, "message": f"Error rebuilding index: {str(e)}"}), 500


@job_bp.route('/jobs/<int:job_id>/delete', methods=['POST'])
@login_required
def delete_job(job_id):
    job = JobPosting.query.get_or_404(job_id)
    job.is_active = False
    db.session.commit()

    try:
        build_job_faiss_index()
    except Exception as e:
        print(f"Warning: Failed to rebuild job index after deletion: {e}")

    flash('Job deactivated successfully', 'success')
    return redirect(url_for('jobs.list_jobs'))


@job_bp.route('/api/jobs', methods=['GET'])
@login_required
def get_jobs_api():
    jobs = JobPosting.query.filter_by(is_active=True).all()
    return jsonify([job.to_dict() for job in jobs])


@job_bp.route('/api/matches/<int:resume_id>')
@login_required
def get_matches_api(resume_id):
    Resume.query.filter_by(id=resume_id, user_id=current_user.id).first_or_404()
    matches = JobMatch.query.filter_by(resume_id=resume_id, user_id=current_user.id).order_by(
        JobMatch.match_score.desc()
    ).all()
    return jsonify([match.to_dict() for match in matches])


@job_bp.route('/api/jobs/<int:job_id>/match', methods=['POST'])
@login_required
def check_job_match(job_id):
    try:
        job = JobPosting.query.get(job_id)
        if not job or not job.is_active:
            return jsonify({"error": "Job not found or is no longer active"}), 404

        latest_resume = current_user.resumes.filter_by(is_active=True).order_by(
            Resume.uploaded_at.desc()
        ).first()
        if not latest_resume:
            return jsonify({"error": "No resume found. Please upload your resume first."}), 400

        resume_text = extract_text_from_pdf(latest_resume.file_path)

        try:
            analysis_result = get_job_matching_chain().run(
                resume=resume_text[:3000],
                job_title=job.title,
                company=job.company,
                job_description=job.description[:1000],
                job_requirements=job.requirements[:1000] if job.requirements else "Not specified"
            )
            analysis = json.loads(analysis_result)

            existing_match = JobMatch.query.filter_by(
                user_id=current_user.id,
                resume_id=latest_resume.id,
                job_id=job.id
            ).first()

            if existing_match:
                existing_match.match_score = analysis.get('match_score', 0)
                existing_match.matched_skills = json.dumps(analysis.get('matched_skills', []))
                existing_match.gaps = json.dumps(analysis.get('skill_gaps', []))
                existing_match.recommendation = analysis.get('recommendation', '')
                existing_match.resume_filename = latest_resume.filename
            else:
                job_match = JobMatch(
                    user_id=current_user.id,
                    resume_id=latest_resume.id,
                    resume_filename=latest_resume.filename,
                    job_id=job.id,
                    match_score=analysis.get('match_score', 0),
                    matched_skills=json.dumps(analysis.get('matched_skills', [])),
                    gaps=json.dumps(analysis.get('skill_gaps', [])),
                    recommendation=analysis.get('recommendation', '')
                )
                db.session.add(job_match)
            db.session.commit()
            return jsonify(analysis)

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed for job {job.id}: {e}")
            return jsonify({"error": "Failed to parse analysis results. Please try again."}), 500

    except Exception as e:
        print(f"Error in check_job_match: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
