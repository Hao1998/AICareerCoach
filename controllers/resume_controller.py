"""
Resume Controller

Handles resume upload, Q&A, and job matching triggered from resume actions.
Blueprint: 'resume'
"""

import os
import json
from datetime import datetime

import magic
from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from flask_limiter.errors import RateLimitExceeded
from werkzeug.utils import secure_filename
from langchain_community.vectorstores import FAISS

ALLOWED_MIME_TYPES = {'application/pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

from models import db, Resume, JobMatch, JobPosting
from services.db_lock import safe_commit, safe_flush
from jobs.utils import get_embeddings
from services.resume_service import extract_text_from_pdf, get_resume_text, perform_qa, text_splitter, invalidate_resume_cache
from services.llm_service import get_resume_analysis_chain, run_job_matching, get_preparation_roadmap_chain, invoke_chain_with_retry, RETRYABLE_ERRORS
from services.input_guard import scan_with_llm
from services.job_service import find_matching_jobs
from factory import limiter
from schemas.request_schemas import RoadmapRequest
from schemas.validate import validate_json

resume_bp = Blueprint('resume', __name__)


@resume_bp.errorhandler(RateLimitExceeded)
def handle_rate_limit(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429


@resume_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('auth.index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('auth.index'))

    if file:
        # Validate size — seek to end, read position, rewind
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        if size > MAX_FILE_SIZE:
            flash('File too large (max 10MB)', 'error')
            return redirect(url_for('auth.index'))

        # Validate MIME by sniffing file content, not by extension
        header = file.read(2048)
        file.seek(0)
        try:
            mime = magic.from_buffer(header, mime=True)
        except Exception:
            flash('Unable to verify file type. Please upload a valid PDF.', 'error')
            return redirect(url_for('auth.index'))
        if mime not in ALLOWED_MIME_TYPES:
            flash('Only PDF files are allowed', 'error')
            return redirect(url_for('auth.index'))

        original_filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{original_filename}"

        user_upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(current_user.id))
        os.makedirs(user_upload_dir, exist_ok=True)

        file_path = os.path.join(user_upload_dir, filename)
        file.save(file_path)

        resume = Resume(
            user_id=current_user.id,
            filename=filename,
            original_filename=original_filename,
            file_path=file_path
        )
        db.session.add(resume)
        safe_flush()

        resume_text = extract_text_from_pdf(file_path)

        # LLM-as-Judge guard: catch sophisticated injections that bypass regex
        guard = scan_with_llm(resume_text, context="resume")
        if not guard["safe"]:
            os.remove(file_path)
            db.session.rollback()
            flash('Your resume contains content that cannot be processed. Please upload a valid resume.', 'error')
            return redirect(url_for('auth.index'))

        # Delete stale match results so they're recalculated against the new resume.
        # match_score is NOT NULL so we can't set it to None; deletion is safer.
        JobMatch.query.filter_by(user_id=current_user.id).delete()

        resume.text_content = resume_text  # cache so future requests skip PDF I/O
        splitted_text = text_splitter.split_text(resume_text)

        user_vector_dir = os.path.join('vector_index', str(current_user.id))
        os.makedirs(user_vector_dir, exist_ok=True)

        vectorstore = FAISS.from_texts(splitted_text, get_embeddings())
        vectorstore.save_local(user_vector_dir)
        invalidate_resume_cache(current_user.id)

        try:
            resume_analysis = invoke_chain_with_retry(get_resume_analysis_chain(), resume=resume_text)
        except RETRYABLE_ERRORS:
            flash('AI service is temporarily unavailable. Resume saved — analysis will be available shortly.', 'warning')
            safe_commit()
            return redirect(url_for('auth.index'))

        resume.analysis = resume_analysis
        safe_commit()

        matching_jobs = find_matching_jobs(resume_text, top_k=5)

        for match in matching_jobs:
            job_match = JobMatch(
                user_id=current_user.id,
                resume_id=resume.id,
                resume_filename=filename,
                job_id=match['job'].id,
                match_score=match['analysis']['match_score'],
                matched_skills=json.dumps(match['analysis']['matched_skills']),
                gaps=json.dumps(match['analysis']['skill_gaps']),
                recommendation=match['analysis']['recommendation']
            )
            db.session.add(job_match)
        safe_commit()

        flash('Resume uploaded and analyzed successfully!', 'success')
        return render_template('results.html',
                               resume_analysis=resume_analysis,
                               matching_jobs=matching_jobs,
                               filename=original_filename,
                               user=current_user)


@resume_bp.route('/ask', methods=['GET', 'POST'])
@login_required
def ask_query():
    if request.method == 'POST':
        query = request.form['query']
        result = perform_qa(query, current_user.id)
        return render_template('qa_results.html', query=query, result=result, user=current_user)
    return render_template('ask.html', user=current_user)


@resume_bp.route('/check-resume-status', methods=['GET'])
@login_required
def check_resume_status():
    try:
        user_vector_dir = os.path.join('vector_index', str(current_user.id))
        return jsonify({"hasResume": os.path.exists(user_vector_dir)})
    except Exception as e:
        return jsonify({"hasResume": False, "error": str(e)})


@resume_bp.route('/find-matching-jobs', methods=['POST'])
@login_required
@limiter.limit("5 per minute; 50 per day")
def find_matching_jobs_endpoint():
    try:
        latest_resume = current_user.resumes.filter_by(is_active=True).order_by(
            Resume.uploaded_at.desc()
        ).first()

        if not latest_resume:
            return jsonify({"error": "No resume found. Please upload your resume first."}), 400

        resume_text = get_resume_text(latest_resume)
        resume_analysis = invoke_chain_with_retry(get_resume_analysis_chain(), resume=resume_text)
        matching_jobs = find_matching_jobs(resume_text, top_k=5)

        return render_template('results.html',
                               resume_analysis=resume_analysis,
                               matching_jobs=matching_jobs,
                               filename=latest_resume.original_filename,
                               user=current_user)
    except RETRYABLE_ERRORS:
        return jsonify({"error": "Our AI service is temporarily unavailable. Please try again in a few minutes."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@resume_bp.route('/api/prepare-roadmap', methods=['POST'])
@login_required
@limiter.limit("5 per minute; 30 per day")
@validate_json(RoadmapRequest)
def prepare_roadmap(validated: RoadmapRequest):
    try:
        job_id = validated.job_id
        timeline_months = validated.timeline_months

        job = JobPosting.query.get(job_id)
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404

        latest_resume = current_user.resumes.filter_by(is_active=True).order_by(
            Resume.uploaded_at.desc()
        ).first()
        if not latest_resume:
            return jsonify({"success": False, "error": "No resume uploaded"}), 400

        resume_text = get_resume_text(latest_resume)

        job_match = JobMatch.query.filter_by(
            user_id=current_user.id,
            resume_id=latest_resume.id,
            job_id=job_id
        ).first()

        skill_gaps = []
        if job_match and job_match.gaps:
            skill_gaps = json.loads(job_match.gaps) if job_match.gaps else []
        else:
            try:
                analysis_result = run_job_matching(
                    resume=resume_text[:3000],
                    job_title=job.title,
                    company=job.company,
                    job_description=job.description[:1000],
                    job_requirements=job.requirements[:1000] if job.requirements else "Not specified",
                )
                skill_gaps = analysis_result.skill_gaps
            except Exception:
                skill_gaps = ["General skill development needed"]

        skill_gaps_str = ", ".join(skill_gaps) if skill_gaps else "No specific gaps identified"

        roadmap_result = invoke_chain_with_retry(
            get_preparation_roadmap_chain(),
            resume=resume_text[:3000],
            job_title=job.title,
            company=job.company,
            job_description=job.description[:1500],
            job_requirements=job.requirements[:1500] if job.requirements else "Not specified",
            skill_gaps=skill_gaps_str,
            timeline_months=timeline_months
        )

        roadmap = json.loads(roadmap_result)
        return jsonify({"success": True, "roadmap": roadmap, "job_title": job.title,
                        "timeline_months": timeline_months})

    except RETRYABLE_ERRORS:
        return jsonify({"success": False, "error": "Our AI service is temporarily unavailable. Please try again in a few minutes."}), 503
    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"Failed to parse roadmap: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"Error generating roadmap: {str(e)}"}), 500
