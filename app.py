from flask import Flask, request, render_template, redirect, url_for, jsonify, flash
import os
from werkzeug.utils import secure_filename
import PyPDF2
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_xai import ChatXAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

from langchain.text_splitter import CharacterTextSplitter
from models import db, JobPosting, JobMatch, User, Resume
from form import LoginForm, RegistrationForm
import numpy as np
import json
from datetime import datetime
from job_fetcher import fetch_jobs_from_adzuna
from job_utils import (
    embeddings,
    JOB_VECTOR_INDEX,
    compute_job_embedding,
    compute_all_job_embeddings,
    build_job_faiss_index,
    get_job_faiss_index
)

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)


def perform_qa(query, user_id):
    """Perform Q&A on user-specific resume vector index"""
    user_vector_dir = os.path.join('vector_index', str(user_id))

    if not os.path.exists(user_vector_dir):
        return "Please upload a resume first before asking questions."

    vector_db = FAISS.load_local(user_vector_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = rqa.invoke(query)
    return result['result']


app = Flask(__name__)

# Security configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///career_coach.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)



# Flask-Login configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# File upload configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(JOB_VECTOR_INDEX):
    os.makedirs(JOB_VECTOR_INDEX)

# Create database tables
with app.app_context():
    db.create_all()


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


xai_api_key = os.getenv("XAI_API_KEY")
if not xai_api_key:
    raise RuntimeError("XAI_API_KEY environment variable is not set")

llm = ChatXAI(
    model="grok-3",
    temperature=0,
    api_key=os.getenv('XAI_API_KEY', xai_api_key)  # Set XAI_API_KEY env var or replace here
)

resume_summary_template = """
Role: You are an AI Career Coach.

Task: Given the candidate's resume, provide a comprehensive summary that includes the following key aspects:

- Career Objective
- Skills and Expertise
- Professional Experience
- Educational Background
- Notable Achievements

Instructions:
Provide a concise summary of the resume, focusing on the candidate's skills, experience, and career trajectory. Ensure the summary is well-structured, clear, and highlights the candidate's strengths in alignment with industry standards.

Requirements:
{resume}

"""

resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=resume_summary_template,
)

resume_analysis_chain = LLMChain(
    llm=llm,
    prompt=resume_prompt,
)

preparation_roadmap_template = """
Role: You are an AI Career Coach creating personalized interview preparation roadmaps.

Task: Given a candidate's resume, job posting details, current skill gaps, and a preparation timeline, create a detailed month-by-month preparation roadmap with progressive technical interview questions.

Resume:
{resume}

Job Details:

Title: {job_title}
Company: {company}
Description: {job_description}
Requirements: {job_requirements}

Current Skill Gaps:
{skill_gaps}

Timeline: {timeline_months} months

Create a comprehensive preparation roadmap divided into phases. For a {timeline_months}-month timeline:
- If 3 months: Create 3 phases (Foundation, Intermediate, Advanced)
- If 6 months: Create 4-5 phases with more detailed preparation
- If 9 months: Create 5-6 phases with comprehensive coverage

For each phase, provide:

1. Phase name and duration
2. Specific skills to learn (related to the skill gaps)
3. Recommended learning resources (courses, books, documentation)
4. Hands-on projects to build (that demonstrate the required skills)
5. Milestones to achieve
6. **Technical interview questions** - Generate 4-6 role-specific questions that:
   - Are relevant to the skills being learned in this phase
   - Progress in difficulty from phase to phase (Easy → Medium → Hard)
   - Mix conceptual understanding, practical application, and problem-solving
   - Are realistic questions for this specific role ({job_title})
   - Include the difficulty level and topic area
Also provide an overview and final tips for interview success.
Provide your roadmap in the following JSON format:
{{
    "overview": "Brief overview of the preparation plan and interview strategy",
    "phases": [
        {{
            "phase_name": "Phase name (e.g., 'Foundation Building')",
            "duration": "Time period (e.g., 'Month 1-2' or 'Weeks 1-4')",
            "difficulty_level": "Easy|Medium|Hard",
            "skills": ["skill1", "skill2", "skill3"],
            "resources": ["resource1 with description", "resource2 with description"],
            "projects": ["project1 description", "project2 description"],
            "milestones": ["milestone1", "milestone2"],
            "interview_questions": [
                {{
                    "question": "The technical interview question",
                    "difficulty": "Easy|Medium|Hard",
                    "topic": "Topic area (e.g., 'Data Structures', 'System Design', 'API Design')",
                    "hint": "Brief hint or approach to tackle this question"
                }}
            ]
        }}
    ],
    "final_tips": "Important advice for interview success, including behavioral tips and common pitfalls to avoid"
}}
Make sure the roadmap is:
- Specific to the skill gaps and target role identified
- Realistic for the given timeline
- Actionable with concrete steps
- Progressive from foundational to advanced topics
- Interview questions progress in difficulty and relevance across phases
- Questions reflect real interview scenarios for {job_title} positions
"""
preparation_roadmap_prompt = PromptTemplate(
    input_variables=["resume", "job_title", "company", "job_description", "job_requirements", "skill_gaps",
                     "timeline_months"],
    template=preparation_roadmap_template,
)
preparation_roadmap_chain = LLMChain(
    llm=llm,
    prompt=preparation_roadmap_prompt,
)

# Job matching prompt template
job_matching_template = """
Role: You are an AI Career Coach analyzing job matches.

Task: Given a candidate's resume and a job posting, analyze the match and provide:
1. Match Score (0-100): Overall compatibility percentage
2. Matched Skills: List of candidate's skills that match job requirements
3. Skill Gaps: Skills required by the job that the candidate lacks
4. Recommendation: Brief advice for the candidate

Resume:
{resume}

Job Posting:
Title: {job_title}
Company: {company}
Description: {job_description}
Requirements: {job_requirements}

Provide your analysis in the following JSON format:
{{
    "match_score": <number between 0-100>,
    "matched_skills": ["skill1", "skill2", ...],
    "skill_gaps": ["gap1", "gap2", ...],
    "recommendation": "Your recommendation text here"
}}
"""

job_matching_prompt = PromptTemplate(
    input_variables=["resume", "job_title", "company", "job_description", "job_requirements"],
    template=job_matching_template,
)

job_matching_chain = LLMChain(
    llm=llm,
    prompt=job_matching_prompt,
)


def calculate_embedding_similarity(resume_embedding, job_embedding):
    """Calculate cosine similarity between resume and job embeddings"""
    similarity = np.dot(resume_embedding, job_embedding) / (
            np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
    )
    return float(similarity)


def find_matching_jobs_old(resume_text, top_k=5):
    """[DEPRECATED] Old brute-force method - kept for fallback
    Find top matching jobs for a given resume using naive approach (slow for 1000+ jobs)
    """
    # Get all active jobs
    jobs = JobPosting.query.filter_by(is_active=True).all()

    if not jobs:
        return []

    # Create resume embedding
    resume_embedding = embeddings.embed_query(resume_text)

    matches = []
    for job in jobs:
        # Create job text from description and requirements
        job_text = f"{job.title} {job.description} {job.requirements or ''}"
        job_embedding = embeddings.embed_query(job_text)

        # Calculate similarity
        similarity_score = calculate_embedding_similarity(resume_embedding, job_embedding)

        # Get detailed analysis from LLM
        try:
            analysis_result = job_matching_chain.run(
                resume=resume_text[:3000],  # Limit text size
                job_title=job.title,
                company=job.company,
                job_description=job.description[:1000],
                job_requirements=job.requirements[:1000] if job.requirements else "Not specified"
            )

            # Parse JSON response
            analysis = json.loads(analysis_result)
        except:
            # Fallback if JSON parsing fails
            analysis = {
                "match_score": similarity_score * 100,
                "matched_skills": [],
                "skill_gaps": [],
                "recommendation": "Analysis not available"
            }

        matches.append({
            'job': job,
            'similarity_score': similarity_score,
            'analysis': analysis
        })

    # Sort by similarity score
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)

    return matches[:top_k]


def find_matching_jobs(resume_text, top_k=5, candidate_k=20):
    """
    [OPTIMIZED] Find top matching jobs using two-stage retrieval with FAISS

    Performance: ~60-100x faster than old method for 1000+ jobs
    - Stage 1: Fast FAISS vector search to get top candidate_k jobs (~0.5 seconds)
    - Stage 2: Detailed LLM analysis only for top_k jobs (5 LLM calls instead of 1000!)

    Args:
        resume_text: The resume text to match against
        top_k: Number of final matches to return with full LLM analysis (default: 5)
        candidate_k: Number of candidates to retrieve in stage 1 (default: 20)

    Returns:
        List of job matches with similarity scores and LLM analysis
    """
    try:
        # Stage 1: Fast FAISS vector search
        job_index = get_job_faiss_index()

        if job_index is None:
            print("No job index available, falling back to old method")
            return find_matching_jobs_old(resume_text, top_k)

        # Search for top candidate_k similar jobs using FAISS
        docs_with_scores = job_index.similarity_search_with_score(
            resume_text,
            k=min(candidate_k, job_index.index.ntotal)  # Don't request more than available
        )

        if not docs_with_scores:
            return []

        # Stage 2: Detailed LLM analysis for top_k candidates only
        matches = []
        for idx, (doc, distance) in enumerate(docs_with_scores[:top_k]):
            # Get job from database
            job_id = doc.metadata.get("job_id")
            job = JobPosting.query.get(job_id)

            if not job or not job.is_active:
                continue

            # Convert FAISS distance to similarity score
            # FAISS returns L2 distance, convert to cosine similarity approximation
            # For normalized vectors: similarity ≈ 1 - (distance² / 2)
            similarity_score = 1 - (distance ** 2 / 2)
            similarity_score = max(0, min(1, similarity_score))  # Clamp to [0, 1]

            # Get detailed LLM analysis (only for top_k!)
            try:
                analysis_result = job_matching_chain.run(
                    resume=resume_text[:3000],
                    job_title=job.title,
                    company=job.company,
                    job_description=job.description[:1000],
                    job_requirements=job.requirements[:1000] if job.requirements else "Not specified"
                )

                analysis = json.loads(analysis_result)
            except Exception as e:
                print(f"LLM analysis failed for job {job.id}: {e}")
                # Fallback if JSON parsing fails
                analysis = {
                    "match_score": similarity_score * 100,
                    "matched_skills": [],
                    "skill_gaps": [],
                    "recommendation": "Analysis not available"
                }

            matches.append({
                'job': job,
                'similarity_score': similarity_score,
                'analysis': analysis
            })

        return matches

    except Exception as e:
        print(f"Error in optimized job matching: {e}")
        # Fallback to old method if something goes wrong
        print("Falling back to old matching method")
        return find_matching_jobs_old(resume_text, top_k)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ============================================

# Authentication Routes

# ============================================



@app.route('/login', methods=['GET', 'POST'])

def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()

        login_user(user, remember=form.remember_me.data)
        flash(f'Welcome back, {user.username}!', 'success')

        # Redirect to next page or index
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', form=form)




@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            full_name=form.full_name.data
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()


        # Create user-specific directories
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user.id))
        user_vector_dir = os.path.join('vector_index', str(user.id))
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_vector_dir, exist_ok=True)
        flash('Congratulations, you are now registered! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    resumes = current_user.resumes.filter_by(is_active=True).order_by(Resume.uploaded_at.desc()).all()
    recent_matches = current_user.job_matches.order_by(JobMatch.created_at.desc()).limit(10).all()

    return render_template('profile.html',
                           user=current_user,
                           resumes=resumes,
                           recent_matches=recent_matches)

# ============================================
# Main Application Routes
# ============================================

@app.route('/')
@login_required
def index():
    return render_template('index.html', user=current_user)


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    if file:
        # Save the uploaded file in user-specific directory
        original_filename = secure_filename(file.filename)
        # Generate unique filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{original_filename}"

        # User-specific upload directory
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        os.makedirs(user_upload_dir, exist_ok=True)

        file_path = os.path.join(user_upload_dir, filename)
        file.save(file_path)

        # Create Resume record
        resume = Resume(
            user_id=current_user.id,
            filename=filename,
            original_filename=original_filename,
            file_path=file_path
        )
        db.session.add(resume)
        db.session.flush()  # Get resume.id

        # Extract text from the PDF
        resume_text = extract_text_from_pdf(file_path)
        splitted_text = text_splitter.split_text(resume_text)

        # User-specific vector index
        user_vector_dir = os.path.join('vector_index', str(current_user.id))
        os.makedirs(user_vector_dir, exist_ok=True)

        vectorstore = FAISS.from_texts(splitted_text, embeddings)
        vectorstore.save_local(user_vector_dir)

        # Run resume analysis using the LLM chain
        resume_analysis = resume_analysis_chain.run(resume=resume_text)

        # Save analysis to Resume record
        resume.analysis = resume_analysis
        db.session.commit()

        # Find matching jobs
        matching_jobs = find_matching_jobs(resume_text, top_k=5)

        # Save matches to database with user_id and resume_id
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
        db.session.commit()

        flash('Resume uploaded and analyzed successfully!', 'success')
        return render_template('results.html',
                               resume_analysis=resume_analysis,
                               matching_jobs=matching_jobs,
                               filename=original_filename,
                               user=current_user)


@app.route('/ask', methods=['GET', 'POST'])
@login_required
def ask_query():
    if request.method == 'POST':
        query = request.form['query']
        result = perform_qa(query, current_user.id)
        return render_template('qa_results.html', query=query, result=result, user=current_user)
    return render_template('ask.html', user=current_user)


@app.route('/jobs')
@login_required
def list_jobs():
    """Display all active job postings"""
    jobs = JobPosting.query.filter_by(is_active=True).order_by(JobPosting.posted_date.desc()).all()
    return render_template('jobs.html', jobs=jobs, user=current_user)


@app.route('/jobs/fetch', methods=['GET', 'POST'])
@login_required
def fetch_jobs():
    """Fetch jobs from Adzuna API"""
    if request.method == 'POST':
        try:
            # Get parameters from form
            keywords = request.form.get('keywords', '').strip() or None
            location = request.form.get('location', '').strip() or None
            max_jobs = int(request.form.get('max_jobs', 50))
            max_days_old = int(request.form.get('max_days_old', 30))

            # Validate max_jobs
            if max_jobs < 1 or max_jobs > 200:
                return render_template('fetch_jobs.html',
                                       error="Please enter a number between 1 and 200 for max jobs")

            # Fetch jobs from Adzuna
            stats = fetch_jobs_from_adzuna(
                keywords=keywords,
                location=location,
                max_jobs=max_jobs,
                max_days_old=max_days_old
            )

            # Check if there were errors
            if stats['errors'] > 0:
                error_msg = '; '.join(stats['error_messages'])
                return render_template('fetch_jobs.html',
                                       error=error_msg,
                                       stats=stats)

            # Success - redirect to jobs list with success message
            return render_template('fetch_jobs.html',
                                   success=True,
                                   stats=stats)

        except ValueError as e:
            return render_template('fetch_jobs.html',
                                   error=str(e))
        except Exception as e:
            return render_template('fetch_jobs.html',
                                   error=f"Unexpected error: {str(e)}")

    # GET request - show form
    return render_template('fetch_jobs.html')


@app.route('/api/jobs/fetch', methods=['POST'])
@login_required
def fetch_jobs_api():
    """API endpoint to fetch jobs from Adzuna"""
    try:
        data = request.get_json() or {}
        keywords = data.get('keywords')
        location = data.get('location')
        max_jobs = int(data.get('max_jobs', 50))
        max_days_old = int(data.get('max_days_old', 30))
        # Validate max_jobs
        if max_jobs < 1 or max_jobs > 200:
            return jsonify({
                'success': False,
                'error': 'max_jobs must be between 1 and 200'
            }), 400
        # Fetch jobs from Adzuna
        stats = fetch_jobs_from_adzuna(
            keywords=keywords,
            location=location,
            max_jobs=max_jobs,
            max_days_old=max_days_old
        )
        if stats['errors'] > 0:
            return jsonify({
                'success': False,
                'stats': stats,
                'error': '; '.join(stats['error_messages'])
            }), 500
        return jsonify({
            'success': True,
            'stats': stats
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Unexpected error: {str(e)}"
        }), 500


@app.route('/check-resume-status', methods=['GET'])
@login_required
def check_resume_status():
    """Check if user has already uploaded a resume (vector index exists)"""
    try:
        user_vector_dir = os.path.join('vector_index', str(current_user.id))
        if os.path.exists(user_vector_dir):
            return jsonify({"hasResume": True})
        else:
            return jsonify({"hasResume": False})
    except Exception as e:
        return jsonify({"hasResume": False, "error": str(e)})


@app.route('/api/prepare-roadmap', methods=['POST'])
@login_required
def prepare_roadmap():
    """Generate a preparation roadmap for a specific job"""
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        timeline_months = data.get('timeline_months')
        if not job_id or not timeline_months:
            return jsonify({
                "success": False,
                "error": "Missing job_id or timeline_months"
            }), 400
        # Get the job details
        job = JobPosting.query.get(job_id)
        if not job:
            return jsonify({
                "success": False,
                "error": "Job not found"
            }), 404
        # Get the latest resume for current user
        latest_resume = current_user.resumes.filter_by(is_active=True).order_by(Resume.uploaded_at.desc()).first()
        if not latest_resume:
            return jsonify({
                "success": False,
                "error": "No resume uploaded"
            }), 400

        # Extract resume text
        resume_text = extract_text_from_pdf(latest_resume.file_path)
        # Get or create job match to find skill gaps
        job_match = JobMatch.query.filter_by(
            user_id=current_user.id,
            resume_id=latest_resume.id,
            job_id=job_id
        ).first()
        skill_gaps = []
        if job_match and job_match.skill_gaps:
            skill_gaps = job_match.skill_gaps
        else:
            # If no existing match, run quick analysis to get skill gaps
            try:
                analysis_result = job_matching_chain.run(
                    resume=resume_text[:3000],
                    job_title=job.title,
                    company=job.company,
                    job_description=job.description[:1000],
                    job_requirements=job.requirements[:1000] if job.requirements else "Not specified"
                )
                analysis = json.loads(analysis_result)
                skill_gaps = analysis.get('skill_gaps', [])
            except:
                skill_gaps = ["General skill development needed"]
        # Convert skill gaps list to string for prompt
        skill_gaps_str = ", ".join(skill_gaps) if skill_gaps else "No specific gaps identified"
        # Generate the preparation roadmap
        roadmap_result = preparation_roadmap_chain.run(
            resume=resume_text[:3000],
            job_title=job.title,
            company=job.company,
            job_description=job.description[:1500],
            job_requirements=job.requirements[:1500] if job.requirements else "Not specified",
            skill_gaps=skill_gaps_str,
            timeline_months=timeline_months
        )
        # Parse the JSON response
        roadmap = json.loads(roadmap_result)
        return jsonify({
            "success": True,
            "roadmap": roadmap,
            "job_title": job.title,
            "timeline_months": timeline_months
        })
    except json.JSONDecodeError as e:
        return jsonify({
            "success": False,
            "error": f"Failed to parse roadmap: {str(e)}"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error generating roadmap: {str(e)}"
        }), 500


@app.route('/jobs/add', methods=['GET', 'POST'])
@login_required
def add_job():
    """Add a new job posting"""
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

        # Pre-compute embedding for the new job
        compute_job_embedding(job)

        db.session.add(job)
        db.session.commit()

        # Rebuild FAISS index to include new job
        try:
            build_job_faiss_index()
        except Exception as e:
            print(f"Warning: Failed to rebuild job index: {e}")

        flash('Job posted successfully!', 'success')
        return redirect(url_for('list_jobs'))
    return render_template('add_job.html', user=current_user)


@app.route('/jobs/<int:job_id>')
@login_required
def view_job(job_id):
    """View a specific job posting"""
    job = JobPosting.query.get_or_404(job_id)
    return render_template('view_job.html', job=job, user=current_user)


@app.route('/find-matching-jobs', methods=['POST'])
@login_required
def find_matching_jobs_endpoint():
    """Find matching jobs using the latest uploaded resume"""
    try:
        # Get latest resume for current user
        latest_resume = current_user.resumes.filter_by(is_active=True).order_by(Resume.uploaded_at.desc()).first()

        if not latest_resume:
            return jsonify({"error": "No resume found. Please upload your resume first."}), 400

        resume_text = extract_text_from_pdf(latest_resume.file_path)

        resume_analysis = resume_analysis_chain.run(resume=resume_text)

        matching_jobs = find_matching_jobs(resume_text, top_k=5)

        # Store the results in session or temporary storage
        # For now, we'll return success and redirect to results page
        return render_template('results.html',
                               resume_analysis=resume_analysis,
                               matching_jobs=matching_jobs,
                               filename=latest_resume.original_filename,
                               user=current_user)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/jobs/rebuild-index', methods=['POST'])
@login_required
def rebuild_job_index():
    """Rebuild FAISS index for all job embeddings (admin endpoint)"""
    try:
        # Compute embeddings for all jobs without embeddings
        updated_count = compute_all_job_embeddings()

        # Rebuild FAISS index
        vectorstore = build_job_faiss_index()

        if vectorstore:
            total_jobs = JobPosting.query.filter_by(is_active=True).count()
            return jsonify({
                "success": True,
                "message": f"Successfully rebuilt job index with {total_jobs} jobs",
                "updated_embeddings": updated_count
            })
        else:
            return jsonify({
                "success": False,
                "message": "No jobs available to build index"
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error rebuilding index: {str(e)}"
        }), 500


@app.route('/jobs/<int:job_id>/delete', methods=['POST'])
@login_required
def delete_job(job_id):
    """Deactivate a job posting"""
    job = JobPosting.query.get_or_404(job_id)
    job.is_active = False
    db.session.commit()

    # Rebuild index after deleting job
    try:
        build_job_faiss_index()
    except Exception as e:
        print(f"Warning: Failed to rebuild job index after deletion: {e}")

    flash('Job deactivated successfully', 'success')
    return redirect(url_for('list_jobs'))


@app.route('/api/jobs', methods=['GET'])
@login_required
def get_jobs_api():
    """API endpoint to get all jobs"""
    jobs = JobPosting.query.filter_by(is_active=True).all()
    return jsonify([job.to_dict() for job in jobs])


@app.route('/api/matches/<int:resume_id>')
@login_required
def get_matches_api(resume_id):
    """API endpoint to get matches for a resume"""
    # Ensure the resume belongs to current user
    resume = Resume.query.filter_by(id=resume_id, user_id=current_user.id).first_or_404()
    matches = JobMatch.query.filter_by(resume_id=resume_id, user_id=current_user.id).order_by(JobMatch.match_score.desc()).all()
    return jsonify([match.to_dict() for match in matches])


if __name__ == "__main__":
    # Build job index on startup if it doesn't exist
    try:
        if not os.path.exists(os.path.join(JOB_VECTOR_INDEX, "index.faiss")):
            print("Job index not found, building initial index...")
            build_job_faiss_index()
            print("Job index built successfully")
    except Exception as e:
        print(f"Warning: Could not build job index on startup: {e}")

    app.run(host='0.0.0.0', port=5001)
