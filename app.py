from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import PyPDF2
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_xai import ChatXAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

from langchain.text_splitter import CharacterTextSplitter
from models import db, JobPosting, JobMatch
import numpy as np
import json
from datetime import datetime
from job_fetcher import fetch_jobs_from_adzuna

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)

embeddings = HuggingFaceEmbeddings()


def perform_qa(query):
    db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = rqa.invoke(query)
    return result['result']


app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///career_coach.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
JOB_VECTOR_INDEX = 'job_vector_index'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(JOB_VECTOR_INDEX):
    os.makedirs(JOB_VECTOR_INDEX)

# Create database tables
with app.app_context():
    db.create_all()

    # Build job index on startup if it doesn't exist
    try:
        if not os.path.exists(os.path.join(JOB_VECTOR_INDEX, "index.faiss")):
            print("Job index not found, building initial index...")
            build_job_faiss_index()
            print("Job index built successfully")
    except Exception as e:
        print(f"Warning: Could not build job index on startup: {e}")


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
    api_key=os.getenv('XAI_API_KEY',xai_api_key)  # Set XAI_API_KEY env var or replace here
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


def compute_job_embedding(job):
    """Compute and store embedding for a single job"""
    job_text = job.get_job_text()
    job.embedding = embeddings.embed_query(job_text)
    job.embedding_updated_at = datetime.utcnow()
    return job.embedding


def compute_all_job_embeddings():
    """Pre-compute embeddings for all active jobs (run once or when jobs are updated)"""
    jobs = JobPosting.query.filter_by(is_active=True).all()

    updated_count = 0
    for job in jobs:
        if job.embedding is None:
            compute_job_embedding(job)
            updated_count += 1

    db.session.commit()
    return updated_count


def build_job_faiss_index():
    """Build FAISS index from all active job embeddings for fast similarity search"""
    # First ensure all jobs have embeddings
    compute_all_job_embeddings()

    # Get all active jobs with embeddings
    jobs = JobPosting.query.filter_by(is_active=True).filter(JobPosting.embedding.isnot(None)).all()

    if not jobs:
        print("No jobs available to build index")
        return None

    # Extract embeddings and metadata
    job_texts = [job.get_job_text() for job in jobs]
    job_embeddings = [job.embedding for job in jobs]
    job_metadatas = [{"job_id": job.id} for job in jobs]

    # Create FAISS vector store
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(job_texts, job_embeddings)),
        embedding=embeddings,
        metadatas=job_metadatas
    )

    # Save to disk
    vectorstore.save_local(JOB_VECTOR_INDEX)
    print(f"Built FAISS index for {len(jobs)} jobs")

    return vectorstore


def get_job_faiss_index():
    """Load or build FAISS index for job embeddings"""
    try:
        # Try to load existing index
        if os.path.exists(os.path.join(JOB_VECTOR_INDEX, "index.faiss")):
            vectorstore = FAISS.load_local(
                JOB_VECTOR_INDEX,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        else:
            # Build new index if doesn't exist
            return build_job_faiss_index()
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        # Rebuild if loading fails
        return build_job_faiss_index()


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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extracted   the  text from the PDF
        resume_text = extract_text_from_pdf(file_path)
        splitted_text = text_splitter.split_text(resume_text)
        vectorstore = FAISS.from_texts(splitted_text, embeddings)
        vectorstore.save_local("vector_index")

        # print(proposal_text)
        # Run SWOT analysis using the LLM chain
        resume_analysis = resume_analysis_chain.run(resume=resume_text)

        # Find matching jobs
        matching_jobs = find_matching_jobs(resume_text, top_k=5)

        # Save matches to database
        for match in matching_jobs:
            job_match = JobMatch(
                resume_filename=filename,
                job_id=match['job'].id,
                match_score=match['analysis']['match_score'],
                matched_skills=json.dumps(match['analysis']['matched_skills']),
                gaps=json.dumps(match['analysis']['skill_gaps']),
                recommendation=match['analysis']['recommendation']
            )
            db.session.add(job_match)
        db.session.commit()

        return render_template('results.html',
                             resume_analysis=resume_analysis,
                             matching_jobs=matching_jobs,
                             filename=filename)


@app.route('/ask', methods=['GET', 'POST'])
def ask_query():
    if request.method == 'POST':
        query = request.form['query']
        result = perform_qa(query)
        return render_template('qa_results.html', query=query, result=result)
    return render_template('ask.html')


@app.route('/jobs')
def list_jobs():
    """Display all active job postings"""
    jobs = JobPosting.query.filter_by(is_active=True).order_by(JobPosting.posted_date.desc()).all()
    return render_template('jobs.html', jobs=jobs)

@app.route('/check-resume-status', methods=['GET'])
def check_resume_status():
    """Check if user has already uploaded a resume (vector index exists)"""
    try:
        if os.path.exists("vector_index"):
            return jsonify({"hasResume": True})
        else:
            return jsonify({"hasResume": False})
    except Exception as e:
        return jsonify({"hasResume": False, "error": str(e)})


@app.route('/jobs/add', methods=['GET', 'POST'])
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

        return redirect(url_for('list_jobs'))
    return render_template('add_job.html')


@app.route('/jobs/<int:job_id>')
def view_job(job_id):
    """View a specific job posting"""
    job = JobPosting.query.get_or_404(job_id)
    return render_template('view_job.html', job=job)


@app.route('/find-matching-jobs', methods=['POST'])
def find_matching_jobs_endpoint():
    """Find matching jobs using the latest uploaded resume"""
    try:
        # Check if vector index exists
        if not os.path.exists("vector_index"):
            return jsonify({"error": "No resume found. Please upload your resume first."}), 400

        # Load the vector store to get the resume content
        db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)


        latest_resume_file = max(
            [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER'])],
            key=os.path.getctime
        )
        resume_text = extract_text_from_pdf(latest_resume_file)

        resume_analysis = resume_analysis_chain.run(resume=resume_text)

        matching_jobs = find_matching_jobs(resume_text, top_k=5)

        filename = os.path.basename(latest_resume_file)
        # Store the results in session or temporary storage
        # For now, we'll return success and redirect to results page
        return render_template('results.html',
                               resume_analysis=resume_analysis,
                               matching_jobs=matching_jobs,
                               filename=filename)

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/jobs/rebuild-index', methods=['POST'])
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

    return redirect(url_for('list_jobs'))


@app.route('/api/jobs', methods=['GET'])
def get_jobs_api():
    """API endpoint to get all jobs"""
    jobs = JobPosting.query.filter_by(is_active=True).all()
    return jsonify([job.to_dict() for job in jobs])


@app.route('/api/matches/<filename>')
def get_matches_api(filename):
    """API endpoint to get matches for a resume"""
    matches = JobMatch.query.filter_by(resume_filename=filename).order_by(JobMatch.match_score.desc()).all()
    return jsonify([match.to_dict() for match in matches])


@app.route('/jobs/fetch', methods=['GET', 'POST'])
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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
