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


def find_matching_jobs(resume_text, top_k=5):
    """Find top matching jobs for a given resume"""
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
        db.session.add(job)
        db.session.commit()
        return redirect(url_for('list_jobs'))
    return render_template('add_job.html')


@app.route('/jobs/<int:job_id>')
def view_job(job_id):
    """View a specific job posting"""
    job = JobPosting.query.get_or_404(job_id)
    return render_template('view_job.html', job=job)


@app.route('/jobs/<int:job_id>/delete', methods=['POST'])
def delete_job(job_id):
    """Deactivate a job posting"""
    job = JobPosting.query.get_or_404(job_id)
    job.is_active = False
    db.session.commit()
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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
